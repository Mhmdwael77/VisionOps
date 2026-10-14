"""Verify persisted artifacts, served byte ranges, and desktop/mobile browser behavior."""
from pathlib import Path
import csv
import json
import sys
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visionops.vision.geometry import Point, point_in_polygon
from visionops.vision.retail_pipeline import write_json, write_report


def main() -> None:
    output = Path("outputs/retail_store_test")
    summary = json.loads((output / "summary.json").read_text())
    events = json.loads((output / "events.json").read_text())["events"]
    with (output / "events.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(events)
    for row, event in zip(csv_rows, events):
        assert row["id"] == str(event["id"]) and row["track_id"] == str(event["track_id"])
        assert float(row["timestamp"]) == event["timestamp"] and row["event_type"] == event["event_type"]
        if event.get("dwell_seconds") is not None:
            assert float(row["dwell_seconds"]) == event["dwell_seconds"]
    assert len({(e["line_name"], e["track_id"], e["event_type"]) for e in events if "line_name" in e}) == sum("line_name" in e for e in events)
    visits = json.loads((output / "zone_visits.json").read_text())
    for visit in visits:
        assert abs(visit["dwell_seconds"] - (visit["exited_at"]-visit["entered_at"])) < 1e-7
    rows = [json.loads(line) for line in (output / "tracks.jsonl").read_text().splitlines()]
    occupancy = json.loads((output / "occupancy.json").read_text())
    assert len(rows) == len(occupancy) == summary["source"]["decoded_frames"]
    assert summary["tracker"]["implementation"] == "BYTETracker"
    width, height = summary["source"]["width"], summary["source"]["height"]
    for tracks, counts in zip(rows, occupancy):
        for zone in summary["configuration"]["zones"]:
            polygon = [Point(x*width, y*height) for x, y in zone["polygon"]]
            expected = sum(point_in_polygon(Point((d["bbox"][0]+d["bbox"][2])/2, d["bbox"][3]), polygon)
                           for d in tracks["detections"] if d["track_id"] is not None)
            assert counts["zones"][zone["name"]] == expected
    base = "http://localhost:8000/api/retail-test"
    with urlopen(base) as response:
        served = json.load(response)
    assert served["summary"]["run_id"] == summary["run_id"]
    assert len(served["events"]) == len(events)
    with urlopen(Request(base+"/artifacts/annotated.mp4", headers={"Range": "bytes=0-1023"})) as response:
        assert response.status == 206 and len(response.read()) == 1024
    with urlopen(base+"/config") as response:
        config = json.load(response)
    # Round-trip the unchanged configuration without marking the line reviewed.
    request = Request(base+"/config", data=json.dumps(config).encode(), method="PUT", headers={"Content-Type": "application/json"})
    with urlopen(request) as response:
        assert json.load(response) == config

    from playwright.sync_api import sync_playwright
    errors = []
    browser_checks = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for name, viewport in [("desktop", {"width": 1440, "height": 1100}), ("mobile", {"width": 390, "height": 844})]:
            page = browser.new_page(viewport=viewport)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto("http://localhost:5173", wait_until="networkidle")
            page.get_by_role("button", name="Retail Test", exact=True).click()
            page.locator(".test-stat span").get_by_text("Line crossings", exact=True).wait_for()
            player = page.locator("video")
            player.evaluate("video => video.play()")
            page.wait_for_function("document.querySelector('video').currentTime > 0.3")
            player.evaluate("video => video.pause()")
            video_data = player.evaluate("video => ({width:video.videoWidth,height:video.videoHeight,duration:video.duration,time:video.currentTime,error:video.error?.message})")
            assert video_data["width"] == width and video_data["height"] == height
            assert abs(video_data["duration"]-summary["source"]["duration_seconds"]) < 0.1
            assert not video_data.get("error")
            pixels = player.evaluate("""video => {
                const canvas=document.createElement('canvas'); canvas.width=80; canvas.height=45;
                const context=canvas.getContext('2d'); context.drawImage(video,0,0,80,45);
                return Array.from(context.getImageData(0,0,80,45).data);
            }""")
            assert np.std(np.array(pixels).reshape(-1,4)[:,:3]) > 15
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(output/f"dashboard_{name}.png"), full_page=True)
            page.screenshot(path=str(output/f"dashboard_{name}_viewport.png"))
            page.get_by_role("button", name="Source", exact=True).click()
            page.wait_for_function("document.querySelector('.evidence-image').complete && document.querySelector('.evidence-image').naturalWidth > 0")
            assert "_source.jpg" in page.locator(".evidence-image").get_attribute("src")
            page.get_by_role("button", name="Annotated", exact=True).click()
            page.get_by_label("Event filter").select_option("line")
            assert page.locator("tbody tr").count() == summary["analytics"]["entries"] + summary["analytics"]["exits"]
            first_line = next(event for event in events if "line_name" in event)
            page.locator("tbody .time-link").first.click()
            page.wait_for_function("expected => Math.abs(document.querySelector('video').currentTime-expected)<0.15", arg=first_line["timestamp"])
            page.get_by_role("button", name="Geometry", exact=True).click()
            page.get_by_alt_text("Retail store geometry reference").wait_for()
            if name == "desktop":
                # Exercise dragging in a local draft; leave persisted geometry untouched.
                handle = page.locator(".geometry-handle").first.bounding_box()
                assert handle
                page.mouse.move(handle["x"]+handle["width"]/2, handle["y"]+handle["height"]/2)
                page.mouse.down()
                page.mouse.move(handle["x"]+handle["width"]/2+12, handle["y"]+handle["height"]/2+4, steps=4)
                page.mouse.up()
                assert page.get_by_role("button", name="Save Geometry").is_enabled()
                assert not page.get_by_label("Entrance location and direction reviewed").is_checked()
                page.get_by_label("Geometry region").select_option("0")
                vertex = page.locator(".geometry-handle").first
                previous_x = float(vertex.get_attribute("cx"))
                handle = vertex.bounding_box()
                page.mouse.move(handle["x"]+handle["width"]/2, handle["y"]+handle["height"]/2)
                page.mouse.down()
                page.mouse.move(handle["x"]+handle["width"]/2+8, handle["y"]+handle["height"]/2+4, steps=4)
                page.mouse.up()
                assert float(vertex.get_attribute("cx")) != previous_x
                assert not page.get_by_label("Cashier and shelf polygons reviewed").is_checked()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(output/f"geometry_{name}.png"), full_page=True)
            browser_checks[name] = {"video": video_data, "nonblank_video_pixels": True, "no_horizontal_overflow": True,
                                    "geometry_preview": True, "line_and_polygon_draft_drag": name == "desktop",
                                    "event_filter_and_video_seek": True, "source_annotated_evidence_toggle": True}
            page.close()
        browser.close()
    assert not errors, errors
    verification = {"run_id": summary["run_id"], "events_csv_json_values_match": True,
                    "actual_bytetrack_implementation": True,
                    "zone_occupancy_recomputed_all_frames": True, "video_time_dwell_recomputed_all_visits": True,
                    "duplicate_line_events_absent": True, "api_matches_run": True, "video_http_range": True,
                    "config_round_trip": True, "browser": browser_checks, "browser_errors": errors}
    write_json(output / "verification.json", verification)
    summary["verification"].update({key: value for key, value in verification.items() if key not in {"run_id", "browser"}})
    summary["verification"]["desktop_mobile_video_playback"] = True
    summary["verification"]["desktop_mobile_geometry_preview"] = True
    summary["verification"]["desktop_geometry_draft_drag"] = True
    summary["verification"]["desktop_mobile_event_filter_and_video_seek"] = True
    summary["verification"]["desktop_mobile_evidence_toggle"] = True
    write_json(output / "summary.json", summary)
    write_report(output, summary)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()

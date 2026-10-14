"""Verify real comparison artifacts, exclusive IDs, Settings and video playback."""
import csv
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import numpy as np
from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionops.vision.retail_pipeline import digest, write_json


API = "http://localhost:8000"
ROOT = Path("outputs/retail_store_test/tracking_experiment")


def api(path, body=None):
    request = Request(API+path, data=json.dumps(body).encode() if body is not None else None,
                      headers={"Content-Type": "application/json"} if body is not None else {},
                      method="PUT" if body is not None else "GET")
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    result = api("/api/tracking-experiments")
    assert result["status"]["state"] == "completed", result["status"]
    comparison = result["comparison"]
    output = ROOT/comparison["relative_dir"]
    assert len(comparison["results"]) == 10
    metadata = comparison["detector"]
    assert metadata["checkpoint_sha256"] == digest(Path(metadata["checkpoint"]))
    assert metadata["cache_sha256"] == digest(output/"detections.npz")
    original = api("/api/config")["tracking"]
    overview = api("/api/analytics/overview")
    errors = []
    for row in comparison["results"]:
        folder = output/row["variant"]
        observations = {}
        with (folder/"tracks.jsonl").open(encoding="utf-8") as stream:
            tracks = [json.loads(line) for line in stream]
        assert len(tracks) == metadata["frames"]
        for frame in tracks:
            detections = frame["detections"]
            ids = [detection["track_id"] for detection in detections]
            indices = [detection["detection_index"] for detection in detections]
            assert len(ids) == len(set(ids)) and len(indices) == len(set(indices))
            observations[frame["timestamp"]] = set(ids)
        events = json.loads((folder/"events.json").read_text())["events"]
        with (folder/"events.csv").open(newline="", encoding="utf-8") as stream:
            assert len(list(csv.DictReader(stream))) == len(events)
        assert sum(event["event_type"] == "entry" for event in events) == row["entries"]
        assert sum(event["event_type"] == "exit" for event in events) == row["exits"]
        for event in events:
            assert event["track_id"] in observations[event["timestamp"]]
        for visit in json.loads((folder/"zone_visits.json").read_text()):
            assert abs(visit["dwell_seconds"]-(visit["exited_at"]-visit["entered_at"])) < 1e-7
            if visit["reason"] == "recovered_gap":
                assert visit["censored"]
        assert row["video_fully_decoded"] and row["identity_metrics"] is None
        with urlopen(Request(f"{API}/api/tracking-experiments/artifacts/{comparison['relative_dir']}/{row['variant']}/annotated.mp4",
                             headers={"Range": "bytes=0-1023"})) as response:
            assert response.status == 206 and len(response.read()) == 1024
    verification = {"run_id": comparison["run_id"], "variants": 10, "identical_detector_cache": True,
                    "checkpoint_unchanged": True, "unique_active_ids_and_detection_indices_all_frames": True,
                    "events_use_observed_engine_ids": True, "dwell_matches_video_timestamps": True,
                    "csv_json_event_counts_match": True, "all_videos_http_ranges": True,
                    "identity_accuracy_not_claimed": True, "viewports": {}, "browser_errors": errors}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            for name, viewport in [("desktop", {"width": 1440, "height": 1100}), ("mobile", {"width": 390, "height": 844})]:
                page = browser.new_page(viewport=viewport)
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto("http://localhost:5173", wait_until="networkidle")
                page.get_by_role("button", name="Settings", exact=True).click()
                expect(page.get_by_role("heading", name="Tracking Experiment")).to_be_visible()
                expect(page.locator(".tracking-table tbody tr")).to_have_count(10)
                if name == "desktop":
                    page.get_by_label("Tracker", exact=True).select_option("botsort_reid")
                    page.get_by_label("Lost-track buffer", exact=True).select_option("60")
                    page.get_by_role("button", name="Save Tracker", exact=True).click()
                    expect(page.get_by_text("Tracker saved for future video jobs. Existing analyses are unchanged.")).to_be_visible()
                    assert api("/api/config")["tracking"]["mode"] == "botsort_reid"
                    page.reload(wait_until="networkidle")
                    page.get_by_role("button", name="Settings", exact=True).click()
                    expect(page.get_by_label("Tracker", exact=True)).to_have_value("botsort_reid")
                    expect(page.get_by_label("Lost-track buffer", exact=True)).to_have_value("60")
                    api("/api/config", {"tracking": original})
                    page.reload(wait_until="networkidle")
                    page.get_by_role("button", name="Settings", exact=True).click()
                expect(page.get_by_label("Tracker", exact=True)).to_have_value(original["mode"])
                for variant in ["bytetrack_30", "botsort_reid_30"]:
                    page.get_by_label("Comparison video", exact=True).select_option(variant)
                    player = page.locator(".tracking-settings video")
                    player.evaluate("video => video.play()")
                    page.wait_for_function("document.querySelector('.tracking-settings video').currentTime > .3")
                    player.evaluate("video => video.pause()")
                    pixels = player.evaluate("""video => {
                        const c=document.createElement('canvas'); c.width=80; c.height=45;
                        const x=c.getContext('2d'); x.drawImage(video,0,0,80,45);
                        return Array.from(x.getImageData(0,0,80,45).data);
                    }""")
                    assert np.std(np.array(pixels).reshape(-1,4)[:,:3]) > 15
                    assert player.evaluate("video => video.videoWidth===1270 && video.videoHeight===720 && !video.error")
                    selected = next(row for row in comparison["results"] if row["variant"] == variant)
                    expect(page.locator('[data-tracker-diagnostic="recovered"]')).to_have_text(str(selected["last_diagnostics"]["recovered"]))
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=str(output/f"settings_{name}.png"), full_page=True)
                page.screenshot(path=str(output/f"settings_{name}_viewport.png"))
                verification["viewports"][name] = {"comparison_table": True, "diagnostics_match": True,
                    "baseline_and_reid_video_nonblank": True, "no_horizontal_overflow": True}
                page.close()
            browser.close()
    finally:
        api("/api/config", {"tracking": original})
    assert not errors, errors
    assert api("/api/config")["tracking"] == original
    assert api("/api/analytics/overview") == overview
    verification.update(settings_saved_and_restored=True, settings_survive_page_reload=True, operational_overview_unchanged=True)
    write_json(output/"verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()

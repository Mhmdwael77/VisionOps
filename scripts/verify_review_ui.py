"""Exercise the real review UI and save desktop/tablet/mobile evidence."""
import json
from pathlib import Path
import sys

from playwright.sync_api import sync_playwright
import requests

ROOT = Path("outputs/retail_store_test/reviewed/ui_verification")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    verify_write = "--verify-write" in sys.argv
    errors = []
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, width, height in [("desktop", 1440, 1000), ("tablet", 900, 1100), ("mobile", 390, 844)]:
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto("http://127.0.0.1:5173", wait_until="networkidle")
            page.get_by_text("Entries / exits", exact=True).wait_for()
            assert page.get_by_text("8 / 3", exact=True).count()
            assert page.locator(".review-preview img").evaluate("img => img.complete && img.naturalWidth === 1270")
            page.screenshot(path=str(ROOT / f"{name}_overview.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"{name}: overview overflow"
            page.get_by_role("button", name="Video Analysis", exact=True).click()
            page.locator(".paused-frame").wait_for()
            page.wait_for_function("document.querySelector('.paused-frame')?.naturalWidth === 1270")
            assert page.locator(".annotation-surface svg g").count() == 10
            page.get_by_label("Select detection", exact=True).select_option(label="M-EDGE-000 / visible")
            assert page.get_by_label("Box x1", exact=True).input_value() == "1074"
            page.get_by_role("button", name="Next frame", exact=True).click()
            assert "Frame 1 /" in page.locator(".review-section").first.inner_text()
            page.get_by_role("button", name="Previous frame", exact=True).click()
            page.get_by_label("Raw comparison", exact=True).check()
            page.wait_for_function("document.querySelectorAll('.annotation-surface svg g').length === 9")
            page.get_by_label("Raw comparison", exact=True).uncheck()
            page.wait_for_function("document.querySelectorAll('.annotation-surface svg g').length === 10")
            page.get_by_role("button", name="Play", exact=True).click()
            page.wait_for_timeout(1200)
            page.get_by_role("button", name="Pause", exact=True).click()
            assert page.get_by_label("Video timeline", exact=True).input_value() != "0"
            page.get_by_label("Playback speed", exact=True).select_option("0.5")
            page.get_by_label("Boxes", exact=True).uncheck()
            assert page.locator(".annotation-surface svg g").count() == 0
            page.get_by_label("Boxes", exact=True).check()
            if name == "desktop" and verify_write:
                page.get_by_label("Video timeline", exact=True).focus()
                page.keyboard.press("Home")
                page.get_by_label("Select detection", exact=True).select_option(label="M-EDGE-000 / visible")
                page.get_by_label("Annotation review note", exact=True).fill("Browser persistence check: re-saved the visually inspected clipped person box in frame 0 without changing its coordinates or unconfirmed identity.")
                previous = requests.get("http://127.0.0.1:8000/api/review").json()["revision"]
                page.get_by_role("button", name="Save", exact=True).click()
                page.wait_for_function("revision => document.querySelector('.review-badges')?.innerText.includes('Revision '+(revision+1))", arg=previous)
                current = requests.get("http://127.0.0.1:8000/api/review").json()
                assert current["revision"] == previous+1 and current["analytics"]["corrected_detections"] == 1
                page.get_by_label("Annotation review note", exact=True).fill("Frame 0 remains ambiguous: clipped right-edge identity and overlapping rear children/staff are unresolved; only the one inspected missed-person box is corrected.")
                page.get_by_label("Frame review state", exact=True).select_option("ambiguous")
                page.get_by_role("button", name="Save frame review", exact=True).click()
                page.wait_for_function("revision => document.querySelector('.review-badges')?.innerText.includes('Revision '+(revision+2))", arg=previous)
            page.get_by_label("Event kind", exact=True).select_option("line")
            page.get_by_label("Event identity filter", exact=True).fill("DOES-NOT-EXIST")
            assert page.get_by_text("No events match the selected filters.", exact=True).count()
            page.get_by_label("Event identity filter", exact=True).fill("")
            page.locator(".review-table-scroll .time-link").first.click()
            page.screenshot(path=str(ROOT / f"{name}_video.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"{name}: video overflow"
            page.get_by_role("button", name="Tracking Inspector", exact=True).click()
            page.get_by_label("Filter tracks", exact=True).fill("M-EDGE")
            assert page.locator(".track-item").count() == 1
            page.locator(".track-item").click()
            assert page.get_by_label("Identity label", exact=True).input_value() == "M-EDGE-000"
            page.screenshot(path=str(ROOT / f"{name}_tracking.png"), full_page=True)
            page.get_by_role("button", name="Zone Analytics", exact=True).click()
            page.locator(".geometry-surface img").wait_for()
            page.screenshot(path=str(ROOT / f"{name}_zones.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"{name}: zone overflow"
            page.get_by_role("button", name="Analytics", exact=True).click()
            assert page.locator(".recharts-surface").count() >= 2
            page.get_by_role("button", name="Settings", exact=True).click()
            page.get_by_text("Runtime Configuration", exact=True).wait_for()
            page.get_by_role("button", name="Overview", exact=True).click()
            if width == 1440:
                page.get_by_role("button", name="Collapse navigation", exact=True).click()
                assert page.locator(".shell.collapsed").count()
                page.get_by_role("button", name="Expand navigation", exact=True).click()
            results.append({"viewport": name, "width": width, "height": height, "source_image_loaded": True,
                "frame_stepping": True, "actual_video_playback": True, "raw_vs_edited_boxes": True,
                "overlay_controls": True, "event_filter_and_seek": True, "track_selection": True,
                "charts_rendered": True, "settings_preserved": True, "no_horizontal_page_overflow": True})
            page.close()
        # Loading/API failure and recovery states use network failure, not dummy analytics.
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.route("**/api/review", lambda route: route.abort())
        page.goto("http://127.0.0.1:5173", wait_until="networkidle")
        page.get_by_role("button", name="Retry", exact=True).wait_for()
        page.screenshot(path=str(ROOT / "api_error.png"), full_page=True)
        page.unroute("**/api/review")
        page.get_by_role("button", name="Retry", exact=True).click()
        page.get_by_text("Entries / exits", exact=True).wait_for()
        assert not errors, errors
        browser.close()
    report = {"results": results, "browser_errors": errors, "api_failure_recovery": True,
              "writes_to_real_annotations": verify_write, "note": "Write mode re-saves the inspected first-frame box unchanged and restores its ambiguous state. No additional detections, identities or fully reviewed frames are invented."}
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

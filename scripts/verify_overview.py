"""Check real Overview data, responsive UI, refresh, and isolated failure states."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from playwright.sync_api import expect, sync_playwright


API = "http://localhost:8000"
OUTPUT = Path("outputs/retail_store_test/integration")


def fetch(path, method="GET"):
    with urlopen(Request(API+path, method=method), timeout=60) as response:
        return json.load(response)


def save(name, value):
    (OUTPUT/name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def check_metrics(page, data):
    for label, value in [("Entries", data["entries"]), ("Exits", data["exits"]), ("Tracker IDs", data["tracker_id_count"]),
                         ("Live occupancy", "Unavailable"), ("Unique customers", "Unavailable")]:
        expect(page.locator(f'[data-metric="{label}"] strong')).to_have_text(str(value))
    expect(page.locator(".overview-dashboard")).to_have_attribute("data-analysis-ids", ",".join(data["selected_analysis_ids"]))
    expect(page.get_by_role("heading", name="Recent Events", exact=True)).to_be_visible()
    assert page.locator(".recharts-bar-rectangle").count() > 0
    assert f"{data['event_count']} video events" in page.locator('[data-metric="Completed videos"]').inner_text()
    assert "Entrance geometry: unreviewed" in page.locator(".test-warning").inner_text()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", action="store_true", help="Execute the actual retail video and verify browser auto-refresh")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = fetch("/api/analytics/overview")
    assert before["selected_analysis_ids"], "A completed real retail analysis is required"
    save("overview_before.json", before)
    errors = []
    verification = {"real_persisted_data": True, "browser_errors": errors, "viewports": {}}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://localhost:5173", wait_until="networkidle")
        check_metrics(page, before)
        if args.process:
            result = fetch("/api/retail-test/run", "POST")
            assert result["state"] == "queued"
            print("Actual video queued; checking the already-open Overview", flush=True)
            expect(page.get_by_text("running", exact=True)).to_be_visible(timeout=30000)
            page.wait_for_function("""previous => {
                const ids = document.querySelector('.overview-dashboard')?.dataset.analysisIds;
                return ids && ids !== previous;
            }""",
                                   arg=",".join(before["selected_analysis_ids"]), timeout=300000)
            after = fetch("/api/analytics/overview")
            check_metrics(page, after)
            assert after["selected_analysis_ids"] != before["selected_analysis_ids"]
            assert fetch("/api/retail-test")["analysis_id"] == after["selected_analysis_ids"][0]
            verification["real_processing_completion_auto_refreshed"] = True
            verification["previous_analysis_id"] = before["selected_analysis_ids"][0]
            before = after
            save("processing_refresh_verification.json", {**verification, "completed_analysis_id": after["selected_analysis_ids"][0]})
        page.close()
        save("overview_after.json", before)
        for name, viewport in [("desktop", {"width": 1440, "height": 1100}), ("mobile", {"width": 390, "height": 844})]:
            page = browser.new_page(viewport=viewport)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto("http://localhost:5173", wait_until="networkidle")
            check_metrics(page, before)
            page.screenshot(path=str(OUTPUT/f"overview_{name}.png"), full_page=True)
            page.screenshot(path=str(OUTPUT/f"overview_{name}_viewport.png"))
            selector = page.get_by_label("Analysis selection")
            selector.select_option("all")
            expect(page.locator(".overview-dashboard")).to_have_attribute("data-selection", "all")
            expect(page.locator('[data-metric="Entries"] strong')).to_have_text(str(before["entries"]))
            expect(page.locator(".overview-dashboard")).to_have_attribute("data-analysis-ids", ",".join(fetch("/api/analytics/overview?analysis_id=all")["selected_analysis_ids"]))
            selector.select_option(before["selected_analysis_ids"][0])
            expect(page.locator(".overview-dashboard")).to_have_attribute("data-selection", before["selected_analysis_ids"][0])
            check_metrics(page, before)
            verification["viewports"][name] = {"real_metrics": True, "chart_nonempty": True, "selectors": True, "no_horizontal_overflow": True}
            page.close()

        # Intercept only this disposable browser context; never seed live data.
        page = browser.new_page()
        held = []
        pattern = "**/api/analytics/overview?*"
        page.route(pattern, lambda route: held.append(route))
        page.goto("http://localhost:5173", wait_until="domcontentloaded")
        expect(page.get_by_role("status")).to_have_text("Loading persisted analytics...")
        page.screenshot(path=str(OUTPUT/"overview_loading.png"))
        assert held
        held[0].fulfill(status=503, json={"detail": "Isolated API-error verification"})
        page.unroute(pattern)
        expect(page.get_by_role("alert")).to_contain_text("Analytics unavailable")
        assert page.locator(".summary-metric").count() == 0
        page.screenshot(path=str(OUTPUT/"overview_api_error.png"))
        page.get_by_role("button", name="Retry", exact=True).click()
        check_metrics(page, before)
        empty = {**before, "analyses": [], "selected_analysis_ids": [], "history": []}
        page.route(pattern, lambda route: route.fulfill(json=empty))
        page.get_by_role("button", name="Refresh analytics", exact=True).click()
        expect(page.get_by_role("heading", name="No completed analyses")).to_be_visible()
        assert page.locator(".summary-metric").count() == 0
        page.screenshot(path=str(OUTPUT/"overview_empty.png"))
        verification.update(loading_state=True, empty_state=True, api_error_state_without_zero_metrics=True, retry_recovers=True)
        page.close()
        browser.close()
    assert not errors, errors
    verification.update(analysis_id=before["selected_analysis_ids"][0], entries=before["entries"], exits=before["exits"],
                        event_count=before["event_count"], tracker_ids=before["tracker_id_count"], checked_at=time.time())
    save("overview_verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()

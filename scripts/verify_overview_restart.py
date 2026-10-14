"""Capture and compare live API data across an actual backend/frontend restart."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


OUTPUT = Path("outputs/retail_store_test/integration")


def snapshot():
    result = {}
    for name, route in [("overview", "/api/analytics/overview"), ("all_videos", "/api/analytics/overview?analysis_id=all"),
                        ("retail", "/api/retail-test"), ("jobs", "/api/videos")]:
        with urlopen("http://localhost:8000"+route, timeout=30) as response:
            result[name] = json.load(response)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["capture", "verify"])
    for service in ["backend", "frontend"]:
        for phase in ["before", "after"]:
            parser.add_argument(f"--{service}-{phase}", type=int)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    current = snapshot()
    if args.action == "capture":
        (OUTPUT/"restart_before.json").write_text(json.dumps(current, indent=2), encoding="utf-8")
        print("Captured persisted API payloads before service restart")
        return
    before = json.loads((OUTPUT/"restart_before.json").read_text(encoding="utf-8"))
    assert before == current, "Persisted API payload changed across restart"
    assert args.backend_before != args.backend_after and args.frontend_before != args.frontend_after
    assert all([args.backend_before, args.backend_after, args.frontend_before, args.frontend_after])
    with urlopen(Request("http://localhost:8000/api/retail-test/artifacts/annotated.mp4", headers={"Range": "bytes=0-1023"})) as response:
        assert response.status == 206 and len(response.read()) == 1024
    original = next(job for job in current["jobs"] if job["id"] == "a98721bc-af13-5abe-809d-6fed92902545")
    with urlopen("http://localhost:8000/api/analytics/overview?analysis_id="+original["id"]) as response:
        original_analysis = json.load(response)
    assert (original_analysis["entries"], original_analysis["exits"], original_analysis["event_count"]) == (8, 3, 375)
    report = {"all_api_payloads_identical_after_restart": True, "archived_video_range_after_restart": True,
              "original_375_event_analysis_preserved": True, "backend_pids": [args.backend_before, args.backend_after],
              "frontend_pids": [args.frontend_before, args.frontend_after],
              "analysis_id": current["overview"]["selected_analysis_ids"][0], "job_count": len(current["jobs"]),
              "latest_events": current["overview"]["event_count"], "entries": current["overview"]["entries"],
              "exits": current["overview"]["exits"], "all_videos_analysis_ids": current["all_videos"]["selected_analysis_ids"]}
    (OUTPUT/"restart_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

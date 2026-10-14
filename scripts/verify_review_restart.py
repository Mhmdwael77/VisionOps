"""Compare exact persisted review and baseline API payloads across service restarts."""
import argparse
import hashlib
import json
from pathlib import Path

import requests

OUTPUT = Path("outputs/retail_store_test/reviewed")
API = "http://127.0.0.1:8000"


def snapshot():
    result = {}
    value = requests.get(API+"/api/review", timeout=30).json()
    paths = {"review": "/api/review", "export": f"/api/review/{value['id']}/export/json",
             "history": f"/api/review/{value['id']}/history", "renders": f"/api/review/{value['id']}/renders",
             "raw": f"/api/review/{value['id']}/export/json?raw=true", "baseline": "/api/analytics/overview", "jobs": "/api/videos"}
    for key, path in paths.items():
        response = requests.get(API+path, timeout=30)
        response.raise_for_status()
        result[key] = response.json()
    for name in ("events.csv", "visits.csv", "annotations.csv"):
        response = requests.get(API+f"/api/review/{value['id']}/export/{name}", timeout=30)
        response.raise_for_status()
        result[name+"_sha256"] = hashlib.sha256(response.content).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["capture", "verify"])
    for service in ("backend", "frontend"):
        for phase in ("before", "after"):
            parser.add_argument(f"--{service}-{phase}", type=int)
    args = parser.parse_args()
    current = snapshot()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.action == "capture":
        (OUTPUT / "restart_before.json").write_text(json.dumps(current), encoding="utf-8")
        print("Captured canonical annotations, raw inference, CSVs, history, renders, jobs and Overview before restart")
        return
    before = json.loads((OUTPUT / "restart_before.json").read_text(encoding="utf-8"))
    assert before == current, "Persisted review/baseline data changed after restart"
    for service in ("backend", "frontend"):
        old, new = getattr(args, service+"_before"), getattr(args, service+"_after")
        assert old and new and old != new
    value = current["review"]
    jobs = [r for r in current["renders"] if r["revision"] == value["revision"] and r["status"] == "completed"]
    assert jobs, "Latest canonical revision must have a completed matching presentation"
    response = requests.get(API+f"/api/review/{value['id']}/renders/{jobs[0]['id']}.mp4", headers={"Range": "bytes=0-1023"})
    assert response.status_code == 206 and len(response.content) == 1024
    result = {"exact_payloads_and_exports_persisted": True, "presentation_range_after_restart": True,
        "revision": value["revision"], "dataset_id": value["id"], "backend_pids": [args.backend_before, args.backend_after],
        "frontend_pids": [args.frontend_before, args.frontend_after], "baseline_job_count": len(current["jobs"]),
        "protected_raw_annotations_unchanged": current["raw"]["document"] == before["raw"]["document"]}
    (OUTPUT / "restart_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

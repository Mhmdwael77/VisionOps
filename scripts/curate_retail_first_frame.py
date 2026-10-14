"""Apply the explicitly inspected first-frame correction, not a full-video certificate."""
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.database import SessionLocal, init_db
from backend.app.services.review_store import OUTPUT, edit, initialize, sha256


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    backup = OUTPUT / "before_manual_review.sqlite3"
    if not backup.exists():
        with sqlite3.connect("visionops.db") as original, sqlite3.connect(backup) as target:
            original.backup(target)
    protected = [Path("VisionOps_best.pt"), Path("data/videos/retail_store.mp4")]
    protected += [Path("outputs/retail_store_test") / name for name in ("tracks.jsonl", "events.json", "summary.json", "annotated.mp4")]
    before = {str(path): sha256(path) for path in protected}
    init_db()
    with SessionLocal() as db:
        value = initialize(db)
        doc = json.loads(value.document_json)
        if not any(a["person_id"] == "M-EDGE-000" for a in doc["annotations"]):
            edit(db, value, value.revision, "annotation", {"frame_index": 0, "person_id": "M-EDGE-000", "bbox": [1074, 468, 1269, 719],
                "visibility": "visible", "uncertainty": "Partially clipped at right/bottom frame boundary; identity beyond this frame unconfirmed"},
                "Visual review of source frame 0: grey/black-clothed person at right edge missed by raw detections. Single-frame box only; no inferred persistent identity.")
            edit(db, value, value.revision, "review_frames", {"start_frame": 0, "end_frame": 0, "state": "ambiguous"},
                "Inspected frame 0 and added one missed clipped person. Rear children/staff and overlap remain unresolved; this is not an exhaustive or fully reviewed frame. Entrance threshold may lie outside the lower-right frame.")
        analytics = json.loads(value.analytics_json)
        after = {str(path): sha256(path) for path in protected}
        assert before == after
        report = {"dataset_id": value.id, "revision": value.revision, "protected_original_hashes": after,
                  "originals_unchanged": True, "reviewed_frames": analytics["reviewed_frames"],
                  "fully_reviewed_frames": analytics["fully_reviewed_frames"], "corrected_detections": analytics["corrected_detections"],
                  "note": "Only the explicitly inspected first-frame observation was corrected. No person identity links or entrance geometry were certified."}
        (OUTPUT / "curation_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

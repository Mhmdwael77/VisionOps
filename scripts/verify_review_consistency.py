"""Validate real canonical data, exports, stream concurrency and decoded presentation pixels."""
from collections import defaultdict, deque
import csv
import io
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import requests
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.core.database import SessionLocal
from backend.app.models.tables import ReviewDataset, ReviewRender, ReviewRevision
from backend.app.services.review_renderer import paint
from backend.app.services.review_store import OUTPUT, sha256
from visionops.vision.review_analytics import derive, fingerprint, frame_annotations

API = "http://127.0.0.1:8000/api/review"


def main():
    response = requests.get(API, timeout=30)
    response.raise_for_status()
    public = response.json()
    data = requests.get(f"{API}/{public['id']}/export/json", timeout=30).json()
    analytics = derive(data["document"], data["metadata"])
    assert fingerprint(analytics) == fingerprint(public["analytics"]) == fingerprint(data["analytics"])
    events = analytics["events"]
    for direction in ("entry", "exit"):
        accepted = [e for e in events if e["event_type"] == direction and e["kind"] == "line" and e["review_status"] != "rejected" and not e["suppressed_duplicate"]]
        key = "entries" if direction == "entry" else "exits"
        assert len(accepted) == analytics[key] == sum(b[key] for b in analytics["traffic"])
        assert sum(e["confirmed"] for e in accepted) == analytics["reviewed_"+key]
    for kind, expected in (("events", events), ("visits", analytics["visits"]), ("annotations", data["document"]["annotations"])):
        rows = list(csv.DictReader(io.StringIO(requests.get(f"{API}/{public['id']}/export/{kind}.csv", timeout=30).text)))
        assert len(rows) == len(expected)
        for row, actual in zip(rows, expected):
            assert int(row["revision"]) == public["revision"]
            for key, value in actual.items():
                if isinstance(value, (dict, list)):
                    assert json.loads(row[key]) == value
                elif value is None:
                    assert row[key] == ""
                else:
                    # CSV formula protection is intentionally reversible.
                    assert row[key] == str(value) or row[key] == "'"+str(value)
    streams = []
    try:
        for _ in range(20):
            stream = requests.get(f"{API}/{public['id']}/source.mp4", stream=True, timeout=20)
            stream.raise_for_status()
            streams.append(stream)
        health = requests.get(API, timeout=10)
        assert health.status_code == 200, "Video streams must not retain all DB pool connections"
    finally:
        for stream in streams:
            stream.close()
    with SessionLocal() as db:
        value = db.get(ReviewDataset, public["id"])
        job = db.scalar(select(ReviewRender).where(ReviewRender.dataset_id == value.id,
            ReviewRender.revision == value.revision, ReviewRender.status == "completed").order_by(ReviewRender.created_at.desc()))
        assert job, "Render the current revision before validation"
        revision = db.get(ReviewRevision, (value.id, value.revision))
        assert fingerprint(json.loads(revision.document_json)) == data["fingerprint"]
        source_path, output_path = value.source_path, job.output_path
        render_report = json.loads(job.report_json)
    grouped = frame_annotations(data["document"])
    trails = defaultdict(lambda: deque(maxlen=18))
    source, output = cv2.VideoCapture(source_path), cv2.VideoCapture(output_path)
    decoded, pixel_samples = 0, []
    try:
        while True:
            ok, frame = source.read()
            produced, encoded = output.read()
            assert ok == produced, "Presentation/source frame counts differ"
            if not ok:
                break
            expected = frame.copy()
            ledger = paint(expected, decoded, data["document"], data["metadata"], analytics, grouped[decoded], trails)
            assert ledger["metrics"] == analytics["timeline"][decoded]
            if decoded % 120 == 0 or decoded == analytics["frame_count"]-1:
                error = np.abs(expected.astype(np.int16)-encoded.astype(np.int16))
                mae = float(error.mean())
                assert mae < 12, f"Encoded overlay/frame alignment differs at frame {decoded}: MAE={mae}"
                pixel_samples.append({"frame_index": decoded, "mean_absolute_codec_error": mae})
            decoded += 1
    finally:
        source.release()
        output.release()
    assert decoded == analytics["frame_count"] == render_report["total_frames"]
    originals = json.loads((OUTPUT / "curation_verification.json").read_text(encoding="utf-8"))
    assert all(sha256(path) == digest for path, digest in originals["protected_original_hashes"].items())
    report = {"dataset_id": public["id"], "revision": public["revision"], "canonical_fingerprint": data["fingerprint"],
        "database_export_and_recomputed_analytics_identical": True, "all_csv_values_match_canonical_records": True,
        "traffic_and_counts_match_event_ledger": True, "video_streams_do_not_exhaust_database_pool": True,
        "concurrent_open_video_streams_tested": 20, "decoded_presentation_frames": decoded,
        "representative_decoded_pixels_match_replayed_overlays_with_codec_tolerance": pixel_samples,
        "original_artifacts_and_checkpoint_unchanged": True, "render": render_report,
        "presentation_path": output_path, "remaining_limitations": [
            "Only frame 0 inspected; it remains ambiguous, not fully reviewed.",
            "Only one missing, clipped first-frame person corrected; no persistent identity asserted.",
            "Entrance and zone geometry unreviewed. Counts remain provisional.",
            "Identity-switch accuracy, unique visitors, initial/live occupancy and reviewed dwell unavailable.",
            "Exact stored boxes are rendered; raw box jitter is preserved instead of inventing smoothed detections.",
            "Pixel checks cover representative frames; full-video checks cover decode, record ledger and metrics, not semantic identity accuracy."]}
    (OUTPUT / "consistency_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

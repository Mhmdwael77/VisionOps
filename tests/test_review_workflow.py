"""Integration tests use the existing real retail inference, never seed demo records."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path

import pytest
from sqlalchemy import select

from backend.app.models.tables import AnalysisRecord, ProcessingJob, ReviewDataset, ReviewRevision
from backend.app.services import review_store as store
from backend.app.services.review_renderer import paint
from visionops.vision.review_analytics import derive, fingerprint


@pytest.fixture
def workspace(database):
    path = Path("outputs/retail_store_test/summary.json")
    if not path.exists():
        pytest.skip("Real retail inference artifacts required")
    summary = json.loads(path.read_text(encoding="utf-8"))
    with database() as db:
        db.add(ProcessingJob(id="real-retail", source_path=summary["source"]["path"], status="completed"))
        db.flush()
        db.add(AnalysisRecord(job_id="real-retail", run_id=summary["run_id"], kind="retail", video_key="real-retail", summary_json=json.dumps(summary)))
        db.commit()
        dataset = store.initialize(db)
        yield dataset.id


def test_raw_import_matches_real_counts_and_is_idempotent(database, workspace):
    with database() as db:
        value = db.get(ReviewDataset, workspace)
        assert store.initialize(db).id == workspace
        a = json.loads(value.analytics_json)
        assert (a["entries"], a["exits"], a["frame_count"]) == (8, 4, 1452)
        assert a["reviewed_frames"] == a["reviewed_entries"] == 0
        assert a["visitors"] is None and a["live_occupancy"] is None and a["reviewed_dwell"] is None
        assert a["unconfirmed_identities"] == 46


def test_edit_exports_persistence_and_raw_immutable(api_client, database, workspace):
    client = api_client
    original = client.get(f"/api/review/{workspace}/export/json?raw=true").json()
    a = original["document"]["annotations"][0]
    body = {"revision": 0, "action": "annotation", "payload": {**a, "bbox": [810, 255, 996, 605]}, "note": "Real source box adjustment for integration test"}
    response = client.post(f"/api/review/{workspace}/edit", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["revision"] == 1
    assert client.post(f"/api/review/{workspace}/edit", json=body).status_code == 409
    raw = client.get(f"/api/review/{workspace}/export/json?raw=true").json()
    assert raw["document"] == original["document"]
    assert raw["fingerprint"] == fingerprint(raw["document"]) and raw["revision"] == 0
    current = client.get(f"/api/review/{workspace}/export/json").json()
    assert current["document"]["annotations"][-1]["bbox"] == body["payload"]["bbox"]
    csv_rows = list(csv.DictReader(io.StringIO(client.get(f"/api/review/{workspace}/export/events.csv").text)))
    assert len(csv_rows) == len(current["analytics"]["events"])
    assert sum(row["event_type"] == "entry" and row["review_status"] != "rejected" and row["suppressed_duplicate"] == "False" for row in csv_rows) == current["analytics"]["entries"]
    with database() as reopened:
        saved = reopened.get(ReviewDataset, workspace)
        assert saved.revision == 1
        assert json.loads(saved.document_json) == current["document"]
        assert json.loads(saved.analytics_json) == current["analytics"]
    assert len(client.get(f"/api/review/{workspace}/history").json()) == 2
    image = client.get(f"/api/review/{workspace}/images/0.jpg")
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"


def test_merge_overlap_rejected_split_nonoverlapping_merge_supported(database, workspace):
    with database() as db:
        value = db.get(ReviewDataset, workspace)
        with pytest.raises(ValueError, match="one observation"):
            store.edit(db, value, 0, "merge", {"source": "P001", "target": "P002"}, "Overlapping identities must not merge")
        db.rollback()
        value = db.get(ReviewDataset, workspace)
        store.edit(db, value, 0, "split", {"source": "P001", "target": "SPLIT-1", "start_frame": 1, "end_frame": 10}, "Test a bounded split of real observations")
        doc = json.loads(value.document_json)
        assert any(a["person_id"] == "SPLIT-1" for a in doc["annotations"])
        assert all(a["person_id"] != "P001" for a in doc["annotations"] if 1 <= a["frame_index"] <= 10)
        store.edit(db, value, 1, "merge", {"source": "SPLIT-1", "target": "P001"}, "Restore nonoverlapping real track fragments")
        assert not any(a["person_id"] == "SPLIT-1" for a in json.loads(value.document_json)["annotations"])


def test_occluded_has_no_box_and_edits_invalidate_frame_review(database, workspace):
    with database() as db:
        value = db.get(ReviewDataset, workspace)
        first = json.loads(value.document_json)["annotations"][0]
        store.edit(db, value, 0, "review_frames", {"start_frame": 0, "end_frame": 0, "state": "reviewed"}, "Unit-test review flag, not actual human review")
        with pytest.raises(ValueError, match="must not invent"):
            store.edit(db, value, 1, "annotation", {**first, "visibility": "occluded"}, "Invalid box during occlusion")
        db.rollback()
        value = db.get(ReviewDataset, workspace)
        store.edit(db, value, 1, "annotation", {**first, "visibility": "occluded", "bbox": None}, "Occlusion record without invented box")
        doc = json.loads(value.document_json)
        assert doc["frame_reviews"][0]["state"] == "unreviewed"
        assert json.loads(value.analytics_json)["timeline"][0]["visible"] == 7


def test_reviewed_crossing_requires_geometry_identity_adjacent_frames_and_decision(database, workspace):
    with database() as db:
        value = db.get(ReviewDataset, workspace)
        doc, metadata = json.loads(value.document_json), json.loads(value.metadata_json)
        event = next(e for e in json.loads(value.analytics_json)["events"] if e["kind"] == "line" and not e["censored"])
        doc["event_corrections"][event["id"]] = {"status": "accepted", "evidence_hash": event["evidence_hash"]}
        assert not next(e for e in derive(doc, metadata)["events"] if e["id"] == event["id"])["confirmed"]
        doc["geometry"]["counting_line"]["reviewed"] = True
        # Geometry edits invalidate earlier event decisions.
        result = derive(doc, metadata)
        assert event["id"] in result["stale_event_reviews"]
        new = next(e for e in result["events"] if e["id"] == event["id"])
        doc["event_corrections"][new["id"]]["evidence_hash"] = new["evidence_hash"]
        doc["identities"][event["person_id"]]["confirmed"] = True
        for i in (event["frame_index"]-1, event["frame_index"]):
            doc["frame_reviews"][i]["state"] = "reviewed"
        result = derive(doc, metadata)
        assert next(e for e in result["events"] if e["id"] == event["id"])["confirmed"]
        doc["frame_reviews"][event["frame_index"]-1]["state"] = "ambiguous"
        assert not next(e for e in derive(doc, metadata)["events"] if e["id"] == event["id"])["confirmed"]


def test_manual_duplicate_suppressed_and_zone_censoring_consistent(database, workspace):
    with database() as db:
        value = db.get(ReviewDataset, workspace)
        doc, metadata = json.loads(value.document_json), json.loads(value.metadata_json)
        baseline = derive(doc, metadata)
        e = next(e for e in baseline["events"] if e["kind"] == "line")
        doc["manual_events"] = [{key: e[key] for key in ("person_id", "frame_index", "kind", "region", "event_type")} |
                                {"id": "manual-crossing", "review_status": "accepted", "censored": False}]
        result = derive(doc, metadata)
        assert (result["entries"], result["exits"]) == (baseline["entries"], baseline["exits"])
        assert len(result["suppressed_duplicate_events"]) == 1
        assert all(abs(v["dwell_seconds"] - (v["exited_at"]-v["entered_at"])) < 1e-6 for v in result["visits"])
        assert all(z["reviewed_dwell"] is None for z in result["zones"])
        for zone in result["zones"]:
            visits = [v for v in result["visits"] if v["zone_name"] == zone["name"]]
            assert zone["visits"] == len(visits)
            assert zone["censored"] == sum(v["censored"] for v in visits)
        assert sum(b["entries"] for b in result["traffic"]) == result["entries"]


def test_import_resets_certificates_and_rejects_wrong_source(database, workspace):
    with database() as db:
        value = db.get(ReviewDataset, workspace)
        doc = json.loads(value.document_json)
        doc["frame_reviews"][0]["state"] = "reviewed"
        with pytest.raises(ValueError, match="different video"):
            store.edit(db, value, 0, "import", {"source_sha256": "wrong", "document": doc}, "Wrong source must be rejected")
        store.edit(db, value, 0, "import", {"source_sha256": value.source_sha256, "document": doc}, "Imported draft is not certified review")
        assert json.loads(value.document_json)["frame_reviews"][0]["state"] == "unreviewed"
        assert json.loads(value.analytics_json)["reviewed_frames"] == 0


def test_render_records_exact_boxes_without_occlusion_box(database, workspace):
    from collections import defaultdict, deque
    with database() as db:
        value = db.get(ReviewDataset, workspace)
        doc, metadata = json.loads(value.document_json), json.loads(value.metadata_json)
        annotations = [a for a in doc["annotations"] if a["frame_index"] == 0]
        image = store.read_frame(value, 0)
        ledger = paint(image, 0, doc, metadata, json.loads(value.analytics_json), annotations, defaultdict(lambda: deque(maxlen=18)))
        assert ledger["visible_annotations"] == [{key: a[key] for key in ("id", "person_id", "bbox", "origin")} for a in annotations]
        assert ledger["metrics"] == json.loads(value.analytics_json)["timeline"][0]
        assert image.var() > 100


def test_database_optimistic_lock_stops_concurrent_edit(database, workspace):
    with database() as first, database() as second:
        a, b = first.get(ReviewDataset, workspace), second.get(ReviewDataset, workspace)
        store.edit(first, a, 0, "review_frames", {"start_frame": 0, "end_frame": 0, "state": "ambiguous"}, "First reviewer")
        with pytest.raises(RuntimeError, match="Concurrent"):
            store.edit(second, b, 0, "review_frames", {"start_frame": 1, "end_frame": 1, "state": "ambiguous"}, "Stale second reviewer")


def test_complete_reviewed_dwell_excludes_censored_and_rejected_boundaries(database, workspace):
    with database() as db:
        value = db.get(ReviewDataset, workspace)
        doc, metadata = json.loads(value.document_json), json.loads(value.metadata_json)
        initial = derive(doc, metadata)
        visit = next(v for v in initial["visits"] if not v["censored"] and v["dwell_seconds"] > 0)
        original_events = {e["id"]: e for e in initial["events"]}
        entry, exit = original_events[visit["entry_event"]], original_events[visit["exit_event"]]
        next(z for z in doc["geometry"]["zones"] if z["name"] == visit["zone_name"])["reviewed"] = True
        doc["identities"][visit["person_id"]]["confirmed"] = True
        for i in range(max(0, entry["frame_index"]-1), exit["frame_index"]+1):
            doc["frame_reviews"][i]["state"] = "reviewed"
        generated = {e["id"]: e for e in derive(doc, metadata)["events"]}
        for event_id in (entry["id"], exit["id"]):
            doc["event_corrections"][event_id] = {"status": "accepted", "evidence_hash": generated[event_id]["evidence_hash"]}
        result = derive(doc, metadata)
        assert result["reviewed_dwell"] == visit["dwell_seconds"]
        assert sum(v["confirmed"] for v in result["visits"]) == 1
        doc["event_corrections"][entry["id"]]["status"] = "rejected"
        rejected = derive(doc, metadata)
        assert rejected["reviewed_dwell"] is None and exit["id"] in rejected["incomplete_zone_events"]


def test_annotation_changes_invalidate_accepted_event_evidence_even_when_timestamp_unchanged(database, workspace):
    with database() as db:
        value = db.get(ReviewDataset, workspace)
        doc, metadata = json.loads(value.document_json), json.loads(value.metadata_json)
        event = next(e for e in derive(doc, metadata)["events"] if e["kind"] == "line")
        doc["event_corrections"][event["id"]] = {"status": "accepted", "evidence_hash": event["evidence_hash"]}
        a = next(a for a in doc["annotations"] if a["person_id"] == event["person_id"] and a["frame_index"] == event["frame_index"])
        a["bbox"][0] += .01
        assert event["id"] in derive(doc, metadata)["stale_event_reviews"]

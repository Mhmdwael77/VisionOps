from datetime import datetime, timedelta, timezone
import csv
import io
import json
from pathlib import Path
import shutil

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from backend.app.models.tables import AnalysisRecord, Event, ProcessingJob, ZoneVisit
from backend.app.services.analysis_store import execute_retail_analysis, import_retail_analysis
from backend.app.services.analytics import dwell_stats, overview as query_overview
from visionops.vision.test_config import load_retail_config


def completed_artifacts(tmp_path):
    output = tmp_path / "retail"
    shutil.copytree(Path("tests/fixtures/retail_store"), output)
    summary = json.loads((output / "summary.json").read_text())
    (output / "status.json").write_text(json.dumps({"state": "completed", "run_id": summary["run_id"]}))
    return output


def test_real_completed_retail_appears_in_overview_and_retail_database_views(api_client, database, tmp_path):
    output = completed_artifacts(tmp_path)
    with database() as db:
        job = import_retail_analysis(db, output)
        job_id = job.id
        import_retail_analysis(db, output)
        assert db.scalar(select(func.count()).select_from(ProcessingJob)) == 1
        assert db.scalar(select(func.count()).select_from(Event)) == 375
        assert db.scalar(select(func.count()).select_from(ZoneVisit)) == 182
    response = api_client.get("/api/analytics/overview")
    assert response.status_code == 200
    overview = response.json()
    assert (overview["entries"], overview["exits"], overview["event_count"]) == (8, 3, 375)
    assert overview["completed_jobs"] == 1 and overview["tracker_id_count"] == 138
    assert overview["line_counts_provisional"] and overview["entrance_geometry_status"] == "unreviewed"
    assert overview["total_visitors"] is None and overview["current_occupancy"] is None
    assert overview["historical_visible_people"] == 6
    cashier = next(zone for zone in overview["zones"] if zone["zone_name"] == "Cashier area")
    assert (cashier["historical_final_occupancy"], cashier["historical_peak_occupancy"]) == (2, 7)
    assert overview["recent_events"] and all(event["job_id"] == job_id for event in overview["recent_events"])
    assert sum(point["entries"] for point in overview["traffic"]) == 8
    assert sum(point["exits"] for point in overview["traffic"]) == 3
    assert overview["censored_visits"] == 105 and overview["complete_visits"] == 77
    assert abs(overview["average_dwell_seconds"]-3.1016489811265626) < 1e-9
    retail = api_client.get("/api/retail-test").json()
    assert retail["analysis_id"] == job_id and len(retail["events"]) == overview["event_count"]
    assert retail["summary"]["analytics"]["entries"] == overview["entries"]
    for zone in retail["summary"]["analytics"]["zones"]:
        persisted = next(row for row in overview["zones"] if row["zone_name"] == zone["name"])
        assert zone["average_dwell_seconds"] == persisted["average_dwell_seconds"]
    # Completed reads must survive both source artifacts disappearing and fresh sessions.
    shutil.rmtree(output)
    assert api_client.get("/api/analytics/overview").json()["event_count"] == 375
    assert api_client.get("/api/retail-test").json()["summary"]["analytics"]["entries"] == 8
    original_engine = database.kw["bind"]
    url = original_engine.url
    original_engine.dispose()
    reopened = create_engine(url)
    with Session(reopened) as db:
        restored = query_overview(db)
        assert restored.selected_analysis_ids == [job_id]
        assert (restored.entries, restored.exits, restored.event_count) == (8, 3, 375)
    reopened.dispose()


def test_video_filters_all_latest_per_video_and_failed_runs(api_client, database, tmp_path):
    output = completed_artifacts(tmp_path)
    with database() as db:
        first = import_retail_analysis(db, output)
        first_id = first.id
    summary = json.loads((output / "summary.json").read_text())
    newer = (datetime.fromisoformat(summary["run_id"])+timedelta(minutes=1)).isoformat()
    summary["run_id"] = newer
    (output / "summary.json").write_text(json.dumps(summary))
    events = json.loads((output / "events.json").read_text())
    events["run_id"] = newer
    (output / "events.json").write_text(json.dumps(events))
    (output / "status.json").write_text(json.dumps({"state": "completed", "run_id": newer}))
    with database() as db:
        second = import_retail_analysis(db, output)
        second_id = second.id
        # This represents a separate video in an isolated test DB, never live data.
        other = ProcessingJob(id="other", source_path="other.mp4", status="completed", entries=1, exits=0,
                              started_at=datetime.now(timezone.utc))
        db.add(other)
        db.flush()
        db.add(AnalysisRecord(job_id="other", run_id="other", video_key="other", summary_json=json.dumps({"configuration": {"counting_line": {"reviewed": True}}, "source": {"duration_seconds": 2}, "analytics": {"unique_track_ids": 1}})))
        db.add(Event(job_id="other", track_id=1, event_type="entry", timestamp=1, label="door", payload_json="{}"))
        db.add(ProcessingJob(id="failed", source_path="bad.mp4", status="failed", entries=999))
        db.commit()
    all_videos = api_client.get("/api/analytics/overview?analysis_id=all").json()
    assert all_videos["entries"] == 9 and all_videos["exits"] == 3
    assert set(all_videos["selected_analysis_ids"]) == {second_id, "other"}
    assert all_videos["historical_visible_people"] is None
    assert all(zone["historical_final_occupancy"] is None for zone in all_videos["zones"])
    assert all_videos["tracker_id_count"] == 139  # Job-scoped IDs; not people.
    assert all_videos["entrance_geometry_status"] == "mixed"
    for job_id in [first_id, second_id]:
        assert api_client.get(f"/api/analytics/overview?analysis_id={job_id}").json()["entries"] == 8
        traffic = api_client.get(f"/api/analytics/traffic?analysis_id={job_id}").json()
        assert sum(point["entries"] for point in traffic) == 8
        event_rows = api_client.get(f"/api/events?analysis_id={job_id}&limit=500").json()
        assert len(event_rows) == 375 and {row["job_id"] for row in event_rows} == {job_id}
        exported = api_client.get(f"/api/exports/events.csv?analysis_id={job_id}")
        csv_rows = list(csv.DictReader(io.StringIO(exported.text)))
        assert len(csv_rows) == 375 and {row["job_id"] for row in csv_rows} == {job_id}
        assert all(isinstance(json.loads(row["payload_json"]), dict) for row in csv_rows)
    assert len(api_client.get("/api/events?analysis_id=all&limit=500").json()) == 376
    assert api_client.get("/api/analytics/overview?analysis_id=failed").status_code == 404
    assert api_client.get("/api/analytics/overview?analysis_id=missing").status_code == 404


def test_empty_metrics_are_unavailable_not_zero(api_client):
    overview = api_client.get("/api/analytics/overview").json()
    assert overview["selected_analysis_ids"] == [] and overview["recent_events"] == []
    assert overview["entries"] is None and overview["average_dwell_seconds"] is None
    assert overview["current_occupancy"] is None and overview["total_visitors"] is None


def test_censored_open_and_unknown_visits_have_distinct_aggregation():
    def visit(start, end, dwell, censored, reason="observed"):
        return ZoneVisit(job_id="test", track_id=1, zone_name="cashier", entered_at=start,
                         exited_at=end, dwell_seconds=dwell, censored=censored, left_censored=False, close_reason=reason)
    stats = dwell_stats([visit(0, 10, 10, False), visit(0, 20, 20, True, "track_lost"),
                         visit(0, None, None, None), visit(0, 30, 30, None), visit(0, 2, 9, False)])
    assert stats["average_dwell_seconds"] == 20 and stats["average_uncensored_dwell_seconds"] == 10
    assert stats["censored_visits"] == 1 and stats["incomplete_visits"] == 2
    assert stats["unknown_censoring_visits"] == 2 and stats["complete_visits"] == 1


def test_processing_completion_persists_before_api_reports_complete(api_client, database, tmp_path, monkeypatch):
    output = completed_artifacts(tmp_path)
    config = load_retail_config()
    config.output_dir = str(output)
    monkeypatch.setattr("backend.app.services.analysis_store.SessionLocal", database)
    def processed(_config, on_progress):
        on_progress({"state": "running", "progress": 0.5, "processed_frames": 726})
        assert api_client.get("/api/retail-test").json()["status"]["state"] == "running"
        return json.loads((output / "summary.json").read_text())
    monkeypatch.setattr("visionops.vision.retail_pipeline.run_retail_test", processed)
    execute_retail_analysis(config)
    assert api_client.get("/api/analytics/overview").json()["entries"] == 8
    assert api_client.get("/api/retail-test").json()["status"]["state"] == "completed"


def test_failed_run_persists_error_without_replacing_completed_analysis(api_client, database, tmp_path, monkeypatch):
    with database() as db:
        original = import_retail_analysis(db, completed_artifacts(tmp_path)).id
    monkeypatch.setattr("backend.app.services.analysis_store.SessionLocal", database)
    def fail(_config, on_progress):
        raise RuntimeError("Processing failure regression test")
    monkeypatch.setattr("visionops.vision.retail_pipeline.run_retail_test", fail)
    with pytest.raises(RuntimeError, match="Processing failure regression test"):
        execute_retail_analysis(load_retail_config())
    data = api_client.get("/api/analytics/overview").json()
    assert data["selected_analysis_ids"] == [original] and data["entries"] == 8
    failed = next(row for row in data["history"] if row["status"] == "failed")
    assert failed["entries"] is None and failed["error"] == "Processing failure regression test"


def test_websocket_serializes_persisted_analysis_timestamps(api_client, database, tmp_path, monkeypatch):
    with database() as db:
        import_retail_analysis(db, completed_artifacts(tmp_path))
    monkeypatch.setattr("backend.app.main.SessionLocal", database)
    with api_client.websocket_connect("/ws/analytics") as websocket:
        data = websocket.receive_json()
        assert data["entries"] == 8 and isinstance(data["history"][0]["started_at"], str)


def test_line_and_zone_review_flags_are_independent(api_client, database):
    with database() as db:
        job = ProcessingJob(id="review", source_path="review.mp4", status="completed", entries=1, exits=0)
        db.add(job)
        db.flush()
        db.add(AnalysisRecord(job_id=job.id, run_id="review", video_key="review", summary_json=json.dumps({
            "configuration": {"counting_line": {"reviewed": True}, "zones_reviewed": False}})))
        for kind in ["entry", "zone_entry"]:
            db.add(Event(job_id=job.id, track_id=1, event_type=kind, timestamp=1, label="region", payload_json="{}"))
        db.commit()
    data = api_client.get("/api/analytics/overview").json()
    flags = {event["event_type"]: event["provisional"] for event in data["recent_events"]}
    assert flags == {"entry": False, "zone_entry": True}

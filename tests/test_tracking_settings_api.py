import json

from backend.app.models.tables import AppConfig
from backend.app.models.tables import AnalysisRecord, Event, ProcessingJob


def test_tracker_defaults_and_settings_persist_across_sessions(api_client, database):
    assert api_client.get("/api/config").json()["tracking"]["mode"] == "bytetrack"
    settings = {"mode": "botsort_reid", "lost_buffer_frames": 60, "min_embedding_similarity": .92}
    saved = api_client.put("/api/config", json={"tracking": settings})
    assert saved.status_code == 200
    assert api_client.get("/api/config").json()["tracking"] == saved.json()["tracking"]
    assert api_client.get("/api/retail-test/config").json()["tracking"] == saved.json()["tracking"]
    with database() as db:
        persisted = json.loads(db.get(AppConfig, "tracking").value_json)
        assert persisted["lost_buffer_frames"] == 60 and persisted["mode"] == "botsort_reid"
    # Editing detector settings must not reset the independently saved tracker.
    api_client.put("/api/config", json={"detector_confidence": .4})
    assert api_client.get("/api/config").json()["tracking"]["mode"] == "botsort_reid"


def test_invalid_tracking_settings_and_artifact_escape_are_rejected(api_client):
    for config in [{"lost_buffer_frames": 0}, {"mode": "unknown"}, {"gallery_size": 2, "gallery_min_observations": 4},
                   {"mode": "botsort_reid", "embedding_backend": "onnx"}, {"min_embedding_similarity": 1.5}]:
        assert api_client.put("/api/config", json={"tracking": config}).status_code == 422
    assert api_client.get("/api/tracking-experiments/artifacts/%2e%2e/%2e%2e/visionops.db").status_code == 404
    assert api_client.get("/api/tracking-experiments/artifacts/experiment.lock").status_code == 404


def test_crossing_over_recovered_gap_stays_provisional_with_reviewed_geometry(api_client, database):
    with database() as db:
        db.add(ProcessingJob(id="gap", source_path="gap.mp4", status="completed", entries=1, exits=0))
        db.flush()
        db.add(AnalysisRecord(job_id="gap", run_id="gap", video_key="gap", summary_json=json.dumps({
            "configuration": {"counting_line": {"reviewed": True}}})))
        db.add(Event(job_id="gap", track_id=1, event_type="entry", timestamp=3,
                     payload_json=json.dumps({"provisional": True, "censored": True, "reason": "crossing_across_unobserved_gap"})))
        db.commit()
    response = api_client.get("/api/analytics/overview").json()
    assert response["line_counts_provisional"]
    assert response["entrance_geometry_status"] == "reviewed"
    assert response["recent_events"][0]["censored"]

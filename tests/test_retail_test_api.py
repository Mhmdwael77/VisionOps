import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from backend.app import retail_test
from backend.app.core.database import get_db
from backend.app.models.tables import AnalysisRecord, ProcessingJob
from visionops.vision.test_config import RetailTestConfig


@pytest.fixture
def retail_client(tmp_path, monkeypatch, database):
    config = json.loads(Path("configs/retail_store_test.json").read_text())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(retail_test, "OUTPUT", tmp_path)
    monkeypatch.setattr(retail_test, "CONFIG", config_path)
    app = FastAPI()
    app.include_router(retail_test.router)
    def session():
        with database() as db:
            yield db
    app.dependency_overrides[get_db] = session
    return TestClient(app), tmp_path, config


def test_stale_summary_hidden_during_rerun(retail_client, database):
    client, root, _ = retail_client
    (root / "status.json").write_text(json.dumps({"state": "running", "run_id": "new"}))
    (root / "summary.json").write_text(json.dumps({"run_id": "old"}))
    with database() as db:
        db.add(ProcessingJob(id="new", source_path="data/videos/retail_store.mp4", status="running"))
        db.flush()
        db.add(AnalysisRecord(job_id="new", run_id="new", video_key="retail", kind="retail", summary_json="{}"))
        db.commit()
    response = client.get("/api/retail-test").json()
    assert response["summary"] is None and response["events"] == []


def test_running_pipeline_blocks_configuration_and_another_run(retail_client):
    client, root, config = retail_client
    (root / "run.lock").touch()
    assert client.put("/api/retail-test/config", json=config).status_code == 409
    assert client.post("/api/retail-test/run").status_code == 409


def test_line_stays_unreviewed_after_config_roundtrip(retail_client):
    client, _, config = retail_client
    config["counting_line"]["reviewed"] = False
    assert client.put("/api/retail-test/config", json=config).json()["counting_line"]["reviewed"] is False


def test_polygon_and_line_changes_are_persisted(retail_client):
    client, root, config = retail_client
    config["zones"][0]["polygon"][0] = [0.05, 0.31]
    config["counting_line"]["start"] = [0.59, 0.89]
    response = client.put("/api/retail-test/config", json=config)
    assert response.status_code == 200
    expected = RetailTestConfig.model_validate(config).model_dump(mode="json")
    assert client.get("/api/retail-test/config").json() == expected
    assert json.loads((root / "config.json").read_text()) == expected


def test_no_artifacts_outside_scoped_directory(retail_client):
    client, _, _ = retail_client
    assert client.get("/api/retail-test/artifacts/%2e%2e/configs/retail_store_test.json").status_code == 404
    assert client.get("/api/retail-test/artifacts/run.lock").status_code == 404


def test_geometry_and_output_constraints(retail_client):
    _, _, config = retail_client
    config["counting_line"]["start"] = [-0.2, 0.5]
    with pytest.raises(ValueError):
        RetailTestConfig.model_validate(config)
    config["counting_line"]["start"] = [0.1, 0.5]
    config["output_dir"] = "../elsewhere"
    with pytest.raises(ValueError):
        RetailTestConfig.model_validate(config)

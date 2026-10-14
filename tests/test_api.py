def test_health_reports_model_state(api_client) -> None:
    response = api_client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_path" in body
    assert "model_available" in body


def test_invalid_job_source_rejected(api_client) -> None:
    response = api_client.post("/api/jobs", json={"source_path": "notes.txt"})
    assert response.status_code == 400


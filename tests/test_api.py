import time

import pytest
from fastapi.testclient import TestClient

from app import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "QUEUE_DB_PATH", str(tmp_path / "test_queue.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "generator": "mock"}


def test_create_and_poll_job_to_completion(client):
    resp = client.post("/jobs", json={"num_samples": 2})
    assert resp.status_code == 201
    job_id = resp.json()["job_id"]

    body = None
    for _ in range(50):
        status_resp = client.get(f"/jobs/{job_id}")
        assert status_resp.status_code == 200
        body = status_resp.json()
        if body["status"] == "done":
            break
        time.sleep(0.05)
    else:
        pytest.fail("job did not complete in time")

    assert body["status"] == "done"
    assert len(body["images"]) == 2
    assert body["error"] is None


def test_job_starts_pending_or_running(client):
    resp = client.post("/jobs", json={"num_samples": 1})
    job_id = resp.json()["job_id"]

    status_resp = client.get(f"/jobs/{job_id}")
    assert status_resp.json()["status"] in ("pending", "running", "done")


def test_unknown_job_returns_404(client):
    resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


def test_num_samples_must_be_positive(client):
    resp = client.post("/jobs", json={"num_samples": 0})
    assert resp.status_code == 422


def test_num_samples_capped_at_max(client):
    resp = client.post("/jobs", json={"num_samples": config.MAX_SAMPLES_PER_JOB + 1})
    assert resp.status_code == 422


def test_default_num_samples_is_one(client):
    resp = client.post("/jobs", json={})
    assert resp.status_code == 201

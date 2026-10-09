import os
import time
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("PYTEST_RUNNING", "1")

from app.main import app
from app.database import Base, engine


@pytest.fixture(scope="module")
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c


def _wait_for_job(client, job_id, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        r = client.get(f"/jobs/{job_id}/status")
        assert r.status_code == 200
        s = r.json()["status"]
        if s in ("completed", "partial", "failed"):
            return r.json()
        time.sleep(0.2)
    raise TimeoutError(f"Job {job_id} did not finish")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_create_job_and_generate(client):
    payload = {
        "recipients": [
            {"name": "Alice Johnson", "email": "alice@example.com",
             "course_name": "Advanced Python", "issue_date": "2024-06-01"},
            {"name": "Bob Smith", "email": "bob@example.com",
             "course_name": "Advanced Python", "issue_date": "2024-06-01"},
        ]
    }
    r = client.post("/jobs", json=payload)
    assert r.status_code == 202
    job = r.json()
    assert job["total"] == 2
    assert job["status"] in ("pending", "processing", "completed", "partial")

    final = _wait_for_job(client, job["id"])
    assert final["succeeded"] == 2
    assert final["failed"] == 0
    assert final["status"] == "completed"

    # Retrieve certificates
    r = client.get(f"/jobs/{job['id']}/certificates")
    assert r.status_code == 200
    data = r.json()
    assert data["succeeded"] == 2
    for c in data["certificates"]:
        assert c["status"] == "success"
        assert c["download_url"]
        # Download the actual file
        dl = client.get(c["download_url"])
        assert dl.status_code == 200
        assert dl.headers["content-type"] == "image/png"
        assert len(dl.content) > 1000


def test_input_validation(client):
    # Empty name should fail
    r = client.post("/jobs", json={"recipients": [{"name": ""}]})
    assert r.status_code == 422

    # Blank-ish name
    r = client.post("/jobs", json={"recipients": [{"name": "   "}]})
    assert r.status_code == 422

    # Empty list
    r = client.post("/jobs", json={"recipients": []})
    assert r.status_code == 422

    # Bad email
    r = client.post("/jobs", json={"recipients": [{"name": "X", "email": "not-an-email"}]})
    assert r.status_code == 422


def test_individual_certificate_failure(client, monkeypatch):
    """
    Force one certificate to fail by monkeypatching generate_certificate
    to raise for a specific name; the other should still succeed.
    """
    import app.worker as worker_mod
    import app.certificate as cert_mod

    original = cert_mod.generate_certificate

    def flaky(name, course_name=None, issue_date=None, template_path=None):
        if name == "Bad Person":
            raise RuntimeError("simulated failure")
        return original(name, course_name, issue_date, template_path)

    monkeypatch.setattr(worker_mod, "generate_certificate", flaky)
    monkeypatch.setattr(cert_mod, "generate_certificate", flaky)

    payload = {
        "recipients": [
            {"name": "Good Person"},
            {"name": "Bad Person"},
            {"name": "Another Good Person"},
        ]
    }
    r = client.post("/jobs", json=payload)
    assert r.status_code == 202
    job_id = r.json()["id"]

    final = _wait_for_job(client, job_id)
    assert final["succeeded"] == 2
    assert final["failed"] == 1
    assert final["status"] == "partial"

    detail = client.get(f"/jobs/{job_id}").json()
    failed = [c for c in detail["certificates"] if c["status"] == "failed"]
    assert len(failed) == 1
    assert "simulated failure" in failed[0]["error"]
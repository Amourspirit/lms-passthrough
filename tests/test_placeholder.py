from fastapi.testclient import TestClient

from lms_passthrough import create_app


def test_health_live() -> None:
    client = TestClient(create_app())
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

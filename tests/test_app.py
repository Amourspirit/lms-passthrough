from pathlib import Path

from fastapi.testclient import TestClient

from lms_passthrough import create_app

from .conftest import StubHttp


def make_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app())


def test_unknown_provider_header_returns_400(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/v1/models", headers={"X-LMS-Provider": "nope"})
    assert response.status_code == 400


def test_health_ready_shape(stub_http: StubHttp) -> None:
    """The default (no config file) app: one provider, reported by name.

    Readiness is a real signal now, so the probe has to be stubbed rather than
    pointed at 127.0.0.1:1234 — see `tests/test_readiness.py` for the contract.
    """
    app = create_app()
    stub_http(app, {"lm-studio": 200})

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "providers": {"lm-studio": True}}

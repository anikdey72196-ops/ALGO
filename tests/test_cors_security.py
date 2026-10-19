"""
test_cors_security.py — Security tests for CORS policy enforcement on FastAPI endpoints.
Validates CWE-942 remediation: prevents unauthorized cross-origin access while
allowing legitimate trusted origins (loopback, LAN, and configured domains).
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from fastapi.testclient import TestClient
from web_app import app, get_allowed_origins


@pytest.fixture
def client():
    return TestClient(app)


def test_unauthorized_origin_rejected(client):
    """Verify unauthorized external origins do not receive CORS allow headers."""
    unauthorized_origins = [
        "https://evil.com",
        "http://malicious-site.org",
        "https://attacker.io",
        "http://evil.localhost.com",
        "http://127.0.0.1.attacker.com",
    ]

    for origin in unauthorized_origins:
        res = client.get("/api/state", headers={"Origin": origin})
        # The request itself may complete (or be un-corsed), but CORS headers MUST NOT permit the origin
        cors_allow_origin = res.headers.get("access-control-allow-origin")
        assert cors_allow_origin != origin, f"Origin {origin} was improperly allowed!"
        assert cors_allow_origin != "*", f"Wildcard origin returned for {origin}!"


def test_unauthorized_preflight_rejected(client):
    """Verify preflight OPTIONS requests from unauthorized origins do not receive allow headers."""
    res = client.options(
        "/api/state",
        headers={
            "Origin": "https://evil.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    cors_allow_origin = res.headers.get("access-control-allow-origin")
    assert cors_allow_origin != "https://evil.com"
    assert cors_allow_origin != "*"


def test_authorized_loopback_allowed(client):
    """Verify legitimate loopback origins are granted CORS access with credentials."""
    for origin in ["http://localhost:8000", "http://127.0.0.1:8000"]:
        res = client.get("/api/state", headers={"Origin": origin})
        assert res.headers.get("access-control-allow-origin") == origin
        assert res.headers.get("access-control-allow-credentials") == "true"


def test_authorized_preflight_allowed(client):
    """Verify preflight OPTIONS requests from trusted origins receive appropriate headers."""
    res = client.options(
        "/api/state",
        headers={
            "Origin": "http://127.0.0.1:8000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert res.headers.get("access-control-allow-origin") == "http://127.0.0.1:8000"
    assert res.headers.get("access-control-allow-credentials") == "true"
    allow_methods = res.headers.get("access-control-allow-methods", "")
    assert "POST" in allow_methods or "GET" in allow_methods


def test_get_allowed_origins_env_parsing(monkeypatch):
    """Verify custom origins from environment variables are parsed and wildcards rejected."""
    # Ensure wildcard is discarded
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://trading.secure-domain.com, *, http://another-allowed.com/")
    origins = get_allowed_origins()

    assert "https://trading.secure-domain.com" in origins
    assert "http://another-allowed.com" in origins
    assert "*" not in origins

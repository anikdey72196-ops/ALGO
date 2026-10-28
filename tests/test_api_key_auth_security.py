"""
test_api_key_auth_security.py — Comprehensive security test suite for API Key authentication.
Tests:
- Local development behavior (open when unconfigured in non-prod).
- Strict 401 rejection on missing/invalid API key when ALGO_API_KEY is configured.
- Valid API key via X-API-Key header, Authorization header, and query param.
- Fail-Secure enforcement in production (blocks open access when key is missing).
- WebSocket handshake security enforcement (CWE-306).
"""

import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from web_app import app, API_KEY_ENV, get_api_key_header


@pytest.fixture
def client():
    return TestClient(app)


def test_api_open_in_dev_when_unconfigured(client, monkeypatch):
    """In development without ALGO_API_KEY, API routes are accessible."""
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("ALGO_ENV", raising=False)

    res = client.get("/api/state")
    assert res.status_code == 200


def test_api_rejects_missing_key_when_configured(client, monkeypatch):
    """When ALGO_API_KEY is configured, requests without key must return 401."""
    monkeypatch.setenv(API_KEY_ENV, "super-secret-key-12345")
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    res = client.get("/api/state")
    assert res.status_code == 401
    assert "Unauthorized" in res.text


def test_api_rejects_wrong_key(client, monkeypatch):
    """When ALGO_API_KEY is configured, wrong key returns 401."""
    monkeypatch.setenv(API_KEY_ENV, "super-secret-key-12345")

    res = client.get("/api/state", headers={"X-API-Key": "wrong-key"})
    assert res.status_code == 401


def test_api_accepts_valid_header_key(client, monkeypatch):
    """When ALGO_API_KEY is configured, valid X-API-Key returns 200."""
    secret = "super-secret-key-12345"
    monkeypatch.setenv(API_KEY_ENV, secret)

    res = client.get("/api/state", headers={"X-API-Key": secret})
    assert res.status_code == 200


def test_api_accepts_bearer_token(client, monkeypatch):
    """Bearer token in Authorization header is accepted."""
    secret = "super-secret-key-12345"
    monkeypatch.setenv(API_KEY_ENV, secret)

    res = client.get("/api/state", headers={"Authorization": f"Bearer {secret}"})
    assert res.status_code == 200


def test_api_fail_secure_in_production_without_key(client, monkeypatch):
    """CWE-1188 Remediation: In production environment, missing ALGO_API_KEY blocks access (500), does NOT fail-open."""
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")

    res = client.get("/api/state")
    assert res.status_code == 500
    assert "ALGO_API_KEY required in production" in res.text


def test_websocket_rejects_unauthorized_when_key_configured(client, monkeypatch):
    """WebSocket handshake rejects connection without key when configured."""
    monkeypatch.setenv(API_KEY_ENV, "ws-secret-key")

    with pytest.raises(Exception):
        with client.websocket_connect("/ws/state") as ws:
            pass


def test_websocket_accepts_valid_key_in_query(client, monkeypatch):
    """WebSocket accepts connection with valid api_key query param."""
    secret = "ws-secret-key"
    monkeypatch.setenv(API_KEY_ENV, secret)

    with client.websocket_connect(f"/ws/state?api_key={secret}") as ws:
        data = ws.receive_json()
        assert "is_active" in data


def test_clear_trades_endpoint_auth(client, monkeypatch):
    """Verify /api/trades/clear enforces authentication dependency."""
    secret = "trades-secret-key-123"
    monkeypatch.setenv(API_KEY_ENV, secret)

    # 1. Missing auth returns 401
    res_unauth = client.post("/api/trades/clear")
    assert res_unauth.status_code == 401

    # 2. Invalid auth returns 401
    res_wrong = client.post("/api/trades/clear", headers={"X-API-Key": "wrong-key"})
    assert res_wrong.status_code == 401

    # 3. Valid X-API-Key returns 200
    res_auth = client.post("/api/trades/clear", headers={"X-API-Key": secret})
    assert res_auth.status_code == 200
    assert res_auth.json().get("status") == "success"

    # 4. Valid Bearer token returns 200
    res_bearer = client.post("/api/trades/clear", headers={"Authorization": f"Bearer {secret}"})
    assert res_bearer.status_code == 200
    assert res_bearer.json().get("status") == "success"


def test_halt_endpoint_auth(client, monkeypatch):
    """Verify /api/halt enforces authentication dependency."""
    secret = "halt-secret-key-456"
    monkeypatch.setenv(API_KEY_ENV, secret)

    # 1. Missing auth returns 401
    res_unauth = client.post("/api/halt")
    assert res_unauth.status_code == 401

    # 2. Invalid auth returns 401
    res_wrong = client.post("/api/halt", headers={"X-API-Key": "wrong-key"})
    assert res_wrong.status_code == 401

    # 3. Valid X-API-Key allows authorized execution
    res_auth = client.post("/api/halt", headers={"X-API-Key": secret})
    assert res_auth.status_code == 200


def test_deactivate_endpoint_auth(client, monkeypatch):
    """Verify /api/deactivate enforces authentication dependency."""
    secret = "deactivate-secret-key-789"
    monkeypatch.setenv(API_KEY_ENV, secret)

    # 1. Missing auth returns 401
    res_unauth = client.post("/api/deactivate")
    assert res_unauth.status_code == 401

    # 2. Invalid auth returns 401
    res_wrong = client.post("/api/deactivate", headers={"X-API-Key": "wrong-key"})
    assert res_wrong.status_code == 401

    # 3. Valid X-API-Key allows authorized execution
    res_auth = client.post("/api/deactivate", headers={"X-API-Key": secret})
    assert res_auth.status_code == 200




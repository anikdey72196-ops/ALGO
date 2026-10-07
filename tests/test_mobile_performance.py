import pytest
from fastapi.testclient import TestClient
from web_app import app, bot_instance

@pytest.fixture
def client():
    return TestClient(app)

def test_gzip_compression_active(client):
    """Test that responses larger than threshold are compressed with gzip when requested."""
    response = client.get("/api/state", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"

def test_compact_state_endpoint(client):
    """Test /api/state/compact returns minimal essential keys without heavy logs."""
    response = client.get("/api/state/compact")
    assert response.status_code == 200
    data = response.json()
    assert "is_active" in data
    assert "equity" in data
    assert "daily_pnl" in data
    assert "trades_today" in data
    assert "circuit_breaker_active" in data
    assert "winning_trades" in data
    assert "losing_trades" in data
    assert "pairs_config" in data
    # Ensure heavy verbose arrays are stripped
    assert "recent_logs" not in data
    assert "ml_logs" not in data
    assert "ml_trap_detector" not in data

def test_state_endpoint_compact_query_param(client):
    """Test /api/state?compact=true returns lightweight payload."""
    response = client.get("/api/state?compact=true")
    assert response.status_code == 200
    data = response.json()
    assert "is_active" in data
    assert "recent_logs" not in data

def test_compact_logs_endpoint(client):
    """Test /api/logs/compact returns trimmed log lines."""
    bot_instance.log("A" * 200)
    response = client.get("/api/logs/compact?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "logs" in data
    for log in data["logs"]:
        assert len(log) <= 165  # 160 + ellipsis

def test_streaming_csv_exports(client):
    """Test /api/trades/download_csv and /api/sessions/export_csv stream CSV content."""
    res_trades = client.get("/api/trades/download_csv")
    assert res_trades.status_code == 200
    assert "text/csv" in res_trades.headers["content-type"]
    assert "id,timestamp" in res_trades.text

    res_sessions = client.get("/api/sessions/export_csv")
    assert res_sessions.status_code == 200
    assert "text/csv" in res_sessions.headers["content-type"]
    assert "id,activation_time" in res_sessions.text

def test_network_info_cached(client):
    """Test that /api/network-info is fast and cached."""
    res1 = client.get("/api/network-info")
    assert res1.status_code == 200
    res2 = client.get("/api/network-info")
    assert res2.status_code == 200
    assert res1.json() == res2.json()

def test_batched_daily_summary():
    """Test State.get_daily_summary returns all daily fields in single query."""
    summary = bot_instance.state.get_daily_summary()
    assert "realized_pnl" in summary
    assert "trade_count" in summary
    assert "circuit_breaker_active" in summary

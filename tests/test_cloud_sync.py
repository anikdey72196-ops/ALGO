import os
import sqlite3
import tempfile
from pathlib import Path
import pytest
import joblib
from fastapi.testclient import TestClient

from core.cloud_sync import CloudSyncEngine, CloudSyncClient
from web_app import app


@pytest.fixture
def temp_sync_env():
    """Create isolated temporary database and artifact directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "test_trading_state.db"
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Create tables
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE ml_events (
                event_id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                kind TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry REAL NOT NULL,
                stop REAL NOT NULL,
                target REAL NOT NULL,
                features TEXT NOT NULL,
                label INTEGER,
                outcome TEXT,
                r_multiple REAL,
                label_ts TEXT,
                p_genuine REAL,
                model_version TEXT,
                allowed INTEGER,
                strategy TEXT NOT NULL DEFAULT 'SMC'
            );
        """)
        conn.execute("""
            CREATE TABLE trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                lot_size REAL,
                realized_pnl REAL DEFAULT 0.0,
                status TEXT DEFAULT 'OPEN',
                strategy_name TEXT DEFAULT 'SMC',
                magic_number INTEGER DEFAULT 123456
            );
        """)
        conn.execute("""
            CREATE TABLE partial_tp_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                direction TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                current_price REAL NOT NULL,
                r_multiple REAL NOT NULL,
                p_reversal REAL NOT NULL,
                p_full_tp REAL NOT NULL,
                predicted_max_r REAL,
                action TEXT NOT NULL,
                close_pct REAL NOT NULL,
                closed_lot REAL DEFAULT 0.0,
                remaining_lot REAL NOT NULL,
                nearest_resistance TEXT,
                structure_confluence_count INTEGER DEFAULT 0,
                features_json TEXT,
                realized_max_r REAL,
                outcome_label TEXT,
                is_shadow INTEGER DEFAULT 0
            );
        """)
        # Insert seed event
        conn.execute("""
            INSERT INTO ml_events (event_id, ts, symbol, timeframe, kind, direction, entry, stop, target, features, strategy)
            VALUES ('evt_1', '2026-10-09T10:00:00Z', 'EURUSD', '5m', 'SWEEP', 'BUY', 1.0850, 1.0830, 1.0890, '{}', 'SMC');
        """)
        conn.execute("""
            INSERT INTO trade_log (timestamp, symbol, direction, entry_price, stop_loss, take_profit, lot_size, status)
            VALUES ('2026-10-09T10:00:00Z', 'EURUSD', 'BUY', 1.0850, 1.0830, 1.0890, 0.1, 'CLOSED');
        """)
        conn.commit()
        conn.close()

        # Save dummy model artifact
        dummy_model = {"model_name": "test_detector", "weights": [1, 2, 3]}
        model_file = artifacts_dir / "trap_detector_SMC.joblib"
        joblib.dump(dummy_model, model_file)

        engine = CloudSyncEngine(db_path=db_path, artifacts_dir=artifacts_dir)
        yield engine, db_path, artifacts_dir


def test_engine_export_events(temp_sync_env):
    engine, _, _ = temp_sync_env
    data = engine.export_events()
    assert data["counts"]["ml_events"] == 1
    assert data["counts"]["trade_log"] == 1
    assert data["ml_events"][0]["event_id"] == "evt_1"


def test_engine_import_events_deduplication(temp_sync_env):
    engine, db_path, _ = temp_sync_env
    # Prepare payload with 1 duplicate (evt_1) and 1 new (evt_2)
    payload = {
        "ml_events": [
            {
                "event_id": "evt_1",
                "ts": "2026-10-09T10:00:00Z",
                "symbol": "EURUSD",
                "timeframe": "5m",
                "kind": "SWEEP",
                "direction": "BUY",
                "entry": 1.0850,
                "stop": 1.0830,
                "target": 1.0890,
                "features": "{}",
                "strategy": "SMC",
            },
            {
                "event_id": "evt_2",
                "ts": "2026-10-09T11:00:00Z",
                "symbol": "GBPUSD",
                "timeframe": "15m",
                "kind": "SWEEP",
                "direction": "SELL",
                "entry": 1.3050,
                "stop": 1.3080,
                "target": 1.3000,
                "features": "{}",
                "strategy": "ICT",
            },
        ],
        "trade_log": [
            {
                "timestamp": "2026-10-09T10:00:00Z",
                "symbol": "EURUSD",
                "direction": "BUY",
                "entry_price": 1.0850,
                "stop_loss": 1.0830,
                "take_profit": 1.0890,
                "lot_size": 0.1,
                "magic_number": 123456,
            },
            {
                "timestamp": "2026-10-09T12:00:00Z",
                "symbol": "XAUUSD",
                "direction": "BUY",
                "entry_price": 2650.0,
                "stop_loss": 2640.0,
                "take_profit": 2670.0,
                "lot_size": 0.01,
                "magic_number": 123456,
            },
        ],
    }

    stats = engine.import_events(payload)
    assert stats["ml_events_inserted"] == 1  # Only evt_2 inserted
    assert stats["trade_log_inserted"] == 1  # Only XAUUSD inserted

    # Verify total count in DB
    conn = engine.get_db_connection()
    ml_total = conn.execute("SELECT count(*) FROM ml_events").fetchone()[0]
    trade_total = conn.execute("SELECT count(*) FROM trade_log").fetchone()[0]
    conn.close()
    assert ml_total == 2
    assert trade_total == 2


def test_models_manifest_and_atomic_replace(temp_sync_env):
    engine, _, artifacts_dir = temp_sync_env
    manifest = engine.get_models_manifest()
    assert "trap_detector_SMC.joblib" in manifest
    assert len(manifest["trap_detector_SMC.joblib"]["sha256"]) == 64

    # Test saving corrupted model (should be rejected)
    ok, msg = engine.save_model_artifact("trap_detector_SMC.joblib", b"CORRUPTED_BYTES_NOT_JOBLIB")
    assert not ok
    assert "verification failed" in msg.lower()

    # Test saving valid model
    import io
    bio = io.BytesIO()
    joblib.dump({"valid": True}, bio)
    valid_bytes = bio.getvalue()

    ok, msg = engine.save_model_artifact("trap_detector_SMC.joblib", valid_bytes)
    assert ok

    # Test path traversal attempts (should be rejected)
    ok_trav1, _ = engine.save_model_artifact("../evil.joblib", valid_bytes)
    assert not ok_trav1
    ok_trav2, _ = engine.save_model_artifact("subdir/evil.joblib", valid_bytes)
    assert not ok_trav2
    ok_trav3, _ = engine.save_model_artifact("evil.exe", valid_bytes)
    assert not ok_trav3

    # Verify loaded
    loaded = joblib.load(artifacts_dir / "trap_detector_SMC.joblib")
    assert loaded.get("valid") is True


def test_api_sync_endpoints(monkeypatch):
    token = os.getenv("SYNC_AUTH_TOKEN", "test_sync_token")
    monkeypatch.setenv("SYNC_AUTH_TOKEN", token)
    headers = {"Authorization": f"Bearer {token}"}
    client = TestClient(app)

    # 1. Status
    res = client.get("/api/sync/status", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "counts" in data
    assert "models_count" in data

    # 2. Manifest
    res = client.get("/api/sync/models/manifest", headers=headers)
    assert res.status_code == 200
    assert "models" in res.json()

    # 3. Export
    res = client.post("/api/sync/events/export?limit=5", headers=headers)
    assert res.status_code == 200
    exp = res.json()
    assert "ml_events" in exp
    assert "trade_log" in exp

    # 4. Trigger Retrain Endpoint
    res = client.post("/api/sync/trigger_retrain", headers=headers)
    assert res.status_code == 200
    retrain_data = res.json()
    assert retrain_data.get("status") == "success"

    # 5. Unauthorized Check
    res_unauth = client.post("/api/sync/trigger_retrain")
    assert res_unauth.status_code == 401

    # 6. run_client_sync Auth & Validation Checks
    res_sync_unauth = client.post("/api/sync/run_client_sync")
    assert res_sync_unauth.status_code == 401

    res_sync_bad_auth = client.post("/api/sync/run_client_sync", headers={"Authorization": "Bearer wrong-token"})
    assert res_sync_bad_auth.status_code == 401

    monkeypatch.delenv("CLOUD_SYNC_URL", raising=False)
    res_sync_no_url = client.post("/api/sync/run_client_sync", headers=headers)
    assert res_sync_no_url.status_code == 400

    res_sync_bad_url = client.post("/api/sync/run_client_sync?target_url=ftp://evil.com", headers=headers)
    assert res_sync_bad_url.status_code == 400

    from unittest.mock import patch
    with patch("web_app.CloudSyncClient.full_two_way_sync", return_value={"status": "ok"}):
        res_sync_ok = client.post("/api/sync/run_client_sync?target_url=https://cloud.example.com", headers=headers)
        assert res_sync_ok.status_code == 200



def test_engine_import_partial_tp_events_batch(temp_sync_env):
    """Verify batch insertion and deduplication of partial_tp_events."""
    engine, db_path, _ = temp_sync_env

    # 1. Insert initial batch of 3 events
    initial_events = [
        {
            "trade_id": 100,
            "symbol": "XAUUSD",
            "strategy_name": "SMC",
            "direction": "BUY",
            "timestamp": "2026-10-18T10:00:00Z",
            "current_price": 2650.0,
            "r_multiple": 1.5,
            "p_reversal": 0.25,
            "p_full_tp": 0.75,
            "predicted_max_r": 3.0,
            "action": "PARTIAL_CLOSE",
            "close_pct": 50.0,
            "closed_lot": 0.05,
            "remaining_lot": 0.05,
            "nearest_resistance": "2660.0",
            "structure_confluence_count": 2,
            "features_json": "{}",
            "realized_max_r": 2.0,
            "outcome_label": "WIN",
            "is_shadow": 0,
        },
        {
            "trade_id": 101,
            "symbol": "EURUSD",
            "strategy_name": "ICT",
            "direction": "SELL",
            "timestamp": "2026-10-18T10:05:00Z",
            "current_price": 1.0850,
            "r_multiple": 2.0,
            "p_reversal": 0.40,
            "p_full_tp": 0.60,
            "predicted_max_r": 2.5,
            "action": "PARTIAL_CLOSE",
            "close_pct": 50.0,
            "closed_lot": 0.10,
            "remaining_lot": 0.10,
            "nearest_resistance": "1.0830",
            "structure_confluence_count": 1,
            "features_json": "{}",
            "realized_max_r": 2.2,
            "outcome_label": "WIN",
            "is_shadow": 0,
        },
    ]

    stats1 = engine.import_events({"partial_tp_events": initial_events})
    assert stats1["partial_tp_inserted"] == 2

    # 2. Second batch: 1 duplicate (trade_id 100) + 1 new (trade_id 102)
    second_batch = [
        initial_events[0],  # Duplicate: same trade_id, timestamp, action
        {
            "trade_id": 102,
            "symbol": "GBPUSD",
            "strategy_name": "ORDER_FLOW",
            "direction": "BUY",
            "timestamp": "2026-10-18T10:15:00Z",
            "current_price": 1.3050,
            "r_multiple": 1.8,
            "p_reversal": 0.30,
            "p_full_tp": 0.70,
            "predicted_max_r": 2.8,
            "action": "PARTIAL_CLOSE",
            "close_pct": 50.0,
            "closed_lot": 0.08,
            "remaining_lot": 0.08,
            "nearest_resistance": "1.3100",
            "structure_confluence_count": 3,
            "features_json": "{}",
            "realized_max_r": 2.5,
            "outcome_label": "WIN",
            "is_shadow": 0,
        },
    ]

    stats2 = engine.import_events({"partial_tp_events": second_batch})
    assert stats2["partial_tp_inserted"] == 1  # Only trade_id 102 inserted

    # Verify total count in table
    conn = engine.get_db_connection()
    total = conn.execute("SELECT count(*) FROM partial_tp_events").fetchone()[0]
    conn.close()
    assert total == 3


def test_api_sync_models_path_traversal_rejected(monkeypatch):
    """Verify download and upload endpoints reject path traversal attacks (CWE-22)."""
    token = os.getenv("SYNC_AUTH_TOKEN", "test_sync_token")
    monkeypatch.setenv("SYNC_AUTH_TOKEN", token)
    headers = {"Authorization": f"Bearer {token}"}
    client = TestClient(app)

    # 1. Traversal attempts in download
    bad_filenames = [
        "../../etc/passwd",
        "..\\..\\trading_state.db",
        "subdir/model.joblib",
        "model.joblib/../../evil.joblib",
        "/absolute/path/model.joblib",
    ]

    for bad in bad_filenames:
        res = client.get(f"/api/sync/models/download/{bad}", headers=headers)
        # Should be rejected with 400 or 404, never 200 or 500
        assert res.status_code in (400, 404), f"Path traversal for {bad} should fail, got {res.status_code}"

    # 2. Traversal attempts in upload
    for bad in bad_filenames:
        res = client.post(f"/api/sync/models/upload/{bad}", headers=headers, content=b"payload")
        assert res.status_code in (400, 404), f"Upload path traversal for {bad} should fail, got {res.status_code}"


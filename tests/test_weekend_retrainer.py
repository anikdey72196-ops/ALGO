import pytest
from fastapi.testclient import TestClient

from ml.weekend_retrainer import WeekendRetrainService, weekend_retrain_service
from web_app import app
from core.config import TradingConfig


def test_weekend_retrain_service_init():
    svc = WeekendRetrainService()
    assert svc.is_running is False
    info = svc.get_last_run_info()
    assert isinstance(info, dict)
    assert "status" in info


def test_config_weekend_retrain_defaults():
    cfg = TradingConfig()
    assert cfg.weekend_retrain_enabled is True
    assert cfg.weekend_retrain_day == "sat"
    assert cfg.weekend_retrain_hour_utc == 2
    assert cfg.weekend_retrain_sync_cloud is True


def test_api_weekend_retrain_status():
    client = TestClient(app)
    res = client.get("/api/ml/weekend_retrain/status")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "is_running" in data


def test_api_weekend_retrain_trigger():
    client = TestClient(app)
    # Background execution trigger test
    res = client.post("/api/ml/weekend_retrain?background=true")
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") in ("success", "in_progress")

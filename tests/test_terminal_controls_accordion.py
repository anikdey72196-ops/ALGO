import pytest
from fastapi.testclient import TestClient
from web_app import app, bot_instance

client = TestClient(app)

def test_controls_page_accordion_structure():
    """Verify all 12 pairs and accordion controls render properly on /terminal/controls."""
    res = client.get("/terminal/controls")
    assert res.status_code == 200
    html = res.text

    # Expected currency pairs from user drawing and catalog
    expected_pairs = [
        "EURUSD", "EURJPY", "USDCHF", "AUDJPY", "GBPUSD",
        "CADCHF", "NZDCAD", "NZDCHF", "NZDUSD", "XAUUSD", "BTCUSD", "ETHUSD"
    ]

    for pair in expected_pairs:
        assert f'data-symbol="{pair}"' in html, f"Missing pair card for {pair}"
        assert f"{pair} Settings" in html or f"{pair} Controls" in html or pair in html

    # Accordion markers and interactive controls
    assert "pair-accordion-header" in html
    assert "pair-accordion-content" in html
    assert "pair-switch" in html
    assert "input-lot" in html
    assert "input-sl" in html
    assert "est-risk" in html
    assert "toggleAllPairsBtn" in html
    assert "applyParamsBtn" in html

def test_pair_toggle_and_security_validation():
    """Verify security validation and per-pair parameter update."""
    client.post("/api/deactivate")

    # 1. Valid update with enable toggle, lot and sl
    res = client.post("/api/update", json={
        "action": "update_pair_settings",
        "pair": "EURUSD",
        "lot": 0.15,
        "sl": 18.0,
        "enabled": True
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["symbol"] == "EURUSD"
    assert data["lot"] == 0.15
    assert data["sl"] == 18.0
    assert data["enabled"] is True

    # 2. Security validation: reject injection attempts
    malicious_payloads = [
        {"action": "update_pair_settings", "pair": "<script>alert(1)</script>"},
        {"action": "update_pair_settings", "pair": "EUR/USD; DROP TABLE;"},
        {"action": "update_pair_settings", "pair": "A" * 50},
        {"action": "update_pair_settings", "pair": "E"},
    ]
    for p in malicious_payloads:
        res_sec = client.post("/api/update", json=p)
        assert res_sec.status_code == 400

    # 3. Save all pairs bulk update security & validity
    res_bulk = client.post("/api/update", json={
        "action": "save_all_pairs",
        "pairs": {
            "EURUSD": {"lot": 0.10, "sl": 15.0, "enabled": True},
            "EURJPY": {"lot": 0.12, "sl": 25.0, "enabled": False},
            "USDCHF": {"lot": 0.08, "sl": 20.0, "enabled": True}
        }
    })
    assert res_bulk.status_code == 200
    assert res_bulk.json()["status"] == "success"

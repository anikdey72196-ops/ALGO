"""
test_web_api.py — Automated test script to verify FastAPI endpoints and lock enforcement.
"""

import sys
import asyncio
from fastapi.testclient import TestClient
from web_app import app

def test_api_workflow():
    client = TestClient(app)

    print("1. Testing GET /api/state...")
    res = client.get("/api/state")
    assert res.status_code == 200, f"Status code: {res.status_code}"
    state = res.json()
    print(f"   Initial state: is_active={state['is_active']}, symbols={state['selected_symbols']}, equity={state['equity']}")
    assert state['is_active'] is False, "Bot should start deactivated"

    print("\n2. Testing POST /api/configure (set symbols to XAUUSD + BTCUSD and lot size to 0.10)...")
    res = client.post("/api/configure", json={
        "selected_symbols": ["XAUUSD", "BTCUSD"],
        "fixed_lot_size": 0.10
    })
    assert res.status_code == 200, f"Configure failed: {res.text}"
    conf_data = res.json()
    print(f"   Updated config: symbols={conf_data['selected_symbols']}, fixed_lot={conf_data['fixed_lot_size']}")

    print("\n3. Testing POST /api/activate...")
    res = client.post("/api/activate")
    assert res.status_code == 200
    state = client.get("/api/state").json()
    print(f"   State after activation: is_active={state['is_active']}")
    assert state['is_active'] is True, "Bot should now be active"

    print("\n4. Testing STRICT LOCK: attempting to change symbols while ACTIVE (must fail with 400)...")
    res = client.post("/api/configure", json={
        "selected_symbols": ["EURUSD", "GBPUSD"],
        "fixed_lot_size": 0.05
    })
    print(f"   Response status: {res.status_code}, error detail: {res.json().get('detail')}")
    assert res.status_code == 400, "Should reject symbol change while active!"
    print("   [PASSED] Lock enforcement verified.")

    print("\n5. Testing POST /api/trigger_tick (while active)...")
    res = client.post("/api/trigger_tick")
    assert res.status_code == 200
    print(f"   Trigger result: {res.json()['message']}")

    print("\n6. Testing POST /api/deactivate...")
    res = client.post("/api/deactivate")
    assert res.status_code == 200
    state = client.get("/api/state").json()
    print(f"   State after deactivation: is_active={state['is_active']}")
    assert state['is_active'] is False, "Bot should now be deactivated"

    print("\n8. Testing GET /api/trades (Trade history ledger)...")
    res = client.get("/api/trades")
    assert res.status_code == 200
    trades_resp = res.json()
    print(f"   Trade history: total_trades={trades_resp['total_trades']}, total_pnl=${trades_resp['total_pnl']}")
    assert "trades" in trades_resp

    print("\n9. Testing AI Confirmation Gate & Fixed Stop Loss in pips...")
    res = client.post("/api/configure", json={
        "selected_symbols": ["XAUUSD", "BTCUSD"],
        "fixed_lot_size": 0.05,
        "fixed_sl_pips": 25.0,
        "ai_confirmation_enabled": True
    })
    assert res.status_code == 200
    state = client.get("/api/state").json()
    assert state['ai_confirmation_enabled'] is True
    assert state['fixed_sl_pips'] == 25.0
    print(f"   AI Gate status: enabled={state['ai_confirmation_enabled']}, threshold={state['ai_confidence_threshold']}%")
    print("\n10. Testing Chart 1 = XAUUSD and Chart 2 = NONE (Single pair mode)...")
    res = client.post("/api/configure", json={
        "selected_symbols": ["XAUUSD", "NONE"],
        "fixed_lot_size": 0.05,
        "fixed_sl_pips": 20.0
    })
    assert res.status_code == 200
    state = client.get("/api/state").json()
    assert state['selected_symbols'] == ["XAUUSD"], f"Symbols: {state['selected_symbols']}"
    print(f"   Single pair configured successfully: active_symbols={state['selected_symbols']}")

    print("\nAll Web API integration tests passed successfully!")


if __name__ == "__main__":
    test_api_workflow()


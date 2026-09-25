"""
test_web_api.py — Automated test script to verify FastAPI endpoints and lock enforcement.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import asyncio
from fastapi.testclient import TestClient
from web_app import app, bot_instance

def test_api_workflow():
    client = TestClient(app)

    print("1. Testing GET /api/state...")
    res = client.get("/api/state")
    assert res.status_code == 200, f"Status code: {res.status_code}"
    state = res.json()
    print(f"   Initial state: is_active={state['is_active']}, symbols={state['selected_symbols']}, strategy_type={state['strategy_type']}, equity={state['equity']}")
    assert state['is_active'] is False, "Bot should start deactivated"
    assert "strategy_type" in state, "strategy_type must be in BotStateResponse"
    assert state['strategy_type'] in ["SMC", "SMC_SCALP_5M", "ICT", "ORDER_FLOW"], f"Expected valid strategy_type, got {state['strategy_type']}"
    assert "available_strategies" in state, "available_strategies must be in BotStateResponse"
    assert "SMC_SCALP_5M" in state['available_strategies'], "SMC_SCALP_5M must be in available_strategies"

    assert "pair1" in state and "pair2" in state and "pair3" in state, "pair1, pair2, and pair3 must all be in BotStateResponse"

    print("\n2. Testing POST /api/configure (set symbols to XAUUSD + BTCUSD, strategy_type to SMC_SCALP_5M and lot size to 0.10)...")
    res = client.post("/api/configure", json={
        "selected_symbols": ["XAUUSD", "BTCUSD"],
        "strategy_type": "SMC_SCALP_5M",
        "fixed_lot_size": 0.10
    })
    assert res.status_code == 200, f"Configure failed: {res.text}"
    conf_data = res.json()
    print(f"   Updated config: symbols={conf_data['selected_symbols']}, strategy_type={conf_data['strategy_type']}, fixed_lot={conf_data['fixed_lot_size']}")
    assert conf_data['strategy_type'] == "SMC_SCALP_5M"

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
    print("   [PASSED] Symbol lock enforcement verified.")

    print("\n4b. Testing STRICT LOCK: attempting to change strategy_type while ACTIVE (must fail with 400)...")
    res = client.post("/api/configure", json={
        "selected_symbols": ["XAUUSD", "BTCUSD"],
        "strategy_type": "SMC",
        "fixed_lot_size": 0.10
    })
    print(f"   Response status: {res.status_code}, error detail: {res.json().get('detail')}")
    assert res.status_code == 400, "Should reject strategy change while active!"
    assert "Strategy type changes are LOCKED" in res.json().get("detail", ""), "Detail should state strategy changes are locked"
    print("   [PASSED] Strategy lock enforcement verified.")

    print("\n4c. Testing STRICT LOCK: attempting to change fixed_lot_size while ACTIVE (must fail with 400)...")
    res = client.post("/api/configure", json={
        "fixed_lot_size": 0.25
    })
    print(f"   Response status: {res.status_code}, error detail: {res.json().get('detail')}")
    assert res.status_code == 400, "Should reject lot size change while active!"
    assert "Lot size is LOCKED" in res.json().get("detail", "")
    print("   [PASSED] Global lot size lock enforcement verified.")

    print("\n4d. Testing STRICT LOCK: attempting to change fixed_sl_pips while ACTIVE (must fail with 400)...")
    res = client.post("/api/configure", json={
        "fixed_sl_pips": 50.0
    })
    print(f"   Response status: {res.status_code}, error detail: {res.json().get('detail')}")
    assert res.status_code == 400, "Should reject SL change while active!"
    assert "Stop Loss (SL) is LOCKED" in res.json().get("detail", "")
    print("   [PASSED] Global SL lock enforcement verified.")

    print("\n4e. Testing STRICT LOCK: attempting to change Pair 1, Pair 2 & Pair 3 lot size / SL while ACTIVE (must fail with 400)...")
    res = client.post("/api/configure", json={
        "pair1": {
            "symbol": "XAUUSD",
            "fixed_lot_size": 0.50,
            "fixed_sl_pips": 30.0,
            "enabled": True
        }
    })
    assert res.status_code == 400, "Should reject Pair 1 lot change while active!"
    assert "Pair 1 lot size is LOCKED" in res.json().get("detail", "")

    res = client.post("/api/configure", json={
        "pair2": {
            "symbol": "BTCUSD",
            "fixed_lot_size": bot_instance.config.pair2.fixed_lot_size,
            "fixed_sl_pips": 99.0,
            "enabled": True
        }
    })
    assert res.status_code == 400, "Should reject Pair 2 SL change while active!"
    assert "Pair 2 Stop Loss (SL) is LOCKED" in res.json().get("detail", "")

    res = client.post("/api/configure", json={
        "pair3": {
            "symbol": bot_instance.config.pair3.symbol,
            "fixed_lot_size": 0.88,
            "fixed_sl_pips": 20.0,
            "enabled": True
        }
    })
    assert res.status_code == 400, "Should reject Pair 3 lot change while active!"
    assert "Pair 3 lot size is LOCKED" in res.json().get("detail", "")
    print("   [PASSED] Per-pair lot and SL lock enforcement verified for Pair 1, 2, and 3.")

    print("\n4f. Testing STRICT LOCK: attempting to change max_open_positions while ACTIVE (must fail with 400)...")
    res = client.post("/api/configure", json={
        "max_open_positions": 3
    })
    assert res.status_code == 400, "Should reject max open positions limit change while active!"
    assert "Max open positions limit is LOCKED" in res.json().get("detail", "")
    print("   [PASSED] Max open positions lock enforcement verified.")

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

    print("\n7. Testing configuration change when DEACTIVATED (should succeed)...")
    res = client.post("/api/configure", json={
        "selected_symbols": ["XAUUSD", "EURUSD"],
        "strategy_type": "SMC",
        "fixed_lot_size": 0.05
    })
    assert res.status_code == 200
    conf_data = res.json()
    assert conf_data['strategy_type'] == "SMC"
    print("   [PASSED] Deactivated strategy change verified.")

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

    print("\n11. Testing Persistence to bot_settings.json (167 pips SL, SMC_SCALP_5M strategy)...")
    res = client.post("/api/configure", json={
        "selected_symbols": ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD", "ETHUSD"],
        "strategy_type": "SMC_SCALP_5M",
        "fixed_lot_size": 0.05,
        "fixed_sl_pips": 167.0,
        "ai_confirmation_enabled": True
    })
    assert res.status_code == 200
    state = client.get("/api/state").json()
    assert state['strategy_type'] == "SMC_SCALP_5M"
    assert state['fixed_sl_pips'] == 167.0

    from pathlib import Path
    import json
    settings_file = Path("bot_settings.json")
    assert settings_file.exists(), "bot_settings.json must be created upon configuration update!"
    with open(settings_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["strategy_type"] == "SMC_SCALP_5M", f"Saved strategy: {saved_data.get('strategy_type')}"
    assert saved_data["fixed_sl_pips"] == 167.0, f"Saved SL: {saved_data.get('fixed_sl_pips')}"
    print(f"   [PASSED] Persisted settings verified in bot_settings.json: {saved_data}")

    print("\nAll Web API integration tests passed successfully!")


if __name__ == "__main__":
    test_api_workflow()


"""
test_activation_history.py — Verify bot activation & deactivation history tracking.
"""

import time
from fastapi.testclient import TestClient
from web_app import app

def test_activation_history_flow():
    client = TestClient(app)

    print("1. Testing GET /api/activation_history initially...")
    res = client.get("/api/activation_history")
    assert res.status_code == 200
    data = res.json()
    assert "sessions" in data
    assert "stats" in data
    initial_session_count = len(data["sessions"])
    print(f"   Initial session count: {initial_session_count}")

    print("\n2. Activating bot via POST /api/activate...")
    act_res = client.post("/api/activate")
    assert act_res.status_code == 200
    act_data = act_res.json()
    assert act_data["status"] == "success"
    session_id = act_data.get("session_id")
    print(f"   Bot activated successfully! Session ID: {session_id}")
    assert session_id is not None

    print("\n3. Verifying GET /api/state contains active_session...")
    state_res = client.get("/api/state")
    assert state_res.status_code == 200
    state_data = state_res.json()
    assert state_data["is_active"] is True
    assert state_data["active_session"] is not None
    assert state_data["active_session"]["status"] == "ACTIVE"
    print(f"   Live Active Session: ID={state_data['active_session']['id']}, Duration={state_data['active_session']['formatted_duration']}")

    # Wait a brief moment to accumulate duration
    time.sleep(1.5)

    print("\n4. Verifying /api/activation_history shows the active session...")
    hist_res = client.get("/api/activation_history")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert len(hist_data["sessions"]) == initial_session_count + 1
    current = hist_data["sessions"][0]
    assert current["id"] == session_id
    assert current["status"] == "ACTIVE"
    assert current["duration_seconds"] >= 1.0
    print(f"   Active session confirmed: ID={current['id']}, Elapsed={current['formatted_duration']}, Reason={current['deactivation_reason']}")

    print("\n5. Deactivating bot via POST /api/deactivate...")
    deact_res = client.post("/api/deactivate")
    assert deact_res.status_code == 200
    deact_data = deact_res.json()
    assert deact_data["status"] == "success"
    print(f"   Deactivated session ID: {deact_data.get('session_id')}")

    print("\n6. Verifying session is COMPLETED with recorded duration...")
    hist_res2 = client.get("/api/activation_history")
    assert hist_res2.status_code == 200
    hist_data2 = hist_res2.json()
    closed = hist_data2["sessions"][0]
    assert closed["id"] == session_id
    assert closed["status"] == "COMPLETED"
    assert closed["deactivation_time"] is not None
    assert closed["deactivation_reason"] == "Manual User Stop"
    assert closed["duration_seconds"] >= 1.0
    print(f"   Session #{closed['id']} COMPLETED: Start={closed['activation_time_formatted']} | End={closed['deactivation_time_formatted']} | Duration={closed['formatted_duration']}")

    print("\n7. Verifying /api/activation_history stats...")
    stats = hist_data2["stats"]
    assert stats["is_active"] is False
    assert stats["total_sessions"] >= 1
    assert stats["total_uptime_seconds"] >= 1.0
    print(f"   Stats: Total Sessions={stats['total_sessions']}, Total Uptime={stats['formatted_total_uptime']}, Avg Duration={stats['formatted_avg_duration']}")

    print("\nAll Activation History integration tests PASSED successfully!")


if __name__ == "__main__":
    test_activation_history_flow()

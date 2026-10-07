import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from fastapi.testclient import TestClient
from web_app import app, bot_instance

def test_websocket_stream():
    # Ensure bot is in standby initially
    bot_instance.is_active = False
    client = TestClient(app)

    with client.websocket_connect("/ws/state") as websocket:
        # Upon connection, server pushes the immediate state
        data = websocket.receive_json()
        assert "is_active" in data
        assert "broker_info" in data
        assert "equity" in data
        print("[OK] Received initial state snapshot via WebSocket:", data["is_active"])

        # Test ping pong
        websocket.send_text("ping")
        msg = websocket.receive_text()
        assert msg == "pong"
        print("[OK] WebSocket ping/pong responded correctly")

        # Test refresh command
        websocket.send_text("refresh")
        refreshed = websocket.receive_json()
        assert "is_active" in refreshed
        print("[OK] WebSocket refresh responded with fresh state")

if __name__ == "__main__":
    test_websocket_stream()
    print("All WebSocket tests passed!")

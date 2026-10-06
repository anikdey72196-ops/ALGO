import unittest
from datetime import datetime, timezone
import pandas as pd
from unittest.mock import MagicMock

from core.config import TradingConfig, InstrumentConfig, MarketBias
from core.state import StateManager
from main import TradingBot
from web_app import app
from fastapi.testclient import TestClient


class TestDailyBiasCard(unittest.TestCase):
    def setUp(self):
        self.state = StateManager(db_path=":memory:")
        self.config = TradingConfig(
            instruments=[
                InstrumentConfig(symbol="XAUUSD", digits=2, pip_value=1.0, min_lot=0.01, max_lot=10.0, lot_step=0.01, point_size=0.01, tick_size=0.01),
                InstrumentConfig(symbol="EURUSD", digits=5, pip_value=10.0, min_lot=0.01, max_lot=10.0, lot_step=0.01, point_size=0.00001, tick_size=0.00001),
            ],
            selected_symbols=["XAUUSD", "EURUSD"],
            use_mock_broker=True,
            load_saved_settings=False,
        )

    def tearDown(self):
        self.state.close()

    def test_state_manager_daily_bias_persistence(self):
        """Test that daily bias records are saved and queried correctly."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        bias_data = {
            "date": today,
            "symbol": "XAUUSD",
            "bias": "BULLISH",
            "trend_clarity_score": 30.0,
            "ema_value": 2650.50,
            "current_price": 2665.00,
            "dealing_range_low": 2620.00,
            "dealing_range_high": 2680.00,
            "fib_50": 2650.00,
            "is_discount": True,
            "is_premium": False,
            "zone_status": "DISCOUNT",
            "first_zone_type": "FIRST_FVG_>0.5",
            "all_zones_failed": False,
            "reversal_risk": "LOW",
            "choch_detected": False,
            "summary": "Bullish bias in discount territory.",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        self.state.save_daily_bias(bias_data)
        record = self.state.get_daily_bias("XAUUSD", today)
        self.assertIsNotNone(record)
        self.assertEqual(record["symbol"], "XAUUSD")
        self.assertEqual(record["bias"], "BULLISH")
        self.assertEqual(record["zone_status"], "DISCOUNT")
        self.assertEqual(record["is_discount"], 1)

        # Test querying all daily biases
        all_biases = self.state.get_all_daily_biases(today)
        self.assertIn("XAUUSD", all_biases)

        # Test history query
        history = self.state.get_daily_bias_history("XAUUSD", limit=5)
        self.assertGreaterEqual(len(history), 1)

    def test_trading_bot_compute_daily_bias(self):
        """Test that TradingBot computes and summarizes daily bias."""
        bot = TradingBot(self.config, load_saved_settings=False)
        bot.startup()

        # Compute bias for XAUUSD (in mock mode)
        bias_res = bot.compute_daily_bias("XAUUSD")
        self.assertEqual(bias_res["symbol"], "XAUUSD")
        self.assertIn(bias_res["bias"], ["BULLISH", "BEARISH", "NEUTRAL"])
        self.assertIn("clarity_pct", bias_res)
        self.assertIn("zone_status", bias_res)
        self.assertIn("summary", bias_res)

        # Summary across all active symbols
        summary = bot.get_daily_bias_summary()
        self.assertIn("symbols", summary)
        self.assertIn("XAUUSD", summary["symbols"])
        self.assertIn("EURUSD", summary["symbols"])
        self.assertEqual(summary["primary_symbol"], "XAUUSD")
        self.assertIn(summary["primary_bias"], ["BULLISH", "BEARISH", "NEUTRAL"])

        bot.shutdown()

    def test_web_app_daily_bias_endpoints(self):
        """Test FastAPI endpoints /api/state and /api/daily-bias."""
        client = TestClient(app)
        res = client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("daily_bias", data)
        self.assertIn("primary_bias", data["daily_bias"])

        res_bias = client.get("/api/daily-bias")
        self.assertEqual(res_bias.status_code, 200)
        bias_data = res_bias.json()
        self.assertIn("symbols", bias_data)

        res_history = client.get("/api/daily-bias/history")
        self.assertEqual(res_history.status_code, 200)
        self.assertIsInstance(res_history.json(), list)


if __name__ == "__main__":
    unittest.main()

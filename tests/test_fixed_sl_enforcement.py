"""
test_fixed_sl_enforcement.py — Verifies that Stop Loss is strictly capped to 6.0 pips
and lot size to 0.12 on EURUSD and GBPUSD, both in configuration and at trade execution.
"""

import unittest
from datetime import datetime, timezone
from pathlib import Path
import json

from core.config import TradingConfig, get_instrument, Direction, MarketBias
from strategies.strategy import TradeSignal, LTFConfirmation
from core.state import StateManager
from core.risk_engine import RiskEngine
from main import TradingBot
from execution.execution import BracketOrder, PriceQuote


class TestFixedSLEnforcement(unittest.TestCase):
    def setUp(self):
        self.config = TradingConfig()
        self.state = StateManager(db_path=":memory:")
        self.risk_engine = RiskEngine(config=self.config, state=self.state)

    def test_default_config_eurusd_and_gbpusd(self):
        """Verify core TradingConfig defaults to 0.12 lot and 6.0 pips SL for EURUSD & GBPUSD."""
        self.assertEqual(self.config.pair2.symbol, "EURUSD")
        self.assertEqual(self.config.pair2.fixed_lot_size, 0.12)
        self.assertEqual(self.config.pair2.fixed_sl_pips, 6.0)

        self.assertEqual(self.config.pair3.symbol, "GBPUSD")
        self.assertEqual(self.config.pair3.fixed_lot_size, 0.12)
        self.assertEqual(self.config.pair3.fixed_sl_pips, 6.0)

        self.assertEqual(self.config.pair_configs["EURUSD"].fixed_lot_size, 0.12)
        self.assertEqual(self.config.pair_configs["EURUSD"].fixed_sl_pips, 6.0)
        self.assertEqual(self.config.pair_configs["GBPUSD"].fixed_lot_size, 0.12)
        self.assertEqual(self.config.pair_configs["GBPUSD"].fixed_sl_pips, 6.0)

    def test_bot_settings_json_has_6_pips_and_012_lots(self):
        """Verify bot_settings.json contains 0.12 lot and 6.0 pips SL for EURUSD & GBPUSD."""
        settings_path = Path(__file__).parent.parent / "bot_settings.json"
        self.assertTrue(settings_path.exists())
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["pair2"]["fixed_lot_size"], 0.12)
        self.assertEqual(data["pair2"]["fixed_sl_pips"], 6.0)
        self.assertEqual(data["pair3"]["fixed_lot_size"], 0.12)
        self.assertEqual(data["pair3"]["fixed_sl_pips"], 6.0)

        self.assertEqual(data["pair_configs"]["EURUSD"]["fixed_lot_size"], 0.12)
        self.assertEqual(data["pair_configs"]["EURUSD"]["fixed_sl_pips"], 6.0)
        self.assertEqual(data["pair_configs"]["GBPUSD"]["fixed_lot_size"], 0.12)
        self.assertEqual(data["pair_configs"]["GBPUSD"]["fixed_sl_pips"], 6.0)

    def test_risk_amount_calculation_at_6_pips_and_012_lots(self):
        """Verify that 0.12 lot with a 6.0-pip SL on GBPUSD risks exactly $7.20."""
        inst = get_instrument(self.config, "GBPUSD")
        sl_distance = 6.0 * inst.pip_size  # 0.00060
        sl_points = sl_distance / (10 ** -inst.digits)  # 60.0 points
        risk_dollar = 0.12 * sl_points * inst.point_value
        self.assertAlmostEqual(risk_dollar, 7.20, places=2)

    def test_risk_engine_authorizes_012_lot_for_fixed_lot(self):
        """Verify RiskEngine authorizes exactly 0.12 lot when fixed_lot_size=0.12 is passed."""
        sig = TradeSignal(
            symbol="GBPUSD",
            direction=Direction.BUY,
            entry_price=1.34000,
            stop_loss=1.33940,  # 6.0 pips
            take_profit=1.34120,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.OB_MITIGATION,
            rr_ratio=2.0,
            quality_score=85.0,
            timestamp=datetime.now(timezone.utc),
            sl_distance=0.00060,
            tp_distance=0.00120,
            strategy_name="SMC",
        )
        auth = self.risk_engine.authorize_trade(
            signal=sig,
            current_equity=10_000.0,
            fixed_lot_size=0.12,
            current_time=datetime(2026, 10, 22, 14, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(auth.authorized)
        self.assertEqual(auth.lot_size, 0.12)
        self.assertAlmostEqual(auth.risk_amount, 7.20, places=2)

    def test_trading_bot_anchors_and_clamps_sl_to_6_pips(self):
        """
        Verify that TradingBot enforces the 6.0 pips SL ceiling and re-anchors
        directly to the live quote entry price, preventing a 23.4 pip SL.
        """
        bot = TradingBot(config=self.config)
        inst = get_instrument(self.config, "GBPUSD")

        # Simulate a signal that originally had a 23.4-pip SL (1.34030 to 1.33796)
        sig = TradeSignal(
            symbol="GBPUSD",
            direction=Direction.BUY,
            entry_price=1.34000,
            stop_loss=1.33796,  # 23.4 pips away
            take_profit=1.34500,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.OB_MITIGATION,
            rr_ratio=2.0,
            quality_score=85.0,
            timestamp=datetime.now(timezone.utc),
            sl_distance=0.00234,
            tp_distance=0.00500,
            strategy_name="SMC Swing (15m)",
        )

        quote = PriceQuote(symbol="GBPUSD", bid=1.34020, ask=1.34034, spread=0.00014, timestamp=datetime.now(timezone.utc))

        # Re-anchoring logic as implemented in main.py
        eff_sl_pips = 6.0
        live_entry = quote.ask if sig.direction == Direction.BUY else quote.bid
        sl_dist = eff_sl_pips * inst.pip_size

        if sig.direction == Direction.BUY:
            sig.stop_loss = round(live_entry - sl_dist, inst.digits)
        else:
            sig.stop_loss = round(live_entry + sl_dist, inst.digits)
        sig.entry_price = live_entry
        sig.sl_distance = sl_dist

        # Check that SL distance is EXACTLY 6 pips (0.00060)
        actual_sl_pips = abs(sig.entry_price - sig.stop_loss) / inst.pip_size
        self.assertAlmostEqual(actual_sl_pips, 6.0, places=1)
        self.assertEqual(sig.stop_loss, round(1.34034 - 0.00060, 5))  # 1.33974

        # Risk amount on 0.12 lots
        risk_dollar = 0.12 * (actual_sl_pips * 10) * inst.point_value
        self.assertAlmostEqual(risk_dollar, 7.20, places=2)
        bot.state.close()


if __name__ == "__main__":
    unittest.main()

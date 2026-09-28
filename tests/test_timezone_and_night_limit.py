"""
test_timezone_and_night_limit.py — Comprehensive tests for timezone resolution
and the automatic night limit window (11 PM - 8 AM).
"""

import unittest
from datetime import datetime, timezone, timedelta
from core.config import TradingConfig, RiskConfig, resolve_timezone


class TestTimezoneAndNightLimit(unittest.TestCase):

    def test_resolve_timezone_formats(self):
        """Verify resolve_timezone parses named zones, abbreviations, and offsets."""
        now = datetime.now()
        tz_ist = resolve_timezone("Asia/Kolkata")
        self.assertEqual(tz_ist.utcoffset(now), timedelta(hours=5, minutes=30))

        tz_ist_abbr = resolve_timezone("IST")
        self.assertEqual(tz_ist_abbr.utcoffset(now), timedelta(hours=5, minutes=30))

        tz_offset = resolve_timezone("+05:30")
        self.assertEqual(tz_offset.utcoffset(now), timedelta(hours=5, minutes=30))

        tz_utc = resolve_timezone("UTC")
        self.assertEqual(tz_utc.utcoffset(now), timedelta(0))

        tz_est = resolve_timezone("EST")
        self.assertEqual(tz_est.utcoffset(now), timedelta(hours=-5))

    def test_daytime_at_1300_ist_is_not_night(self):
        """
        At 13:00 IST (07:30 UTC), is_night_window must return False
        and effective max open positions must be 15 (not 2).
        """
        risk = RiskConfig(
            night_limit_enabled=True,
            night_start_hour=23,
            night_end_hour=8,
            night_max_open_positions=2,
            max_open_positions=15,
            night_timezone_mode="Asia/Kolkata",
        )

        # 07:30 UTC == 13:00 IST (Afternoon)
        dt_1300_ist_in_utc = datetime(2026, 9, 28, 7, 30, tzinfo=timezone.utc)
        self.assertFalse(risk.is_night_window(dt_1300_ist_in_utc))
        self.assertEqual(risk.get_effective_max_open_positions(dt_1300_ist_in_utc), 15)

    def test_morning_at_0830_ist_is_not_night(self):
        """At 08:30 IST (03:00 UTC), night window has ended."""
        risk = RiskConfig(
            night_limit_enabled=True,
            night_start_hour=23,
            night_end_hour=8,
            night_max_open_positions=2,
            max_open_positions=15,
            night_timezone_mode="Asia/Kolkata",
        )
        dt_0830_ist_in_utc = datetime(2026, 9, 28, 3, 0, tzinfo=timezone.utc)
        self.assertFalse(risk.is_night_window(dt_0830_ist_in_utc))
        self.assertEqual(risk.get_effective_max_open_positions(dt_0830_ist_in_utc), 15)

    def test_night_at_2330_ist_is_night(self):
        """At 23:30 IST (18:00 UTC), night window is active."""
        risk = RiskConfig(
            night_limit_enabled=True,
            night_start_hour=23,
            night_end_hour=8,
            night_max_open_positions=2,
            max_open_positions=15,
            night_timezone_mode="Asia/Kolkata",
        )
        dt_2330_ist_in_utc = datetime(2026, 9, 28, 18, 0, tzinfo=timezone.utc)
        self.assertTrue(risk.is_night_window(dt_2330_ist_in_utc))
        self.assertEqual(risk.get_effective_max_open_positions(dt_2330_ist_in_utc), 2)

    def test_overnight_at_0300_ist_is_night(self):
        """At 03:00 IST (21:30 UTC prev day), night window is active."""
        risk = RiskConfig(
            night_limit_enabled=True,
            night_start_hour=23,
            night_end_hour=8,
            night_max_open_positions=2,
            max_open_positions=15,
            night_timezone_mode="Asia/Kolkata",
        )
        dt_0300_ist_in_utc = datetime(2026, 9, 27, 21, 30, tzinfo=timezone.utc)
        self.assertTrue(risk.is_night_window(dt_0300_ist_in_utc))
        self.assertEqual(risk.get_effective_max_open_positions(dt_0300_ist_in_utc), 2)

    def test_disabled_night_limit(self):
        """When night limit is disabled, is_night_window always returns False."""
        risk = RiskConfig(
            night_limit_enabled=False,
            night_start_hour=23,
            night_end_hour=8,
            night_max_open_positions=2,
            max_open_positions=15,
            night_timezone_mode="Asia/Kolkata",
        )
        dt_night = datetime(2026, 9, 28, 18, 0, tzinfo=timezone.utc)
        self.assertFalse(risk.is_night_window(dt_night))
        self.assertEqual(risk.get_effective_max_open_positions(dt_night), 15)

    def test_2254_ist_is_before_night_limit(self):
        """
        At 22:54 IST (10:54 PM), 11 PM (23:00) has NOT started yet.
        is_night_window must return False and max open positions must be 15.
        """
        risk = RiskConfig(
            night_limit_enabled=True,
            night_start_hour=23,
            night_end_hour=8,
            night_max_open_positions=2,
            max_open_positions=15,
            night_timezone_mode="Asia/Kolkata",
        )
        # 22:54:42 IST == 17:24:42 UTC
        dt_2254_ist_in_utc = datetime(2026, 9, 28, 17, 24, 42, tzinfo=timezone.utc)
        self.assertFalse(risk.is_night_window(dt_2254_ist_in_utc))
        self.assertEqual(risk.get_effective_max_open_positions(dt_2254_ist_in_utc), 15)

    def test_authorize_trade_uses_execution_time_not_candle_timestamp(self):
        """
        Ensure authorize_trade evaluates circuit breakers/night limits using the current execution time,
        NOT the candle timestamp (which may have broker timezone offsets or be from an earlier bar).
        """
        import tempfile
        import os
        from core.state import StateManager, TradeRecord
        from core.risk_engine import RiskEngine
        from strategies.strategy import TradeSignal, Direction, MarketBias, LTFConfirmation

        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db.close()
        try:
            config = TradingConfig(use_mock_broker=True)
            config.risk.night_limit_enabled = True
            config.risk.night_start_hour = 23
            config.risk.night_end_hour = 8
            config.risk.night_max_open_positions = 2
            config.risk.max_open_positions = 15
            config.risk.night_timezone_mode = "Asia/Kolkata"

            state = StateManager(db_path=temp_db.name)
            # Record 2 open positions
            t1 = TradeRecord(
                id=9001, timestamp=datetime.now(timezone.utc), symbol="EURUSD",
                direction=Direction.BUY, entry_price=1.08, stop_loss=1.07, take_profit=1.10,
                lot_size=0.10, realized_pnl=0.0, status="OPEN", strategy_name="SMC", magic_number=124456
            )
            t2 = TradeRecord(
                id=9002, timestamp=datetime.now(timezone.utc), symbol="GBPUSD",
                direction=Direction.BUY, entry_price=1.25, stop_loss=1.24, take_profit=1.27,
                lot_size=0.10, realized_pnl=0.0, status="OPEN", strategy_name="ICT", magic_number=124457
            )
            state.record_trade(t1)
            state.record_trade(t2)
            self.assertEqual(len(state.get_open_positions()), 2)

            risk_engine = RiskEngine(config, state)

            # Signal timestamp is broker candle time (02:20 AM IST equivalent),
            # but current execution time is 22:54:42 IST (17:24:42 UTC)
            sig_with_candle_time = TradeSignal(
                symbol="XAUUSD",
                direction=Direction.BUY,
                entry_price=2000.0,
                stop_loss=1990.0,
                take_profit=2025.0,
                htf_bias=MarketBias.BULLISH,
                ltf_confirmation=LTFConfirmation.LIQUIDITY_SWEEP,
                rr_ratio=2.5,
                quality_score=90.0,
                timestamp=datetime(2026, 9, 28, 20, 50, tzinfo=timezone.utc), # Broker time offset
                sl_distance=10.0,
                tp_distance=25.0,
                strategy_id="SMC",
                strategy_name="SMC",
                magic_number=1001,
            )

            current_exec_time = datetime(2026, 9, 28, 17, 24, 42, tzinfo=timezone.utc)
            auth = risk_engine.authorize_trade(
                sig_with_candle_time, current_equity=100000.0, fixed_lot_size=0.01, current_time=current_exec_time
            )
            # Must NOT be rejected by night limit because 22:54 IST is daytime (limit is 15)!
            self.assertTrue(auth.authorized, f"Trade was unexpectedly rejected: {auth.rejection_reason}")
            state.close()
        finally:
            try:
                os.remove(temp_db.name)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()

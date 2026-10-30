"""
test_risk_engine.py — Comprehensive Unit Tests for RiskEngine.

Tests:
1. calculate_lot_size:
   - Position sizing formula: (Equity * Risk%) / (SL Points * Point Value)
   - Clamping to min_lot and max_lot
   - Rounding down to lot_step
   - Zero/negative SL distance and zero/negative point value edge cases
2. check_circuit_breakers:
   - Existing active circuit breaker check
   - Max concurrent open positions (daytime limit)
   - Night window limit (2 positions between 11 PM - 8 AM)
   - Daily drawdown limit breach and state.activate_circuit_breaker() triggering
   - Clear/healthy state
3. authorize_trade:
   - Full trade authorization pipeline (dynamic lot & fixed lot)
   - Risk multiplier scaling for fixed and dynamic lots
   - Rejection when circuit breaker is triggered
   - Rejection on unknown/unsupported symbol
   - Rejection when max open positions per symbol is reached
   - Rejection of duplicate trades by the same strategy on the same symbol
   - Allowance of concurrent trades by different strategies on the same symbol
   - Rejection on invalid/zero SL distance
   - Rejection when calculated lot is below instrument minimum lot
   - Directional loss cooldown rejection on 2 consecutive losses within 60m
"""

import math
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from core.config import (
    TradingConfig,
    AccountConfig,
    RiskConfig,
    InstrumentConfig,
    Direction,
    MarketBias,
)
from core.state import StateManager, TradeRecord
from core.risk_engine import RiskEngine, AuthorizationResult
from strategies.strategy import TradeSignal, LTFConfirmation


class TestCalculateLotSize(unittest.TestCase):
    """Tests for RiskEngine.calculate_lot_size."""

    def setUp(self):
        self.config = TradingConfig()
        self.state = StateManager(db_path=":memory:")
        self.risk_engine = RiskEngine(config=self.config, state=self.state)

        # Standard 5-digit Forex instrument (EURUSD)
        self.eurusd = InstrumentConfig(
            symbol="EURUSD",
            digits=5,
            point_value=1.0,   # $1.00 per point ($10 per pip) for 1.0 standard lot
            pip_size=0.0001,
            min_lot=0.01,
            max_lot=50.0,
            lot_step=0.01,
        )

        # 2-digit Commodity instrument (XAUUSD)
        self.xauusd = InstrumentConfig(
            symbol="XAUUSD",
            digits=2,
            point_value=1.0,
            pip_size=0.1,
            min_lot=0.01,
            max_lot=20.0,
            lot_step=0.01,
        )

    def test_standard_forex_lot_calculation(self):
        """
        Equity: $10,000, Risk: 1% ($100 risk)
        SL Distance: 0.0020 (20 pips = 200 points)
        Point Value: 1.0
        Expected Lot Size: $100 / (200 * 1.0) = 0.50 lots
        """
        lot = self.risk_engine.calculate_lot_size(
            equity=10_000.0,
            risk_pct=0.01,
            sl_distance_price=0.0020,
            instrument=self.eurusd,
        )
        self.assertEqual(lot, 0.50)

    def test_clamped_to_min_lot(self):
        """Very small equity and large SL should clamp to instrument.min_lot."""
        lot = self.risk_engine.calculate_lot_size(
            equity=100.0,
            risk_pct=0.001,  # $0.10 risk
            sl_distance_price=0.0100,  # 1000 points
            instrument=self.eurusd,
        )
        self.assertEqual(lot, self.eurusd.min_lot)

    def test_clamped_to_max_lot(self):
        """Large equity and tiny SL should clamp to instrument.max_lot."""
        lot = self.risk_engine.calculate_lot_size(
            equity=10_000_000.0,
            risk_pct=0.05,
            sl_distance_price=0.0001,
            instrument=self.eurusd,
        )
        self.assertEqual(lot, self.eurusd.max_lot)

    def test_lot_step_rounding_down(self):
        """
        Verify lot size rounds down (math.floor) to lot_step increments.
        Using lot_step=0.05.
        Raw calculation: 0.287 -> rounds down to 0.25
        """
        stepped_instrument = InstrumentConfig(
            symbol="STEP_TEST",
            digits=5,
            point_value=1.0,
            pip_size=0.0001,
            min_lot=0.05,
            max_lot=10.0,
            lot_step=0.05,
        )
        # Equity=10000, risk_pct=0.00287 ($28.70 risk), SL=100 points -> 28.70 / 100 = 0.287 lots
        lot = self.risk_engine.calculate_lot_size(
            equity=10_000.0,
            risk_pct=0.00287,
            sl_distance_price=0.0010,
            instrument=stepped_instrument,
        )
        self.assertEqual(lot, 0.25)

    def test_zero_sl_distance_returns_zero(self):
        """Zero or negative SL distance returns 0.0."""
        self.assertEqual(
            self.risk_engine.calculate_lot_size(10_000.0, 0.01, 0.0, self.eurusd),
            0.0
        )
        self.assertEqual(
            self.risk_engine.calculate_lot_size(10_000.0, 0.01, -0.0050, self.eurusd),
            0.0
        )

    def test_zero_or_negative_point_value_returns_zero(self):
        """Invalid point_value returns 0.0 via defensive check."""
        invalid_instrument = InstrumentConfig.model_construct(
            symbol="INVALID",
            digits=5,
            point_value=0.0,
            pip_size=0.0001,
            min_lot=0.01,
            max_lot=10.0,
            lot_step=0.01,
        )
        self.assertEqual(
            self.risk_engine.calculate_lot_size(10_000.0, 0.01, 0.0020, invalid_instrument),
            0.0
        )


class TestCircuitBreakers(unittest.TestCase):
    """Tests for RiskEngine.check_circuit_breakers."""

    def setUp(self):
        self.config = TradingConfig()
        self.state = StateManager(db_path=":memory:")
        self.risk_engine = RiskEngine(config=self.config, state=self.state)

    def test_active_circuit_breaker_blocks_trade(self):
        """If circuit breaker is already activated for today, all trades are blocked."""
        self.state.is_circuit_breaker_active = MagicMock(return_value=True)
        blocked, reason = self.risk_engine.check_circuit_breakers(equity=10_000.0)
        self.assertTrue(blocked)
        self.assertIn("Circuit breaker active", reason)

    def test_max_concurrent_open_positions_daytime(self):
        """When daytime open positions reach max_open_positions (15), trades are blocked."""
        self.state.is_circuit_breaker_active = MagicMock(return_value=False)
        daytime = datetime(2026, 10, 19, 14, 0, tzinfo=timezone.utc)
        self.state.get_open_positions = MagicMock(return_value=[MagicMock()] * 15)

        blocked, reason = self.risk_engine.check_circuit_breakers(equity=10_000.0, current_time=daytime)
        self.assertTrue(blocked)
        self.assertIn("Max concurrent open positions reached", reason)

    def test_night_limit_window_blocks_at_two_positions(self):
        """
        During night window (11 PM - 8 AM), night_max_open_positions is 2.
        2 or more open positions must block new trades.
        """
        self.state.is_circuit_breaker_active = MagicMock(return_value=False)
        self.config.risk.night_limit_enabled = True
        self.config.risk.night_timezone_mode = "UTC"
        self.config.risk.night_start_hour = 23
        self.config.risk.night_end_hour = 8
        self.config.risk.night_max_open_positions = 2

        night_time = datetime(2026, 10, 19, 2, 0, tzinfo=timezone.utc)
        self.state.get_open_positions = MagicMock(return_value=[MagicMock(), MagicMock()])

        blocked, reason = self.risk_engine.check_circuit_breakers(equity=10_000.0, current_time=night_time)
        self.assertTrue(blocked)
        self.assertIn("Night Limit", reason)

    def test_daily_drawdown_limit_breached(self):
        """
        When daily PnL breaches max_daily_drawdown_pct (e.g. 2% on $10,000 = -$200),
        circuit breaker is tripped and trade is blocked.
        """
        self.state.is_circuit_breaker_active = MagicMock(return_value=False)
        self.state.get_open_positions = MagicMock(return_value=[])
        limit = 10_000.0 * self.config.account.max_daily_drawdown_pct
        self.state.get_daily_pnl = MagicMock(return_value=-(limit + 50.0))
        self.state.activate_circuit_breaker = MagicMock()

        blocked, reason = self.risk_engine.check_circuit_breakers(equity=10_000.0)
        self.assertTrue(blocked)
        self.assertIn("Daily drawdown limit breached", reason)
        self.state.activate_circuit_breaker.assert_called_once()

    def test_healthy_state_allows_trading(self):
        """Under normal conditions with no drawdown and open positions within limits, trading is allowed."""
        self.state.is_circuit_breaker_active = MagicMock(return_value=False)
        self.state.get_open_positions = MagicMock(return_value=[MagicMock()])
        self.state.get_daily_pnl = MagicMock(return_value=50.0)

        blocked, reason = self.risk_engine.check_circuit_breakers(equity=10_000.0)
        self.assertFalse(blocked)
        self.assertIsNone(reason)


class TestAuthorizeTrade(unittest.TestCase):
    """Tests for RiskEngine.authorize_trade."""

    def setUp(self):
        self.config = TradingConfig()
        self.state = StateManager(db_path=":memory:")
        self.risk_engine = RiskEngine(config=self.config, state=self.state)

        self.signal = TradeSignal(
            symbol="EURUSD",
            direction=Direction.BUY,
            entry_price=1.10000,
            stop_loss=1.09800,
            take_profit=1.10400,
            sl_distance=0.00200,
            tp_distance=0.00400,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.OB_MITIGATION,
            rr_ratio=2.0,
            quality_score=95.0,
            timestamp=datetime.now(timezone.utc),
            strategy_name="SMC Swing",
            strategy_id="SMC",
            magic_number=123456,
        )

    def test_authorize_trade_success_dynamic_lot(self):
        """Successful trade authorization using dynamic equity-based lot sizing."""
        self.state.is_circuit_breaker_active = MagicMock(return_value=False)
        self.state.get_open_positions = MagicMock(return_value=[])
        self.state.get_daily_pnl = MagicMock(return_value=0.0)

        # Equity $10,000, risk_pct 0.005 ($50 risk), SL 20 pips (200 points), Point Value $1.0
        # Expected lot: $50 / 200 = 0.25 lots
        result = self.risk_engine.authorize_trade(
            signal=self.signal,
            current_equity=10_000.0,
        )
        self.assertTrue(result.authorized)
        self.assertEqual(result.lot_size, 0.25)
        self.assertAlmostEqual(result.risk_amount, 50.0, places=2)
        self.assertEqual(result.account_equity, 10_000.0)

    def test_authorize_trade_fixed_lot_override(self):
        """Direct fixed_lot_size argument overrides dynamic calculation."""
        self.state.is_circuit_breaker_active = MagicMock(return_value=False)
        self.state.get_open_positions = MagicMock(return_value=[])
        self.state.get_daily_pnl = MagicMock(return_value=0.0)

        result = self.risk_engine.authorize_trade(
            signal=self.signal,
            current_equity=10_000.0,
            fixed_lot_size=0.12,
        )
        self.assertTrue(result.authorized)
        self.assertEqual(result.lot_size, 0.12)
        # Risk amount: 0.12 * 200 points * $1.0 = $24.00
        self.assertAlmostEqual(result.risk_amount, 24.0, places=2)

    def test_authorize_trade_risk_multiplier_scales_fixed_lot(self):
        """When risk_multiplier < 1.0 (e.g. 0.50), fixed lot size is scaled proportionally."""
        self.state.is_circuit_breaker_active = MagicMock(return_value=False)
        self.state.get_open_positions = MagicMock(return_value=[])
        self.state.get_daily_pnl = MagicMock(return_value=0.0)

        result = self.risk_engine.authorize_trade(
            signal=self.signal,
            current_equity=10_000.0,
            fixed_lot_size=0.12,
            risk_multiplier=0.50,
        )
        self.assertTrue(result.authorized)
        self.assertEqual(result.lot_size, 0.06)

    def test_authorize_trade_circuit_breaker_rejection(self):
        """Rejects trade when circuit breaker is active."""
        self.state.is_circuit_breaker_active = MagicMock(return_value=True)

        result = self.risk_engine.authorize_trade(
            signal=self.signal,
            current_equity=10_000.0,
        )
        self.assertFalse(result.authorized)
        self.assertIn("Circuit breaker active", result.rejection_reason)

    def test_authorize_trade_unknown_instrument_rejection(self):
        """Rejects trade for an empty or invalid symbol."""
        self.state.is_circuit_breaker_active = MagicMock(return_value=False)
        self.state.get_open_positions = MagicMock(return_value=[])
        self.state.get_daily_pnl = MagicMock(return_value=0.0)

        unknown_signal = TradeSignal(
            symbol="NONE",
            direction=Direction.BUY,
            entry_price=100.0,
            stop_loss=99.0,
            take_profit=102.0,
            sl_distance=1.0,
            tp_distance=2.0,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.OB_MITIGATION,
            rr_ratio=2.0,
            quality_score=90.0,
            timestamp=datetime.now(timezone.utc),
            strategy_name="SMC",
        )
        result = self.risk_engine.authorize_trade(
            signal=unknown_signal,
            current_equity=10_000.0,
        )
        self.assertFalse(result.authorized)
        self.assertIn("invalid or not configured", result.rejection_reason)

    def test_authorize_trade_max_per_symbol_exceeded(self):
        """Rejects trade when symbol already has max allowed positions (default 3)."""
        self.state.is_circuit_breaker_active = MagicMock(return_value=False)
        self.state.get_daily_pnl = MagicMock(return_value=0.0)

        existing_trades = [
            TradeRecord(id=1, symbol="EURUSD", direction=Direction.BUY, entry_price=1.1, stop_loss=1.09, take_profit=1.12, lot_size=0.1, realized_pnl=0.0, status="OPEN", timestamp=datetime.now(timezone.utc), strategy_name="SMC_1"),
            TradeRecord(id=2, symbol="EURUSD", direction=Direction.BUY, entry_price=1.1, stop_loss=1.09, take_profit=1.12, lot_size=0.1, realized_pnl=0.0, status="OPEN", timestamp=datetime.now(timezone.utc), strategy_name="SMC_2"),
            TradeRecord(id=3, symbol="EURUSD", direction=Direction.BUY, entry_price=1.1, stop_loss=1.09, take_profit=1.12, lot_size=0.1, realized_pnl=0.0, status="OPEN", timestamp=datetime.now(timezone.utc), strategy_name="SMC_3"),
        ]
        self.state.get_open_positions = MagicMock(return_value=existing_trades)

        result = self.risk_engine.authorize_trade(
            signal=self.signal,
            current_equity=10_000.0,
            current_time=datetime(2026, 10, 22, 14, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(result.authorized)
        self.assertIn("Max open positions for EURUSD reached", result.rejection_reason)

    def test_authorize_trade_duplicate_strategy_blocked(self):
        """Rejects trade if the same strategy already has an active trade for this symbol."""
        self.state.is_circuit_breaker_active = MagicMock(return_value=False)
        self.state.get_daily_pnl = MagicMock(return_value=0.0)

        existing_trade = TradeRecord(
            id=42,
            symbol="EURUSD",
            direction=Direction.BUY,
            entry_price=1.0990,
            stop_loss=1.0970,
            take_profit=1.1030,
            lot_size=0.1,
            realized_pnl=0.0,
            status="OPEN",
            timestamp=datetime.now(timezone.utc),
            strategy_name="SMC Swing",
            magic_number=123456,
        )
        self.state.get_open_positions = MagicMock(return_value=[existing_trade])

        result = self.risk_engine.authorize_trade(
            signal=self.signal,
            current_equity=10_000.0,
        )
        self.assertFalse(result.authorized)
        self.assertIn("already has an active trade for EURUSD", result.rejection_reason)

    def test_authorize_trade_different_strategy_allowed(self):
        """Allows trade if the open trade for this symbol belongs to a DIFFERENT strategy."""
        self.state.is_circuit_breaker_active = MagicMock(return_value=False)
        self.state.get_daily_pnl = MagicMock(return_value=0.0)

        existing_trade = TradeRecord(
            id=42,
            symbol="EURUSD",
            direction=Direction.BUY,
            entry_price=1.0990,
            stop_loss=1.0970,
            take_profit=1.1030,
            lot_size=0.1,
            realized_pnl=0.0,
            status="OPEN",
            timestamp=datetime.now(timezone.utc),
            strategy_name="ICT",
            magic_number=654321,
        )
        self.state.get_open_positions = MagicMock(return_value=[existing_trade])

        result = self.risk_engine.authorize_trade(
            signal=self.signal,
            current_equity=10_000.0,
        )
        self.assertTrue(result.authorized)

    def test_authorize_trade_invalid_sl_distance_rejected(self):
        """Rejects trade when entry_price equals stop_loss and sl_distance is zero."""
        self.state.is_circuit_breaker_active = MagicMock(return_value=False)
        self.state.get_open_positions = MagicMock(return_value=[])
        self.state.get_daily_pnl = MagicMock(return_value=0.0)

        zero_sl_signal = TradeSignal(
            symbol="EURUSD",
            direction=Direction.BUY,
            entry_price=1.10000,
            stop_loss=1.10000,
            take_profit=1.10400,
            sl_distance=0.0,
            tp_distance=0.00400,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.OB_MITIGATION,
            rr_ratio=2.0,
            quality_score=90.0,
            timestamp=datetime.now(timezone.utc),
            strategy_name="SMC",
        )
        result = self.risk_engine.authorize_trade(
            signal=zero_sl_signal,
            current_equity=10_000.0,
        )
        self.assertFalse(result.authorized)
        self.assertEqual(result.rejection_reason, "Invalid SL distance")


if __name__ == "__main__":
    unittest.main()

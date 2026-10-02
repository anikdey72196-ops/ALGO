import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from core.config import TradingConfig, Direction, MarketBias
from core.risk_engine import RiskEngine
from core.state import StateManager, TradeRecord
from strategies.strategy import TradeSignal, LTFConfirmation


class TestDirectionalLossCooldown(unittest.TestCase):
    def setUp(self):
        self.config = TradingConfig()
        self.state = StateManager(db_path=":memory:")
        self.risk_engine = RiskEngine(config=self.config, state=self.state)

    def test_directional_loss_cooldown_deactivated_by_default(self):
        """Verify that when directional_loss_cooldown_enabled is False (default), trade is authorized."""
        self.assertFalse(self.config.risk.directional_loss_cooldown_enabled)

        now = datetime.now(timezone.utc)
        # Record 2 consecutive losing SELL trades on GBPUSD 10 minutes ago
        self.state.get_all_trades = MagicMock(return_value=[
            TradeRecord(
                id=101,
                symbol="GBPUSD",
                direction=Direction.SELL,
                entry_price=1.3220,
                stop_loss=1.3250,
                take_profit=1.3150,
                lot_size=0.01,
                status="CLOSED_SL",
                realized_pnl=-50.0,
                timestamp=now - timedelta(minutes=10),
                closed_at=now - timedelta(minutes=10),
            ),
            TradeRecord(
                id=100,
                symbol="GBPUSD",
                direction=Direction.SELL,
                entry_price=1.3230,
                stop_loss=1.3260,
                take_profit=1.3160,
                lot_size=0.01,
                status="CLOSED_SL",
                realized_pnl=-45.0,
                timestamp=now - timedelta(minutes=25),
                closed_at=now - timedelta(minutes=25),
            ),
        ])

        signal = TradeSignal(
            symbol="GBPUSD",
            direction=Direction.SELL,
            entry_price=1.3200,
            stop_loss=1.3230,
            take_profit=1.3100,
            htf_bias=MarketBias.BEARISH,
            ltf_confirmation=LTFConfirmation.OB_MITIGATION,
            rr_ratio=3.33,
            quality_score=90.0,
            timestamp=now,
            sl_distance=0.0030,
            tp_distance=0.0100,
            strategy_name="SMC",
            strategy_id="SMC",
            magic_number=12345,
        )

        auth = self.risk_engine.authorize_trade(signal, current_equity=10000.0, current_time=now)
        self.assertTrue(auth.authorized)
        self.assertIsNone(auth.rejection_reason)
        self.assertGreater(auth.lot_size, 0.0)

    def test_directional_loss_cooldown_activated(self):
        """Verify that when directional_loss_cooldown_enabled is True, 2 consecutive losses trigger rejection."""
        self.config.risk.directional_loss_cooldown_enabled = True

        now = datetime.now(timezone.utc)
        self.state.get_all_trades = MagicMock(return_value=[
            TradeRecord(
                id=101,
                symbol="GBPUSD",
                direction=Direction.SELL,
                entry_price=1.3220,
                stop_loss=1.3250,
                take_profit=1.3150,
                lot_size=0.01,
                status="CLOSED_SL",
                realized_pnl=-50.0,
                timestamp=now - timedelta(minutes=10),
                closed_at=now - timedelta(minutes=10),
            ),
            TradeRecord(
                id=100,
                symbol="GBPUSD",
                direction=Direction.SELL,
                entry_price=1.3230,
                stop_loss=1.3260,
                take_profit=1.3160,
                lot_size=0.01,
                status="CLOSED_SL",
                realized_pnl=-45.0,
                timestamp=now - timedelta(minutes=25),
                closed_at=now - timedelta(minutes=25),
            ),
        ])

        signal = TradeSignal(
            symbol="GBPUSD",
            direction=Direction.SELL,
            entry_price=1.3200,
            stop_loss=1.3230,
            take_profit=1.3100,
            htf_bias=MarketBias.BEARISH,
            ltf_confirmation=LTFConfirmation.OB_MITIGATION,
            rr_ratio=3.33,
            quality_score=90.0,
            timestamp=now,
            sl_distance=0.0030,
            tp_distance=0.0100,
            strategy_name="SMC",
            strategy_id="SMC",
            magic_number=12345,
        )

        auth = self.risk_engine.authorize_trade(signal, current_equity=10000.0, current_time=now)
        self.assertFalse(auth.authorized)
        self.assertIn("Directional Loss Guard", auth.rejection_reason)

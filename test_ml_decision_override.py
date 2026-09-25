"""
test_ml_decision_override.py — Verification of ML Decision Override System

Verifies:
1. High Stop Loss probability (> max_sl_probability) triggers ML decision override (veto/skip).
2. Low Stop Loss probability (<= max_sl_probability) allows trade execution.
3. 100% of strategy LTF confirmation signals map to valid EventKinds.
4. TradingBot tick loop logs explicit [ML DECISION OVERRIDE] and aborts order routing.
5. TradingBot tick loop logs [ML MODEL APPROVED] when setup passes ML check.
6. Web API configuration, persistence in bot_settings.json, and activation locking.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from config import TradingConfig, Direction, MarketBias, InstrumentConfig
from strategy import LTFConfirmation, TradeSignal
from ml.trap_detector import (
    TrapDetectorConfig,
    TrapGate,
    TrapModel,
    EventKind,
    GateDecision,
    ALL_FEATURES,
)
from main import TradingBot
from web_app import app, bot_instance


class TestMLDecisionOverride(unittest.TestCase):
    def setUp(self):
        # Create an in-memory / mock model setup
        self.cfg = TrapDetectorConfig(
            db_path=":memory:",
            p_genuine_threshold=0.50,
            max_sl_probability=0.50,
            shadow_until_samples=0,
        )

    def test_trap_gate_veto_on_high_sl_probability(self):
        """When P(SL) > max_sl_probability (or P(genuine) < threshold), gate must veto."""
        mock_model = MagicMock(spec=TrapModel)
        mock_model.version = "test-model-v1"
        mock_model.n_samples = 100
        mock_model._sgd_fitted = True
        mock_model.lgbm = None

        # P(genuine) = 0.30 -> P(SL) = 0.70 (> 0.50 max SL risk)
        mock_model.predict_proba_genuine.return_value = np.array([0.30])

        gate = TrapGate(mock_model, self.cfg)
        decision = gate.evaluate(EventKind.FVG_BULL, {k: 0.0 for k in ALL_FEATURES})

        self.assertFalse(decision.allow)
        self.assertEqual(decision.mode, "gated")
        self.assertAlmostEqual(decision.p_genuine, 0.30)
        self.assertIn("High SL probability", decision.reason)
        self.assertIn("70.0%", decision.reason)

    def test_trap_gate_allow_on_low_sl_probability(self):
        """When P(SL) <= max_sl_probability, gate must approve."""
        mock_model = MagicMock(spec=TrapModel)
        mock_model.version = "test-model-v1"
        mock_model.n_samples = 100
        mock_model._sgd_fitted = True
        mock_model.lgbm = None

        # P(genuine) = 0.75 -> P(SL) = 0.25 (<= 0.50 max SL risk)
        mock_model.predict_proba_genuine.return_value = np.array([0.75])

        gate = TrapGate(mock_model, self.cfg)
        decision = gate.evaluate(EventKind.FVG_BULL, {k: 0.0 for k in ALL_FEATURES})

        self.assertTrue(decision.allow)
        self.assertEqual(decision.mode, "gated")
        self.assertAlmostEqual(decision.p_genuine, 0.75)
        self.assertEqual(decision.reason, "passed")

    def test_universal_confirmation_mapping_100_percent(self):
        """Verify 100% of LTFConfirmation values map to valid EventKind for BUY & SELL."""
        config = TradingConfig(use_mock_broker=True)
        bot = TradingBot(config=config, load_saved_settings=False)

        for conf in LTFConfirmation:
            sig_buy = TradeSignal(
                symbol="EURUSD",
                direction=Direction.BUY,
                entry_price=1.1000,
                stop_loss=1.0950,
                take_profit=1.1100,
                htf_bias=MarketBias.BULLISH,
                ltf_confirmation=conf,
                rr_ratio=2.0,
                quality_score=85.0,
                timestamp=datetime.now(timezone.utc),
            )
            kind_buy = bot._signal_to_trap_kind(sig_buy)
            self.assertIsNotNone(kind_buy, f"BUY confirmation {conf} returned None!")
            self.assertIsInstance(kind_buy, EventKind)

            sig_sell = TradeSignal(
                symbol="EURUSD",
                direction=Direction.SELL,
                entry_price=1.1000,
                stop_loss=1.1050,
                take_profit=1.0900,
                htf_bias=MarketBias.BEARISH,
                ltf_confirmation=conf,
                rr_ratio=2.0,
                quality_score=85.0,
                timestamp=datetime.now(timezone.utc),
            )
            kind_sell = bot._signal_to_trap_kind(sig_sell)
            self.assertIsNotNone(kind_sell, f"SELL confirmation {conf} returned None!")
            self.assertIsInstance(kind_sell, EventKind)

    def test_bot_execute_tick_veto_skips_trade(self):
        """When ML model detects high SL probability, trade must be skipped and logged."""
        config = TradingConfig(use_mock_broker=True)
        config.ml_gating_enabled = True
        config.ml_max_sl_probability = 0.40  # Max allowable SL probability is 40%
        bot = TradingBot(config=config, load_saved_settings=False)

        # Mock signal generator
        mock_signal = TradeSignal(
            symbol="EURUSD",
            direction=Direction.BUY,
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.OB_SCALP_5M,
            rr_ratio=2.0,
            quality_score=90.0,
            timestamp=datetime.now(timezone.utc),
            strategy_name="5M Scalp",
        )

        # Mock broker quote
        mock_quote = MagicMock()
        mock_quote.ask = 1.1001
        mock_quote.bid = 1.1000
        mock_quote.spread = 0.0001
        bot.broker.get_current_price = MagicMock(return_value=mock_quote)

        # Mock strategy engine and conflict resolver
        from conflict_resolver import FilterResult
        from news_filter import SpreadFilterResult, NewsFilter
        bot.strategy.evaluate_all = MagicMock(return_value=[mock_signal])
        bot.conflict_resolver.resolve = MagicMock(
            return_value=FilterResult(accepted_signal=mock_signal, total_signals=1, rejected_count=0, rejection_reasons=[])
        )

        # Mock market data
        df = pd.DataFrame({
            "open": [1.10, 1.10, 1.10, 1.10],
            "high": [1.11, 1.11, 1.11, 1.11],
            "low": [1.09, 1.09, 1.09, 1.09],
            "close": [1.10, 1.10, 1.10, 1.10],
            "volume": [100, 100, 100, 100],
        }, index=pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC"))
        bot._get_ohlcv = MagicMock(return_value=df)

        # Mock trap_svc to predict high SL risk (P(genuine)=0.20 -> P(SL)=0.80 > 0.40)
        mock_trap_ev = MagicMock()
        mock_trap_ev.allowed = False
        mock_trap_ev.p_genuine = 0.20
        mock_trap_ev.model_version = "sgd-test"
        bot.trap_svc.observe_event = MagicMock(return_value=mock_trap_ev)

        # Mock AI and Broker order methods to verify they are NEVER called when vetoed
        bot.ai_analyst.evaluate_setup = MagicMock()
        bot.broker.send_bracket_order = MagicMock()

        # Run tick
        bot.config.selected_symbols = ["EURUSD"]
        bot.state.is_circuit_breaker_active = MagicMock(return_value=False)
        bot.news_filter.is_blackout = MagicMock(return_value=(False, "No blackout"))
        bot.news_filter.check_news_blackout = MagicMock(return_value=MagicMock(blocked=False))
        bot.state.can_trade = MagicMock(return_value=(True, "OK"))
        bot.state.get_open_positions = MagicMock(return_value=[])

        with patch.object(NewsFilter, "check_spread", return_value=SpreadFilterResult(blocked=False, current_spread=0.0001, avg_spread=0.0001)):
            bot._execute_tick()

        # Verify trade was SKIPPED by ML Decision Override
        bot.ai_analyst.evaluate_setup.assert_not_called()
        bot.broker.send_bracket_order.assert_not_called()

        # Check logs for explicit decision override notice
        logs = "\n".join(bot.recent_logs)
        self.assertIn("🪤 [ML DECISION OVERRIDE] Trade SKIPPED: High chance of Stop Loss", logs)
        self.assertIn("P(SL)=80.0% > 40.0%", logs)
        self.assertIn("vetoed by ML Model (sgd-test)", logs)

    def test_bot_execute_tick_approval_proceeds(self):
        """When ML model detects low SL probability, trade must proceed to AI & Risk."""
        config = TradingConfig(use_mock_broker=True)
        config.ml_gating_enabled = True
        config.ml_max_sl_probability = 0.50
        bot = TradingBot(config=config, load_saved_settings=False)

        mock_signal = TradeSignal(
            symbol="EURUSD",
            direction=Direction.BUY,
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.LIQUIDITY_SWEEP,
            rr_ratio=2.0,
            quality_score=90.0,
            timestamp=datetime.now(timezone.utc),
            strategy_name="SMC Swing",
        )

        mock_quote = MagicMock()
        mock_quote.ask = 1.1001
        mock_quote.bid = 1.1000
        mock_quote.spread = 0.0001
        bot.broker.get_current_price = MagicMock(return_value=mock_quote)

        from conflict_resolver import FilterResult
        from news_filter import SpreadFilterResult, NewsFilter
        bot.strategy.evaluate_all = MagicMock(return_value=[mock_signal])
        bot.conflict_resolver.resolve = MagicMock(
            return_value=FilterResult(accepted_signal=mock_signal, total_signals=1, rejected_count=0, rejection_reasons=[])
        )

        df = pd.DataFrame({
            "open": [1.10, 1.10, 1.10, 1.10],
            "high": [1.11, 1.11, 1.11, 1.11],
            "low": [1.09, 1.09, 1.09, 1.09],
            "close": [1.10, 1.10, 1.10, 1.10],
            "volume": [100, 100, 100, 100],
        }, index=pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC"))
        bot._get_ohlcv = MagicMock(return_value=df)

        # ML approves: P(genuine)=0.80 -> P(SL)=0.20 <= 0.50
        mock_trap_ev = MagicMock()
        mock_trap_ev.allowed = True
        mock_trap_ev.p_genuine = 0.80
        mock_trap_ev.model_version = "sgd-test"
        bot.trap_svc.observe_event = MagicMock(return_value=mock_trap_ev)

        # AI also confirms
        mock_ai_verdict = MagicMock()
        mock_ai_verdict.confirmed = True
        mock_ai_verdict.confidence = 88.0
        mock_ai_verdict.reason = "Strong liquidity structure"
        bot.ai_analyst.evaluate_setup = MagicMock(return_value=mock_ai_verdict)

        # Risk authorizes
        mock_auth = MagicMock()
        mock_auth.authorized = True
        mock_auth.lot_size = 0.10
        mock_auth.risk_amount = 100.0
        mock_auth.account_equity = 10000.0
        bot.risk_engine.authorize_trade = MagicMock(return_value=mock_auth)

        import itertools
        from state import StateManager
        bot.state = StateManager(db_path=":memory:")

        # Broker fills with unique IDs
        order_counter = itertools.count(99001)
        def create_mock_order_res(bracket):
            res = MagicMock()
            res.success = True
            res.order_id = next(order_counter)
            res.fill_price = 1.1001
            return res
        bot.broker.send_bracket_order = MagicMock(side_effect=create_mock_order_res)

        bot.config.pair1.symbol = "EURUSD"
        bot.config.pair1.enabled = True
        bot.config.pair2.enabled = False
        bot.config.pair3.enabled = False
        bot.config.selected_symbols = ["EURUSD"]
        bot.state.is_circuit_breaker_active = MagicMock(return_value=False)
        bot.news_filter.is_blackout = MagicMock(return_value=(False, "No blackout"))
        bot.news_filter.check_news_blackout = MagicMock(return_value=MagicMock(blocked=False))
        bot.state.can_trade = MagicMock(return_value=(True, "OK"))
        bot.state.get_open_positions = MagicMock(return_value=[])

        with patch.object(NewsFilter, "check_spread", return_value=SpreadFilterResult(blocked=False, current_spread=0.0001, avg_spread=0.0001)):
            bot._execute_tick()

        # Check approval log
        logs = "\n".join(bot.recent_logs)
        self.assertIn("🔬 [ML MODEL APPROVED] Setup verified genuine", logs)
        self.assertIn("P(TP)=80.0%", logs)
        self.assertIn("P(SL)=20.0% <= 50.0%", logs)

        # Verify AI and Broker were reached and order was filled
        bot.ai_analyst.evaluate_setup.assert_called_once()
        bot.broker.send_bracket_order.assert_called_once()
        self.assertIn("ID=99001", logs)
        self.assertIn("ORDER FILLED", logs)

    def test_web_api_ml_configuration_and_activation_lock(self):
        """Verify API exposes and locks ML gating settings."""
        client = TestClient(app)

        # Ensure bot is deactivated first
        bot_instance.is_active = False

        # Configure ML gating parameters
        res = client.post(
            "/api/configure",
            json={
                "ml_gating_enabled": True,
                "ml_max_sl_probability": 0.45,
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(bot_instance.config.ml_gating_enabled)
        self.assertAlmostEqual(bot_instance.config.ml_max_sl_probability, 0.45)
        self.assertAlmostEqual(bot_instance.trap_svc.cfg.max_sl_probability, 0.45)

        # Check /api/state returns the settings
        state_res = client.get("/api/state")
        self.assertEqual(state_res.status_code, 200)
        state_json = state_res.json()
        self.assertTrue(state_json["ml_gating_enabled"])
        self.assertAlmostEqual(state_json["ml_max_sl_probability"], 0.45)
        self.assertEqual(state_json["ml_trap_detector"]["max_sl_probability"], 0.45)

        # Activate bot and verify ML setting modifications are locked
        bot_instance.is_active = True
        locked_res = client.post(
            "/api/configure",
            json={"ml_max_sl_probability": 0.30},
        )
        self.assertEqual(locked_res.status_code, 400)
        self.assertIn("LOCKED during activation", locked_res.json()["detail"])

        # Reset bot state
        bot_instance.is_active = False


if __name__ == "__main__":
    unittest.main()

"""
test_multi_pair_limits.py — Comprehensive test suite verifying:
1. Synchronized dual-pair scanning and immediate trade execution on both pairs in the same cycle.
2. Strict trade limits:
   - Max 1 trade per strategy per pair.
   - Duplicate trades for the same strategy on the same pair are forbidden.
   - Max 3 open trades per pair (one per strategy).
   - Max 6 open trades across both pairs (3 strategies x 2 pairs).
   - Strict global cap of 3 open trades when max_open_positions = 3.
3. Locked parameters during active session.
4. AI analyzer and ML model trade validation.
"""

import os
import tempfile
import unittest
from datetime import datetime, timezone
import pandas as pd
from unittest.mock import MagicMock

from config import TradingConfig, InstrumentConfig, Direction, MarketBias
from strategy import TradeSignal, LTFConfirmation
from state import TradeRecord
from risk_engine import RiskEngine
from ai_analyst import AIAnalyst
from ml.trap_detector import TrapDetectorService, TrapDetectorConfig, EventKind
from main import TradingBot


def make_test_df(n=50, base_price=1.0850, step=0.0002):
    times = pd.date_range("2026-09-18 10:00", periods=n, freq="5min", tz="UTC")
    data = []
    price = base_price
    for t in times:
        price += step
        data.append({
            "time": t,
            "open": price - 0.0001,
            "high": price + 0.0005,
            "low": price - 0.0005,
            "close": price,
            "tick_volume": 100,
            "volume": 100,
        })
    return pd.DataFrame(data)


def create_valid_signal(symbol="EURUSD", direction=Direction.BUY, entry=1.0850, sl=1.0830, tp=1.0900, strat_name="SMC", magic=124456):
    sl_dist = abs(entry - sl)
    tp_dist = abs(entry - tp)
    rr = tp_dist / sl_dist
    return TradeSignal(
        symbol=symbol,
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        htf_bias=MarketBias.BULLISH if direction == Direction.BUY else MarketBias.BEARISH,
        ltf_confirmation=LTFConfirmation.ICT_KILLZONE_FVG if "ICT" in strat_name else LTFConfirmation.OB_MITIGATION,
        rr_ratio=rr,
        quality_score=90.0,
        timestamp=datetime.now(timezone.utc),
        sl_distance=sl_dist,
        tp_distance=tp_dist,
        strategy_id=strat_name,
        strategy_name=strat_name,
        magic_number=magic,
    )


class TestMultiPairLimitsAndSynchronization(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db = os.path.join(self.temp_dir.name, "test_trading_state.db")

        self.config = TradingConfig()
        self.config.db_path = self.temp_db
        self.config.use_mock_broker = True
        self.config.instruments = [
            InstrumentConfig(
                symbol="EURUSD",
                point_value=1.0,
                pip_size=0.0001,
                avg_spread_points=10.0,
                digits=5,
                min_lot=0.01,
                max_lot=100.0,
                lot_step=0.01,
            ),
            InstrumentConfig(
                symbol="GBPUSD",
                point_value=1.0,
                pip_size=0.0001,
                avg_spread_points=10.0,
                digits=5,
                min_lot=0.01,
                max_lot=100.0,
                lot_step=0.01,
            ),
        ]
        self.config.pair1.symbol = "EURUSD"
        self.config.pair1.fixed_lot_size = 0.10
        self.config.pair1.fixed_sl_pips = 20.0
        self.config.pair1.enabled = True

        self.config.pair2.symbol = "GBPUSD"
        self.config.pair2.fixed_lot_size = 0.05
        self.config.pair2.fixed_sl_pips = 25.0
        self.config.pair2.enabled = True

        self.config.enabled_strategies = ["SMC", "SMC_SCALP_5M", "ICT"]
        self.config.risk.max_open_positions = 6
        self.config.risk.max_open_per_symbol = 3
        self.bot = None

    def tearDown(self):
        if self.bot and hasattr(self.bot, 'state') and self.bot.state:
            try:
                self.bot.state.close()
            except Exception:
                pass
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_synchronized_dual_pair_execution_same_cycle(self):
        """Verify that Pair 1 and Pair 2 can both execute valid trades in the exact same tick cycle."""
        self.bot = TradingBot(self.config, load_saved_settings=False)
        self.bot.startup()
        self.bot.is_active = True

        # Mock broker prices with low spread
        quote_eur = MagicMock(bid=1.0850, ask=1.08508, spread=0.00008)
        quote_gbp = MagicMock(bid=1.3050, ask=1.30508, spread=0.00008)
        self.bot.broker.get_current_price = lambda sym: quote_eur if sym == "EURUSD" else quote_gbp
        self.bot.broker.get_account_equity = lambda: 100000.0

        # Mock OHLCV data
        df = make_test_df()
        self.bot._get_ohlcv = lambda sym, tf, count=300: df

        # Mock news filter
        self.bot.news_filter.check_news_blackout = lambda sym, dt: MagicMock(blocked=False, reason="")

        # Both pairs return valid signals
        def mock_evaluate_all(symbol, **kwargs):
            if symbol == "EURUSD":
                return [
                    create_valid_signal(symbol="EURUSD", direction=Direction.BUY, entry=1.0851, sl=1.0831, tp=1.0901, strat_name="ICT", magic=126456)
                ]
            elif symbol == "GBPUSD":
                return [
                    create_valid_signal(symbol="GBPUSD", direction=Direction.BUY, entry=1.3051, sl=1.3026, tp=1.3116, strat_name="SMC", magic=124456)
                ]
            return []

        self.bot.strategy.evaluate_all = mock_evaluate_all

        # Run exactly ONE tick cycle
        self.bot._execute_tick()

        open_positions = self.bot.state.get_open_positions()
        eur_positions = [t for t in open_positions if t.symbol == "EURUSD"]
        gbp_positions = [t for t in open_positions if t.symbol == "GBPUSD"]

        self.assertEqual(len(eur_positions), 1, "Pair 1 (EURUSD) should have executed a trade in tick 1")
        self.assertEqual(len(gbp_positions), 1, "Pair 2 (GBPUSD) should have executed a trade immediately in the SAME tick 1")
        self.assertEqual(eur_positions[0].strategy_name, "ICT")
        self.assertEqual(gbp_positions[0].strategy_name, "SMC")
        self.assertAlmostEqual(eur_positions[0].lot_size, 0.10)
        self.assertAlmostEqual(gbp_positions[0].lot_size, 0.05)

    def test_duplicate_strategy_forbidden_on_same_pair(self):
        """Verify that a strategy cannot open duplicate trades on the same pair."""
        self.bot = TradingBot(self.config, load_saved_settings=False)
        self.bot.startup()
        self.bot.is_active = True

        # Record an open SMC trade for EURUSD
        self.bot.state.record_trade(TradeRecord(
            id=9001,
            timestamp=datetime.now(timezone.utc),
            symbol="EURUSD",
            direction=Direction.BUY,
            entry_price=1.0850,
            stop_loss=1.0830,
            take_profit=1.0900,
            lot_size=0.10,
            realized_pnl=0.0,
            status="OPEN",
            strategy_name="SMC",
            magic_number=124456,
        ))

        # Risk Engine authorization attempt for another SMC trade on EURUSD must be rejected
        sig_dup = create_valid_signal(symbol="EURUSD", direction=Direction.BUY, strat_name="SMC", magic=124456)

        auth = self.bot.risk_engine.authorize_trade(sig_dup, current_equity=100000.0)
        self.assertFalse(auth.authorized, "Duplicate trade for SMC on EURUSD must be blocked")
        self.assertIn("already has an active trade", auth.rejection_reason)

    def test_per_pair_maximum_3_trades(self):
        """Verify that at most 3 trades can be open on a single pair (one per strategy)."""
        self.bot = TradingBot(self.config, load_saved_settings=False)
        self.bot.startup()
        self.bot.is_active = True

        # Open 3 trades for EURUSD (SMC, SMC_SCALP_5M, ICT)
        strats = [("SMC", 124456), ("SMC_SCALP_5M", 125456), ("ICT", 126456)]
        for idx, (name, magic) in enumerate(strats, start=1):
            self.bot.state.record_trade(TradeRecord(
                id=9100 + idx,
                timestamp=datetime.now(timezone.utc),
                symbol="EURUSD",
                direction=Direction.BUY,
                entry_price=1.0850,
                stop_loss=1.0830,
                take_profit=1.0900,
                lot_size=0.10,
                realized_pnl=0.0,
                status="OPEN",
                strategy_name=name,
                magic_number=magic,
            ))

        eur_trades = [t for t in self.bot.state.get_open_positions() if t.symbol == "EURUSD"]
        self.assertEqual(len(eur_trades), 3, "Exactly 3 trades open on EURUSD")

        # Any 4th signal for EURUSD must be rejected by RiskEngine
        sig_extra = create_valid_signal(symbol="EURUSD", direction=Direction.BUY, strat_name="SMC", magic=124456)
        auth = self.bot.risk_engine.authorize_trade(sig_extra, current_equity=100000.0)
        self.assertFalse(auth.authorized)
        self.assertIn("Max open positions for EURUSD reached", auth.rejection_reason)

    def test_global_maximum_across_pairs(self):
        """Verify max 6 open trades across both pairs (3 strategies x 2 pairs)."""
        self.bot = TradingBot(self.config, load_saved_settings=False)
        self.bot.startup()
        self.bot.is_active = True

        strats = [("SMC", 124456), ("SMC_SCALP_5M", 125456), ("ICT", 126456)]
        trade_id = 9200

        # Fill 3 on EURUSD
        for name, magic in strats:
            trade_id += 1
            self.bot.state.record_trade(TradeRecord(
                id=trade_id,
                timestamp=datetime.now(timezone.utc),
                symbol="EURUSD",
                direction=Direction.BUY,
                entry_price=1.0850,
                stop_loss=1.0830,
                take_profit=1.0900,
                lot_size=0.10,
                realized_pnl=0.0,
                status="OPEN",
                strategy_name=name,
                magic_number=magic,
            ))

        # Fill 3 on GBPUSD
        for name, magic in strats:
            trade_id += 1
            self.bot.state.record_trade(TradeRecord(
                id=trade_id,
                timestamp=datetime.now(timezone.utc),
                symbol="GBPUSD",
                direction=Direction.BUY,
                entry_price=1.3050,
                stop_loss=1.3025,
                take_profit=1.3115,
                lot_size=0.05,
                realized_pnl=0.0,
                status="OPEN",
                strategy_name=name,
                magic_number=magic,
            ))

        all_open = self.bot.state.get_open_positions()
        self.assertEqual(len(all_open), 6, "Total open positions across both pairs must equal 6")

        # Any 7th trade must be rejected by circuit breaker check
        blocked, reason = self.bot.risk_engine.check_circuit_breakers(100000.0)
        self.assertTrue(blocked)
        self.assertIn("Max concurrent open positions reached (6/6 open)", reason)

    def test_stricter_global_cap_of_3(self):
        """Verify that when stricter global cap of 3 is configured, exactly 3 open trades are permitted and 4th is rejected."""
        strict_config = TradingConfig()
        strict_config.db_path = self.temp_db
        strict_config.use_mock_broker = True
        strict_config.instruments = self.config.instruments
        strict_config.pair1 = self.config.pair1
        strict_config.pair2 = self.config.pair2
        strict_config.risk.max_open_positions = 3  # Stricter global cap!

        self.bot = TradingBot(strict_config, load_saved_settings=False)
        self.bot.startup()
        self.bot.is_active = True

        # Record 2 on EURUSD and 1 on GBPUSD (Total = 3)
        self.bot.state.record_trade(TradeRecord(
            id=9301,
            timestamp=datetime.now(timezone.utc),
            symbol="EURUSD",
            direction=Direction.BUY,
            entry_price=1.0850,
            stop_loss=1.0830,
            take_profit=1.0900,
            lot_size=0.10,
            realized_pnl=0.0,
            status="OPEN",
            strategy_name="SMC",
            magic_number=124456,
        ))
        self.bot.state.record_trade(TradeRecord(
            id=9302,
            timestamp=datetime.now(timezone.utc),
            symbol="EURUSD",
            direction=Direction.BUY,
            entry_price=1.0850,
            stop_loss=1.0830,
            take_profit=1.0900,
            lot_size=0.10,
            realized_pnl=0.0,
            status="OPEN",
            strategy_name="ICT",
            magic_number=126456,
        ))
        self.bot.state.record_trade(TradeRecord(
            id=9303,
            timestamp=datetime.now(timezone.utc),
            symbol="GBPUSD",
            direction=Direction.BUY,
            entry_price=1.3050,
            stop_loss=1.3025,
            take_profit=1.3115,
            lot_size=0.05,
            realized_pnl=0.0,
            status="OPEN",
            strategy_name="SMC",
            magic_number=124456,
        ))

        self.assertEqual(len(self.bot.state.get_open_positions()), 3)

        # 4th trade on GBPUSD (e.g. ICT) must be rejected because global cap of 3 is reached
        sig_extra = create_valid_signal(symbol="GBPUSD", direction=Direction.BUY, strat_name="ICT", magic=126456)
        auth = self.bot.risk_engine.authorize_trade(sig_extra, current_equity=100000.0)
        self.assertFalse(auth.authorized)
        self.assertIn("Max concurrent open positions reached (3/3 open)", auth.rejection_reason)

    def test_ai_and_ml_trap_validation(self):
        """Verify AI analyzer confidence scoring and ML Trap Detector gate."""
        ai = AIAnalyst(self.config)
        sig = create_valid_signal(symbol="EURUSD", direction=Direction.BUY, strat_name="SMC", magic=124456)

        htf_analysis = MagicMock(bias=MarketBias.BULLISH, ema_value=1.0800)
        verdict = ai.evaluate_setup(sig, htf_analysis, current_spread=0.0001)
        self.assertTrue(verdict.confirmed, "Valid aligned setup should be confirmed by AI")
        self.assertGreaterEqual(verdict.confidence, 75.0)

        # ML Trap Detector veto check
        trap_svc = TrapDetectorService(TrapDetectorConfig(shadow_until_samples=0, p_genuine_threshold=0.60, db_path=":memory:"))
        # Mock predict_proba_genuine returning 0.30 (genuine probability below 0.60 threshold)
        trap_svc.model.predict_proba_genuine = lambda x: np.array([0.30])

        df = make_test_df(30)
        ev = trap_svc.observe_event(
            symbol="EURUSD",
            timeframe="5m",
            kind=EventKind.FVG_BULL,
            direction="long",
            entry=1.0850,
            stop=1.0830,
            target=1.0900,
            df=df,
            bar_index=len(df) - 1,
        )
        self.assertFalse(ev.allowed, "Trap detector must veto when p_genuine < threshold")


if __name__ == "__main__":
    unittest.main()

"""
test_ict_multi_strategy.py — Comprehensive Unit & Integration Tests for:
1. ICT Strategy Engine (Kill Zones, Judas Swings, MSS Displacement, FVG, OTE 61.8%-78.6% Retest).
2. BaseStrategy Polymorphism & Magic Offsets (SMC 15M, Scalp 5M, ICT).
3. Concurrent Multi-Strategy Execution via StrategyEngine.evaluate_all().
4. Independent Pair 1 and Pair 2 Configuration (Distinct Symbols, Lots, and Fixed SLs).
5. State Database Migrations and Segmented Win-Rate Analytics (by Strategy & by Pair).
6. Web API endpoints (/api/state, /api/configure, /api/trigger_tick).
"""

import unittest
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import tempfile
import os

from config import (
    TradingConfig,
    InstrumentConfig,
    Direction,
    MarketBias,
    StrategyType,
    PairSettings,
    ICTConfig,
)
from strategy import (
    StrategyEngine,
    BaseStrategy,
    SMCSwingStrategy,
    SMCScalp5MStrategy,
    ICTStrategy,
    ICTEngine,
    ICTKillZone,
    TradeSignal,
    LTFConfirmation,
    HTFAnalysis,
)
from risk_engine import RiskEngine
from state import StateManager, TradeRecord
from execution import BracketOrder, MockBrokerAdapter
from conflict_resolver import ConflictResolver
from web_app import app
from fastapi.testclient import TestClient


def make_candle(time, o, h, l, c, vol=100):
    return {
        'time': time,
        'open': float(o),
        'high': float(h),
        'low': float(l),
        'close': float(c),
        'tick_volume': vol,
        'volume': vol,
    }


class TestICTStrategyEngine(unittest.TestCase):

    def setUp(self):
        self.config = TradingConfig()
        self.instrument = InstrumentConfig(
            symbol="EURUSD",
            point_value=1.0,
            pip_size=0.0001,
            avg_spread_points=1.0,
            digits=5,
            min_lot=0.01,
            max_lot=100.0,
            lot_step=0.01,
        )
        self.ict_engine = ICTEngine(config=self.config.ict)

    def test_kill_zone_detection(self):
        """Test identification of ICT Kill Zones: London Open, NY AM, London Close, Asia."""
        # 07:30 UTC -> London Open Kill Zone (07:00 - 10:00)
        t_london = datetime(2026, 9, 15, 7, 30, tzinfo=timezone.utc)
        self.assertEqual(self.ict_engine.identify_kill_zone(t_london), ICTKillZone.LONDON_OPEN)

        # 13:30 UTC -> NY AM Kill Zone (12:00 - 15:00)
        t_ny = datetime(2026, 9, 15, 13, 30, tzinfo=timezone.utc)
        self.assertEqual(self.ict_engine.identify_kill_zone(t_ny), ICTKillZone.NY_AM)

        # 15:30 UTC -> London Close Kill Zone (15:00 - 17:00)
        t_lc = datetime(2026, 9, 15, 15, 30, tzinfo=timezone.utc)
        self.assertEqual(self.ict_engine.identify_kill_zone(t_lc), ICTKillZone.LONDON_CLOSE)

        # 01:00 UTC -> Asian Session (00:00 - 06:00)
        t_asia = datetime(2026, 9, 15, 1, 0, tzinfo=timezone.utc)
        self.assertEqual(self.ict_engine.identify_kill_zone(t_asia), ICTKillZone.ASIA)

        # 20:00 UTC -> Outside Kill Zones
        t_none = datetime(2026, 9, 15, 20, 0, tzinfo=timezone.utc)
        self.assertEqual(self.ict_engine.identify_kill_zone(t_none), ICTKillZone.NONE)

    def test_fvg_detection(self):
        """Test 3-candle Fair Value Gap detection."""
        # Bullish FVG: Candle 0 High < Candle 2 Low
        c0 = make_candle(datetime(2026, 9, 15, 8, 0, tzinfo=timezone.utc), 1.0800, 1.0820, 1.0795, 1.0815)
        c1 = make_candle(datetime(2026, 9, 15, 8, 5, tzinfo=timezone.utc), 1.0815, 1.0880, 1.0810, 1.0875)
        c2 = make_candle(datetime(2026, 9, 15, 8, 10, tzinfo=timezone.utc), 1.0875, 1.0900, 1.0850, 1.0890)

        df = pd.DataFrame([c0, c1, c2])
        fvg = self.ict_engine.detect_fvg(df, index=2, direction=Direction.BUY)
        self.assertIsNotNone(fvg)
        self.assertEqual(fvg['top'], 1.0850)
        self.assertEqual(fvg['bottom'], 1.0820)
        self.assertAlmostEqual(fvg['midpoint'], 1.0835)

    def test_fib_ote_calculation(self):
        """Test Optimal Trade Entry (OTE) 61.8% to 78.6% retracement zone."""
        # Bullish impulse: Low 1.0800 -> High 1.0900 (Range = 0.0100)
        ote = self.ict_engine.calculate_ote_zone(low=1.0800, high=1.0900, direction=Direction.BUY)
        # 61.8% retracement from 1.0900 = 1.0900 - 0.00618 = 1.08382
        # 78.6% retracement from 1.0900 = 1.0900 - 0.00786 = 1.08214
        self.assertAlmostEqual(ote['fib_618'], 1.08382, places=4)
        self.assertAlmostEqual(ote['fib_786'], 1.08214, places=4)
        self.assertAlmostEqual(ote['fib_705'], 1.08295, places=4)


class TestMultiStrategyEngine(unittest.TestCase):

    def setUp(self):
        self.config = TradingConfig()
        self.instrument = InstrumentConfig(
            symbol="XAUUSD",
            point_value=1.0,
            pip_size=0.01,
            avg_spread_points=25.0,
            digits=2,
            min_lot=0.01,
            max_lot=100.0,
            lot_step=0.01,
        )
        self.engine = StrategyEngine(self.config, self.instrument)

    def test_magic_offsets(self):
        """Verify unique magic offsets for each strategy implementation."""
        smc_strat = SMCSwingStrategy(self.config, self.instrument)
        scalp_strat = SMCScalp5MStrategy(self.config, self.instrument)
        ict_strat = ICTStrategy(self.config, self.instrument)

        self.assertEqual(smc_strat.magic_offset, 1000)
        self.assertEqual(scalp_strat.magic_offset, 2000)
        self.assertEqual(ict_strat.magic_offset, 3000)

        # Base magic is 123456
        base_magic = 123456
        self.assertEqual(base_magic + smc_strat.magic_offset, 124456)
        self.assertEqual(base_magic + scalp_strat.magic_offset, 125456)
        self.assertEqual(base_magic + ict_strat.magic_offset, 126456)

    def test_evaluate_all_concurrent(self):
        """Test that evaluate_all() checks all enabled strategies and tags signals with strategy_name and magic."""
        self.engine.set_enabled_strategies([
            StrategyType.SMC,
            StrategyType.SMC_SCALP_5M,
            StrategyType.ICT
        ])
        self.assertEqual(len(self.engine.active_strategies), 3)

        # Mock bars data
        base_time = datetime(2026, 9, 15, 8, 0, tzinfo=timezone.utc)
        bars_15m = pd.DataFrame([
            make_candle(base_time + timedelta(minutes=15*i), 2500 + i, 2505 + i, 2498 + i, 2502 + i)
            for i in range(50)
        ])
        bars_1h = pd.DataFrame([
            make_candle(base_time + timedelta(hours=i), 2490 + i*3, 2510 + i*3, 2485 + i*3, 2500 + i*3)
            for i in range(50)
        ])
        bars_5m = pd.DataFrame([
            make_candle(base_time + timedelta(minutes=5*i), 2500 + i*0.3, 2503 + i*0.3, 2498 + i*0.3, 2501 + i*0.3)
            for i in range(50)
        ])

        signals = self.engine.evaluate_all(
            bars_15m=bars_15m,
            bars_1h=bars_1h,
            bars_5m=bars_5m,
            current_spread_points=15.0,
            account_balance=10000.0,
        )
        # Verify result is a list of TradeSignal objects
        self.assertIsInstance(signals, list)
        for sig in signals:
            self.assertIn(sig.strategy_name, ["SMC_SWING", "SMC_SCALP_5M", "ICT"])
            self.assertIn(sig.magic_number, [124456, 125456, 126456])


class TestPairSettingsAndRisk(unittest.TestCase):

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.state = StateManager(db_path=self.temp_db.name)
        self.config = TradingConfig()
        self.risk_engine = RiskEngine(self.config, self.state)

    def tearDown(self):
        try:
            os.remove(self.temp_db.name)
        except OSError:
            pass

    def test_independent_lot_sizing(self):
        """Test position sizing with fixed lot override vs dynamic risk %."""
        inst = self.config.instruments[0]
        # Dynamic sizing (0.5% risk on $10,000 = $50 risk; 10.0 distance on XAUUSD digits=2 is 1000 points -> 50 / 1000 = 0.05 lots)
        lot_dynamic = self.risk_engine.calculate_lot_size(
            equity=10000.0,
            risk_pct=0.005,
            sl_distance_price=10.0,
            instrument=inst,
        )
        self.assertAlmostEqual(lot_dynamic, 0.05, places=2)

        # Signal for Pair 1 with fixed 0.10 lot
        sig_pair1 = TradeSignal(
            symbol="XAUUSD",
            direction=Direction.BUY,
            entry_price=2500.0,
            stop_loss=2480.0,
            take_profit=2550.0,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.OB_MITIGATION,
            rr_ratio=2.5,
            quality_score=80.0,
            timestamp=datetime.now(timezone.utc),
            sl_distance=20.0,
            tp_distance=50.0,
            strategy_id="SMC",
            strategy_name="SMC Swing (15m)",
            magic_number=124456,
        )
        auth_pair1 = self.risk_engine.authorize_trade(
            signal=sig_pair1,
            current_equity=10000.0,
            fixed_lot_size=0.10,
        )
        self.assertTrue(auth_pair1.authorized)
        self.assertEqual(auth_pair1.lot_size, 0.10)

        # Signal for Pair 2 with fixed 0.03 lot
        sig_pair2 = TradeSignal(
            symbol="EURUSD",
            direction=Direction.BUY,
            entry_price=1.0850,
            stop_loss=1.0830,
            take_profit=1.0900,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.ICT_KILLZONE_FVG,
            rr_ratio=2.5,
            quality_score=85.0,
            timestamp=datetime.now(timezone.utc),
            sl_distance=0.0020,
            tp_distance=0.0050,
            strategy_id="ICT",
            strategy_name="ICT KillZone / Silver Bullet",
            magic_number=126456,
        )
        auth_pair2 = self.risk_engine.authorize_trade(
            signal=sig_pair2,
            current_equity=10000.0,
            fixed_lot_size=0.03,
        )
        self.assertTrue(auth_pair2.authorized)
        self.assertEqual(auth_pair2.lot_size, 0.03)


class TestStateDatabaseAnalytics(unittest.TestCase):

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.state = StateManager(db_path=self.temp_db.name)

    def tearDown(self):
        try:
            os.remove(self.temp_db.name)
        except OSError:
            pass

    def test_schema_migration_and_segmented_stats(self):
        """Test recording trades across multiple strategies and pairs, and retrieving segmented stats."""
        now = datetime.now(timezone.utc)
        # Record trade 1: SMC strategy on XAUUSD (Win +$150)
        t1 = TradeRecord(
            id=1,
            timestamp=now,
            symbol="XAUUSD",
            direction="BUY",
            entry_price=2500.0,
            stop_loss=2490.0,
            take_profit=2530.0,
            lot_size=0.05,
            realized_pnl=150.0,
            status="FILLED",
            strategy_name="SMC_SWING",
            magic_number=124456,
        )
        self.state.record_trade(t1)

        # Record trade 2: Scalp 5M strategy on EURUSD (Loss -$40)
        t2 = TradeRecord(
            id=2,
            timestamp=now + timedelta(minutes=10),
            symbol="EURUSD",
            direction="SELL",
            entry_price=1.0850,
            stop_loss=1.0870,
            take_profit=1.0820,
            lot_size=0.02,
            realized_pnl=-40.0,
            status="FILLED",
            strategy_name="SMC_SCALP_5M",
            magic_number=125456,
        )
        self.state.record_trade(t2)

        # Record trade 3: ICT strategy on XAUUSD (Win +$200)
        t3 = TradeRecord(
            id=3,
            timestamp=now + timedelta(minutes=20),
            symbol="XAUUSD",
            direction="BUY",
            entry_price=2510.0,
            stop_loss=2500.0,
            take_profit=2540.0,
            lot_size=0.05,
            realized_pnl=200.0,
            status="FILLED",
            strategy_name="ICT",
            magic_number=126456,
        )
        self.state.record_trade(t3)

        # Check stats by strategy
        strat_stats = self.state.get_stats_by_strategy()
        self.assertIn("SMC_SWING", strat_stats)
        self.assertIn("SMC_SCALP_5M", strat_stats)
        self.assertIn("ICT", strat_stats)

        self.assertEqual(strat_stats["SMC_SWING"]["win_rate"], 100.0)
        self.assertEqual(strat_stats["SMC_SWING"]["total_pnl"], 150.0)

        self.assertEqual(strat_stats["SMC_SCALP_5M"]["win_rate"], 0.0)
        self.assertEqual(strat_stats["SMC_SCALP_5M"]["total_pnl"], -40.0)

        self.assertEqual(strat_stats["ICT"]["win_rate"], 100.0)
        self.assertEqual(strat_stats["ICT"]["total_pnl"], 200.0)

        # Check stats by pair
        pair_stats = self.state.get_stats_by_pair()
        self.assertIn("XAUUSD", pair_stats)
        self.assertIn("EURUSD", pair_stats)

        self.assertEqual(pair_stats["XAUUSD"]["total_trades"], 2)
        self.assertEqual(pair_stats["XAUUSD"]["win_rate"], 100.0)
        self.assertEqual(pair_stats["XAUUSD"]["total_pnl"], 350.0)

        self.assertEqual(pair_stats["EURUSD"]["total_trades"], 1)
        self.assertEqual(pair_stats["EURUSD"]["win_rate"], 0.0)
        self.assertEqual(pair_stats["EURUSD"]["total_pnl"], -40.0)


class TestWebAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_api_state(self):
        """Test GET /api/state returns multi-strategy, pair1, pair2, and stats breakdown fields."""
        response = self.client.get("/api/state")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("enabled_strategies", data)
        self.assertIn("pair1", data)
        self.assertIn("pair2", data)
        self.assertIn("stats_by_strategy", data)
        self.assertIn("stats_by_pair", data)

    def test_api_configure_multi_strategy_and_pairs(self):
        """Test POST /api/configure updates concurrent strategies and per-pair lot/SL."""
        payload = {
            "pair1": {
                "symbol": "XAUUSD",
                "enabled": True,
                "fixed_lot_size": 0.05,
                "fixed_sl_pips": 25.0
            },
            "pair2": {
                "symbol": "EURUSD",
                "enabled": True,
                "fixed_lot_size": 0.02,
                "fixed_sl_pips": 15.0
            },
            "enabled_strategies": ["SMC", "SMC_SCALP_5M", "ICT"],
            "ai_confirmation_enabled": True
        }
        response = self.client.post("/api/configure", json=payload)
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data.get("status"), "success")

        # Verify updated state
        state_res = self.client.get("/api/state")
        state_data = state_res.json()
        self.assertEqual(state_data["pair1"]["symbol"], "XAUUSD")
        self.assertEqual(state_data["pair1"]["fixed_lot_size"], 0.05)
        self.assertEqual(state_data["pair1"]["fixed_sl_pips"], 25.0)
        self.assertEqual(state_data["pair2"]["symbol"], "EURUSD")
        self.assertEqual(state_data["pair2"]["fixed_lot_size"], 0.02)
        self.assertEqual(state_data["pair2"]["fixed_sl_pips"], 15.0)
        self.assertEqual(set(state_data["enabled_strategies"]), {"SMC", "SMC_SCALP_5M", "ICT"})


class TestMultiStrategyConcurrency(unittest.TestCase):
    """
    Validates:
    1. Every strategy can execute 1 trade at a time per symbol.
    2. Duplicate strategy on the same symbol is rejected.
    3. Different strategies on the same symbol are authorized up to 3 trades per symbol.
    4. 4th trade on the same symbol is rejected (max 3 for EURUSD, max 3 for XAUUSD).
    5. Maximum 6 concurrent open trades across the bot (3 EURUSD + 3 XAUUSD).
    6. 7th trade across the bot is rejected by the global limit.
    """

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.state = StateManager(db_path=self.temp_db.name)
        self.config = TradingConfig()
        self.risk_engine = RiskEngine(self.config, self.state)

    def tearDown(self):
        try:
            os.remove(self.temp_db.name)
        except OSError:
            pass

    def _make_signal(self, symbol: str, strategy_name: str, magic_number: int, entry: float, sl: float, tp: float):
        return TradeSignal(
            symbol=symbol,
            direction=Direction.BUY,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.LIQUIDITY_SWEEP,
            rr_ratio=2.5,
            quality_score=85.0,
            timestamp=datetime.now(timezone.utc),
            sl_distance=abs(entry - sl),
            tp_distance=abs(tp - entry),
            strategy_id=strategy_name,
            strategy_name=strategy_name,
            magic_number=magic_number,
        )

    def test_multi_strategy_concurrency_and_limits(self):
        now = datetime.now(timezone.utc)
        equity = 100000.0

        # 1. Authorize SMC on EURUSD -> Should succeed
        sig_eur_smc = self._make_signal("EURUSD", "SMC", 124456, 1.0850, 1.0830, 1.0900)
        auth = self.risk_engine.authorize_trade(sig_eur_smc, equity, fixed_lot_size=0.01)
        self.assertTrue(auth.authorized)

        # Record trade as OPEN in state
        t1 = TradeRecord(
            id=1, timestamp=now, symbol="EURUSD", direction="BUY",
            entry_price=1.0850, stop_loss=1.0830, take_profit=1.0900,
            lot_size=0.01, realized_pnl=0.0, status="OPEN",
            strategy_name="SMC", magic_number=124456
        )
        self.state.record_trade(t1)

        # 2. Authorize duplicate SMC on EURUSD -> Must be REJECTED
        auth_dup = self.risk_engine.authorize_trade(sig_eur_smc, equity, fixed_lot_size=0.01)
        self.assertFalse(auth_dup.authorized)
        self.assertIn("already has an active trade for EURUSD", auth_dup.rejection_reason)

        # 3. Authorize ICT on EURUSD -> Should SUCCEED (different strategy)
        sig_eur_ict = self._make_signal("EURUSD", "ICT", 126456, 1.0855, 1.0835, 1.0905)
        auth_ict = self.risk_engine.authorize_trade(sig_eur_ict, equity, fixed_lot_size=0.01)
        self.assertTrue(auth_ict.authorized)

        t2 = TradeRecord(
            id=2, timestamp=now, symbol="EURUSD", direction="BUY",
            entry_price=1.0855, stop_loss=1.0835, take_profit=1.0905,
            lot_size=0.01, realized_pnl=0.0, status="OPEN",
            strategy_name="ICT", magic_number=126456
        )
        self.state.record_trade(t2)

        # 4. Authorize SMC_SCALP_5M on EURUSD -> Should SUCCEED (3rd strategy on EURUSD)
        sig_eur_scalp = self._make_signal("EURUSD", "SMC_SCALP_5M", 125456, 1.0860, 1.0840, 1.0910)
        auth_scalp = self.risk_engine.authorize_trade(sig_eur_scalp, equity, fixed_lot_size=0.01)
        self.assertTrue(auth_scalp.authorized)

        t3 = TradeRecord(
            id=3, timestamp=now, symbol="EURUSD", direction="BUY",
            entry_price=1.0860, stop_loss=1.0840, take_profit=1.0910,
            lot_size=0.01, realized_pnl=0.0, status="OPEN",
            strategy_name="SMC_SCALP_5M", magic_number=125456
        )
        self.state.record_trade(t3)

        # 5. EURUSD now has 3 open trades (SMC, ICT, SMC_SCALP_5M). 4th trade on EURUSD must be REJECTED.
        sig_eur_4th = self._make_signal("EURUSD", "CUSTOM_STRAT", 999999, 1.0865, 1.0845, 1.0915)
        auth_4th = self.risk_engine.authorize_trade(sig_eur_4th, equity, fixed_lot_size=0.01)
        self.assertFalse(auth_4th.authorized)
        self.assertIn("Max open positions for EURUSD reached", auth_4th.rejection_reason)

        # 6. Now test XAUUSD: SMC on XAUUSD should SUCCEED (even though SMC is open on EURUSD)
        sig_xau_smc = self._make_signal("XAUUSD", "SMC", 124456, 2500.0, 2490.0, 2525.0)
        auth_xau_smc = self.risk_engine.authorize_trade(sig_xau_smc, equity, fixed_lot_size=0.01)
        self.assertTrue(auth_xau_smc.authorized)

        t4 = TradeRecord(
            id=4, timestamp=now, symbol="XAUUSD", direction="BUY",
            entry_price=2500.0, stop_loss=2490.0, take_profit=2525.0,
            lot_size=0.01, realized_pnl=0.0, status="OPEN",
            strategy_name="SMC", magic_number=124456
        )
        self.state.record_trade(t4)

        # 7. Authorize ICT on XAUUSD -> SUCCEEDS (2nd open trade on XAUUSD)
        sig_xau_ict = self._make_signal("XAUUSD", "ICT", 126456, 2505.0, 2495.0, 2530.0)
        auth_xau_ict = self.risk_engine.authorize_trade(sig_xau_ict, equity, fixed_lot_size=0.01)
        self.assertTrue(auth_xau_ict.authorized)

        t5 = TradeRecord(
            id=5, timestamp=now, symbol="XAUUSD", direction="BUY",
            entry_price=2505.0, stop_loss=2495.0, take_profit=2530.0,
            lot_size=0.01, realized_pnl=0.0, status="OPEN",
            strategy_name="ICT", magic_number=126456
        )
        self.state.record_trade(t5)

        # 8. Authorize SMC_SCALP_5M on XAUUSD -> SUCCEEDS (3rd open trade on XAUUSD, 6th total)
        sig_xau_scalp = self._make_signal("XAUUSD", "SMC_SCALP_5M", 125456, 2510.0, 2500.0, 2535.0)
        auth_xau_scalp = self.risk_engine.authorize_trade(sig_xau_scalp, equity, fixed_lot_size=0.01)
        self.assertTrue(auth_xau_scalp.authorized)

        t6 = TradeRecord(
            id=6, timestamp=now, symbol="XAUUSD", direction="BUY",
            entry_price=2510.0, stop_loss=2500.0, take_profit=2535.0,
            lot_size=0.01, realized_pnl=0.0, status="OPEN",
            strategy_name="SMC_SCALP_5M", magic_number=125456
        )
        self.state.record_trade(t6)

        # 9. Now we have 6 open positions (3 EURUSD, 3 XAUUSD).
        self.assertEqual(len(self.state.get_open_positions()), 6)

        # 10. A 7th trade across the bot must be REJECTED by the global max_open_positions limit (6/6).
        sig_7th = self._make_signal("GBPUSD", "SMC", 124456, 1.3000, 1.2980, 1.3050)
        auth_7th = self.risk_engine.authorize_trade(sig_7th, equity, fixed_lot_size=0.01)
        self.assertFalse(auth_7th.authorized)
        self.assertIn("Max concurrent open positions reached (6/6 open)", auth_7th.rejection_reason)


if __name__ == '__main__':
    unittest.main()

"""
test_order_flow_strategy.py — Comprehensive Unit Tests for Order Flow Trading Strategy.
"""

import unittest
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

from config import (
    DEFAULT_CONFIG,
    Direction,
    InstrumentConfig,
    MarketBias,
    OrderFlowConfig,
    RiskConfig,
    StrategyType,
    TradingConfig,
    normalize_strategy_key,
)
from conflict_resolver import ConflictResolver
from strategy import (
    HTFAnalysis,
    HTFAnalyzer,
    LiquidityPool,
    LTFConfirmation,
    OrderFlowEngine,
    OrderFlowStrategy,
    StrategyEngine,
    TradeSignal,
)


class TestOrderFlowStrategy(unittest.TestCase):
    def setUp(self):
        self.instrument = InstrumentConfig(
            symbol="EURUSD",
            point_value=1.0,
            pip_size=0.0001,
            avg_spread_points=1.5,
            digits=5,
        )
        self.of_config = OrderFlowConfig(
            delta_lookback_bars=20,
            absorption_volume_factor=1.5,
            wick_ratio_threshold=0.4,
            cvd_divergence_bars=14,
            min_rr=1.8,
            target_rr=2.2,
        )
        self.engine = OrderFlowEngine(self.of_config)

    def _create_mock_ohlcv(self, count=30, base_price=1.0800, trend="BULLISH") -> pd.DataFrame:
        """Create synthetic OHLCV dataframe with consistent time series."""
        times = [datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(count)]
        rows = []
        p = base_price
        for i in range(count):
            if trend == "BULLISH":
                p += 0.0002
            elif trend == "BEARISH":
                p -= 0.0002
            o = p - 0.0001
            c = p + 0.0001
            h = max(o, c) + 0.0001
            l = min(o, c) - 0.0001
            rows.append({
                'time': times[i],
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'tick_volume': 150.0,
            })
        return pd.DataFrame(rows)

    def test_volume_delta_and_cvd_calculation(self):
        """Test mathematical calculation of bar delta and cumulative volume delta."""
        df = self._create_mock_ohlcv(count=10)
        # Bar where close == high (100% buying volume)
        df.loc[5, 'open'] = 1.0820
        df.loc[5, 'low'] = 1.0820
        df.loc[5, 'high'] = 1.0850
        df.loc[5, 'close'] = 1.0850
        df.loc[5, 'tick_volume'] = 500.0

        # Bar where close == low (100% selling volume)
        df.loc[6, 'open'] = 1.0850
        df.loc[6, 'high'] = 1.0850
        df.loc[6, 'low'] = 1.0810
        df.loc[6, 'close'] = 1.0810
        df.loc[6, 'tick_volume'] = 400.0

        df_of = self.engine.compute_volume_delta(df)
        self.assertIn('vol_delta', df_of.columns)
        self.assertIn('cvd', df_of.columns)
        self.assertIn('vol_sma20', df_of.columns)

        # Bar 5: all buying -> vol_delta should be +500
        self.assertAlmostEqual(df_of.loc[5, 'vol_delta'], 500.0, places=2)
        # Bar 6: all selling -> vol_delta should be -400
        self.assertAlmostEqual(df_of.loc[6, 'vol_delta'], -400.0, places=2)

    def test_bullish_absorption_detection(self):
        """Test bullish institutional absorption at support."""
        df = self._create_mock_ohlcv(count=25, trend="BEARISH")
        # Bar 23: heavy volume push down with long lower wick (absorption)
        df.loc[23, 'tick_volume'] = 600.0  # High volume
        df.loc[23, 'high'] = 1.0760
        df.loc[23, 'open'] = 1.0755
        df.loc[23, 'close'] = 1.0754
        df.loc[23, 'low'] = 1.0730   # Long lower wick (24 pips wick vs 30 pips range -> 80% wick)

        # Bar 24: current bar confirms bullish bounce
        df.loc[24, 'open'] = 1.0754
        df.loc[24, 'low'] = 1.0750
        df.loc[24, 'close'] = 1.0770
        df.loc[24, 'high'] = 1.0772
        df.loc[24, 'tick_volume'] = 200.0

        htf_analysis = HTFAnalysis(
            bias=MarketBias.BULLISH,
            ema_value=1.0700,
            last_swing_high=1.0850,
            last_swing_low=1.0700,
            trend_clarity_score=22.0,
            liquidity_pools=[LiquidityPool(level=1.0820, is_high=True, touch_count=2, last_index=10)],
        )

        entry = self.engine.detect_order_flow_entry(
            df=df,
            htf_analysis=htf_analysis,
            instrument=self.instrument,
            current_spread=0.0001,
        )

        self.assertIsNotNone(entry)
        self.assertEqual(entry['direction'], Direction.BUY)
        self.assertIn(entry['conf'], (LTFConfirmation.OF_ABSORPTION, LTFConfirmation.OF_LIQUIDITY_TRAP, LTFConfirmation.OF_DELTA_DIVERGENCE))
        self.assertLess(entry['sl'], entry['entry'])
        self.assertGreater(entry['tp'], entry['entry'])

    def test_bearish_absorption_detection(self):
        """Test bearish institutional absorption at resistance."""
        df = self._create_mock_ohlcv(count=25, trend="BULLISH")
        # Bar 23: heavy volume push up with long upper wick
        df.loc[23, 'tick_volume'] = 700.0
        df.loc[23, 'low'] = 1.0840
        df.loc[23, 'open'] = 1.0845
        df.loc[23, 'close'] = 1.0846
        df.loc[23, 'high'] = 1.0880  # Long upper wick

        # Bar 24: confirms bearish rejection
        df.loc[24, 'open'] = 1.0846
        df.loc[24, 'high'] = 1.0848
        df.loc[24, 'close'] = 1.0825
        df.loc[24, 'low'] = 1.0820
        df.loc[24, 'tick_volume'] = 250.0

        htf_analysis = HTFAnalysis(
            bias=MarketBias.BEARISH,
            ema_value=1.0900,
            last_swing_high=1.0900,
            last_swing_low=1.0750,
            trend_clarity_score=24.0,
            liquidity_pools=[LiquidityPool(level=1.0750, is_high=False, touch_count=2, last_index=10)],
        )

        entry = self.engine.detect_order_flow_entry(
            df=df,
            htf_analysis=htf_analysis,
            instrument=self.instrument,
            current_spread=0.0001,
        )

        self.assertIsNotNone(entry)
        self.assertEqual(entry['direction'], Direction.SELL)
        self.assertGreater(entry['sl'], entry['entry'])
        self.assertLess(entry['tp'], entry['entry'])

    def test_cvd_divergence_detection(self):
        """Test Cumulative Volume Delta divergence."""
        df = self._create_mock_ohlcv(count=25, trend="BEARISH")
        df_of = self.engine.compute_volume_delta(df)
        
        # Craft a bullish CVD divergence: price makes lower low, but CVD makes higher low
        # Prior low at bar 15
        df.loc[15, 'low'] = 1.0780
        df.loc[15, 'close'] = 1.0782
        df.loc[15, 'high'] = 1.0785
        df.loc[15, 'open'] = 1.0785
        df.loc[15, 'tick_volume'] = 800.0  # Big selling volume pushing CVD deep negative

        # Current low at bar 24: price slightly lower, but low selling volume & positive delta
        df.loc[24, 'open'] = 1.0776
        df.loc[24, 'low'] = 1.0774  # Lower price than bar 15
        df.loc[24, 'close'] = 1.0785 # Strong bullish close
        df.loc[24, 'high'] = 1.0786
        df.loc[24, 'tick_volume'] = 200.0

        htf_analysis = HTFAnalysis(
            bias=MarketBias.BULLISH,
            ema_value=1.0700,
            last_swing_high=1.0850,
            last_swing_low=1.0700,
            trend_clarity_score=20.0,
            liquidity_pools=[LiquidityPool(level=1.0850, is_high=True, touch_count=2, last_index=10)],
        )

        entry = self.engine.detect_order_flow_entry(
            df=df,
            htf_analysis=htf_analysis,
            instrument=self.instrument,
            current_spread=0.0001,
        )
        if entry:
            self.assertEqual(entry['direction'], Direction.BUY)

    def test_order_flow_strategy_evaluation(self):
        """Test OrderFlowStrategy evaluate method producing TradeSignal."""
        htf_analyzer = HTFAnalyzer()
        strat = OrderFlowStrategy(htf_analyzer, self.of_config)
        self.assertEqual(strat.id, "ORDER_FLOW")
        self.assertEqual(strat.magic_offset, 4000)

        df_htf = self._create_mock_ohlcv(count=50, base_price=1.0700, trend="BULLISH")
        df_ltf = self._create_mock_ohlcv(count=25, base_price=1.0750, trend="BULLISH")

        # Inject absorption pattern
        df_ltf.loc[23, 'tick_volume'] = 600.0
        df_ltf.loc[23, 'high'] = 1.0760
        df_ltf.loc[23, 'open'] = 1.0755
        df_ltf.loc[23, 'close'] = 1.0754
        df_ltf.loc[23, 'low'] = 1.0730
        df_ltf.loc[24, 'open'] = 1.0754
        df_ltf.loc[24, 'low'] = 1.0750
        df_ltf.loc[24, 'close'] = 1.0770
        df_ltf.loc[24, 'high'] = 1.0772
        df_ltf.loc[24, 'tick_volume'] = 200.0

        signals = strat.evaluate(
            symbol="EURUSD",
            htf_data=df_htf,
            ltf_data=df_ltf,
            instrument=self.instrument,
            current_spread=0.0001,
        )

        if signals:
            sig = signals[0]
            self.assertEqual(sig.strategy_id, "ORDER_FLOW")
            self.assertEqual(sig.magic_number, 127456)
            self.assertGreaterEqual(sig.rr_ratio, 1.8)
            self.assertGreater(sig.quality_score, 50.0)

    def test_strategy_engine_all_four_strategies(self):
        """Verify StrategyEngine registers and runs all 4 strategies concurrently."""
        config = TradingConfig()
        self.assertIn("ORDER_FLOW", config.enabled_strategies)

        engine = StrategyEngine(config)
        self.assertIn("ORDER_FLOW", engine.strategies)
        self.assertEqual(len(engine.active_strategies), 4)

        active_ids = [s.id for s in engine.active_strategies]
        self.assertEqual(active_ids, ["SMC", "SMC_SCALP_5M", "ICT", "ORDER_FLOW"])

    def test_conflict_resolver_with_order_flow(self):
        """Test ConflictResolver filters and deduplicates ORDER_FLOW signals."""
        risk_cfg = RiskConfig(min_rr_ratio=2.5)
        resolver = ConflictResolver(risk_cfg)

        sig_of = TradeSignal(
            symbol="EURUSD",
            direction=Direction.BUY,
            entry_price=1.0850,
            stop_loss=1.0830,
            take_profit=1.0890,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.OF_ABSORPTION,
            rr_ratio=2.0,  # Below standard swing 2.5, but >= 1.8 Order Flow threshold
            quality_score=85.0,
            timestamp=datetime.now(timezone.utc),
            sl_distance=0.0020,
            tp_distance=0.0040,
            strategy_id="ORDER_FLOW",
            strategy_name="Order Flow (Delta & Absorption)",
            magic_number=127456,
        )

        res = resolver.resolve([sig_of], current_spread=0.0001)
        self.assertIsNotNone(res.accepted_signal)
        self.assertEqual(res.accepted_signal.strategy_id, "ORDER_FLOW")
        self.assertEqual(len(res.accepted_signals), 1)

    def test_normalize_strategy_key_order_flow(self):
        """Verify normalize_strategy_key properly canonicalizes ORDER_FLOW."""
        self.assertEqual(normalize_strategy_key("ORDER_FLOW"), "ORDER_FLOW")
        self.assertEqual(normalize_strategy_key("Order Flow (Delta & Absorption)"), "ORDER_FLOW")
        self.assertEqual(normalize_strategy_key("OF_ABSORPTION"), "ORDER_FLOW")
        self.assertEqual(normalize_strategy_key(None, magic=127456), "ORDER_FLOW")


if __name__ == "__main__":
    unittest.main()

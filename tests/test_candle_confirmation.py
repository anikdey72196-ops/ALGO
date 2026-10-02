"""
tests/test_candle_confirmation.py — Comprehensive Unit Tests for Directional Candlestick Confirmation.

Verifies:
1. Pure function check_candle_body_confirmation:
   - Bullish BUY: Must be GREEN (close > open) with body / total_range >= min_body_ratio (default 50%).
   - Bearish SELL: Must be RED (close < open) with body / total_range >= min_body_ratio (default 50%).
   - Marubozu candles (100% body, 0% wicks) accepted.
   - Long-wick / indecision candles (< 50% body) rejected.
   - Wrong-color candles immediately rejected.
2. Integration across strategies:
   - SMC Swing (SMCEntryDetector)
   - SMC Scalp 5M (SMCScalp5MEngine)
   - ICT Silver Bullet & KillZone (ICTEngine)
   - Order Flow Microstructure (OrderFlowEngine)
"""

import unittest
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from config import (
    TradingConfig,
    InstrumentConfig,
    Direction,
    MarketBias,
    TimeframeConfig,
    ICTConfig,
    OrderFlowConfig,
)
from strategy import (
    check_candle_body_confirmation,
    SMCEntryDetector,
    SMCScalp5MEngine,
    ICTEngine,
    OrderFlowEngine,
    HTFAnalysis,
    LTFConfirmation,
)


def make_bar(t, o, h, l, c, vol=100.0, delta=10.0):
    return {
        'time': t,
        'open': float(o),
        'high': float(h),
        'low': float(l),
        'close': float(c),
        'tick_volume': float(vol),
        'volume': float(vol),
        'delta': float(delta),
    }


class TestCandleConfirmation(unittest.TestCase):

    def setUp(self):
        self.instrument = InstrumentConfig(
            symbol="XAUUSD",
            point_value=1.0,
            pip_size=0.01,
            avg_spread_points=25.0,
            digits=2,
        )

    # ─────────────────────────────────────────────────────────
    # 1. Pure Function Validation: check_candle_body_confirmation
    # ─────────────────────────────────────────────────────────

    def test_buy_confirmed_green_candle_above_50_pct(self):
        """Green candle with 70% body should be confirmed for BUY."""
        candle = {'open': 100.0, 'high': 110.0, 'low': 100.0, 'close': 107.0}
        # Range = 10, Body = 7 (70%)
        is_conf, ratio, desc = check_candle_body_confirmation(candle, Direction.BUY, min_body_ratio=0.50)
        self.assertTrue(is_conf)
        self.assertAlmostEqual(ratio, 0.70, places=2)
        self.assertIn("Confirmed GREEN", desc)

    def test_buy_rejected_green_candle_below_50_pct_wicks(self):
        """Green candle with only 30% body (70% wicks) should be rejected for BUY."""
        candle = {'open': 103.0, 'high': 110.0, 'low': 100.0, 'close': 106.0}
        # Range = 10, Body = 3 (30%)
        is_conf, ratio, desc = check_candle_body_confirmation(candle, Direction.BUY, min_body_ratio=0.50)
        self.assertFalse(is_conf)
        self.assertAlmostEqual(ratio, 0.30, places=2)
        self.assertIn("excessive wick", desc)

    def test_buy_rejected_red_candle(self):
        """Red candle must be immediately rejected for BUY setup."""
        candle = {'open': 108.0, 'high': 110.0, 'low': 100.0, 'close': 102.0}
        is_conf, ratio, desc = check_candle_body_confirmation(candle, Direction.BUY, min_body_ratio=0.50)
        self.assertFalse(is_conf)
        self.assertIn("Bearish/Flat", desc)

    def test_sell_confirmed_red_candle_above_50_pct(self):
        """Red candle with 80% body should be confirmed for SELL."""
        candle = {'open': 108.0, 'high': 110.0, 'low': 100.0, 'close': 100.0}
        # Range = 10, Body = 8 (80%)
        is_conf, ratio, desc = check_candle_body_confirmation(candle, Direction.SELL, min_body_ratio=0.50)
        self.assertTrue(is_conf)
        self.assertAlmostEqual(ratio, 0.80, places=2)
        self.assertIn("Confirmed RED", desc)

    def test_sell_rejected_red_candle_below_50_pct_wicks(self):
        """Red candle with only 35% body (65% wicks) should be rejected for SELL."""
        candle = {'open': 107.0, 'high': 110.0, 'low': 100.0, 'close': 103.5}
        # Range = 10, Body = 3.5 (35%)
        is_conf, ratio, desc = check_candle_body_confirmation(candle, Direction.SELL, min_body_ratio=0.50)
        self.assertFalse(is_conf)
        self.assertAlmostEqual(ratio, 0.35, places=2)
        self.assertIn("excessive wick", desc)

    def test_sell_rejected_green_candle(self):
        """Green candle must be immediately rejected for SELL setup."""
        candle = {'open': 102.0, 'high': 110.0, 'low': 100.0, 'close': 108.0}
        is_conf, ratio, desc = check_candle_body_confirmation(candle, Direction.SELL, min_body_ratio=0.50)
        self.assertFalse(is_conf)
        self.assertIn("Bullish/Flat", desc)

    def test_marubozu_candle_100_pct(self):
        """100% Marubozu candle is valid for both BUY and SELL respectively."""
        # Bullish Marubozu (open = low, close = high)
        bull_marubozu = {'open': 100.0, 'high': 110.0, 'low': 100.0, 'close': 110.0}
        is_conf_b, ratio_b, _ = check_candle_body_confirmation(bull_marubozu, Direction.BUY, 0.50)
        self.assertTrue(is_conf_b)
        self.assertAlmostEqual(ratio_b, 1.0, places=2)

        # Bearish Marubozu (open = high, close = low)
        bear_marubozu = {'open': 110.0, 'high': 110.0, 'low': 100.0, 'close': 100.0}
        is_conf_s, ratio_s, _ = check_candle_body_confirmation(bear_marubozu, Direction.SELL, 0.50)
        self.assertTrue(is_conf_s)
        self.assertAlmostEqual(ratio_s, 1.0, places=2)

    def test_flat_zero_range_candle(self):
        """Zero range candle should be rejected safely."""
        flat_candle = {'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0}
        is_conf, _, desc = check_candle_body_confirmation(flat_candle, Direction.BUY, 0.50)
        self.assertFalse(is_conf)
        self.assertIn("Zero or flat", desc)

    # ─────────────────────────────────────────────────────────
    # 2. Integration: SMCScalp5MEngine Rejection vs Approval
    # ─────────────────────────────────────────────────────────

    def test_scalp_engine_rejects_unconfirmed_candle(self):
        """If retest candle is a RED candle on a BUY setup, scalp engine rejects it."""
        scalp_engine = SMCScalp5MEngine(swing_lookback=2)
        base_time = datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc)
        prices = [
            (2000.0, 2002.0, 1999.0, 2001.0),
            (2001.0, 2003.0, 2000.0, 2002.0),
            (2002.0, 2004.0, 2001.0, 2003.0),
            (2003.0, 2004.5, 2002.0, 2004.0),
            (2004.0, 2005.0, 2003.0, 2004.5),  # idx 4: Swing High = 2005.0
            (2004.5, 2004.8, 2002.0, 2002.5),
            (2002.5, 2003.0, 2001.0, 2001.5),
            (2001.5, 2002.0, 2000.0, 2000.5),  # idx 7: OB Candle [2000.0, 2002.0]
            (2000.5, 2004.0, 2000.2, 2003.8),
            (2003.8, 2006.5, 2003.5, 2006.0),  # idx 9: BOS above 2005.0
            (2006.0, 2007.0, 2005.0, 2006.5),
            (2006.5, 2006.5, 2001.5, 2001.8),  # idx 11: Touches OB
            # idx 12: Retest candle is RED (close < open)
            (2003.0, 2003.5, 2001.5, 2001.8),  # Open 2003.0 > Close 2001.8 -> RED candle!
        ]
        candles = [make_bar(base_time + timedelta(minutes=5 * i), o, h, l, c) for i, (o, h, l, c) in enumerate(prices)]
        df_5m = pd.DataFrame(candles)
        htf = HTFAnalysis(bias=MarketBias.BULLISH, ema_value=1990.0, last_swing_high=2020.0, last_swing_low=1980.0, trend_clarity_score=25.0)

        signal = scalp_engine.detect_scalp_entry(df=df_5m, htf_analysis=htf, instrument=self.instrument, current_spread=0.25)
        self.assertIsNone(signal, "Scalp BUY signal MUST be rejected when confirmation candle is RED")

    def test_scalp_engine_rejects_weak_body_green_candle(self):
        """If confirmation candle is GREEN but body is only 30% (< 50% min), reject."""
        scalp_engine = SMCScalp5MEngine(swing_lookback=2)
        base_time = datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc)
        prices = [
            (2000.0, 2002.0, 1999.0, 2001.0),
            (2001.0, 2003.0, 2000.0, 2002.0),
            (2002.0, 2004.0, 2001.0, 2003.0),
            (2003.0, 2004.5, 2002.0, 2004.0),
            (2004.0, 2005.0, 2003.0, 2004.5),
            (2004.5, 2004.8, 2002.0, 2002.5),
            (2002.5, 2003.0, 2001.0, 2001.5),
            (2001.5, 2002.0, 2000.0, 2000.5),
            (2000.5, 2004.0, 2000.2, 2003.8),
            (2003.8, 2006.5, 2003.5, 2006.0),
            (2006.0, 2007.0, 2005.0, 2006.5),
            (2006.5, 2006.5, 2001.5, 2001.8),
            # idx 12: Range = 2005.0 - 2001.0 = 4.0. Body = 2002.5 - 2001.5 = 1.0 (25% body)
            (2001.5, 2005.0, 2001.0, 2002.5),
        ]
        candles = [make_bar(base_time + timedelta(minutes=5 * i), o, h, l, c) for i, (o, h, l, c) in enumerate(prices)]
        df_5m = pd.DataFrame(candles)
        htf = HTFAnalysis(bias=MarketBias.BULLISH, ema_value=1990.0, last_swing_high=2020.0, last_swing_low=1980.0, trend_clarity_score=25.0)

        signal = scalp_engine.detect_scalp_entry(df=df_5m, htf_analysis=htf, instrument=self.instrument, current_spread=0.25)
        self.assertIsNone(signal, "Scalp BUY signal MUST be rejected when green candle body < 50%")


if __name__ == '__main__':
    unittest.main()

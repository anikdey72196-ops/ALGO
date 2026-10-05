"""
test_dealing_range_equilibrium_setup.py — Unit Tests for ICT / SMC Dealing Range, 0.5 Fib Equilibrium & Invalidation Setup.

Verifies:
1. Bullish Dealing Range:
   - Bias remains BULLISH when pulling back into Discount (<0.5 Fib).
   - Buy trades execute only at the 1st FVG / liquidity area below 0.5 upon bullish candle confirmation.
   - Bias changes to BEARISH only when all discount levels (FVG, liquidity, SD) fail.
2. Bearish Dealing Range (Symmetric):
   - Bias remains BEARISH when pulling back into Premium (>0.5 Fib).
   - Sell trades execute only at the 1st FVG / liquidity area above 0.5 upon bearish candle confirmation.
   - Bias changes to BULLISH only when all premium levels (FVG, liquidity, SD) fail.
"""

import unittest
import pandas as pd
import numpy as np

from config import MarketBias, Direction, InstrumentConfig
from strategy import HTFAnalyzer, TrendReversalStrategy, check_candle_body_confirmation


class TestDealingRangeEquilibriumSetup(unittest.TestCase):

    def setUp(self):
        self.analyzer = HTFAnalyzer(ema_period=50)
        self.instrument = InstrumentConfig(
            symbol="EURUSD",
            point_value=1.0,
            pip_size=0.0001,
            digits=5,
            min_lot=0.01,
            max_lot=100.0,
        )

    def test_bullish_dealing_range_discount_buy_and_bias_persistence(self):
        """
        Test that in a Bullish dealing range:
        - Price pulling back into Discount (<0.5 Fib) maintains BULLISH bias.
        - Buy setup triggers at the 1st FVG / discount zone upon bullish candle confirmation.
        """
        # Construct an HTF DataFrame with an expansion leg from 1.0000 to 1.0500 (500 pips)
        # 0.5 Fib level = 1.0250.
        # Then price pulls back to 1.0200 (in Discount < 0.5 Fib).
        timestamps = pd.date_range("2026-09-01 00:00:00", periods=45, freq="1h", tz="UTC")
        data = []

        # Bars 0-30: Rally from 1.0000 to 1.0500 creating a bullish FVG around 1.0180-1.0220
        p = 1.0000
        for i in range(30):
            p += 0.0016
            c_open = p - 0.0005
            c_close = p + 0.0005
            c_high = p + 0.0020
            c_low = p - 0.0020
            # Embed a clear Bullish FVG at bar 15: low of bar 15 > high of bar 13
            if i == 13:
                c_high = 1.0190
            elif i == 15:
                c_low = 1.0210  # FVG between 1.0190 and 1.0210 (< 0.5 Fib 1.0250)
            data.append({
                'open': c_open, 'high': c_high, 'low': c_low, 'close': c_close, 'volume': 200,
            })

        # Bar 30 reaches peak ~1.0500
        data[-1]['high'] = 1.0500
        data[-1]['close'] = 1.0490

        # Bars 31-40: Pullback from 1.0500 down to 1.0200 (inside 1st FVG below 0.5 Fib)
        p = 1.0490
        for i in range(10):
            p -= 0.0028
            data.append({
                'open': p + 0.0005,
                'high': p + 0.0020,
                'low': p - 0.0020,
                'close': p,
                'volume': 150,
            })

        df_htf = pd.DataFrame(data, index=timestamps[:len(data)])

        # Run HTF analysis
        htf_an = self.analyzer.analyze(df_htf)

        # 1. Bias verification: must be BULLISH
        self.assertEqual(htf_an.bias, MarketBias.BULLISH)
        self.assertIsNotNone(htf_an.fib_50)
        # Price is below 0.5 Fib (in Discount)
        self.assertTrue(htf_an.is_discount)
        self.assertFalse(htf_an.all_zones_failed)
        self.assertIsNotNone(htf_an.first_zone_entry)

        # 2. LTF Candle Confirmation (Bullish candle confirming bounce from discount zone)
        ltf_timestamps = pd.date_range("2026-09-02 18:00:00", periods=10, freq="15min", tz="UTC")
        ltf_data = pd.DataFrame([
            {
                'time': t,
                'open': 1.0200,
                'high': 1.0225,
                'low': 1.0195,
                'close': 1.0220,  # Clear bullish body confirmation
                'volume': 300,
            }
            for t in ltf_timestamps
        ], index=ltf_timestamps)

        is_conf, body_ratio, desc = check_candle_body_confirmation(
            ltf_data.iloc[-1], Direction.BUY, min_body_ratio=0.50
        )
        self.assertTrue(is_conf)

        # 3. Strategy Evaluation: Should trigger a BUY signal
        strat = TrendReversalStrategy(self.analyzer)
        signals = strat.evaluate(
            symbol="EURUSD",
            htf_data=df_htf,
            ltf_data=ltf_data,
            instrument=self.instrument,
            current_spread=0.0001,
            htf_analysis=htf_an,
        )

        self.assertGreaterEqual(len(signals), 1)
        self.assertEqual(signals[0].direction, Direction.BUY)
        self.assertEqual(signals[0].htf_bias, MarketBias.BULLISH)
        self.assertLess(signals[0].stop_loss, signals[0].entry_price)
        self.assertGreater(signals[0].take_profit, signals[0].entry_price)

    def test_bullish_bias_invalidated_when_all_discount_zones_fail(self):
        """
        Test that when price completely breaks down below the Anchor Low / all discount levels,
        the bias flips to BEARISH.
        """
        # Construct an HTF DataFrame where price rallied to 1.0500, but then plummeted below 0.9950 (failing all zones)
        timestamps = pd.date_range("2026-09-01 00:00:00", periods=50, freq="1h", tz="UTC")
        data = []
        p = 1.0000
        for i in range(25):
            p += 0.002
            data.append({'open': p - 0.0005, 'high': p + 0.0005, 'low': p - 0.0005, 'close': p, 'volume': 200})

        # Massive breakdown breaking well below starting low 1.0000 down to 0.9850
        for i in range(25):
            p -= 0.0026
            data.append({'open': p + 0.0005, 'high': p + 0.0005, 'low': p - 0.0005, 'close': p, 'volume': 400})

        df_htf = pd.DataFrame(data, index=timestamps)

        htf_an = self.analyzer.analyze(df_htf)

        # All discount levels failed -> bias must flip to BEARISH
        self.assertEqual(htf_an.bias, MarketBias.BEARISH)

    def test_bearish_dealing_range_premium_sell_and_bias_persistence(self):
        """
        Test that in a Bearish dealing range:
        - Price pulling back into Premium (>0.5 Fib) maintains BEARISH bias.
        - Sell setup triggers at the 1st FVG / premium zone upon bearish candle confirmation.
        """
        # Drop from 1.0500 to 1.0000. 0.5 Fib = 1.0250.
        # Pullback rises to 1.0300 (Premium > 0.5 Fib).
        timestamps = pd.date_range("2026-09-01 00:00:00", periods=45, freq="1h", tz="UTC")
        data = []
        p = 1.0500
        for i in range(30):
            p -= 0.0016
            c_open = p + 0.0005
            c_close = p - 0.0005
            c_high = p + 0.0020
            c_low = p - 0.0020
            # Embed a Bearish FVG at bar 15: high of bar 15 < low of bar 13
            if i == 13:
                c_low = 1.0310
            elif i == 15:
                c_high = 1.0290  # FVG between 1.0290 and 1.0310 (> 0.5 Fib 1.0250)
            data.append({
                'open': c_open, 'high': c_high, 'low': c_low, 'close': c_close, 'volume': 200,
            })

        data[-1]['low'] = 1.0000
        data[-1]['close'] = 1.0010

        # Pullback from 1.0000 up to 1.0300 (into Premium 1st FVG)
        p = 1.0010
        for i in range(10):
            p += 0.0028
            data.append({
                'open': p - 0.0005,
                'high': p + 0.0020,
                'low': p - 0.0020,
                'close': p,
                'volume': 150,
            })

        df_htf = pd.DataFrame(data, index=timestamps[:len(data)])

        htf_an = self.analyzer.analyze(df_htf)

        # 1. Bias verification: must be BEARISH
        self.assertEqual(htf_an.bias, MarketBias.BEARISH)
        self.assertIsNotNone(htf_an.fib_50)
        # Price is above 0.5 Fib (in Premium)
        self.assertTrue(htf_an.is_premium)
        self.assertFalse(htf_an.all_zones_failed)
        self.assertIsNotNone(htf_an.first_zone_entry)

        # 2. LTF Candle Confirmation (Bearish candle confirming rejection from premium zone)
        ltf_timestamps = pd.date_range("2026-09-02 18:00:00", periods=10, freq="15min", tz="UTC")
        ltf_data = pd.DataFrame([
            {
                'time': t,
                'open': 1.0300,
                'high': 1.0305,
                'low': 1.0275,
                'close': 1.0280,  # Clear bearish body confirmation
                'volume': 300,
            }
            for t in ltf_timestamps
        ], index=ltf_timestamps)

        is_conf, body_ratio, desc = check_candle_body_confirmation(
            ltf_data.iloc[-1], Direction.SELL, min_body_ratio=0.50
        )
        self.assertTrue(is_conf)

        # 3. Strategy Evaluation: Should trigger a SELL signal
        strat = TrendReversalStrategy(self.analyzer)
        signals = strat.evaluate(
            symbol="EURUSD",
            htf_data=df_htf,
            ltf_data=ltf_data,
            instrument=self.instrument,
            current_spread=0.0001,
            htf_analysis=htf_an,
        )

        self.assertGreaterEqual(len(signals), 1)
        self.assertEqual(signals[0].direction, Direction.SELL)
        self.assertEqual(signals[0].htf_bias, MarketBias.BEARISH)
        self.assertGreater(signals[0].stop_loss, signals[0].entry_price)
        self.assertLess(signals[0].take_profit, signals[0].entry_price)


if __name__ == "__main__":
    unittest.main()

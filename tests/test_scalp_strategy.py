"""
test_scalp_strategy.py — Comprehensive Unit Tests for the 5-Minute Order Block Scalp Strategy (SMC).

Validates:
1. HTF 1H Directional Bias detection.
2. 5M Break of Structure (BOS) identification.
3. Order Block (OB) candle extraction.
4. Retest detection and entry calculation.
5. Invalidation Stop Loss placement beyond the OB wick.
6. 1.5R partial profit target and runner target calculation.
7. London & New York AM session hours gate (07:00 - 16:00 UTC).
8. ConflictResolver adaptive R:R for scalping (accepts 1.5R).
9. AIAnalyst heuristic confirmation of 5M OB scalps (>= 75%).
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
    RiskConfig,
    StrategyType,
)
from strategy import (
    StrategyEngine,
    SMCScalp5MEngine,
    HTFAnalyzer,
    HTFAnalysis,
    TradeSignal,
    LTFConfirmation,
)
from conflict_resolver import ConflictResolver
from ai_analyst import AIAnalyst


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


class TestScalpStrategy(unittest.TestCase):

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
        self.scalp_engine = SMCScalp5MEngine(
            swing_lookback=2,
            session_start_utc=7,
            session_end_utc=16,
            target_rr=1.5,
        )

    def test_session_filter(self):
        """Test that trading hours outside 07:00-16:00 UTC are blocked."""
        # 09:30 UTC (London AM session) -> Should pass
        t_london = datetime(2026, 9, 12, 9, 30, tzinfo=timezone.utc)
        self.assertTrue(self.scalp_engine.check_trading_session(t_london))

        # 14:00 UTC (New York AM session) -> Should pass
        t_ny = datetime(2026, 9, 12, 14, 0, tzinfo=timezone.utc)
        self.assertTrue(self.scalp_engine.check_trading_session(t_ny))

        # 02:00 UTC (Asian session) -> Should fail
        t_asia = datetime(2026, 9, 12, 2, 0, tzinfo=timezone.utc)
        self.assertFalse(self.scalp_engine.check_trading_session(t_asia))

        # 20:00 UTC (New York close) -> Should fail
        t_night = datetime(2026, 9, 12, 20, 0, tzinfo=timezone.utc)
        self.assertFalse(self.scalp_engine.check_trading_session(t_night))

    def test_all_day_trading_session_unrestricted(self):
        """Test that scalping signals trigger even during Asian/overnight hours when enforce_session=False."""
        # 02:00 UTC Asian session
        base_time = datetime(2026, 9, 12, 2, 0, tzinfo=timezone.utc)
        prices = [
            (2000.0, 2002.0, 1999.0, 2001.0),
            (2001.0, 2003.0, 2000.0, 2002.0),
            (2002.0, 2004.0, 2001.0, 2003.0),
            (2003.0, 2004.5, 2002.0, 2004.0),
            (2004.0, 2005.0, 2003.0, 2004.5),  # idx 4: Swing High = 2005.0
            (2004.5, 2004.8, 2002.0, 2002.5),
            (2002.5, 2003.0, 2001.0, 2001.5),
            (2001.5, 2002.0, 2000.0, 2000.5),  # idx 7: OB Candle
            (2000.5, 2004.0, 2000.2, 2003.8),
            (2003.8, 2006.5, 2003.5, 2006.0),  # idx 9: BOS above 2005.0
            (2006.0, 2007.0, 2005.0, 2006.5),
            (2006.5, 2006.5, 2003.0, 2003.5),
            (2003.5, 2004.0, 2001.5, 2001.8),  # idx 12: Retest into OB
        ]
        candles = [make_candle(base_time + timedelta(minutes=5 * i), o, h, l, c) for i, (o, h, l, c) in enumerate(prices)]
        df_5m = pd.DataFrame(candles)
        htf_analysis = HTFAnalysis(
            bias=MarketBias.BULLISH,
            ema_value=1990.0,
            last_swing_high=2020.0,
            last_swing_low=1980.0,
            trend_clarity_score=25.0,
        )
        # Default (enforce_session=False) -> should generate signal even at 02:00 UTC
        signal = self.scalp_engine.detect_scalp_entry(
            df=df_5m,
            htf_analysis=htf_analysis,
            instrument=self.instrument,
            current_spread=0.25,
        )
        self.assertIsNotNone(signal, "Scalp signal should trigger 24/7 when session enforcement is disabled")
        self.assertEqual(signal['direction'], Direction.BUY)

    def test_bullish_scalp_bos_and_ob_retest(self):
        """Test bullish 5M BOS, OB identification, and retest entry."""
        base_time = datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc)
        candles = []

        # 1. Base accumulation: Swing High formed at index 4 (high=2005.0)
        prices = [
            (2000.0, 2002.0, 1999.0, 2001.0),
            (2001.0, 2003.0, 2000.0, 2002.0),
            (2002.0, 2004.0, 2001.0, 2003.0),
            (2003.0, 2004.5, 2002.0, 2004.0),
            (2004.0, 2005.0, 2003.0, 2004.5),  # idx 4: Swing High = 2005.0
            (2004.5, 2004.8, 2002.0, 2002.5),  # idx 5: pullback
            (2002.5, 2003.0, 2001.0, 2001.5),  # idx 6: pullback
            # idx 7: OB Candle (Down candle before impulsive rally):
            (2001.5, 2002.0, 2000.0, 2000.5),  # idx 7: Low=2000.0, High=2002.0
            # idx 8, 9, 10: Impulsive rally creating BOS above 2005.0
            (2000.5, 2004.0, 2000.2, 2003.8),  # idx 8
            (2003.8, 2006.5, 2003.5, 2006.0),  # idx 9: BOS! (Close=2006.0 > 2005.0)
            (2006.0, 2007.0, 2005.0, 2006.5),  # idx 10: extension
            # idx 11: Retrace back towards OB
            (2006.5, 2006.5, 2003.0, 2003.5),  # idx 11
            # idx 12: Retest into the OB zone [2000.0, 2002.0]
            (2003.5, 2004.0, 2001.5, 2001.8),  # idx 12: Low=2001.5 touches OB! Close=2001.8
        ]

        for i, (o, h, l, c) in enumerate(prices):
            t = base_time + timedelta(minutes=5 * i)
            candles.append(make_candle(t, o, h, l, c))

        df_5m = pd.DataFrame(candles)

        htf_analysis = HTFAnalysis(
            bias=MarketBias.BULLISH,
            ema_value=1990.0,
            last_swing_high=2020.0,
            last_swing_low=1980.0,
            trend_clarity_score=25.0,
        )

        signal = self.scalp_engine.detect_scalp_entry(
            df=df_5m,
            htf_analysis=htf_analysis,
            instrument=self.instrument,
            current_spread=0.25,
            enforce_session=True,
        )

        self.assertIsNotNone(signal, "Expected a valid bullish 5M OB scalp signal")
        self.assertEqual(signal['direction'], Direction.BUY)
        self.assertEqual(signal['conf'], LTFConfirmation.OB_SCALP_5M)
        self.assertAlmostEqual(signal['entry'], 2001.8, places=2)

        # Stop loss should be just below the OB candle low (2000.0)
        self.assertLess(signal['sl'], 2000.0)
        sl_dist = signal['entry'] - signal['sl']
        expected_tp = signal['entry'] + (sl_dist * 1.5)
        self.assertAlmostEqual(signal['tp'], expected_tp, places=2)

    def test_bearish_scalp_bos_and_ob_retest(self):
        """Test bearish 5M BOS, OB identification, and retest entry."""
        base_time = datetime(2026, 9, 12, 13, 0, tzinfo=timezone.utc)
        candles = []

        # 1. Base distribution: Swing Low formed at index 4 (low=2010.0)
        prices = [
            (2015.0, 2016.0, 2014.0, 2014.5),
            (2014.5, 2015.0, 2013.0, 2013.5),
            (2013.5, 2014.0, 2012.0, 2012.5),
            (2012.5, 2013.0, 2011.0, 2011.5),
            (2011.5, 2012.0, 2010.0, 2010.5),  # idx 4: Swing Low = 2010.0
            (2010.5, 2013.0, 2010.2, 2012.8),  # idx 5: bounce
            (2012.8, 2014.0, 2012.5, 2013.5),  # idx 6: bounce
            # idx 7: OB Candle (Up candle before displacement dump):
            (2013.5, 2015.0, 2013.0, 2014.8),  # idx 7: High=2015.0, Low=2013.0
            # idx 8, 9, 10: Impulsive drop creating BOS below 2010.0
            (2014.8, 2014.9, 2011.0, 2011.2),  # idx 8
            (2011.2, 2011.5, 2008.0, 2008.5),  # idx 9: BOS! (Close=2008.5 < 2010.0)
            (2008.5, 2009.0, 2007.5, 2008.0),  # idx 10
            # idx 11: Retrace back towards OB
            (2008.0, 2012.0, 2007.8, 2011.5),  # idx 11
            # idx 12: Retest into the OB zone [2013.0, 2015.0]
            (2011.5, 2013.5, 2011.0, 2013.2),  # idx 12: High=2013.5 touches OB! Close=2013.2
        ]

        for i, (o, h, l, c) in enumerate(prices):
            t = base_time + timedelta(minutes=5 * i)
            candles.append(make_candle(t, o, h, l, c))

        df_5m = pd.DataFrame(candles)

        htf_analysis = HTFAnalysis(
            bias=MarketBias.BEARISH,
            ema_value=2030.0,
            last_swing_high=2040.0,
            last_swing_low=2000.0,
            trend_clarity_score=25.0,
        )

        signal = self.scalp_engine.detect_scalp_entry(
            df=df_5m,
            htf_analysis=htf_analysis,
            instrument=self.instrument,
            current_spread=0.25,
            enforce_session=True,
        )

        self.assertIsNotNone(signal, "Expected a valid bearish 5M OB scalp signal")
        self.assertEqual(signal['direction'], Direction.SELL)
        self.assertEqual(signal['conf'], LTFConfirmation.OB_SCALP_5M)
        self.assertAlmostEqual(signal['entry'], 2013.2, places=2)

        # Stop loss should be just above the OB candle high (2015.0)
        self.assertGreater(signal['sl'], 2015.0)
        sl_dist = signal['sl'] - signal['entry']
        expected_tp = signal['entry'] - (sl_dist * 1.5)
        self.assertAlmostEqual(signal['tp'], expected_tp, places=2)

    def test_conflict_resolver_adaptive_rr(self):
        """Test ConflictResolver allows 1.5R scalps through Gate 2."""
        resolver = ConflictResolver(RiskConfig(min_rr_ratio=2.5))

        # A 1.5R 5M scalp signal
        scalp_sig = TradeSignal(
            symbol="XAUUSD",
            direction=Direction.BUY,
            entry_price=2000.0,
            stop_loss=1998.0,
            take_profit=2003.0,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.OB_SCALP_5M,
            rr_ratio=1.5,
            quality_score=85.0,
            timestamp=datetime.now(timezone.utc),
            sl_distance=2.0,
            tp_distance=3.0,
        )

        res = resolver.resolve([scalp_sig], current_spread=0.25)
        self.assertIsNotNone(res.accepted_signal, "1.5R scalp signal should pass adaptive Gate 2")
        self.assertEqual(res.accepted_signal.symbol, "XAUUSD")

        # An sub-threshold 1.2R scalp should be rejected
        low_rr_sig = TradeSignal(
            symbol="XAUUSD",
            direction=Direction.BUY,
            entry_price=2000.0,
            stop_loss=1998.0,
            take_profit=2002.4,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.OB_SCALP_5M,
            rr_ratio=1.2,
            quality_score=85.0,
            timestamp=datetime.now(timezone.utc),
            sl_distance=2.0,
            tp_distance=2.4,
        )
        res_low = resolver.resolve([low_rr_sig], current_spread=0.25)
        self.assertIsNone(res_low.accepted_signal, "Sub-1.4R scalp signal should be rejected")

    def test_ai_analyst_confirms_scalp(self):
        """Test AIAnalyst heuristic mode awards full points to high-quality 5M OB scalps."""
        analyst = AIAnalyst(TradingConfig(ai_confirmation_enabled=True, ai_confidence_threshold=75.0))
        
        scalp_sig = TradeSignal(
            symbol="XAUUSD",
            direction=Direction.BUY,
            entry_price=2000.0,
            stop_loss=1998.0,
            take_profit=2003.0,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.OB_SCALP_5M,
            rr_ratio=1.5,
            quality_score=85.0,
            timestamp=datetime.now(timezone.utc),
            sl_distance=2.0,
            tp_distance=3.0,
        )

        htf_analysis = HTFAnalysis(
            bias=MarketBias.BULLISH,
            ema_value=1990.0,
            last_swing_high=2020.0,
            last_swing_low=1980.0,
            trend_clarity_score=25.0,
        )

        decision = analyst.evaluate_setup(scalp_sig, htf_analysis, current_spread=0.25)
        self.assertTrue(decision.confirmed, f"Expected AI to confirm setup. Score: {decision.confidence}%, Reason: {decision.reason}")
        self.assertGreaterEqual(decision.confidence, 75.0)


if __name__ == "__main__":
    unittest.main()

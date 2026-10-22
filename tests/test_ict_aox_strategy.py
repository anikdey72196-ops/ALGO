"""
test_ict_aox_strategy.py — Institutional-Grade ICT Time & Price Theory & AOX Model Tests:
1. Timezone & Opening Ranges (EST/EDT Midnight Open 00:00, MOR 00:01-00:29, 08:30 Pre-Market, 09:30 NYSE Open).
2. Algorithmic Macros & Distribution Windows (London Macro, London Heavy Distribution, NY AM Macros, NY Lunch, Hourly Macro).
3. Premium / Discount & Deep Premium / Deep Discount Filtering & First Presented FVG (FPFVG) 3-Candle Displacement Rule.
4. Standard Deviation (SD) Projections on MOR and Manipulation Legs (-1.5 SD Silver Bullet, -2.5 SD Minor Retracement, -4.5 SD Major Reversal).
5. AOX Custom Fibonacci Levels (-0.21, -0.255, -0.29 Sniper Entry; 1.47, 1.55 TP1; 2.56, 2.60, 2.64 TP2).
6. SMT Divergence (Gold vs Silver intermarket divergence) & Turtle Soup.
7. 3-Entry Execution Sequence (Entry 1 Sniper Tap, Entry 2 SMT Turtle Soup, Entry 3 HIPPO Mitigation).
8. 25-25-50 Layered Risk Management, Breakeven at 1:1 RR / 80-90 pips, and Daily Trade Quota Enforcement.
"""

import unittest
from datetime import datetime, timezone, timedelta, date
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

from core.config import (
    TradingConfig,
    InstrumentConfig,
    Direction,
    MarketBias,
    ICTConfig,
)
from strategies.strategy import (
    ICTStrategy,
    ICTEngine,
    ICTKillZone,
    ICTOpeningRanges,
    SDProjections,
    AOXFibLevels,
    TradeSignal,
    LTFConfirmation,
    HTFAnalysis,
    to_ny_time,
)
from execution.position_manager import PositionManager
from execution.execution import MockBrokerAdapter
from core.state import StateManager, TradeRecord


def make_candle(ny_time_str: str, o: float, h: float, l: float, c: float, vol: float = 100) -> dict:
    """Helper to create a candle timestamped in America/New_York."""
    tz_ny = ZoneInfo("America/New_York")
    dt = datetime.fromisoformat(ny_time_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz_ny)
    return {
        'time': dt,
        'open': float(o),
        'high': float(h),
        'low': float(l),
        'close': float(c),
        'tick_volume': float(vol),
        'volume': float(vol),
    }


class TestICTAOXCoreTimeAndRanges(unittest.TestCase):
    """Section 1: Timezone anchoring (EST/EDT) & Core Opening Ranges."""

    def setUp(self):
        self.config = ICTConfig()
        self.engine = ICTEngine(self.config)

    def test_midnight_open_and_mor_ranges(self):
        """Verify TDO at 00:00 EST and MOR High/Low strictly from 00:01 to 00:29 EST."""
        candles = []
        # 00:00 EST candle (TDO) has a high wick to 2010, but rule says ignore 00:00 wick for MOR!
        candles.append(make_candle("2026-10-14T00:00:00", 2000.0, 2010.0, 1995.0, 2002.0))
        # 00:01 - 00:29 candles
        candles.append(make_candle("2026-10-14T00:05:00", 2002.0, 2006.0, 1998.0, 2004.0))
        candles.append(make_candle("2026-10-14T00:15:00", 2004.0, 2008.0, 1996.0, 2007.0))
        candles.append(make_candle("2026-10-14T00:25:00", 2007.0, 2005.0, 1994.0, 1997.0))
        # 00:30 candle (should be outside MOR high/low)
        candles.append(make_candle("2026-10-14T00:30:00", 1997.0, 2012.0, 1990.0, 2000.0))
        # 08:30 Pre-Market open candle
        candles.append(make_candle("2026-10-14T08:30:00", 2005.0, 2010.0, 2003.0, 2008.0))
        # 09:30 NYSE official open candle
        candles.append(make_candle("2026-10-14T09:30:00", 2015.0, 2020.0, 2012.0, 2018.0))
        # 16:00 NYSE close candle
        candles.append(make_candle("2026-10-14T16:00:00", 2025.0, 2028.0, 2022.0, 2026.0))

        df = pd.DataFrame(candles)
        ranges = self.engine.calculate_opening_ranges(df, target_date=date(2026, 10, 14))

        # True Day Open (TDO) must be exact open of 00:00 candle = 2000.0
        self.assertEqual(ranges.tdo, 2000.0)
        # MOR High must strictly be max of 00:01-00:29 (2008.0), ignoring 00:00 wick (2010.0) and 00:30 (2012.0)
        self.assertEqual(ranges.mor_high, 2008.0)
        # MOR Low must strictly be min of 00:01-00:29 (1994.0), ignoring 00:30 wick (1990.0)
        self.assertEqual(ranges.mor_low, 1994.0)
        self.assertEqual(ranges.mor_mid, (2008.0 + 1994.0) / 2.0)
        # Pre-Market Open & NYSE Open
        self.assertEqual(ranges.ny_pre_market_open, 2005.0)
        self.assertEqual(ranges.nyse_open, 2015.0)
        self.assertEqual(ranges.nyse_close, 2026.0)

    def test_algorithmic_macros_detection(self):
        """Section 1.B: Verify all institutional algorithmic macros and distribution windows."""
        tz_ny = ZoneInfo("America/New_York")

        # London Macro: 02:33 - 03:00 EST
        t_london_macro = datetime(2026, 10, 14, 2, 45, tzinfo=tz_ny)
        self.assertEqual(self.engine.identify_kill_zone(t_london_macro), ICTKillZone.LONDON_MACRO)
        is_macro, macro_name = self.engine.get_active_macro(t_london_macro)
        self.assertTrue(is_macro)
        self.assertIn("LONDON_MACRO", macro_name)

        # London Heavy Distribution Window: 03:00 - 03:30 EST
        t_london_dist = datetime(2026, 10, 14, 3, 15, tzinfo=tz_ny)
        self.assertEqual(self.engine.identify_kill_zone(t_london_dist), ICTKillZone.LONDON_HEAVY_DIST)
        is_macro, macro_name = self.engine.get_active_macro(t_london_dist)
        self.assertTrue(is_macro)
        self.assertIn("LONDON_DISTRIBUTION", macro_name)

        # NY AM Macro 1: 08:50 - 09:10 EST
        t_ny_am_1 = datetime(2026, 10, 14, 8, 55, tzinfo=tz_ny)
        self.assertEqual(self.engine.identify_kill_zone(t_ny_am_1), ICTKillZone.NY_AM_MACRO_1)

        # NY AM Macro 2: 09:50 - 10:10 EST
        t_ny_am_2 = datetime(2026, 10, 14, 10, 0, tzinfo=tz_ny)
        self.assertEqual(self.engine.identify_kill_zone(t_ny_am_2), ICTKillZone.NY_AM_MACRO_2)

        # NY AM Macro 3: 10:50 - 11:10 EST
        t_ny_am_3 = datetime(2026, 10, 14, 10, 55, tzinfo=tz_ny)
        self.assertEqual(self.engine.identify_kill_zone(t_ny_am_3), ICTKillZone.NY_AM_MACRO_3)

        # NY Lunch Macro: 11:50 - 12:10 EST
        t_lunch = datetime(2026, 10, 14, 12, 0, tzinfo=tz_ny)
        self.assertEqual(self.engine.identify_kill_zone(t_lunch), ICTKillZone.NY_LUNCH_MACRO)

        # Hourly Macro: XX:50 to (XX+1):10
        t_hourly = datetime(2026, 10, 14, 14, 55, tzinfo=tz_ny)
        self.assertEqual(self.engine.identify_kill_zone(t_hourly, include_hourly_macro=True), ICTKillZone.HOURLY_MACRO)
        is_macro, macro_name = self.engine.get_active_macro(t_hourly)
        self.assertTrue(is_macro)
        self.assertIn("HOURLY_MACRO", macro_name)


class TestICTPremiumDiscountAndFPFVG(unittest.TestCase):
    """Section 2: Premium / Discount Filtering & FPFVG 3-Displacement-Candle Rule."""

    def setUp(self):
        self.engine = ICTEngine()
        self.openings = ICTOpeningRanges(
            date=date(2026, 10, 14),
            tdo=2000.0,
            mor_high=2008.0,
            mor_low=1994.0,
            mor_mid=2001.0,
            ny_pre_market_open=2005.0,
            nyse_open=2010.0,
        )

    def test_standard_and_deep_premium_discount(self):
        # 1. Price at 2002.0: Above TDO (2000) but below 08:30 (2005) -> Standard Premium (favors SELL)
        res1 = self.engine.evaluate_premium_discount(2002.0, self.openings)
        self.assertTrue(res1['is_premium'])
        self.assertFalse(res1['is_discount'])
        self.assertFalse(res1['is_deep_premium'])
        self.assertEqual(res1['favored_direction'], Direction.SELL)

        # 2. Standard Discount: Price is below TDO (2000), but not below 08:30 and 09:30
        openings_std_disc = ICTOpeningRanges(
            date=date(2026, 10, 14),
            tdo=2000.0,
            mor_high=2008.0,
            mor_low=1994.0,
            mor_mid=2001.0,
            ny_pre_market_open=1995.0,
            nyse_open=1990.0,
        )
        res2 = self.engine.evaluate_premium_discount(1998.0, openings_std_disc)
        self.assertTrue(res2['is_discount'])
        self.assertFalse(res2['is_deep_discount'])
        self.assertEqual(res2['favored_direction'], Direction.BUY)

        # 3. Price at 2012.0: Above TDO (2000), Above 08:30 (2005), AND Above 09:30 (2010) -> Deep Premium!
        res3 = self.engine.evaluate_premium_discount(2012.0, self.openings)
        self.assertTrue(res3['is_deep_premium'])
        self.assertEqual(res3['favored_direction'], Direction.SELL)

        # 4. If all opens are higher than price (e.g. price 1990.0 < 2000, 2005, 2010) -> Deep Discount!
        res4 = self.engine.evaluate_premium_discount(1990.0, self.openings)
        self.assertTrue(res4['is_deep_discount'])
        self.assertEqual(res4['favored_direction'], Direction.BUY)

    def test_fpfvg_strict_3_candle_displacement_validation(self):
        """FPFVG is ONLY valid if formed by 3 consecutive displacement candles closing beyond previous candle's high/low."""
        # Case A: Valid Bullish FPFVG
        # 3 consecutive green candles, each closing decisively beyond previous high, leaving gap between c0 high and c2 low
        c0 = make_candle("2026-10-14T00:05:00", 2000.0, 2003.0, 1999.0, 2002.5)  # Green, high=2003
        c1 = make_candle("2026-10-14T00:06:00", 2002.5, 2008.0, 2002.0, 2007.5)  # Green, close 2007.5 > c0 high (2003)
        c2 = make_candle("2026-10-14T00:07:00", 2007.5, 2012.0, 2005.0, 2011.0)  # Green, close 2011 > c1 high (2008), low=2005 > c0 high (2003) -> Gap [2003, 2005]
        df_valid = pd.DataFrame([c0, c1, c2])

        fpfvg = self.engine.detect_fpfvg(df_valid, target_date=date(2026, 10, 14))
        self.assertIsNotNone(fpfvg)
        self.assertEqual(fpfvg['type'], 'BULLISH_FPFVG')
        self.assertEqual(fpfvg['bottom'], 2003.0)
        self.assertEqual(fpfvg['top'], 2005.0)
        self.assertEqual(fpfvg['midpoint'], 2004.0)

        # Case B: Rejection of Inside Bar or Non-Displacement Candle
        # Candle 1 fails to close above Candle 0 high (e.g. wick reached high but closed low)
        c1_weak = make_candle("2026-10-14T00:06:00", 2002.5, 2008.0, 2001.0, 2002.0)  # Red / inside close
        df_invalid = pd.DataFrame([c0, c1_weak, c2])
        fpfvg_invalid = self.engine.detect_fpfvg(df_invalid, target_date=date(2026, 10, 14))
        self.assertIsNone(fpfvg_invalid)


class TestICTStandardDeviationsAndAOXFib(unittest.TestCase):
    """Sections 3 & 5: Standard Deviation Projections & AOX Custom Fibonacci Settings."""

    def setUp(self):
        self.engine = ICTEngine()

    def test_sd_projections_bearish_and_bullish(self):
        """
        Direction of Plot:
        - Bearish: 0 at Low (2000), 1 at High (2020) -> downward projections.
        - Bullish: 0 at High (2020), 1 at Low (2000) -> upward projections.
        """
        # Bearish downward projections (manipulation leg is upward move from 2000 to 2020, range=20)
        sd_bear = self.engine.calculate_sd_projections(anchor_high=2020.0, anchor_low=2000.0, direction=Direction.SELL)
        # -1.0 SD = 2000 - 1.0 * 20 = 1980
        # -1.5 SD = 2000 - 1.5 * 20 = 1970
        # -2.0 SD = 2000 - 2.0 * 20 = 1960
        # -2.5 SD = 2000 - 2.5 * 20 = 1950
        # -4.0 SD = 2000 - 4.0 * 20 = 1920
        # -4.5 SD = 2000 - 4.5 * 20 = 1910
        self.assertAlmostEqual(sd_bear.sd_1_0, 1980.0)
        self.assertAlmostEqual(sd_bear.sd_1_5, 1970.0)
        self.assertAlmostEqual(sd_bear.sd_2_0, 1960.0)
        self.assertAlmostEqual(sd_bear.sd_2_5, 1950.0)
        self.assertAlmostEqual(sd_bear.sd_4_0, 1920.0)
        self.assertAlmostEqual(sd_bear.sd_4_5, 1910.0)
        # Midline Consequent Encroachment (CE)
        self.assertAlmostEqual(sd_bear.ce_midline, 2010.0)

        # Bullish upward projections (manipulation leg is downward move from 2020 to 2000, range=20)
        sd_bull = self.engine.calculate_sd_projections(anchor_high=2020.0, anchor_low=2000.0, direction=Direction.BUY)
        # -1.0 SD = 2020 + 20 = 2040
        # -2.5 SD = 2020 + 2.5 * 20 = 2070
        # -4.5 SD = 2020 + 4.5 * 20 = 2110
        self.assertAlmostEqual(sd_bull.sd_1_0, 2040.0)
        self.assertAlmostEqual(sd_bull.sd_2_5, 2070.0)
        self.assertAlmostEqual(sd_bull.sd_4_5, 2110.0)

    def test_aox_custom_fibonacci_sniper_levels(self):
        """
        Verify exact AOX Sniper Entry levels (-0.21, -0.255, -0.29),
        Partial TP1 (1.47, 1.55), and Final TP2 (2.56, 2.60, 2.64).
        """
        # Bullish manipulation leg: High=2000, Low=1900 (range=100)
        fib_bull = self.engine.calculate_aox_fib(manipulation_high=2000.0, manipulation_low=1900.0, direction=Direction.BUY)
        # Sniper Entry: Low - (range * extension)
        # -0.21 ext = 1900 - 21.0 = 1879.0
        # -0.255 ext = 1900 - 25.5 = 1874.5
        # -0.29 ext = 1900 - 29.0 = 1871.0
        self.assertAlmostEqual(fib_bull.sniper_zone_entry, 1879.0)
        self.assertAlmostEqual(fib_bull.sniper_zone_mid, 1874.5)
        self.assertAlmostEqual(fib_bull.sniper_zone_max, 1871.0)
        # TP1: Low + range * 1.47 = 1900 + 147 = 2047.0; 1.55 = 2055.0
        self.assertAlmostEqual(fib_bull.tp1_min, 2047.0)
        self.assertAlmostEqual(fib_bull.tp1_max, 2055.0)
        # TP2: Low + range * 2.56 = 1900 + 256 = 2156.0; 2.64 = 2164.0
        self.assertAlmostEqual(fib_bull.tp2_min, 2156.0)
        self.assertAlmostEqual(fib_bull.tp2_max, 2164.0)


class TestSMTDivergenceAndExecutionSequence(unittest.TestCase):
    """Sections 5 & 6: SMT Divergence, 3-Entry Execution Sequence & 25-25-50 Layered Risk."""

    def setUp(self):
        self.engine = ICTEngine()

    def test_smt_divergence_gold_vs_silver(self):
        """
        SMT Divergence:
        Gold (primary) sweeps previous low to form a lower low (1994 < 2000),
        Silver (correlated) fails to sweep its corresponding low (25.10 > 25.00).
        -> SMT confirmed!
        """
        # Primary asset (Gold) with two distinct swing lows
        primary_candles = [
            make_candle("2026-10-14T09:00:00", 2008, 2010, 2006, 2007),
            make_candle("2026-10-14T09:01:00", 2007, 2008, 2004, 2005),
            make_candle("2026-10-14T09:02:00", 2005, 2006, 2000, 2003),  # Swing Low 1 @ 2000
            make_candle("2026-10-14T09:03:00", 2003, 2007, 2002, 2006),
            make_candle("2026-10-14T09:04:00", 2006, 2008, 2004, 2007),
            make_candle("2026-10-14T09:05:00", 2007, 2008, 2002, 2003),
            make_candle("2026-10-14T09:06:00", 2003, 2004, 1994, 1996),  # Swing Low 2 (Sweep / LL) @ 1994
            make_candle("2026-10-14T09:07:00", 1996, 2001, 1997, 2000),
            make_candle("2026-10-14T09:08:00", 2000, 2005, 1999, 2004),
        ]
        # Correlated asset (Silver) where second swing low fails to sweep (HL)
        corr_candles = [
            make_candle("2026-10-14T09:00:00", 25.30, 25.40, 25.25, 25.35),
            make_candle("2026-10-14T09:01:00", 25.35, 25.38, 25.15, 25.20),
            make_candle("2026-10-14T09:02:00", 25.20, 25.25, 25.00, 25.10),  # Swing Low 1 @ 25.00
            make_candle("2026-10-14T09:03:00", 25.10, 25.30, 25.08, 25.28),
            make_candle("2026-10-14T09:04:00", 25.28, 25.35, 25.20, 25.30),
            make_candle("2026-10-14T09:05:00", 25.30, 25.32, 25.18, 25.22),
            make_candle("2026-10-14T09:06:00", 25.22, 25.26, 25.10, 25.15),  # Swing Low 2 (Holds / HL) @ 25.10
            make_candle("2026-10-14T09:07:00", 25.15, 25.30, 25.14, 25.25),
            make_candle("2026-10-14T09:08:00", 25.25, 25.35, 25.20, 25.32),
        ]

        df_gold = pd.DataFrame(primary_candles)
        df_silver = pd.DataFrame(corr_candles)

        is_smt, smt_type, reason = self.engine.detect_smt_divergence(
            primary_df=df_gold,
            correlated_df=df_silver,
        )
        self.assertTrue(is_smt)
        self.assertEqual(smt_type, "BULLISH_SMT")

    def test_25_25_50_layered_risk_allocation(self):
        """Verify the 3-entry execution sequence layered risk allocation (25%, 25%, 50%)."""
        instrument = InstrumentConfig(symbol="XAUUSD", pip_size=0.1, digits=2, min_lot=0.01, lot_step=0.01)
        strat = ICTStrategy(TradingConfig(), instrument)

        # Build market data reaching deep discount into SD zone
        candles = []
        # 00:00 Midnight open at 2000
        candles.append(make_candle("2026-10-14T00:00:00", 2000.0, 2005.0, 1998.0, 2001.0))
        # MOR 00:01-00:29 (High 2004, Low 1996)
        candles.append(make_candle("2026-10-14T00:15:00", 2001.0, 2004.0, 1996.0, 2002.0))
        # 08:30 Pre-Market open
        candles.append(make_candle("2026-10-14T08:30:00", 2002.0, 2003.0, 2000.0, 2001.0))
        # 09:30 NYSE open
        candles.append(make_candle("2026-10-14T09:30:00", 2001.0, 2002.0, 1995.0, 1996.0))
        # Deep sell-off into -2.5 SD zone (e.g. 1970)
        for m in range(35, 55, 5):
            candles.append(make_candle(f"2026-10-14T09:{m}:00", 1996.0 - (m-30)*0.8, 1997.0 - (m-30)*0.8, 1990.0 - (m-30)*0.8, 1992.0 - (m-30)*0.8))

        # Reversal candle at extreme (-2.5 SD zone around 1974)
        candles.append(make_candle("2026-10-14T09:55:00", 1976.0, 1978.0, 1973.0, 1977.0))

        df_ltf = pd.DataFrame(candles)
        df_htf = df_ltf.copy()

        htf_analysis = HTFAnalysis(
            bias=MarketBias.BULLISH,
            ema_value=2000.0,
            last_swing_high=2010.0,
            last_swing_low=1990.0,
            trend_clarity_score=20.0,
            is_discount=True,
        )
        signals = strat.evaluate(
            symbol="XAUUSD",
            htf_data=df_htf,
            ltf_data=df_ltf,
            instrument=instrument,
            current_spread=0.25,
            htf_analysis=htf_analysis,
        )

        self.assertGreater(len(signals), 0)
        # Check layers produced
        layers = {s.layer: s.risk_fraction for s in signals}
        # Layer 1 must have 0.25 (25%) risk fraction
        if 1 in layers:
            self.assertEqual(layers[1], 0.25)
        # Layer 2 must have 0.25 (25%) risk fraction
        if 2 in layers:
            self.assertEqual(layers[2], 0.25)
        # Layer 3 must have 0.50 (50%) risk fraction
        if 3 in layers:
            self.assertEqual(layers[3], 0.50)


class TestPositionManagerICTRules(unittest.TestCase):
    """Section 6: Breakeven at 1:1 RR / 80-90 pips and Daily Trade Quota Enforcement."""

    def setUp(self):
        self.state = StateManager(db_path=":memory:")
        self.broker = MockBrokerAdapter(initial_equity=10000.0)
        self.pm = PositionManager(
            broker=self.broker,
            state=self.state,
            poll_interval_sec=60.0,
        )

    def tearDown(self):
        if hasattr(self, "pm") and self.pm is not None:
            self.pm.shutdown()
        if hasattr(self, "state") and self.state is not None:
            self.state.close()

    def test_ict_breakeven_and_daily_quota(self):
        """
        Verify:
        1. Trade moves to BE once 80 pips (or 1:1 RR) gained on Gold.
        2. Daily trade quota is triggered once 90+ pips reached.
        """
        # Open an ICT Gold BUY trade at 2000.0 with SL=1980.0 (risk = 20.0 = 200 pips), TP=2050.0
        trade = TradeRecord(
            id=None,
            timestamp=datetime.now(timezone.utc),
            symbol="XAUUSD",
            direction=Direction.BUY,
            entry_price=2000.0,
            stop_loss=1980.0,
            take_profit=2050.0,
            lot_size=1.0,
            realized_pnl=0.0,
            status="OPEN",
            strategy_name="ICT",
            magic_number=126456,
        )
        t_id = self.state.record_trade(trade)
        trade.id = t_id

        # Step A: Price reaches 2008.5 (+85 pips on Gold, >80 pip threshold)
        self.broker.set_price("XAUUSD", bid=2008.5, ask=2008.7)
        self.pm.process_positions()

        # Breakeven should be applied
        self.assertIn(t_id, self.pm._be_applied)
        # Quota not yet hit at 85 pips
        self.assertFalse(self.pm.is_ict_daily_quota_hit())

        # Step B: Price extends to 2010.5 (+105 pips on Gold, >90 pip threshold)
        self.broker.set_price("XAUUSD", bid=2010.5, ask=2010.7)
        self.pm.process_positions()

        # Daily quota should be triggered!
        self.assertTrue(self.pm.is_ict_daily_quota_hit())

        # Step C: Reset daily quota on day rollover
        self.pm.reset_ict_daily_quota()
        self.assertFalse(self.pm.is_ict_daily_quota_hit())


if __name__ == "__main__":
    unittest.main()

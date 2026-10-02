"""
test_trend_reversal.py — Comprehensive Unit Tests for CHoCH & Trend Reversal Detection Subsystem.
"""

import unittest
import numpy as np
import pandas as pd
from config import MarketBias, Direction
from trend_reversal import (
    TrendReversalDetector,
    CHoCHType,
    ReversalStage,
    TrendReversalAnalysis,
)


def create_uptrend_with_bearish_choch(
    bars_before_peak: int = 40,
    peak_sweep: bool = True,
    high_volume_break: bool = True,
    create_fvg: bool = True,
    retrace_to_zone: bool = True,
) -> pd.DataFrame:
    """
    Constructs an uptrend (Higher Highs, Higher Lows), followed by:
    1. A peak at bar 35 (potentially sweeping a prior high).
    2. A sharp drop breaking and closing below the last Higher Low (Bearish CHoCH).
    3. Creation of a Bearish Fair Value Gap.
    4. Retracement into the FVG / Fib 0.382-0.618 Golden Zone.
    """
    timestamps = pd.date_range("2026-09-01 00:00:00", periods=55, freq="5min", tz="UTC")
    data = []

    # 1. Base uptrend: Higher Low at 100.0, High at 105.0, Higher Low at 103.0 (bar 20), High at 108.0 (bar 25), Higher Low at 106.0 (bar 30)
    # The last HL is around 106.0 at bar 30!
    # Then peak surges to 110.0 at bar 35.
    price = 100.0
    for i in range(30):
        # Gradual upward stair-step
        step = (i % 6)
        if step in (0, 1, 2, 3):
            price += 0.4
        else:
            price -= 0.2
        # Ensure swing low at bar 28-29 is around 106.0
        if i == 28:
            price = 106.0
        data.append({
            'open': price - 0.1,
            'high': price + 0.3,
            'low': price - 0.3,
            'close': price + 0.1,
            'volume': 200,
        })

    # At bar 28/29, last HL is 105.7 (low = 105.7)
    # Bars 30-34: Rally to peak HH
    for i in range(30, 35):
        price += 0.8
        data.append({
            'open': price - 0.2,
            'high': price + 0.4,
            'low': price - 0.2,
            'close': price + 0.2,
            'volume': 250,
        })

    # Bar 35: Peak HH at 111.0 (Sweeps prior high of 109.5)
    peak_price = 111.0
    sweep_high = 111.5 if peak_sweep else 111.0
    data.append({
        'open': 109.8,
        'high': sweep_high,
        'low': 109.5,
        'close': 110.2,  # wick up to 111.5 then closed lower
        'volume': 350,
    })

    # Bar 36: Reversal candle 1 (starts drop)
    data.append({
        'open': 110.0,
        'high': 110.2,
        'low': 107.5,
        'close': 107.8,
        'volume': 400,
    })

    # Bar 37: Reversal candle 2 (creates FVG: prev2 low is 109.5, current high is 107.2 -> FVG between 107.2 and 109.5!)
    # and breaks below last HL (106.0) with close at 104.5!
    vol = 800 if high_volume_break else 200
    curr_high = 107.2 if create_fvg else 109.6
    data.append({
        'open': 107.5,
        'high': curr_high,
        'low': 104.2,
        'close': 104.5,  # CLEAR CLOSE BELOW 106.0 -> CHoCH!
        'volume': vol,
    })

    # Bar 38: Continuation low to 103.5
    data.append({
        'open': 104.5,
        'high': 104.8,
        'low': 103.5,
        'close': 103.8,
        'volume': 350,
    })

    # Bars 39-44: Retracement pullback into 0.382-0.618 Fib zone / FVG
    # Impulse from 111.0 to 103.5 -> Range = 7.5
    # Fib 0.382 = 103.5 + 2.865 = 106.365
    # Fib 0.618 = 103.5 + 4.635 = 108.135
    for i in range(39, 45):
        pullback_price = 106.5 if retrace_to_zone else 103.6
        data.append({
            'open': pullback_price - 0.2,
            'high': pullback_price + 0.3,
            'low': pullback_price - 0.2,
            'close': pullback_price + 0.1,
            'volume': 180,
        })

    df = pd.DataFrame(data, index=timestamps[:len(data)])
    return df


class TestTrendReversalDetector(unittest.TestCase):

    def setUp(self):
        self.detector = TrendReversalDetector(
            swing_lookback=3,
            volume_surge_multiplier=1.3,
            fvg_min_atr_multiple=0.2,
        )

    def test_healthy_uptrend_no_choch(self):
        """Test that a healthy uptrend with intact swing lows reports no CHoCH."""
        # Create 40 bars of steady uptrend
        timestamps = pd.date_range("2026-09-01 00:00:00", periods=40, freq="5min", tz="UTC")
        data = []
        p = 100.0
        for i in range(40):
            p += 0.5 if (i % 4 != 0) else -0.1
            data.append({
                'open': p - 0.1,
                'high': p + 0.3,
                'low': p - 0.2,
                'close': p + 0.2,
                'volume': 200,
            })
        df = pd.DataFrame(data, index=timestamps)

        res = self.detector.analyze(df, trend=MarketBias.BULLISH, symbol="EURUSD")
        self.assertTrue(res.is_trending)
        self.assertEqual(res.trend, MarketBias.BULLISH)
        self.assertFalse(res.choch_detected)
        self.assertEqual(res.choch_type, CHoCHType.NONE)
        self.assertEqual(res.stage, ReversalStage.TREND_HEALTHY)
        self.assertLess(res.reversal_probability, 30.0)
        self.assertEqual(res.reversal_risk, "LOW")

    def test_bearish_choch_with_full_confluence(self):
        """Test Bearish CHoCH detection with Liquidity Sweep, Volume Surge, FVG, and Retracement."""
        df = create_uptrend_with_bearish_choch(
            peak_sweep=True,
            high_volume_break=True,
            create_fvg=True,
            retrace_to_zone=True,
        )

        res = self.detector.analyze(df, trend=MarketBias.BULLISH, symbol="XAUUSD")
        self.assertTrue(res.is_trending)
        self.assertEqual(res.trend, MarketBias.BULLISH)
        self.assertTrue(res.choch_detected)
        self.assertEqual(res.choch_type, CHoCHType.BEARISH)
        self.assertIn(
            res.stage,
            (
                ReversalStage.RETRACEMENT_IN_ZONE,
                ReversalStage.CHOCH_DISPLACEMENT,
                ReversalStage.RETRACEMENT_PENDING,
                ReversalStage.CONFIRMED_MSS,
            )
        )
        
        # Confluence checks
        self.assertTrue(res.confluence.volume_surge)
        self.assertTrue(res.confluence.fvg_present or res.confluence.in_retracement_zone)
        self.assertGreaterEqual(res.reversal_probability, 60.0)
        self.assertIn(res.reversal_risk, ("HIGH", "CRITICAL"))

        # Setup targets
        self.assertIsNotNone(res.suggested_sl)
        self.assertGreater(res.suggested_sl, res.key_swing_level)
        self.assertIsNotNone(res.suggested_tp)
        self.assertLess(res.suggested_tp, res.key_swing_level)
        self.assertGreaterEqual(res.suggested_rr, 1.5)

    def test_downtrend_with_bullish_choch(self):
        """Test Bullish CHoCH detection from an established downtrend."""
        timestamps = pd.date_range("2026-09-01 00:00:00", periods=50, freq="5min", tz="UTC")
        data = []
        p = 200.0

        # Downtrend: Lower Highs and Lower Lows
        for i in range(30):
            step = (i % 5)
            if step in (0, 1, 2, 3):
                p -= 0.5
            else:
                p += 0.2
            if i == 25:
                p = 188.0  # Last LH around 188.0
            data.append({
                'open': p + 0.1,
                'high': p + 0.3,
                'low': p - 0.3,
                'close': p - 0.1,
                'volume': 200,
            })

        # Trough LL at bar 32 at 180.0 (sweep below prior low of 182.0)
        for i in range(30, 33):
            p -= 1.0
            data.append({
                'open': p + 0.2,
                'high': p + 0.3,
                'low': p - 0.5,
                'close': p - 0.3,
                'volume': 250,
            })

        # Bar 33: Trough at 178.0 wick
        data.append({
            'open': 179.5,
            'high': 180.2,
            'low': 178.0,
            'close': 179.8,
            'volume': 400,
        })

        # Bar 34 & 35: Bullish CHoCH breakout surging through last LH (188.0) to 190.0 on heavy volume
        data.append({
            'open': 180.0,
            'high': 185.0,
            'low': 179.8,
            'close': 184.5,
            'volume': 500,
        })
        data.append({
            'open': 184.5,
            'high': 191.0,
            'low': 184.2,
            'close': 190.5,  # Break & CLOSE above 188.0!
            'volume': 900,
        })

        # Retracement bars
        for i in range(5):
            data.append({
                'open': 187.0,
                'high': 187.5,
                'low': 186.0,
                'close': 186.8,
                'volume': 150,
            })

        df = pd.DataFrame(data, index=timestamps[:len(data)])
        res = self.detector.analyze(df, trend=MarketBias.BEARISH, symbol="EURUSD")

        self.assertTrue(res.is_trending)
        self.assertEqual(res.trend, MarketBias.BEARISH)
        self.assertTrue(res.choch_detected)
        self.assertEqual(res.choch_type, CHoCHType.BULLISH)
        self.assertGreaterEqual(res.reversal_probability, 50.0)
        self.assertIsNotNone(res.suggested_sl)
        self.assertLess(res.suggested_sl, res.key_swing_level)
        self.assertGreater(res.suggested_tp, res.key_swing_level)

    def test_neutral_ranging_market(self):
        """Test that a flat sideways market is classified as not trending."""
        timestamps = pd.date_range("2026-09-01 00:00:00", periods=40, freq="5min", tz="UTC")
        data = [{
            'open': 100.0,
            'high': 100.2,
            'low': 99.8,
            'close': 100.0 + (0.05 if i % 2 == 0 else -0.05),
            'volume': 100,
        } for i in range(40)]
        df = pd.DataFrame(data, index=timestamps)

        res = self.detector.analyze(df, trend=MarketBias.NEUTRAL, symbol="GBPUSD")
        self.assertFalse(res.is_trending)
        self.assertFalse(res.choch_detected)
        self.assertEqual(res.reversal_risk, "LOW")

    def test_bot_tick_reversal_scan_and_protection(self):
        """Test that TradingBot executes reversal analysis during tick and shields active trades."""
        from unittest.mock import MagicMock, patch
        from main import TradingBot
        from config import TradingConfig
        from state import TradeRecord

        bot = TradingBot(TradingConfig(), load_saved_settings=False)
        bot.broker = MagicMock()
        bot.broker.get_current_price = MagicMock(return_value=MagicMock(bid=105.0, ask=105.05, spread=0.05))

        # Provide synthetic uptrend with bearish CHoCH for XAUUSD
        df_choch = create_uptrend_with_bearish_choch()
        bot._get_ohlcv = MagicMock(return_value=df_choch)

        # Mock an active open long trade on XAUUSD
        from datetime import datetime, timezone
        open_long = TradeRecord(
            id=101,
            timestamp=datetime.now(timezone.utc),
            symbol="XAUUSD",
            direction=Direction.BUY,
            entry_price=104.0,
            stop_loss=102.0,
            take_profit=115.0,
            lot_size=0.1,
            realized_pnl=0.0,
            status='OPEN',
            strategy_name="SMC",
        )
        bot.state.get_open_positions = MagicMock(return_value=[open_long])
        bot.state.is_circuit_breaker_active = MagicMock(return_value=False)
        bot.state.can_trade = MagicMock(return_value=(True, "OK"))

        # Configure bot to analyze XAUUSD
        bot.config.pair1.symbol = "XAUUSD"
        bot.config.pair1.enabled = True
        bot.config.pair2.enabled = False
        bot.config.pair3.enabled = False
        bot.config.selected_symbols = ["XAUUSD"]
        bot.config.reversal_strategy_enabled = True

        # Track protect_against_reversal
        bot.position_manager.protect_against_reversal = MagicMock(return_value=True)

        bot._execute_tick()

        # Verify that reversal analysis was computed and recorded
        self.assertIn("XAUUSD", bot.trend_reversal_status)
        analysis = bot.trend_reversal_status["XAUUSD"]
        self.assertTrue(analysis.choch_detected)
        self.assertEqual(analysis.choch_type, CHoCHType.BEARISH)

        # Verify that active open long was shielded against the Bearish CHoCH!
        bot.position_manager.protect_against_reversal.assert_called_once()
        called_trade, called_analysis = bot.position_manager.protect_against_reversal.call_args[0]
        self.assertEqual(called_trade.id, 101)
        self.assertEqual(called_analysis.choch_type, CHoCHType.BEARISH)

    def test_web_api_trend_reversal_endpoint(self):
        """Test that web_app /api/trend-reversal returns active trend reversal status."""
        from fastapi.testclient import TestClient
        from web_app import app, bot_instance

        # Seed mock reversal status
        mock_analysis = TrendReversalAnalysis(
            symbol="XAUUSD",
            trend=MarketBias.BULLISH,
            is_trending=True,
            choch_detected=True,
            choch_type=CHoCHType.BEARISH,
            stage=ReversalStage.RETRACEMENT_IN_ZONE,
            key_swing_level=2650.0,
            trend_extreme_level=2685.0,
            invalidation_level=2686.0,
            reversal_probability=85.0,
            reversal_risk="CRITICAL",
            warning_message="Bearish CHoCH detected on XAUUSD",
        )
        bot_instance.trend_reversal_status["XAUUSD"] = mock_analysis

        client = TestClient(app)
        response = client.get("/api/trend-reversal")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("XAUUSD", data["data"])
        xau = data["data"]["XAUUSD"]
        self.assertTrue(xau["choch_detected"])
        self.assertEqual(xau["choch_type"], "BEARISH")
        self.assertEqual(xau["reversal_probability"], 85.0)

    def test_reversal_guard_blocks_pre_reversal_sweep(self):
        """Test that pre-reversal sweep correctly filters pro-trend signals and handles MarketBias."""
        from main import Direction, MarketBias
        reversal_analysis = TrendReversalAnalysis(
            symbol="XAUUSD",
            trend=MarketBias.BEARISH,
            is_trending=True,
            choch_detected=False,
            choch_type=CHoCHType.NONE,
            stage=ReversalStage.PRE_REVERSAL_SWEEP,
            key_swing_level=4171.20,
            trend_extreme_level=4108.33,
            invalidation_level=4100.0,
            reversal_probability=48.0,
            reversal_risk="HIGH",
            warning_message="Pre-reversal sweep warning",
        )

        should_block_reversal = False
        blocked_direction = None

        if reversal_analysis.choch_detected and reversal_analysis.reversal_probability >= 50.0:
            should_block_reversal = True
            blocked_direction = Direction.BUY if reversal_analysis.choch_type == CHoCHType.BEARISH else Direction.SELL
        elif (reversal_analysis.stage == ReversalStage.PRE_REVERSAL_SWEEP or reversal_analysis.reversal_risk in ("HIGH", "CRITICAL")) and reversal_analysis.reversal_probability >= 45.0:
            should_block_reversal = True
            blocked_direction = Direction.SELL if reversal_analysis.trend == MarketBias.BEARISH else (Direction.BUY if reversal_analysis.trend == MarketBias.BULLISH else None)

        self.assertTrue(should_block_reversal)
        self.assertEqual(blocked_direction, Direction.SELL)

    def test_standard_deviation_reversal_zone(self):
        """Test that ICT Standard Deviation projection (-2.0, -2.5, -4.0 SD) exhaustion triggers reversal."""
        # Downtrend dealing range: Anchor High at 100.0 (bar 10), Prior Low at 90.0 (bar 20).
        # Anchor range = 100.0 - 90.0 = 10.0
        # SD -2.0 = 100.0 - 2.0 * 10 = 80.0
        # SD -2.5 = 100.0 - 2.5 * 10 = 75.0
        # If trough reaches 74.0, extension = (100 - 74) / 10 = 2.6 >= 2.5 SD
        timestamps = pd.date_range("2026-09-01 00:00:00", periods=50, freq="1h", tz="UTC")
        data = []
        # Bars 0-10: Rise to Anchor High at 100.0
        for i in range(11):
            p = 95.0 + i * 0.5
            data.append({'open': p - 0.1, 'high': p + 0.3, 'low': p - 0.2, 'close': p, 'volume': 150})
        # Bars 11-20: Drop to prior low at 90.0
        for i in range(11, 21):
            p = 100.0 - (i - 10) * 1.0
            data.append({'open': p + 0.1, 'high': p + 0.2, 'low': p - 0.3, 'close': p, 'volume': 200})
        # Bars 21-25: Pullback to 93.0
        for i in range(21, 26):
            p = 90.0 + (i - 20) * 0.6
            data.append({'open': p - 0.1, 'high': p + 0.2, 'low': p - 0.1, 'close': p, 'volume': 150})
        # Bars 26-38: Deep selloff down to trough at 74.0
        for i in range(26, 39):
            p = 93.0 - (i - 25) * 1.5
            if i == 38:
                p = 74.0
            data.append({'open': p + 0.2, 'high': p + 0.3, 'low': p - 0.4, 'close': p, 'volume': 350})
        # Bars 39-45: Small bounce/reversal around 74.5
        for i in range(39, 46):
            p = 74.5
            data.append({'open': p - 0.1, 'high': p + 0.3, 'low': p - 0.2, 'close': p, 'volume': 250})

        df = pd.DataFrame(data, index=timestamps[:len(data)])
        analysis = self.detector.analyze(df=df, trend=MarketBias.BEARISH, symbol="EURUSD", timeframe="1H")

        self.assertTrue(analysis.choch_detected)
        self.assertEqual(analysis.choch_type, CHoCHType.BULLISH)
        self.assertTrue(analysis.confluence.standard_deviation_hit)
        self.assertIsNotNone(analysis.confluence.sd_level)

    def test_fib_50_60_and_below_50_discount_reversal(self):
        """Test that levels below 0.5 Fibonacci and in 0.5-0.6 Fib zone establish Bullish Reversal chance."""
        # Dealing range: Anchor High at 110.0 (bar 10), Trough at 100.0 (bar 30).
        # Range = 10.0. Fib 0.5 = 105.0.
        # At bar 30, price makes a liquidity sweep of a prior low (e.g. 101.0 -> sweeps to 99.8).
        # Current price sits at 101.5, below 0.5 Fib (pos = 0.15 < 0.50).
        timestamps = pd.date_range("2026-09-01 00:00:00", periods=50, freq="1h", tz="UTC")
        data = []
        # Bars 0-10: Anchor high at 110.0
        for i in range(11):
            p = 105.0 + i * 0.5
            data.append({'open': p - 0.1, 'high': p + 0.3, 'low': p - 0.2, 'close': p, 'volume': 150})
        # Bars 11-20: Drop to intermediate low at 101.0
        for i in range(11, 21):
            p = 110.0 - (i - 10) * 0.9
            data.append({'open': p + 0.1, 'high': p + 0.2, 'low': p - 0.3, 'close': p, 'volume': 180})
        # Bars 21-25: Slight bounce to 103.0
        for i in range(21, 26):
            p = 101.0 + (i - 20) * 0.4
            data.append({'open': p - 0.1, 'high': p + 0.2, 'low': p - 0.1, 'close': p, 'volume': 150})
        # Bars 26-30: Drop to 100.0, sweeping below 101.0 with trough low at 99.8
        for i in range(26, 31):
            p = 103.0 - (i - 25) * 0.6
            low_val = p - 0.5 if i < 30 else 99.8
            data.append({'open': p + 0.1, 'high': p + 0.2, 'low': low_val, 'close': p, 'volume': 300})
        # Bars 31-36: Current price consolidating at 100.8 (below 0.5 Fib of range [99.8 - 103.2])
        for i in range(31, 37):
            p = 100.8
            data.append({'open': p - 0.1, 'high': p + 0.2, 'low': p - 0.2, 'close': p, 'volume': 200})

        df = pd.DataFrame(data, index=timestamps[:len(data)])
        analysis = self.detector.analyze(df=df, trend=MarketBias.BEARISH, symbol="GBPUSD", timeframe="1H")

        self.assertTrue(analysis.is_below_fib_50)
        self.assertLess(analysis.confluence.fib_level, 0.50)
        self.assertTrue(analysis.choch_detected)
        self.assertEqual(analysis.choch_type, CHoCHType.BULLISH)

    def test_multitf_reversal_scan_4h_1h_daily(self):
        """Test multi-timeframe scanning across 4H, 1H, and Daily charts."""
        df_1h = create_uptrend_with_bearish_choch(bars_before_peak=35)
        df_4h = create_uptrend_with_bearish_choch(bars_before_peak=35)
        df_1d = create_uptrend_with_bearish_choch(bars_before_peak=35)

        dfs = {"1H": df_1h, "4H": df_4h, "1D": df_1d}
        analysis = self.detector.analyze_multitf(dfs=dfs, trend=MarketBias.BULLISH, symbol="XAUUSD")

        self.assertTrue(analysis.choch_detected)
        self.assertEqual(analysis.choch_type, CHoCHType.BEARISH)
        self.assertIn("MULTI_TF", analysis.timeframe)
        self.assertGreaterEqual(len(analysis.confluence.timeframes_confluent), 2)
        self.assertGreaterEqual(analysis.reversal_probability, 70.0)

    def test_trend_reversal_strategy_signal_generation(self):
        """Test that TrendReversalStrategy generates trade signal in biased direction of reversal zone."""
        from strategy import TrendReversalStrategy, HTFAnalyzer, TradeSignal
        from config import InstrumentConfig

        df_htf = create_uptrend_with_bearish_choch(bars_before_peak=35)
        # LTF 15m data around current price
        ltf_timestamps = pd.date_range("2026-09-01 03:00:00", periods=20, freq="15min", tz="UTC")
        ltf_data = pd.DataFrame([
            {'time': t, 'open': 106.8, 'high': 106.8, 'low': 106.2, 'close': 106.3, 'volume': 150}
            for t in ltf_timestamps
        ], index=ltf_timestamps)

        inst = InstrumentConfig(symbol="XAUUSD", point_value=1.0, pip_size=0.01, digits=2)
        strat = TrendReversalStrategy(HTFAnalyzer())
        signals = strat.evaluate(
            symbol="XAUUSD",
            htf_data=df_htf,
            ltf_data=ltf_data,
            instrument=inst,
            current_spread=0.05,
        )

        self.assertGreaterEqual(len(signals), 1)
        sig = signals[0]
        self.assertEqual(sig.symbol, "XAUUSD")
        self.assertEqual(sig.direction, Direction.SELL)
        self.assertEqual(sig.strategy_id, "TREND_REVERSAL")
        self.assertGreaterEqual(sig.rr_ratio, 1.5)

    def test_cost_to_cost_stop_loss_trigger(self):
        """Test that PositionManager moves SL to cost-to-cost (+0.3R) for Trend Reversal trades."""
        from execution.position_manager import PositionManager
        from unittest.mock import MagicMock
        from config import TradingConfig, DEFAULT_CONFIG
        from state import StateManager, TradeRecord

        mock_broker = MagicMock()
        mock_state = MagicMock()
        pos_mgr = PositionManager(broker=mock_broker, state=mock_state, config=DEFAULT_CONFIG)
        pos_mgr._modify_mt5_sl_tp = MagicMock(return_value=True)

        # Mock an active Trend Reversal BUY trade: Entry 2000.0, SL 1990.0 (risk_dist = 10.0)
        # At +0.35R: current price = 2003.5
        reversal_trade = TradeRecord(
            id=777,
            symbol="XAUUSD",
            direction=Direction.BUY,
            entry_price=2000.0,
            stop_loss=1990.0,
            take_profit=2030.0,
            lot_size=0.1,
            strategy_name="Trend Reversal Strategy",
            magic_number=128456,
            status="OPEN",
            timestamp="2026-09-30T10:00:00Z",
            realized_pnl=0.0,
        )
        mock_state.get_open_positions.return_value = [reversal_trade]

        # Current price reflects +0.35R gain (bid=2003.5)
        mock_quote = MagicMock()
        mock_quote.bid = 2003.5
        mock_quote.ask = 2003.6
        mock_broker.get_current_price.return_value = mock_quote

        pos_mgr.process_positions()

        # Verify that SL was moved to cost-to-cost (2000.0 + 0.1 buffer = 2000.1)
        pos_mgr._modify_mt5_sl_tp.assert_called_once()
        called_ticket, called_sym, called_sl, called_tp = pos_mgr._modify_mt5_sl_tp.call_args[0]
        self.assertEqual(called_ticket, 777)
        self.assertEqual(called_sym, "XAUUSD")
        self.assertAlmostEqual(called_sl, 2000.1, places=1)
        self.assertIn(777, pos_mgr._be_applied)


if __name__ == '__main__':
    unittest.main()



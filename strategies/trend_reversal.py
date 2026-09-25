"""
trend_reversal.py — Institutional CHoCH (Change of Character) & Trend Reversal Detection Subsystem.

Implements Smart Money Concepts (SMC) trend reversal analysis:
1. Trend & Key Swing Points Identification (Last HL in Uptrend, Last LH in Downtrend).
2. CHoCH Signal Detection (Candle break and CLOSE beyond critical swing pivot).
3. Confluence Verification:
   - Liquidity Sweep: Turtle soup / stop run at trend extreme prior to break.
   - Volume Surge: Aggressive displacement volume vs 20-bar rolling average (or ATR displacement).
   - Fair Value Gap (FVG): Imbalance created by the displacement candle.
   - Retracement Entry Zone: Price pullbacks into FVG or Fibonacci Golden Zone (0.382 - 0.618 OTE).
   - Higher Timeframe Bias / S&D Zone alignment.
4. Early Warning & Trade Protection (Shielding active trades against impending reversals).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from loguru import logger

from core.config import Direction, MarketBias
from strategies.strategy import SwingPoint, find_swing_points_logic, compute_atr, HTFAnalysis


class CHoCHType(str, Enum):
    NONE = "NONE"
    BULLISH = "BULLISH"  # Reversal from Downtrend to Uptrend (LH broken upward)
    BEARISH = "BEARISH"  # Reversal from Uptrend to Downtrend (HL broken downward)


class ReversalStage(str, Enum):
    TREND_HEALTHY = "TREND_HEALTHY"                # Trend intact, key swing pivots unviolated
    PRE_REVERSAL_SWEEP = "PRE_REVERSAL_SWEEP"      # Liquidity swept at trend extreme, CHoCH warning
    CHOCH_DISPLACEMENT = "CHOCH_DISPLACEMENT"      # Candle broke & closed past pivot; displacement active
    RETRACEMENT_PENDING = "RETRACEMENT_PENDING"    # CHoCH confirmed, waiting for pullback to key zone
    RETRACEMENT_IN_ZONE = "RETRACEMENT_IN_ZONE"    # Price inside FVG or 0.382-0.618 Fib zone (Prime Entry)
    CONFIRMED_MSS = "CONFIRMED_MSS"                # Market Structure Shift confirmed with new opposite swing


@dataclass
class ReversalConfluence:
    liquidity_sweep: bool = False
    sweep_level: Optional[float] = None
    volume_surge: bool = False
    volume_ratio: float = 1.0
    fvg_present: bool = False
    fvg_top: Optional[float] = None
    fvg_bottom: Optional[float] = None
    fib_382: Optional[float] = None
    fib_618: Optional[float] = None
    in_retracement_zone: bool = False
    htf_alignment: bool = False
    score: float = 0.0
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrendReversalAnalysis:
    symbol: str
    trend: MarketBias
    is_trending: bool
    choch_detected: bool
    choch_type: CHoCHType
    stage: ReversalStage
    key_swing_level: Optional[float] = None        # Last HL in uptrend, last LH in downtrend
    trend_extreme_level: Optional[float] = None    # Peak HH in uptrend, trough LL in downtrend
    invalidation_level: Optional[float] = None     # Invalidation (Stop Loss reference)
    reversal_probability: float = 0.0              # 0.0 to 100.0%
    reversal_risk: str = "LOW"                     # LOW, MODERATE, HIGH, CRITICAL
    confluence: ReversalConfluence = field(default_factory=ReversalConfluence)
    entry_zone: Optional[Tuple[float, float]] = None
    suggested_sl: Optional[float] = None
    suggested_tp: Optional[float] = None
    suggested_rr: Optional[float] = None
    warning_message: str = ""
    timeframe: str = "1H"
    closed_candle_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['trend'] = self.trend.value if hasattr(self.trend, 'value') else str(self.trend)
        d['choch_type'] = self.choch_type.value if hasattr(self.choch_type, 'value') else str(self.choch_type)
        d['stage'] = self.stage.value if hasattr(self.stage, 'value') else str(self.stage)
        d['timeframe'] = self.timeframe
        d['closed_candle_time'] = self.closed_candle_time
        return d


class TrendReversalDetector:
    """
    Real-time analyzer for detecting institutional trend reversals via Change of Character (CHoCH).
    
    In a trending market, constantly monitors whether:
    1. The trend's foundational Higher Low (uptrend) or Lower High (downtrend) has been broken.
    2. Confluence factors confirm that the break is institutional displacement rather than a trap.
    3. Price is retracing into Fair Value Gaps or Golden Fibonacci zones for high-probability setups.
    """

    def __init__(
        self,
        swing_lookback: int = 3,
        volume_surge_multiplier: float = 1.3,
        fvg_min_atr_multiple: float = 0.25,
        displacement_atr_mult: float = 1.2,
    ):
        self.swing_lookback = swing_lookback
        self.volume_surge_multiplier = volume_surge_multiplier
        self.fvg_min_atr_multiple = fvg_min_atr_multiple
        self.displacement_atr_mult = displacement_atr_mult

    def analyze(
        self,
        df: pd.DataFrame,
        trend: Optional[MarketBias | str] = None,
        symbol: str = "UNKNOWN",
        htf_analysis: Optional[HTFAnalysis] = None,
        atr_period: int = 14,
        timeframe: str = "1H",
        current_price: Optional[float] = None,
    ) -> TrendReversalAnalysis:
        """
        Analyze OHLCV price action (e.g. completed 1-Hour candles) in a trending market
        to detect trend reversal risks, CHoCH breaks, and retracement setups.
        """
        candle_time = None
        if df is not None and not df.empty:
            if 'time' in df.columns:
                candle_time = str(df['time'].iloc[-1])
            elif isinstance(df.index, pd.DatetimeIndex):
                candle_time = str(df.index[-1])

        if df is None or len(df) < 25:
            return TrendReversalAnalysis(
                symbol=symbol,
                trend=MarketBias.NEUTRAL,
                is_trending=False,
                choch_detected=False,
                choch_type=CHoCHType.NONE,
                stage=ReversalStage.TREND_HEALTHY,
                warning_message=f"Insufficient data (<25 {timeframe} bars) for trend reversal analysis.",
                timeframe=timeframe,
                closed_candle_time=candle_time,
            )

        n = len(df)
        highs = df['high']
        lows = df['low']
        closes = df['close']

        # ── 1. Determine Trend & Swing Structure ──
        swings = find_swing_points_logic(highs, lows, self.swing_lookback)
        effective_trend = self._resolve_trend(df, swings, trend, htf_analysis)

        if effective_trend == MarketBias.NEUTRAL:
            return TrendReversalAnalysis(
                symbol=symbol,
                trend=MarketBias.NEUTRAL,
                is_trending=False,
                choch_detected=False,
                choch_type=CHoCHType.NONE,
                stage=ReversalStage.TREND_HEALTHY,
                reversal_probability=0.0,
                reversal_risk="LOW",
                warning_message=f"{timeframe} market for {symbol} is currently neutral/ranging. No active trend to reverse.",
                timeframe=timeframe,
                closed_candle_time=candle_time,
            )

        atr_series = compute_atr(df, atr_period)
        current_atr = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 0.001

        # ── 2. Trend-Specific Reversal Analysis ──
        if effective_trend == MarketBias.BULLISH:
            return self._analyze_uptrend_reversal(
                df=df,
                swings=swings,
                symbol=symbol,
                current_atr=current_atr,
                htf_analysis=htf_analysis,
                timeframe=timeframe,
                current_price=current_price,
                candle_time=candle_time,
            )
        else:
            return self._analyze_downtrend_reversal(
                df=df,
                swings=swings,
                symbol=symbol,
                current_atr=current_atr,
                htf_analysis=htf_analysis,
                timeframe=timeframe,
                current_price=current_price,
                candle_time=candle_time,
            )

    def _resolve_trend(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        trend: Optional[MarketBias | str],
        htf_analysis: Optional[HTFAnalysis],
    ) -> MarketBias:
        """Resolve current active trend from explicit bias or internal price action."""
        if trend is not None:
            if isinstance(trend, str):
                t_str = trend.upper()
                if "BULL" in t_str:
                    return MarketBias.BULLISH
                elif "BEAR" in t_str:
                    return MarketBias.BEARISH
                elif "NEUT" in t_str:
                    return MarketBias.NEUTRAL
            elif isinstance(trend, MarketBias):
                return trend

        if htf_analysis and htf_analysis.bias is not None:
            return htf_analysis.bias

        # Auto-detect from swings
        high_swings = [sp for sp in swings if sp.is_high]
        low_swings = [sp for sp in swings if not sp.is_high]
        if len(high_swings) >= 2 and len(low_swings) >= 2:
            if high_swings[-1].price > high_swings[-2].price and low_swings[-1].price > low_swings[-2].price:
                return MarketBias.BULLISH
            elif high_swings[-1].price < high_swings[-2].price and low_swings[-1].price < low_swings[-2].price:
                return MarketBias.BEARISH

        # Fallback to EMA
        closes = df['close']
        ema50 = closes.ewm(span=50, adjust=False).mean().iloc[-1]
        last_close = closes.iloc[-1]
        if last_close > ema50:
            return MarketBias.BULLISH
        elif last_close < ema50:
            return MarketBias.BEARISH

        return MarketBias.NEUTRAL

    def _analyze_uptrend_reversal(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        symbol: str,
        current_atr: float,
        htf_analysis: Optional[HTFAnalysis],
        timeframe: str = "1H",
        current_price: Optional[float] = None,
        candle_time: Optional[str] = None,
    ) -> TrendReversalAnalysis:
        """
        Analyze an established UPTREND (e.g. 1-Hour chart) for signs of Bearish Reversal (Bearish CHoCH):
        - Critical Pivot: Last Higher Low (HL) that supported the trend extreme.
        - Trigger: Candle break and CLOSE below this last HL.
        """
        n = len(df)
        high_swings = [sp for sp in swings if sp.is_high]
        low_swings = [sp for sp in swings if not sp.is_high]

        if not high_swings or not low_swings:
            return TrendReversalAnalysis(
                symbol=symbol,
                trend=MarketBias.BULLISH,
                is_trending=True,
                choch_detected=False,
                choch_type=CHoCHType.NONE,
                stage=ReversalStage.TREND_HEALTHY,
                reversal_probability=5.0,
                reversal_risk="LOW",
                warning_message=f"Uptrend on {symbol} ({timeframe}) lacks mature swing pivots. Trend assumed intact.",
                timeframe=timeframe,
                closed_candle_time=candle_time,
            )

        # 1. Identify the recent trend peak (Highest High / HH)
        recent_window_start = max(0, n - 40)
        recent_high_swings = [sp for sp in high_swings if sp.index >= recent_window_start]
        if not recent_high_swings:
            recent_high_swings = high_swings[-2:]

        peak_sp = max(recent_high_swings, key=lambda sp: sp.price)
        peak_idx = peak_sp.index
        peak_price = peak_sp.price

        # 2. Identify the critical Higher Low (HL) that led directly to that peak
        candidate_hls = [sp for sp in low_swings if sp.index < peak_idx]
        if candidate_hls:
            critical_hl_sp = candidate_hls[-1]
            critical_hl = critical_hl_sp.price
            critical_hl_idx = critical_hl_sp.index
        else:
            pre_peak_lows = df['low'].iloc[max(0, peak_idx - 25):peak_idx]
            if not pre_peak_lows.empty:
                loc = pre_peak_lows.values.argmin()
                critical_hl_idx = max(0, peak_idx - 25) + loc
                critical_hl = float(df['low'].iloc[critical_hl_idx])
            else:
                critical_hl_sp = low_swings[0]
                critical_hl = critical_hl_sp.price
                critical_hl_idx = critical_hl_sp.index

        # 3. Check for CHoCH: Did any candle AFTER the peak break and CLOSE below critical HL?
        choch_detected = False
        break_idx = -1
        break_close = 0.0

        for idx in range(peak_idx, n):
            bar = df.iloc[idx]
            if bar['close'] < critical_hl:
                choch_detected = True
                break_idx = idx
                break_close = float(bar['close'])
                break

        current_candle = df.iloc[-1]
        current_close = float(current_price) if current_price is not None else float(current_candle['close'])

        # 4. Confluence Factors
        confluence_details: List[str] = []
        score = 0.0

        # Confluence A: Liquidity Sweep preceding the reversal
        sweep_detected = False
        sweep_level = None
        # Check if the peak candle or candles near the peak swept a prior swing high
        prior_highs = [sp for sp in high_swings if sp.index < peak_idx]
        if prior_highs:
            prev_high_level = prior_highs[-1].price
            # Check if peak breached prev high but closed below or had a deep upper wick
            peak_bar = df.iloc[peak_idx]
            if peak_bar['high'] > prev_high_level:
                sweep_detected = True
                sweep_level = float(peak_bar['high'])
                confluence_details.append(f"Liquidity Sweep above prior high ({prev_high_level:.5f}) to {sweep_level:.5f}")
                score += 25.0

        # Confluence B: Volume Surge on the break
        volume_surge = False
        vol_ratio = 1.0
        if choch_detected and break_idx >= 0:
            vol_ratio = self._calculate_volume_ratio(df, break_idx)
            break_bar = df.iloc[break_idx]
            body_size = abs(break_bar['close'] - break_bar['open'])
            if vol_ratio >= self.volume_surge_multiplier or body_size >= self.displacement_atr_mult * current_atr:
                volume_surge = True
                confluence_details.append(f"Volume/Displacement Surge ({vol_ratio:.1f}x avg vol or {body_size/current_atr:.1f}x ATR)")
                score += 20.0

        # Confluence C: Fair Value Gap (Bearish FVG)
        fvg_present = False
        fvg_top = None
        fvg_bottom = None
        if choch_detected and break_idx >= 0:
            fvg_present, fvg_top, fvg_bottom = self._detect_bearish_fvg(df, break_idx, current_atr)
            if fvg_present:
                confluence_details.append(f"Bearish FVG created at [{fvg_bottom:.5f} - {fvg_top:.5f}]")
                score += 15.0

        # Confluence D: Fibonacci Retracement Zone (0.382 – 0.618 OTE)
        fib_382 = None
        fib_618 = None
        in_retrace = False

        if choch_detected:
            # Impulse move is from peak (origin) to the lowest point after CHoCH
            displacement_low = df['low'].iloc[peak_idx:].min()
            impulse_range = peak_price - displacement_low
            if impulse_range > 0:
                fib_382 = displacement_low + 0.382 * impulse_range
                fib_618 = displacement_low + 0.618 * impulse_range

                # Check if current price is within Fib zone or within FVG
                if fib_382 <= current_close <= fib_618:
                    in_retrace = True
                    confluence_details.append(f"Price inside Golden Fib zone [0.382: {fib_382:.5f} - 0.618: {fib_618:.5f}]")
                    score += 15.0
                elif fvg_present and fvg_bottom is not None and fvg_top is not None and fvg_bottom <= current_close <= fvg_top:
                    in_retrace = True
                    confluence_details.append(f"Price inside FVG entry zone [{fvg_bottom:.5f} - {fvg_top:.5f}]")
                    score += 15.0

        # Confluence E: Higher Timeframe Bias / S&D Zone Alignment
        htf_align = False
        if htf_analysis:
            # If peak tested an HTF Supply Zone or HTF Liquidity Pool
            for zone in htf_analysis.supply_demand_zones:
                if zone.is_supply and zone.bottom <= peak_price <= zone.top + current_atr * 0.5:
                    htf_align = True
                    confluence_details.append("Reversal initiated at HTF Supply Zone")
                    score += 15.0
                    break

        # Base CHoCH break score
        if choch_detected:
            score += 25.0
        elif sweep_detected:
            score += 10.0  # Pre-warning

        # Cap probability score realistically at 95%
        reversal_prob = min(95.0, round(score, 1))

        # Reversal Risk Level
        if reversal_prob >= 75.0:
            reversal_risk = "CRITICAL"
        elif reversal_prob >= 50.0:
            reversal_risk = "HIGH"
        elif reversal_prob >= 25.0:
            reversal_risk = "MODERATE"
        else:
            reversal_risk = "LOW"

        # Determine Stage
        if not choch_detected:
            if sweep_detected:
                stage = ReversalStage.PRE_REVERSAL_SWEEP
                warning = (
                    f"⚠️ [PRE-REVERSAL SWEEP] {symbol} swept liquidity at {sweep_level:.5f}. "
                    f"Watch for potential Bearish CHoCH if price breaks below last HL ({critical_hl:.5f})."
                )
            else:
                stage = ReversalStage.TREND_HEALTHY
                warning = f"Bullish trend intact on {symbol}. Key swing HL ({critical_hl:.5f}) holding cleanly."
        else:
            # Check for confirmed MSS (subsequent lower high + lower low)
            recent_post_swings = [sp for sp in swings if sp.index > break_idx]
            has_lh = any(sp.is_high and sp.price < peak_price for sp in recent_post_swings)
            has_ll = any(not sp.is_high and sp.price < critical_hl for sp in recent_post_swings)

            if has_lh and has_ll:
                stage = ReversalStage.CONFIRMED_MSS
            elif in_retrace:
                stage = ReversalStage.RETRACEMENT_IN_ZONE
            elif current_close > (critical_hl_sp.price):
                stage = ReversalStage.RETRACEMENT_PENDING
            else:
                stage = ReversalStage.CHOCH_DISPLACEMENT

            warning = (
                f"🚨 [BEARISH CHoCH] {symbol} broke below last HL ({critical_hl:.5f}) with close @ {break_close:.5f}. "
                f"Reversal Risk: {reversal_risk} ({reversal_prob:.0f}%). Stage: {stage.value}."
            )

        # Suggested Setup (Sell Reversal)
        suggested_sl = None
        suggested_tp = None
        suggested_rr = None
        entry_zone = None

        if choch_detected:
            suggested_sl = peak_price + max(0.5 * current_atr, 0.0005)
            # Entry zone is either FVG or Fib 382-618
            if fvg_present and fvg_bottom is not None and fvg_top is not None:
                entry_zone = (fvg_bottom, fvg_top)
            elif fib_382 is not None and fib_618 is not None:
                entry_zone = (min(fib_382, fib_618), max(fib_382, fib_618))
            else:
                entry_zone = (critical_hl, critical_hl + current_atr * 0.5)

            # Target next major low swing or 2.5R (enforcing min 2:1 R:R)
            risk_dist = abs(suggested_sl - current_close)
            target_candidates = [sp.price for sp in low_swings if sp.price < critical_hl]
            valid_targets = [p for p in target_candidates if (current_close - p) >= 2.0 * risk_dist]
            if valid_targets:
                suggested_tp = max(valid_targets)
            elif htf_analysis and htf_analysis.last_swing_low and (current_close - htf_analysis.last_swing_low) >= 2.0 * risk_dist:
                suggested_tp = htf_analysis.last_swing_low
            else:
                suggested_tp = current_close - 2.5 * risk_dist

            reward_dist = abs(current_close - suggested_tp) if suggested_tp else 0.0
            suggested_rr = round(reward_dist / risk_dist, 2) if risk_dist > 0 else 0.0

        confluence_obj = ReversalConfluence(
            liquidity_sweep=sweep_detected,
            sweep_level=sweep_level,
            volume_surge=volume_surge,
            volume_ratio=vol_ratio,
            fvg_present=fvg_present,
            fvg_top=fvg_top,
            fvg_bottom=fvg_bottom,
            fib_382=fib_382,
            fib_618=fib_618,
            in_retracement_zone=in_retrace,
            htf_alignment=htf_align,
            score=reversal_prob,
            details=confluence_details,
        )

        return TrendReversalAnalysis(
            symbol=symbol,
            trend=MarketBias.BULLISH,
            is_trending=True,
            choch_detected=choch_detected,
            choch_type=CHoCHType.BEARISH if choch_detected else CHoCHType.NONE,
            stage=stage,
            key_swing_level=critical_hl,
            trend_extreme_level=peak_price,
            invalidation_level=suggested_sl if choch_detected else critical_hl,
            reversal_probability=reversal_prob,
            reversal_risk=reversal_risk,
            confluence=confluence_obj,
            entry_zone=entry_zone,
            suggested_sl=suggested_sl,
            suggested_tp=suggested_tp,
            suggested_rr=suggested_rr,
            warning_message=warning,
            timeframe=timeframe,
            closed_candle_time=candle_time,
        )

    def _analyze_downtrend_reversal(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        symbol: str,
        current_atr: float,
        htf_analysis: Optional[HTFAnalysis],
        timeframe: str = "1H",
        current_price: Optional[float] = None,
        candle_time: Optional[str] = None,
    ) -> TrendReversalAnalysis:
        """
        Analyze an established DOWNTREND (e.g. 1-Hour chart) for signs of Bullish Reversal (Bullish CHoCH):
        - Critical Pivot: Last Lower High (LH) that supported the trend trough.
        - Trigger: Candle break and CLOSE above this last LH.
        """
        n = len(df)
        high_swings = [sp for sp in swings if sp.is_high]
        low_swings = [sp for sp in swings if not sp.is_high]

        if not high_swings or not low_swings:
            return TrendReversalAnalysis(
                symbol=symbol,
                trend=MarketBias.BEARISH,
                is_trending=True,
                choch_detected=False,
                choch_type=CHoCHType.NONE,
                stage=ReversalStage.TREND_HEALTHY,
                reversal_probability=5.0,
                reversal_risk="LOW",
                warning_message=f"Downtrend on {symbol} ({timeframe}) lacks mature swing pivots. Trend assumed intact.",
                timeframe=timeframe,
                closed_candle_time=candle_time,
            )

        # 1. Identify the recent trend trough (Lowest Low / LL)
        recent_window_start = max(0, n - 40)
        recent_low_swings = [sp for sp in low_swings if sp.index >= recent_window_start]
        if not recent_low_swings:
            recent_low_swings = low_swings[-2:]

        trough_sp = min(recent_low_swings, key=lambda sp: sp.price)
        trough_idx = trough_sp.index
        trough_price = trough_sp.price

        # 2. Identify the critical Lower High (LH) that led directly to that trough
        candidate_lhs = [sp for sp in high_swings if sp.index < trough_idx]
        if candidate_lhs:
            critical_lh_sp = candidate_lhs[-1]
            critical_lh = critical_lh_sp.price
            critical_lh_idx = critical_lh_sp.index
        else:
            # Scan backward from trough_idx - 1 to find the last local peak
            found_lh = False
            for k in range(trough_idx - 1, max(1, trough_idx - 15), -1):
                if df['high'].iloc[k] >= df['high'].iloc[k-1] and df['high'].iloc[k] >= df['high'].iloc[k+1]:
                    critical_lh_idx = k
                    critical_lh = float(df['high'].iloc[k])
                    found_lh = True
                    break
            if not found_lh:
                pre_trough_highs = df['high'].iloc[max(0, trough_idx - 10):trough_idx]
                critical_lh_idx = max(0, trough_idx - 10) + pre_trough_highs.values.argmax()
                critical_lh = float(df['high'].iloc[critical_lh_idx])

        # 3. Check for CHoCH: Did any candle AFTER the trough break and CLOSE above critical LH?
        choch_detected = False
        break_idx = -1
        break_close = 0.0

        for idx in range(trough_idx, n):
            bar = df.iloc[idx]
            if bar['close'] > critical_lh:
                choch_detected = True
                break_idx = idx
                break_close = float(bar['close'])
                break

        current_candle = df.iloc[-1]
        current_close = float(current_price) if current_price is not None else float(current_candle['close'])

        # 4. Confluence Factors
        confluence_details: List[str] = []
        score = 0.0

        # Confluence A: Liquidity Sweep preceding the reversal
        sweep_detected = False
        sweep_level = None
        prior_lows = [sp for sp in low_swings if sp.index < trough_idx]
        if prior_lows:
            prev_low_level = prior_lows[-1].price
            trough_bar = df.iloc[trough_idx]
            if trough_bar['low'] < prev_low_level:
                sweep_detected = True
                sweep_level = float(trough_bar['low'])
                confluence_details.append(f"Liquidity Sweep below prior low ({prev_low_level:.5f}) to {sweep_level:.5f}")
                score += 25.0

        # Confluence B: Volume Surge on the break
        volume_surge = False
        vol_ratio = 1.0
        if choch_detected and break_idx >= 0:
            vol_ratio = self._calculate_volume_ratio(df, break_idx)
            break_bar = df.iloc[break_idx]
            body_size = abs(break_bar['close'] - break_bar['open'])
            if vol_ratio >= self.volume_surge_multiplier or body_size >= self.displacement_atr_mult * current_atr:
                volume_surge = True
                confluence_details.append(f"Volume/Displacement Surge ({vol_ratio:.1f}x avg vol or {body_size/current_atr:.1f}x ATR)")
                score += 20.0

        # Confluence C: Fair Value Gap (Bullish FVG)
        fvg_present = False
        fvg_top = None
        fvg_bottom = None
        if choch_detected and break_idx >= 0:
            fvg_present, fvg_top, fvg_bottom = self._detect_bullish_fvg(df, break_idx, current_atr)
            if fvg_present:
                confluence_details.append(f"Bullish FVG created at [{fvg_bottom:.5f} - {fvg_top:.5f}]")
                score += 15.0

        # Confluence D: Fibonacci Retracement Zone (0.382 – 0.618 OTE)
        fib_382 = None
        fib_618 = None
        in_retrace = False

        if choch_detected:
            displacement_high = df['high'].iloc[trough_idx:].max()
            impulse_range = displacement_high - trough_price
            if impulse_range > 0:
                fib_382 = displacement_high - 0.382 * impulse_range
                fib_618 = displacement_high - 0.618 * impulse_range

                if min(fib_382, fib_618) <= current_close <= max(fib_382, fib_618):
                    in_retrace = True
                    confluence_details.append(f"Price inside Golden Fib zone [0.382: {fib_382:.5f} - 0.618: {fib_618:.5f}]")
                    score += 15.0
                elif fvg_present and fvg_bottom is not None and fvg_top is not None and fvg_bottom <= current_close <= fvg_top:
                    in_retrace = True
                    confluence_details.append(f"Price inside FVG entry zone [{fvg_bottom:.5f} - {fvg_top:.5f}]")
                    score += 15.0

        # Confluence E: Higher Timeframe Bias / S&D Zone Alignment
        htf_align = False
        if htf_analysis:
            for zone in htf_analysis.supply_demand_zones:
                if not zone.is_supply and zone.bottom - current_atr * 0.5 <= trough_price <= zone.top:
                    htf_align = True
                    confluence_details.append("Reversal initiated at HTF Demand Zone")
                    score += 15.0
                    break

        if choch_detected:
            score += 25.0
        elif sweep_detected:
            score += 10.0

        reversal_prob = min(95.0, round(score, 1))

        if reversal_prob >= 75.0:
            reversal_risk = "CRITICAL"
        elif reversal_prob >= 50.0:
            reversal_risk = "HIGH"
        elif reversal_prob >= 25.0:
            reversal_risk = "MODERATE"
        else:
            reversal_risk = "LOW"

        # Stage
        if not choch_detected:
            if sweep_detected:
                stage = ReversalStage.PRE_REVERSAL_SWEEP
                warning = (
                    f"⚠️ [PRE-REVERSAL SWEEP] {symbol} swept liquidity at {sweep_level:.5f}. "
                    f"Watch for potential Bullish CHoCH if price breaks above last LH ({critical_lh:.5f})."
                )
            else:
                stage = ReversalStage.TREND_HEALTHY
                warning = f"Bearish trend intact on {symbol}. Key swing LH ({critical_lh:.5f}) holding cleanly."
        else:
            recent_post_swings = [sp for sp in swings if sp.index > break_idx]
            has_hl = any(not sp.is_high and sp.price > trough_price for sp in recent_post_swings)
            has_hh = any(sp.is_high and sp.price > critical_lh for sp in recent_post_swings)

            if has_hl and has_hh:
                stage = ReversalStage.CONFIRMED_MSS
            elif in_retrace:
                stage = ReversalStage.RETRACEMENT_IN_ZONE
            elif current_close < critical_lh:
                stage = ReversalStage.RETRACEMENT_PENDING
            else:
                stage = ReversalStage.CHOCH_DISPLACEMENT

            warning = (
                f"🚨 [BULLISH CHoCH] {symbol} broke above last LH ({critical_lh:.5f}) with close @ {break_close:.5f}. "
                f"Reversal Risk: {reversal_risk} ({reversal_prob:.0f}%). Stage: {stage.value}."
            )

        # Suggested Setup (Buy Reversal)
        suggested_sl = None
        suggested_tp = None
        suggested_rr = None
        entry_zone = None

        if choch_detected:
            suggested_sl = trough_price - max(0.5 * current_atr, 0.0005)
            if fvg_present and fvg_bottom is not None and fvg_top is not None:
                entry_zone = (fvg_bottom, fvg_top)
            elif fib_382 is not None and fib_618 is not None:
                entry_zone = (min(fib_382, fib_618), max(fib_382, fib_618))
            else:
                entry_zone = (critical_lh - current_atr * 0.5, critical_lh)

            risk_dist = abs(current_close - suggested_sl)
            target_candidates = [sp.price for sp in high_swings if sp.price > critical_lh]
            valid_targets = [p for p in target_candidates if (p - current_close) >= 2.0 * risk_dist]
            if valid_targets:
                suggested_tp = min(valid_targets)
            elif htf_analysis and htf_analysis.last_swing_high and (htf_analysis.last_swing_high - current_close) >= 2.0 * risk_dist:
                suggested_tp = htf_analysis.last_swing_high
            else:
                suggested_tp = current_close + 2.5 * risk_dist

            reward_dist = abs(suggested_tp - current_close) if suggested_tp else 0.0
            suggested_rr = round(reward_dist / risk_dist, 2) if risk_dist > 0 else 0.0

        confluence_obj = ReversalConfluence(
            liquidity_sweep=sweep_detected,
            sweep_level=sweep_level,
            volume_surge=volume_surge,
            volume_ratio=vol_ratio,
            fvg_present=fvg_present,
            fvg_top=fvg_top,
            fvg_bottom=fvg_bottom,
            fib_382=fib_382,
            fib_618=fib_618,
            in_retracement_zone=in_retrace,
            htf_alignment=htf_align,
            score=reversal_prob,
            details=confluence_details,
        )

        return TrendReversalAnalysis(
            symbol=symbol,
            trend=MarketBias.BEARISH,
            is_trending=True,
            choch_detected=choch_detected,
            choch_type=CHoCHType.BULLISH if choch_detected else CHoCHType.NONE,
            stage=stage,
            key_swing_level=critical_lh,
            trend_extreme_level=trough_price,
            invalidation_level=suggested_sl if choch_detected else critical_lh,
            reversal_probability=reversal_prob,
            reversal_risk=reversal_risk,
            confluence=confluence_obj,
            entry_zone=entry_zone,
            suggested_sl=suggested_sl,
            suggested_tp=suggested_tp,
            suggested_rr=suggested_rr,
            warning_message=warning,
            timeframe=timeframe,
            closed_candle_time=candle_time,
        )

    def _calculate_volume_ratio(self, df: pd.DataFrame, target_idx: int) -> float:
        """Calculate volume ratio comparing target bar to 20-period rolling average."""
        vol_col = None
        for col in ('volume', 'tick_volume', 'vol'):
            if col in df.columns:
                vol_col = col
                break

        if vol_col is None:
            return 1.0

        vols = df[vol_col]
        if vols.sum() == 0:
            return 1.0

        start_idx = max(0, target_idx - 20)
        avg_vol = vols.iloc[start_idx:target_idx].mean() if target_idx > start_idx else vols.mean()
        target_vol = vols.iloc[target_idx]

        if avg_vol > 0:
            return float(target_vol / avg_vol)
        return 1.0

    def _detect_bearish_fvg(
        self,
        df: pd.DataFrame,
        break_idx: int,
        current_atr: float,
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """Detect Bearish Fair Value Gap around the break candle (candle i-2 low > candle i high)."""
        n = len(df)
        min_gap = self.fvg_min_atr_multiple * current_atr

        start = max(2, break_idx - 1)
        end = min(n, break_idx + 4)

        for i in range(start, end):
            c_curr = df.iloc[i]
            c_prev2 = df.iloc[i - 2]
            if c_prev2['low'] > c_curr['high']:
                gap = float(c_prev2['low'] - c_curr['high'])
                if gap >= min_gap:
                    return True, float(c_prev2['low']), float(c_curr['high'])

        return False, None, None

    def _detect_bullish_fvg(
        self,
        df: pd.DataFrame,
        break_idx: int,
        current_atr: float,
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """Detect Bullish Fair Value Gap around the break candle (candle i low > candle i-2 high)."""
        n = len(df)
        min_gap = self.fvg_min_atr_multiple * current_atr

        start = max(2, break_idx - 1)
        end = min(n, break_idx + 4)

        for i in range(start, end):
            c_curr = df.iloc[i]
            c_prev2 = df.iloc[i - 2]
            if c_curr['low'] > c_prev2['high']:
                gap = float(c_curr['low'] - c_prev2['high'])
                if gap >= min_gap:
                    return True, float(c_curr['low']), float(c_prev2['high'])

        return False, None, None

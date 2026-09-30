"""
trend_reversal.py — Institutional Trend Reversal Detection Subsystem.

Replaces legacy single-pivot CHoCH with institutional Reversal Zones across 4H, 1H, and Daily (1D) timeframes:
1. ICT Standard Deviation Swing Projections (-2.0, -2.5, -4.0 SD).
2. Fibonacci 0.5 to 0.6 Retracement / Equilibrium Zone.
3. Fair Value Gaps (FVG) on 4H, 1H, and Daily charts.
4. "Below 0.5 Level" Rule:
   - When any key level (FVG, Liquidity Sweep, Standard Deviation exhaustion) sits below the 0.5
     Fibonacci level (in the Discount zone), it establishes a high-probability Bullish Reversal chance.
   - Conversely, when sitting above 0.5 Fibonacci (in the Premium zone), it establishes a Bearish Reversal chance.
5. Generates reversal entry zones with invalidation levels and suggested targets.
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
    """Reversal Direction (maintained for backward compatibility with CHoCH callers)."""
    NONE = "NONE"
    BULLISH = "BULLISH"  # Reversal from Bearish to Bullish (Buying the Reversal)
    BEARISH = "BEARISH"  # Reversal from Bullish to Bearish (Selling the Reversal)


ReversalType = CHoCHType


class ReversalStage(str, Enum):
    TREND_HEALTHY = "TREND_HEALTHY"                  # Trend intact, no reversal zone reached
    PRE_REVERSAL_SWEEP = "PRE_REVERSAL_SWEEP"        # Liquidity swept at trend extreme
    REVERSAL_ZONE_TESTED = "REVERSAL_ZONE_TESTED"    # Price entered SD / Fib 0.5-0.6 / FVG zone
    CHOCH_DISPLACEMENT = "CHOCH_DISPLACEMENT"        # Strong displacement away from reversal zone
    RETRACEMENT_PENDING = "RETRACEMENT_PENDING"      # Reversal initiated, awaiting mitigation pullback
    RETRACEMENT_IN_ZONE = "RETRACEMENT_IN_ZONE"      # Price inside 0.5-0.6 Fib / FVG entry zone
    CONFIRMED_MSS = "CONFIRMED_MSS"                  # Confirmed Market Structure Shift in new direction


@dataclass
class ReversalConfluence:
    liquidity_sweep: bool = False
    sweep_level: Optional[float] = None
    volume_surge: bool = False
    volume_ratio: float = 1.0
    fvg_present: bool = False
    fvg_top: Optional[float] = None
    fvg_bottom: Optional[float] = None
    fvg_timeframe: Optional[str] = None
    fib_382: Optional[float] = None
    fib_50: Optional[float] = None
    fib_60: Optional[float] = None
    fib_618: Optional[float] = None
    fib_level: Optional[float] = None
    is_below_fib_50: bool = False
    is_above_fib_50: bool = False
    in_fib_50_60_zone: bool = False
    standard_deviation_hit: bool = False
    sd_level: Optional[float] = None
    sd_multiple: Optional[float] = None
    reversal_zone_type: str = "NONE"  # SD_PROJECTION, FIB_50_60, FVG, CONFLUENCE
    in_retracement_zone: bool = False
    htf_alignment: bool = False
    timeframes_confluent: List[str] = field(default_factory=list)
    score: float = 0.0
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrendReversalAnalysis:
    symbol: str
    trend: MarketBias
    is_trending: bool
    choch_detected: bool                             # Maintained as reversal_detected alias
    choch_type: CHoCHType                            # Maintained as reversal_type alias
    stage: ReversalStage
    key_swing_level: Optional[float] = None          # Reference swing level or 0.5 Fib equilibrium
    trend_extreme_level: Optional[float] = None      # Peak HH or trough LL of dealing range
    invalidation_level: Optional[float] = None       # Invalidation / Stop Loss reference
    reversal_probability: float = 0.0                # 0.0 to 100.0%
    reversal_risk: str = "LOW"                       # LOW, MODERATE, HIGH, CRITICAL
    confluence: ReversalConfluence = field(default_factory=ReversalConfluence)
    entry_zone: Optional[Tuple[float, float]] = None
    suggested_sl: Optional[float] = None
    suggested_tp: Optional[float] = None
    suggested_rr: Optional[float] = None
    warning_message: str = ""
    timeframe: str = "1H"
    closed_candle_time: Optional[str] = None
    reversal_detected: bool = False
    reversal_type: CHoCHType = CHoCHType.NONE
    reversal_zone_type: str = "NONE"
    fib_level: Optional[float] = None
    is_below_fib_50: bool = False
    sd_level: Optional[float] = None
    tf_confluences: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.choch_detected and not self.reversal_detected:
            self.reversal_detected = True
        elif self.reversal_detected and not self.choch_detected:
            self.choch_detected = True
        if self.choch_type != CHoCHType.NONE and self.reversal_type == CHoCHType.NONE:
            self.reversal_type = self.choch_type
        elif self.reversal_type != CHoCHType.NONE and self.choch_type == CHoCHType.NONE:
            self.choch_type = self.reversal_type

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['trend'] = self.trend.value if hasattr(self.trend, 'value') else str(self.trend)
        d['choch_type'] = self.choch_type.value if hasattr(self.choch_type, 'value') else str(self.choch_type)
        d['reversal_type'] = self.reversal_type.value if hasattr(self.reversal_type, 'value') else str(self.reversal_type)
        d['stage'] = self.stage.value if hasattr(self.stage, 'value') else str(self.stage)
        d['timeframe'] = self.timeframe
        d['closed_candle_time'] = self.closed_candle_time
        return d


class TrendReversalDetector:
    """
    Institutional Trend Reversal Detector.

    Evaluates 4H, 1H, and Daily (1D) timeframes to identify institutional reversal zones:
    1. ICT Standard Deviation Projections (-2.0, -2.5, -4.0 SD from dealing range impulse).
    2. Fibonacci 0.5 to 0.6 Retracement Zone (Equilibrium to Golden Pocket).
    3. Fair Value Gaps (FVG) on 4H, 1H, and Daily charts.
    4. "Below 0.5 Level" Rule:
       - Levels (FVG, liquidity sweep, SD exhaustion) below 0.5 Fibonacci represent high-probability
         Discount accumulation for Bullish Reversals.
       - Levels above 0.5 Fibonacci represent Premium distribution for Bearish Reversals.
    """

    def __init__(
        self,
        swing_lookback: int = 3,
        volume_surge_multiplier: float = 1.3,
        fvg_min_atr_multiple: float = 0.25,
        displacement_atr_mult: float = 1.2,
        sd_multiples: Optional[List[float]] = None,
    ):
        self.swing_lookback = swing_lookback
        self.volume_surge_multiplier = volume_surge_multiplier
        self.fvg_min_atr_multiple = fvg_min_atr_multiple
        self.displacement_atr_mult = displacement_atr_mult
        self.sd_multiples = sd_multiples or [2.0, 2.5, 4.0]

    def analyze_multitf(
        self,
        dfs: Dict[str, pd.DataFrame],
        trend: Optional[MarketBias | str] = None,
        symbol: str = "UNKNOWN",
        htf_analysis: Optional[HTFAnalysis] = None,
        current_price: Optional[float] = None,
    ) -> TrendReversalAnalysis:
        """
        Analyze multi-timeframe DataFrames (e.g. {'1D': df_d1, '4H': df_h4, '1H': df_h1})
        to detect high-conviction institutional reversal zones.
        """
        if not dfs:
            return TrendReversalAnalysis(
                symbol=symbol,
                trend=MarketBias.NEUTRAL,
                is_trending=False,
                choch_detected=False,
                choch_type=CHoCHType.NONE,
                stage=ReversalStage.TREND_HEALTHY,
                warning_message="No multi-TF data provided for reversal analysis.",
            )

        # Primary baseline timeframe is 1H if present, else first available
        base_tf = "1H" if "1H" in dfs else ("4H" if "4H" in dfs else next(iter(dfs.keys())))
        base_df = dfs[base_tf]

        # Analyze base TF first
        base_analysis = self.analyze(
            df=base_df,
            trend=trend,
            symbol=symbol,
            htf_analysis=htf_analysis,
            timeframe=base_tf,
            current_price=current_price,
        )

        tf_details: Dict[str, Any] = {base_tf: base_analysis.to_dict()}
        confluent_tfs = [base_tf] if base_analysis.choch_detected else []
        extra_score = 0.0

        # Scan other available higher timeframes (4H, Daily)
        for tf_name in ("4H", "1D", "DAILY"):
            if tf_name in dfs and tf_name != base_tf:
                df_other = dfs[tf_name]
                if df_other is not None and len(df_other) >= 20:
                    other_analysis = self.analyze(
                        df=df_other,
                        trend=trend,
                        symbol=symbol,
                        htf_analysis=htf_analysis,
                        timeframe=tf_name,
                        current_price=current_price,
                    )
                    tf_details[tf_name] = other_analysis.to_dict()

                    # Check for cross-TF confluence (same reversal direction)
                    if other_analysis.choch_detected:
                        if base_analysis.choch_detected:
                            if other_analysis.choch_type == base_analysis.choch_type:
                                confluent_tfs.append(tf_name)
                                extra_score += 15.0
                                base_analysis.confluence.details.append(
                                    f"Multi-TF Confluence confirmed on {tf_name} ({other_analysis.reversal_zone_type})"
                                )
                        else:
                            # Other TF detected reversal even if base TF was lagging
                            base_analysis = other_analysis
                            confluent_tfs.append(tf_name)

        if len(confluent_tfs) > 1:
            base_analysis.timeframe = f"MULTI_TF({','.join(confluent_tfs)})"
            base_analysis.confluence.timeframes_confluent = confluent_tfs
            base_analysis.confluence.htf_alignment = True
            base_analysis.reversal_probability = min(95.0, base_analysis.reversal_probability + extra_score)
            if base_analysis.reversal_probability >= 75.0:
                base_analysis.reversal_risk = "CRITICAL"
            elif base_analysis.reversal_probability >= 50.0:
                base_analysis.reversal_risk = "HIGH"

        base_analysis.tf_confluences = tf_details
        return base_analysis

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
        Analyze OHLCV price action on a given timeframe (1H, 4H, or Daily) to detect
        reversal zones based on Standard Deviation, Fib 0.5-0.6, and FVG levels.
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

        highs = df['high']
        lows = df['low']
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

        high_swings = [sp for sp in swings if sp.is_high]
        low_swings = [sp for sp in swings if not sp.is_high]
        if len(high_swings) >= 2 and len(low_swings) >= 2:
            if high_swings[-1].price > high_swings[-2].price and low_swings[-1].price > low_swings[-2].price:
                return MarketBias.BULLISH
            elif high_swings[-1].price < high_swings[-2].price and low_swings[-1].price < low_swings[-2].price:
                return MarketBias.BEARISH

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
        Analyze an established UPTREND for signs of Bearish Reversal:
        Replaces legacy CHoCH with:
        1. ICT Standard Deviation Projections (+2.0 to +2.5 and +4.0 SD exhaustion of impulse).
        2. Fibonacci 0.5 to 0.6 Retracement Zone (Equilibrium / Premium boundary).
        3. Bearish Fair Value Gap (FVG) in 4H/1H/Daily.
        4. "Above 0.5 Level" Rule (Premium exhaustion zone).
        """
        n = len(df)
        high_swings = [sp for sp in swings if sp.is_high]
        low_swings = [sp for sp in swings if not sp.is_high]

        if not high_swings:
            max_idx = int(df['high'].argmax())
            high_swings = [SwingPoint(index=max_idx, price=float(df['high'].iloc[max_idx]), is_high=True)]
        if not low_swings:
            min_idx = int(df['low'].argmin())
            low_swings = [SwingPoint(index=min_idx, price=float(df['low'].iloc[min_idx]), is_high=False)]

        # 1. Identify dominant dealing range: Anchor Low (L) to Peak High (H)
        recent_window_start = max(0, n - 40)
        recent_high_swings = [sp for sp in high_swings if sp.index >= recent_window_start]
        if not recent_high_swings:
            recent_high_swings = high_swings[-2:]

        peak_sp = max(recent_high_swings, key=lambda sp: sp.price)
        peak_idx = peak_sp.index
        peak_price = peak_sp.price

        candidate_lows = [sp for sp in low_swings if sp.index < peak_idx]
        if candidate_lows:
            anchor_low_sp = candidate_lows[-1]
            anchor_low = anchor_low_sp.price
            anchor_low_idx = anchor_low_sp.index
        else:
            pre_peak_lows = df['low'].iloc[max(0, peak_idx - 25):peak_idx]
            if not pre_peak_lows.empty:
                loc = pre_peak_lows.values.argmin()
                anchor_low_idx = max(0, peak_idx - 25) + loc
                anchor_low = float(df['low'].iloc[anchor_low_idx])
            else:
                anchor_low = low_swings[0].price
                anchor_low_idx = low_swings[0].index

        impulse_range = max(peak_price - anchor_low, current_atr * 0.5)

        # Current price reference
        current_candle = df.iloc[-1]
        current_close = float(current_price) if current_price is not None else float(current_candle['close'])

        # 2. Fibonacci Retracement Levels of Dealing Range
        # In uptrend, 0.0 is Peak, 1.0 is Anchor Low (or normalized 0 at low, 1 at peak)
        fib_50 = anchor_low + 0.50 * impulse_range
        fib_60 = anchor_low + 0.60 * impulse_range
        fib_618 = anchor_low + 0.618 * impulse_range
        fib_382 = anchor_low + 0.382 * impulse_range

        # Normalized Fibonacci position of current price: 0.0 = low, 1.0 = peak
        curr_fib_pos = (current_close - anchor_low) / impulse_range if impulse_range > 0 else 0.5
        is_above_50 = curr_fib_pos >= 0.50
        is_below_50 = curr_fib_pos < 0.50
        in_fib_50_60 = (min(fib_50, fib_60) <= current_close <= max(fib_50, fib_60))

        # 3. ICT Standard Deviation Projections (+2.0, +2.5, +4.0 SD above anchor range)
        prior_highs = [sp for sp in high_swings if sp.index < peak_idx]
        if prior_highs:
            anchor_range = max(prior_highs[-1].price - anchor_low, current_atr * 0.5)
        else:
            anchor_range = max(impulse_range * 0.5, current_atr * 0.5)

        sd_20 = anchor_low + 2.0 * anchor_range
        sd_25 = anchor_low + 2.5 * anchor_range
        sd_40 = anchor_low + 4.0 * anchor_range

        # Check if peak or current price hit SD exhaustion target
        sd_multiple = (peak_price - anchor_low) / anchor_range if anchor_range > 0 else 1.0
        sd_hit = False
        sd_hit_level = None
        sd_hit_multiple = None
        if sd_multiple >= 2.0 - 1e-4 or peak_price >= (sd_20 - current_atr * 0.25):
            sd_hit = True
            if sd_multiple >= 4.0 - 1e-4 or peak_price >= sd_40 - current_atr * 0.25:
                sd_hit_level = sd_40
                sd_hit_multiple = 4.0
            elif sd_multiple >= 2.5 - 1e-4 or peak_price >= sd_25 - current_atr * 0.25:
                sd_hit_level = sd_25
                sd_hit_multiple = 2.5
            else:
                sd_hit_level = sd_20
                sd_hit_multiple = 2.0

        # 4. Bearish Fair Value Gap (FVG)
        fvg_present, fvg_top, fvg_bottom = self._detect_bearish_fvg(df, peak_idx, current_atr)

        # 5. Liquidity Sweep at Peak
        sweep_detected = False
        sweep_level = None
        if prior_highs:
            prev_high_level = prior_highs[-1].price
            peak_bar = df.iloc[peak_idx]
            if peak_bar['high'] > prev_high_level:
                sweep_detected = True
                sweep_level = float(peak_bar['high'])

        # 6. Volume / Displacement Surge
        vol_ratio = self._calculate_volume_ratio(df, min(n - 1, peak_idx + 1))
        volume_surge = (vol_ratio >= self.volume_surge_multiplier)

        # 7. Check Legacy Break (Break below anchor low) for full MSS confirmation
        legacy_break = any(df['close'].iloc[peak_idx:] < anchor_low)

        # ── Score & Confluence Compilation ──
        confluence_details: List[str] = []
        score = 0.0
        reversal_zone_types: List[str] = []

        if sd_hit:
            score += 30.0
            reversal_zone_types.append(f"SD_{sd_hit_multiple}x")
            confluence_details.append(f"ICT Standard Deviation Exhaustion hit (+{sd_hit_multiple:.1f} SD @ {sd_hit_level:.5f})")

        if in_fib_50_60:
            score += 25.0
            reversal_zone_types.append("FIB_0.5_0.6")
            confluence_details.append(f"Price inside Golden 0.5-0.6 Fib Retracement Zone [{fib_50:.5f} - {fib_60:.5f}]")
        elif is_above_50 and current_close >= fib_60:
            # In premium zone above 0.5 fib where bearish reversal risk is elevated
            score += 15.0
            confluence_details.append(f"Price in Premium zone (Fib {curr_fib_pos:.2f} > 0.50)")

        if is_above_50 and (fvg_present or sweep_detected or sd_hit):
            score += 25.0
            confluence_details.append("Premium level above 0.5 Fib established (prime Bearish Reversal chance)")

        if fvg_present:
            score += 20.0
            reversal_zone_types.append("BEARISH_FVG")
            confluence_details.append(f"Bearish FVG on {timeframe} at [{fvg_bottom:.5f} - {fvg_top:.5f}]")

        if sweep_detected:
            score += 20.0
            confluence_details.append(f"Liquidity Sweep above prior swing high ({sweep_level:.5f})")

        if volume_surge:
            score += 15.0
            confluence_details.append(f"Volume Surge ({vol_ratio:.1f}x avg vol)")

        # HTF S/D alignment
        htf_align = False
        if htf_analysis:
            for zone in htf_analysis.supply_demand_zones:
                if zone.is_supply and zone.bottom <= peak_price <= zone.top + current_atr * 0.5:
                    htf_align = True
                    confluence_details.append(f"Reversal aligned with HTF Supply Zone [{zone.bottom:.5f} - {zone.top:.5f}]")
                    score += 15.0
                    break

        if legacy_break:
            score += 20.0
            confluence_details.append(f"Displacement break below swing anchor ({anchor_low:.5f})")

        # Determine Reversal Trigger:
        reversal_detected = (
            (sd_hit and (sweep_detected or fvg_present or in_fib_50_60 or legacy_break or current_close < peak_price - current_atr * 0.25))
            or in_fib_50_60
            or (is_above_50 and (fvg_present or sweep_detected or (sd_hit and current_close < peak_price - current_atr * 0.25)))
            or (fvg_present and (is_above_50 or sweep_detected or volume_surge))
            or legacy_break
            or (score >= 45.0 and current_close < peak_price - current_atr * 0.25)
        )

        # An active new high without rejection, FVG, sweep, or zone test is trend continuation
        if current_close >= peak_price - current_atr * 0.25 and not (sweep_detected or fvg_present or legacy_break or in_fib_50_60):
            reversal_detected = False
            score = min(score, 20.0)

        if reversal_detected:
            score = max(score, 50.0)

        reversal_prob = min(95.0, round(score, 1))

        if reversal_prob >= 75.0:
            reversal_risk = "CRITICAL"
        elif reversal_prob >= 50.0:
            reversal_risk = "HIGH"
        elif reversal_prob >= 25.0:
            reversal_risk = "MODERATE"
        else:
            reversal_risk = "LOW"

        # Determine Stage
        if not reversal_detected:
            if sweep_detected or (sd_hit and current_close < peak_price - current_atr * 0.25):
                stage = ReversalStage.PRE_REVERSAL_SWEEP
                warning = (
                    f"⚠️ [PRE-REVERSAL ALERT] {symbol} ({timeframe}): Liquidity sweep / premium extension at {peak_price:.5f}. "
                    f"Monitoring for Bearish Reversal Zone (SD / Fib 0.5-0.6 / FVG)."
                )
            else:
                stage = ReversalStage.TREND_HEALTHY
                warning = f"Bullish trend intact on {symbol} ({timeframe}). No reversal zone triggered."
        else:
            if legacy_break:
                stage = ReversalStage.CONFIRMED_MSS
            elif in_fib_50_60 or (fvg_present and fvg_bottom is not None and fvg_bottom <= current_close <= (fvg_top or 0.0)):
                stage = ReversalStage.RETRACEMENT_IN_ZONE
            elif current_close < peak_price:
                stage = ReversalStage.REVERSAL_ZONE_TESTED
            else:
                stage = ReversalStage.CHOCH_DISPLACEMENT

            zone_str = "+".join(reversal_zone_types) if reversal_zone_types else "REVERSAL_ZONE"
            warning = (
                f"🚨 [BEARISH REVERSAL ZONE] {symbol} ({timeframe}): Reversal Zone triggered ({zone_str}). "
                f"Reversal Risk: {reversal_risk} ({reversal_prob:.0f}%). Stage: {stage.value}."
            )

        # Setup Parameters for Sell Reversal Trade
        suggested_sl = peak_price + max(0.5 * current_atr, 0.0005)
        if fvg_present and fvg_bottom is not None and fvg_top is not None:
            entry_zone = (fvg_bottom, fvg_top)
        elif in_fib_50_60:
            entry_zone = (min(fib_50, fib_60), max(fib_50, fib_60))
        else:
            entry_zone = (anchor_low, peak_price)

        risk_dist = abs(suggested_sl - current_close)
        suggested_tp = current_close - 2.5 * max(risk_dist, current_atr)
        reward_dist = abs(current_close - suggested_tp) if suggested_tp else 0.0
        suggested_rr = round(reward_dist / risk_dist, 2) if risk_dist > 0 else 2.5

        primary_zone_type = "+".join(reversal_zone_types) if reversal_zone_types else ("FVG" if fvg_present else "FIB_50_60")

        confluence_obj = ReversalConfluence(
            liquidity_sweep=sweep_detected,
            sweep_level=sweep_level,
            volume_surge=volume_surge,
            volume_ratio=vol_ratio,
            fvg_present=fvg_present,
            fvg_top=fvg_top,
            fvg_bottom=fvg_bottom,
            fvg_timeframe=timeframe,
            fib_382=fib_382,
            fib_50=fib_50,
            fib_60=fib_60,
            fib_618=fib_618,
            fib_level=round(curr_fib_pos, 3),
            is_below_fib_50=is_below_50,
            is_above_fib_50=is_above_50,
            in_fib_50_60_zone=in_fib_50_60,
            standard_deviation_hit=sd_hit,
            sd_level=sd_hit_level,
            sd_multiple=sd_hit_multiple,
            reversal_zone_type=primary_zone_type,
            in_retracement_zone=in_fib_50_60 or stage == ReversalStage.RETRACEMENT_IN_ZONE,
            htf_alignment=htf_align,
            timeframes_confluent=[timeframe],
            score=reversal_prob,
            details=confluence_details,
        )

        return TrendReversalAnalysis(
            symbol=symbol,
            trend=MarketBias.BULLISH,
            is_trending=True,
            choch_detected=reversal_detected,
            choch_type=CHoCHType.BEARISH if reversal_detected else CHoCHType.NONE,
            stage=stage,
            key_swing_level=anchor_low,
            trend_extreme_level=peak_price,
            invalidation_level=suggested_sl if reversal_detected else anchor_low,
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
            reversal_detected=reversal_detected,
            reversal_type=CHoCHType.BEARISH if reversal_detected else CHoCHType.NONE,
            reversal_zone_type=primary_zone_type,
            fib_level=round(curr_fib_pos, 3),
            is_below_fib_50=is_below_50,
            sd_level=sd_hit_level,
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
        Analyze an established DOWNTREND for signs of Bullish Reversal:
        Replaces legacy CHoCH with:
        1. ICT Standard Deviation Projections (-2.0 to -2.5 and -4.0 SD downside exhaustion).
        2. Fibonacci 0.5 to 0.6 Retracement Zone (Equilibrium / Discount boundary).
        3. Bullish Fair Value Gap (FVG) in 4H/1H/Daily.
        4. "Below 0.5 Level" Rule:
           - Levels (FVG, liquidity sweep, SD exhaustion) sitting BELOW 0.5 Fibonacci represent
             prime institutional accumulation for a Bullish Reversal.
        """
        n = len(df)
        high_swings = [sp for sp in swings if sp.is_high]
        low_swings = [sp for sp in swings if not sp.is_high]

        if not high_swings:
            max_idx = int(df['high'].argmax())
            high_swings = [SwingPoint(index=max_idx, price=float(df['high'].iloc[max_idx]), is_high=True)]
        if not low_swings:
            min_idx = int(df['low'].argmin())
            low_swings = [SwingPoint(index=min_idx, price=float(df['low'].iloc[min_idx]), is_high=False)]

        # 1. Identify dominant dealing range: Anchor High (H) to Trough Low (L)
        recent_window_start = max(0, n - 40)
        recent_low_swings = [sp for sp in low_swings if sp.index >= recent_window_start]
        if not recent_low_swings:
            recent_low_swings = low_swings[-2:]

        trough_sp = min(recent_low_swings, key=lambda sp: sp.price)
        trough_idx = trough_sp.index
        trough_price = trough_sp.price

        candidate_highs = [sp for sp in high_swings if sp.index < trough_idx]
        if candidate_highs:
            anchor_high_sp = candidate_highs[-1]
            anchor_high = anchor_high_sp.price
            anchor_high_idx = anchor_high_sp.index
        else:
            pre_trough_highs = df['high'].iloc[max(0, trough_idx - 25):trough_idx]
            if not pre_trough_highs.empty:
                loc = pre_trough_highs.values.argmax()
                anchor_high_idx = max(0, trough_idx - 25) + loc
                anchor_high = float(df['high'].iloc[anchor_high_idx])
            else:
                anchor_high = high_swings[0].price
                anchor_high_idx = high_swings[0].index

        impulse_range = max(anchor_high - trough_price, current_atr * 0.5)

        current_candle = df.iloc[-1]
        current_close = float(current_price) if current_price is not None else float(current_candle['close'])

        # 2. Fibonacci Retracement Levels of Dealing Range
        # In downtrend, 0.0 is Trough Low, 1.0 is Anchor High
        fib_50 = trough_price + 0.50 * impulse_range
        fib_60 = trough_price + 0.60 * impulse_range
        fib_618 = trough_price + 0.618 * impulse_range
        fib_382 = trough_price + 0.382 * impulse_range

        # Normalized Fibonacci position: 0.0 = trough, 1.0 = anchor high
        curr_fib_pos = (current_close - trough_price) / impulse_range if impulse_range > 0 else 0.5
        is_below_50 = curr_fib_pos < 0.50
        is_above_50 = curr_fib_pos >= 0.50
        in_fib_50_60 = (min(fib_50, fib_60) <= current_close <= max(fib_50, fib_60))

        # 3. ICT Standard Deviation Projections (-2.0, -2.5, -4.0 SD below anchor range)
        prior_lows = [sp for sp in low_swings if sp.index < trough_idx]
        if prior_lows:
            anchor_range = max(anchor_high - prior_lows[-1].price, current_atr * 0.5)
        else:
            anchor_range = max(impulse_range * 0.5, current_atr * 0.5)

        sd_20 = anchor_high - 2.0 * anchor_range
        sd_25 = anchor_high - 2.5 * anchor_range
        sd_40 = anchor_high - 4.0 * anchor_range

        sd_multiple = (anchor_high - trough_price) / anchor_range if anchor_range > 0 else 1.0
        sd_hit = False
        sd_hit_level = None
        sd_hit_multiple = None
        if sd_multiple >= 2.0 - 1e-4 or trough_price <= (sd_20 + current_atr * 0.25):
            sd_hit = True
            if sd_multiple >= 4.0 - 1e-4 or trough_price <= sd_40 + current_atr * 0.25:
                sd_hit_level = sd_40
                sd_hit_multiple = -4.0
            elif sd_multiple >= 2.5 - 1e-4 or trough_price <= sd_25 + current_atr * 0.25:
                sd_hit_level = sd_25
                sd_hit_multiple = -2.5
            else:
                sd_hit_level = sd_20
                sd_hit_multiple = -2.0

        # 4. Bullish Fair Value Gap (FVG)
        fvg_present, fvg_top, fvg_bottom = self._detect_bullish_fvg(df, trough_idx, current_atr)

        # 5. Liquidity Sweep at Trough
        sweep_detected = False
        sweep_level = None
        if prior_lows:
            prev_low_level = prior_lows[-1].price
            trough_bar = df.iloc[trough_idx]
            if trough_bar['low'] < prev_low_level:
                sweep_detected = True
                sweep_level = float(trough_bar['low'])

        # 6. Volume / Displacement Surge
        vol_ratio = self._calculate_volume_ratio(df, min(n - 1, trough_idx + 1))
        volume_surge = (vol_ratio >= self.volume_surge_multiplier)

        # 7. Check Legacy Break (Break above anchor high) for full MSS confirmation
        legacy_break = any(df['close'].iloc[trough_idx:] > anchor_high)

        # ── Score & Confluence Compilation ──
        confluence_details: List[str] = []
        score = 0.0
        reversal_zone_types: List[str] = []

        if sd_hit:
            score += 30.0
            reversal_zone_types.append(f"SD_{abs(sd_hit_multiple):.1f}x")
            confluence_details.append(f"ICT Standard Deviation Exhaustion hit ({sd_hit_multiple:.1f} SD @ {sd_hit_level:.5f})")

        # "when any level (fvg, liquidity sweep, standard deviation) below .5 level of fibonacci there can be a chance to reversal"
        if is_below_50:
            discount_levels: List[str] = []
            if fvg_present:
                discount_levels.append("Bullish FVG")
            if sweep_detected:
                discount_levels.append("Liquidity Sweep")
            if sd_hit or trough_price < fib_50:
                discount_levels.append("SD Extension")

            if discount_levels:
                score += 25.0
                reversal_zone_types.append("DISCOUNT_REVERSAL_<0.5")
                confluence_details.append(
                    f"Discount Reversal Opportunity: Key level(s) [{', '.join(discount_levels)}] below 0.5 Fib (pos: {curr_fib_pos:.2f} < 0.50)"
                )
            else:
                score += 15.0
                confluence_details.append(f"Price in Discount accumulation zone (Fib {curr_fib_pos:.2f} < 0.50)")

        if in_fib_50_60:
            score += 25.0
            reversal_zone_types.append("FIB_0.5_0.6")
            confluence_details.append(f"Price inside Golden 0.5-0.6 Fib Retracement Zone [{fib_50:.5f} - {fib_60:.5f}]")

        if fvg_present:
            score += 20.0
            reversal_zone_types.append("BULLISH_FVG")
            confluence_details.append(f"Bullish FVG on {timeframe} at [{fvg_bottom:.5f} - {fvg_top:.5f}]")

        if sweep_detected:
            score += 20.0
            confluence_details.append(f"Liquidity Sweep below prior swing low ({sweep_level:.5f})")

        if volume_surge:
            score += 15.0
            confluence_details.append(f"Volume Surge ({vol_ratio:.1f}x avg vol)")

        htf_align = False
        if htf_analysis:
            for zone in htf_analysis.supply_demand_zones:
                if not zone.is_supply and zone.bottom - current_atr * 0.5 <= trough_price <= zone.top:
                    htf_align = True
                    confluence_details.append(f"Reversal aligned with HTF Demand Zone [{zone.bottom:.5f} - {zone.top:.5f}]")
                    score += 15.0
                    break

        if legacy_break:
            score += 20.0
            confluence_details.append(f"Displacement break above swing anchor ({anchor_high:.5f})")

        # Reversal Trigger:
        reversal_detected = (
            sd_hit
            or in_fib_50_60
            or (is_below_50 and (fvg_present or sweep_detected or sd_hit))
            or (fvg_present and (is_below_50 or sweep_detected or volume_surge))
            or legacy_break
            or (score >= 40.0)
        )

        if reversal_detected:
            score = max(score, 50.0)

        reversal_prob = min(95.0, round(score, 1))

        if reversal_prob >= 75.0:
            reversal_risk = "CRITICAL"
        elif reversal_prob >= 50.0:
            reversal_risk = "HIGH"
        elif reversal_prob >= 25.0:
            reversal_risk = "MODERATE"
        else:
            reversal_risk = "LOW"

        if not reversal_detected:
            if sweep_detected or (sd_hit and current_close > trough_price + current_atr * 0.25):
                stage = ReversalStage.PRE_REVERSAL_SWEEP
                warning = (
                    f"⚠️ [PRE-REVERSAL ALERT] {symbol} ({timeframe}): Liquidity sweep / discount extension at {trough_price:.5f}. "
                    f"Monitoring for Bullish Reversal Zone (SD / Fib 0.5-0.6 / FVG below 0.5)."
                )
            else:
                stage = ReversalStage.TREND_HEALTHY
                warning = f"Bearish trend intact on {symbol} ({timeframe}). No reversal zone triggered."
        else:
            if legacy_break:
                stage = ReversalStage.CONFIRMED_MSS
            elif in_fib_50_60 or (fvg_present and fvg_bottom is not None and (fvg_top or 0.0) >= current_close >= fvg_bottom):
                stage = ReversalStage.RETRACEMENT_IN_ZONE
            elif current_close > trough_price:
                stage = ReversalStage.REVERSAL_ZONE_TESTED
            else:
                stage = ReversalStage.CHOCH_DISPLACEMENT

            zone_str = "+".join(reversal_zone_types) if reversal_zone_types else "REVERSAL_ZONE"
            warning = (
                f"🚨 [BULLISH REVERSAL ZONE] {symbol} ({timeframe}): Reversal Zone triggered ({zone_str}). "
                f"Reversal Risk: {reversal_risk} ({reversal_prob:.0f}%). Stage: {stage.value}."
            )

        # Setup Parameters for Buy Reversal Trade
        suggested_sl = trough_price - max(0.5 * current_atr, 0.0005)
        if fvg_present and fvg_bottom is not None and fvg_top is not None:
            entry_zone = (fvg_bottom, fvg_top)
        elif in_fib_50_60:
            entry_zone = (min(fib_50, fib_60), max(fib_50, fib_60))
        else:
            entry_zone = (trough_price, anchor_high)

        risk_dist = abs(current_close - suggested_sl)
        suggested_tp = current_close + 2.5 * max(risk_dist, current_atr)
        reward_dist = abs(suggested_tp - current_close) if suggested_tp else 0.0
        suggested_rr = round(reward_dist / risk_dist, 2) if risk_dist > 0 else 2.5

        primary_zone_type = "+".join(reversal_zone_types) if reversal_zone_types else ("FVG" if fvg_present else "FIB_50_60")

        confluence_obj = ReversalConfluence(
            liquidity_sweep=sweep_detected,
            sweep_level=sweep_level,
            volume_surge=volume_surge,
            volume_ratio=vol_ratio,
            fvg_present=fvg_present,
            fvg_top=fvg_top,
            fvg_bottom=fvg_bottom,
            fvg_timeframe=timeframe,
            fib_382=fib_382,
            fib_50=fib_50,
            fib_60=fib_60,
            fib_618=fib_618,
            fib_level=round(curr_fib_pos, 3),
            is_below_fib_50=is_below_50,
            is_above_fib_50=is_above_50,
            in_fib_50_60_zone=in_fib_50_60,
            standard_deviation_hit=sd_hit,
            sd_level=sd_hit_level,
            sd_multiple=sd_hit_multiple,
            reversal_zone_type=primary_zone_type,
            in_retracement_zone=in_fib_50_60 or stage == ReversalStage.RETRACEMENT_IN_ZONE,
            htf_alignment=htf_align,
            timeframes_confluent=[timeframe],
            score=reversal_prob,
            details=confluence_details,
        )

        return TrendReversalAnalysis(
            symbol=symbol,
            trend=MarketBias.BEARISH,
            is_trending=True,
            choch_detected=reversal_detected,
            choch_type=CHoCHType.BULLISH if reversal_detected else CHoCHType.NONE,
            stage=stage,
            key_swing_level=anchor_high,
            trend_extreme_level=trough_price,
            invalidation_level=suggested_sl if reversal_detected else anchor_high,
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
            reversal_detected=reversal_detected,
            reversal_type=CHoCHType.BULLISH if reversal_detected else CHoCHType.NONE,
            reversal_zone_type=primary_zone_type,
            fib_level=round(curr_fib_pos, 3),
            is_below_fib_50=is_below_50,
            sd_level=sd_hit_level,
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
        pivot_idx: int,
        current_atr: float,
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """Detect Bearish Fair Value Gap (candle i-2 low > candle i high) around the pivot/displacement."""
        n = len(df)
        min_gap = self.fvg_min_atr_multiple * current_atr

        start = max(2, pivot_idx - 2)
        end = min(n, pivot_idx + 8)

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
        pivot_idx: int,
        current_atr: float,
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """Detect Bullish Fair Value Gap (candle i low > candle i-2 high) around the pivot/displacement."""
        n = len(df)
        min_gap = self.fvg_min_atr_multiple * current_atr

        start = max(2, pivot_idx - 2)
        end = min(n, pivot_idx + 8)

        for i in range(start, end):
            c_curr = df.iloc[i]
            c_prev2 = df.iloc[i - 2]
            if c_curr['low'] > c_prev2['high']:
                gap = float(c_curr['low'] - c_prev2['high'])
                if gap >= min_gap:
                    return True, float(c_curr['low']), float(c_prev2['high'])

        return False, None, None

"""
strategy.py — Smart Money Concepts (SMC) & Institutional Order Flow Strategy Engine.

Core Philosophy:
1. Macro Trend (HTF 1H/4H):
   - 200 EMA + Break of Structure (BOS) determines directional bias.
   - Maps major liquidity pools (Equal Highs/Lows, Swing Highs/Lows) and Supply/Demand zones.
2. Entry & Mitigation (LTF 15m/5m):
   - Step 1: Liquidity Sweep (Inducement / Stop Hunt) of a key level.
   - Step 2: Displacement / Impulsive candle leaving institutional footprints.
   - Step 3: Order Block (OB) and/or Fair Value Gap (FVG) detection.
   - Step 4: Mitigation retest entry with an ultra-tight stop just beyond the swept invalidation level.
   - Step 5: Asymmetric Take Profit targeting opposing liquidity pools or HTF S/D zones (3:1 to 10:1+ R:R).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta, time as dtime
from zoneinfo import ZoneInfo
import math
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger

from core.config import MarketBias, Direction, TimeframeConfig, InstrumentConfig, StrategyType, ICTConfig, OrderFlowConfig



# ─────────────────────────────────────────────
#  Enums & Data Structures
# ─────────────────────────────────────────────

class LTFConfirmation(str, Enum):
    OB_MITIGATION = "OB_MITIGATION"       # Retest of Order Block
    FVG_MITIGATION = "FVG_MITIGATION"     # Retest of Fair Value Gap
    OB_PLUS_FVG = "OB_PLUS_FVG"           # Overlapping OB + FVG (Prime Confluence)
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"   # Direct sweep & reversal confirmation
    OB_SCALP_5M = "OB_SCALP_5M"           # 5M Break of Structure & Order Block Retest Scalp
    ICT_KILLZONE_FVG = "ICT_KILLZONE_FVG" # ICT Kill Zone FVG Retest
    ICT_SILVER_BULLET = "ICT_SILVER_BULLET" # ICT Silver Bullet Model
    ICT_JUDAS_SWING = "ICT_JUDAS_SWING"   # ICT Judas Swing Liquidity Purge
    ICT_OTE_RETEST = "ICT_OTE_RETEST"     # ICT Optimal Trade Entry (61.8%-78.6% Fib)
    ICT_AOX_SNIPER_TAP = "ICT_AOX_SNIPER_TAP"         # Entry 1: POI Tap / AOX Fib -0.21 to -0.29 (25% Risk)
    ICT_AOX_SMT_TURTLE_SOUP = "ICT_AOX_SMT_TURTLE_SOUP" # Entry 2: SMT Divergence + Turtle Soup into First FVG (50% Risk)
    ICT_AOX_HIPPO_MITIGATION = "ICT_AOX_HIPPO_MITIGATION" # Entry 3: HIPPO / Final Mitigation Block (50% Risk)
    ICT_FPFVG_MAGNET = "ICT_FPFVG_MAGNET"             # First Presented FVG Magnet Snap-Back
    ICT_SD_REVERSAL = "ICT_SD_REVERSAL"               # Standard Deviation (-2.0/-2.5 or -4.0/-4.5 SD) Exhaustion
    OF_ABSORPTION = "OF_ABSORPTION"       # Order Flow Institutional Absorption at Key Level
    OF_DELTA_DIVERGENCE = "OF_DELTA_DIVERGENCE" # Cumulative Volume Delta Divergence
    OF_LIQUIDITY_TRAP = "OF_LIQUIDITY_TRAP"     # Stop Sweep with Instant Delta Reversal
    REVERSAL_ZONE_SD = "REVERSAL_ZONE_SD"       # ICT Standard Deviation (-2.0, -2.5, -4.0 SD) Reversal Zone
    REVERSAL_ZONE_FIB = "REVERSAL_ZONE_FIB"     # Fibonacci 0.5 - 0.6 Retracement Reversal Zone
    REVERSAL_ZONE_FVG = "REVERSAL_ZONE_FVG"     # Multi-TF (4H/1H/Daily) Fair Value Gap Reversal Zone
    REVERSAL_CONFLUENCE = "REVERSAL_CONFLUENCE" # Multi-factor Reversal Zone Confluence
    PULLBACK = "PULLBACK"                       # Compatibility fallback
    STRUCTURAL_BREAK = "STRUCTURAL_BREAK"       # Compatibility fallback
    NONE = "NONE"


@dataclass(frozen=True)
class SwingPoint:
    index: int
    price: float
    is_high: bool  # True = swing high, False = swing low


@dataclass(frozen=True)
class LiquidityPool:
    level: float
    is_high: bool          # True = Buy-side liquidity (BSL), False = Sell-side liquidity (SSL)
    touch_count: int       # Number of touches clustering near this level
    last_index: int


@dataclass(frozen=True)
class OrderBlock:
    index: int
    direction: Direction   # BUY = Bullish OB (demand), SELL = Bearish OB (supply)
    low: float
    high: float
    is_mitigated: bool = False


@dataclass(frozen=True)
class FairValueGap:
    index: int
    direction: Direction   # BUY = Bullish FVG, SELL = Bearish FVG
    bottom: float
    top: float
    size: float
    is_mitigated: bool = False


@dataclass(frozen=True)
class SupplyDemandZone:
    is_supply: bool        # True = Supply (resistance), False = Demand (support)
    bottom: float
    top: float


@dataclass
class HTFAnalysis:
    bias: MarketBias
    ema_value: float
    last_swing_high: float | None
    last_swing_low: float | None
    trend_clarity_score: float  # 0-30
    liquidity_pools: list[LiquidityPool] = field(default_factory=list)
    supply_demand_zones: list[SupplyDemandZone] = field(default_factory=list)
    # Institutional Dealing Range & 0.5 Fib Equilibrium Fields:
    dealing_range_low: float | None = None
    dealing_range_high: float | None = None
    fib_50: float | None = None
    is_discount: bool = False
    is_premium: bool = False
    invalidation_level: float | None = None
    first_zone_entry: tuple[float, float] | None = None
    first_zone_type: str = "NONE"
    all_zones_failed: bool = False
    sd_40: float | None = None
    sd_45: float | None = None


@dataclass
class TradeSignal:
    symbol: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    htf_bias: MarketBias
    ltf_confirmation: LTFConfirmation
    rr_ratio: float
    quality_score: float  # 0-100
    timestamp: datetime
    sl_distance: float = 0.0  # |entry - sl| in price
    tp_distance: float = 0.0  # |entry - tp| in price
    strategy_id: str = "SMC"
    strategy_name: str = "SMC Swing"
    magic_number: int = 123456
    runner_tp: float | None = None
    candle_confirmation: str = ""
    layer: int = 1
    risk_fraction: float = 1.0
    aox_tp1: float | None = None
    aox_tp2: float | None = None
    session_killzone: str = ""
    smt_confirmed: bool = False
    sd_zone: str = ""



# ─────────────────────────────────────────────
#  Mathematical & Indicator Helpers
# ─────────────────────────────────────────────

def clamp_tp_to_rr(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: Direction,
    min_rr: float = 1.5,
    max_rr: float = 3.0,
) -> tuple[float, float]:
    """
    Clamps take profit price so that the Risk:Reward ratio strictly stays within [min_rr, max_rr].
    Returns (clamped_tp, effective_rr).
    """
    sl_dist = abs(entry_price - stop_loss)
    if sl_dist <= 1e-9:
        return take_profit, 0.0

    min_tp_dist = sl_dist * min_rr
    max_tp_dist = sl_dist * max_rr

    tp_dist = abs(take_profit - entry_price)

    # Strictly bound between min_rr (1.5) and max_rr (3.0)
    clamped_tp_dist = max(min_tp_dist, min(tp_dist, max_tp_dist))

    if direction == Direction.BUY:
        clamped_tp = entry_price + clamped_tp_dist
    else:
        clamped_tp = entry_price - clamped_tp_dist

    effective_rr = round(clamped_tp_dist / sl_dist, 2)
    return clamped_tp, effective_rr

def check_candle_body_confirmation(
    candle: pd.Series | dict,
    direction: Direction,
    min_body_ratio: float = 0.50,
) -> tuple[bool, float, str]:
    """
    Validates directional candlestick body confirmation:
    - BUY Setup: Candle MUST be GREEN (close > open) with body/range >= min_body_ratio (default 50% body, <= 50% wick).
    - SELL Setup: Candle MUST be RED (close < open) with body/range >= min_body_ratio (default 50% body, <= 50% wick).
    - Max allowed can be a 100% Marubozu candle.
    - If opposite color or body ratio < min_body_ratio (excessive wick), the candle is rejected.
    Returns: (is_confirmed: bool, body_ratio: float, reason: str)
    """
    try:
        open_p = float(candle['open'])
        close_p = float(candle['close'])
        high_p = float(candle['high'])
        low_p = float(candle['low'])
    except Exception as e:
        return False, 0.0, f"Invalid candle data: {e}"

    total_range = high_p - low_p
    if total_range <= 1e-9:
        return False, 0.0, "Zero or flat candle range"

    if direction == Direction.BUY:
        if close_p <= open_p:
            return False, 0.0, f"Bearish/Flat candle ({close_p:.5f} <= {open_p:.5f}) rejected for BUY setup"
        body = close_p - open_p
        ratio = body / total_range
        if ratio < min_body_ratio:
            return False, ratio, f"Bullish body ratio {ratio:.1%} < required {min_body_ratio:.0%} (excessive wick)"
        return True, ratio, f"Confirmed GREEN candle: {ratio:.1%} body (>= {min_body_ratio:.0%})"

    elif direction == Direction.SELL:
        if close_p >= open_p:
            return False, 0.0, f"Bullish/Flat candle ({close_p:.5f} >= {open_p:.5f}) rejected for SELL setup"
        body = open_p - close_p
        ratio = body / total_range
        if ratio < min_body_ratio:
            return False, ratio, f"Bearish body ratio {ratio:.1%} < required {min_body_ratio:.0%} (excessive wick)"
        return True, ratio, f"Confirmed RED candle: {ratio:.1%} body (>= {min_body_ratio:.0%})"

    return False, 0.0, f"Unknown direction: {direction}"


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)."""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    close_prev = np.empty_like(close)
    close_prev[0] = close[0]
    close_prev[1:] = close[:-1]

    tr1 = high - low
    tr2 = np.abs(high - close_prev)
    tr3 = np.abs(low - close_prev)

    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    return pd.Series(tr, index=df.index).rolling(window=period, min_periods=1).mean()


def find_swing_points_logic(highs: pd.Series, lows: pd.Series, lookback: int) -> list[SwingPoint]:
    """Identify swing highs and swing lows using N-bar pivot logic (NumPy optimized)."""
    swings = []
    n = len(highs)
    if n < 2 * lookback + 1:
        return swings

    h_arr = highs.values
    l_arr = lows.values

    for i in range(lookback, n - lookback):
        window_highs = h_arr[i - lookback : i + lookback + 1]
        if h_arr[i] == window_highs.max():
            swings.append(SwingPoint(index=i, price=float(h_arr[i]), is_high=True))

        window_lows = l_arr[i - lookback : i + lookback + 1]
        if l_arr[i] == window_lows.min():
            swings.append(SwingPoint(index=i, price=float(l_arr[i]), is_high=False))

    swings.sort(key=lambda sp: sp.index)
    return swings


# ─────────────────────────────────────────────
#  HTF Macro Structure & Liquidity Analyzer
# ─────────────────────────────────────────────

class HTFAnalyzer:
    """Analyzes macro market structure, trend bias, liquidity pools, and S/D zones."""

    def __init__(self, ema_period: int = 200, swing_lookback: int = 5, equal_level_tolerance: float = 0.15):
        self.ema_period = ema_period
        self.swing_lookback = swing_lookback
        self.equal_level_tolerance = equal_level_tolerance

    def compute_ema(self, closes: pd.Series) -> pd.Series:
        """Compute Exponential Moving Average."""
        return closes.ewm(span=self.ema_period, adjust=False).mean()

    def find_swing_points(self, highs: pd.Series, lows: pd.Series) -> list[SwingPoint]:
        return find_swing_points_logic(highs, lows, self.swing_lookback)

    def detect_market_structure(self, swing_points: list[SwingPoint]) -> MarketBias:
        """
        Determine Break of Structure (BOS) trend bias:
        - Ascending highs and ascending lows -> BULLISH
        - Descending highs and descending lows -> BEARISH
        """
        highs = [sp.price for sp in swing_points if sp.is_high]
        lows = [sp.price for sp in swing_points if not sp.is_high]

        if len(highs) >= 2 and len(lows) >= 2:
            last_2_highs = highs[-2:]
            last_2_lows = lows[-2:]

            if last_2_highs[1] > last_2_highs[0] and last_2_lows[1] > last_2_lows[0]:
                return MarketBias.BULLISH
            elif last_2_highs[1] < last_2_highs[0] and last_2_lows[1] < last_2_lows[0]:
                return MarketBias.BEARISH

        return MarketBias.NEUTRAL

    def find_liquidity_pools(self, swing_points: list[SwingPoint], atr_val: float) -> list[LiquidityPool]:
        """Group swing points into liquidity pools (e.g. Equal Highs/Lows clusters)."""
        pools: list[LiquidityPool] = []
        if not swing_points or atr_val <= 0:
            return pools

        threshold = atr_val * self.equal_level_tolerance

        # Split highs and lows
        highs = [sp for sp in swing_points if sp.is_high]
        lows = [sp for sp in swing_points if not sp.is_high]

        for group, is_high in [(highs, True), (lows, False)]:
            clusters: list[list[SwingPoint]] = []
            for sp in group:
                assigned = False
                for cluster in clusters:
                    avg_price = sum(p.price for p in cluster) / len(cluster)
                    if abs(sp.price - avg_price) <= threshold:
                        cluster.append(sp)
                        assigned = True
                        break
                if not assigned:
                    clusters.append([sp])

            for cluster in clusters:
                avg_level = sum(p.price for p in cluster) / len(cluster)
                last_idx = max(p.index for p in cluster)
                pools.append(LiquidityPool(
                    level=avg_level,
                    is_high=is_high,
                    touch_count=len(cluster),
                    last_index=last_idx,
                ))

        return pools

    def find_supply_demand_zones(self, df: pd.DataFrame, swing_points: list[SwingPoint]) -> list[SupplyDemandZone]:
        """Detect macro Supply and Demand zones from HTF order blocks."""
        zones: list[SupplyDemandZone] = []
        n = len(df)
        if n < 5:
            return zones

        # Demand zones around major swing lows, Supply zones around major swing highs
        for sp in swing_points[-6:]:
            idx = sp.index
            if idx < 1 or idx >= n:
                continue
            candle = df.iloc[idx]
            if sp.is_high:
                # Supply Zone at high
                zones.append(SupplyDemandZone(
                    is_supply=True,
                    bottom=float(min(candle['open'], candle['close'])),
                    top=float(candle['high']),
                ))
            else:
                # Demand Zone at low
                zones.append(SupplyDemandZone(
                    is_supply=False,
                    bottom=float(candle['low']),
                    top=float(max(candle['open'], candle['close'])),
                ))
        return zones

    def detect_dealing_range(
        self,
        df: pd.DataFrame,
        swings: list[SwingPoint],
        current_atr: float,
        current_close: float,
        structure_bias: MarketBias = MarketBias.NEUTRAL,
        base_bias: MarketBias = MarketBias.NEUTRAL,
    ) -> dict[str, Any]:
        """
        Identify Institutional Dealing Range, Equilibrium (0.5 Fib), and Discount/Premium Zones.
        - Bullish Dealing Range: Expansion leg from Anchor Low to Swing High.
          - 0.5 Fib separates Premium from Discount.
          - In Discount (<0.5 Fib): Look for 1st FVG, liquidity point, and -4 to -4.5 SD zone.
          - Market Bias remains BULLISH as price pulls back into Discount.
          - If ALL discount levels fail (price closes below invalidation level), bias flips to BEARISH.
        - Bearish Dealing Range: Expansion leg from Anchor High to Swing Low.
          - 0.5 Fib separates Discount from Premium.
          - In Premium (>0.5 Fib): Look for 1st FVG, liquidity point, and +4 to +4.5 SD zone.
          - Market Bias remains BEARISH as price pulls back into Premium.
          - If ALL premium levels fail (price closes above invalidation level), bias flips to BULLISH.
        """
        res: dict[str, Any] = {
            "dealing_range_low": None,
            "dealing_range_high": None,
            "fib_50": None,
            "is_discount": False,
            "is_premium": False,
            "invalidation_level": None,
            "first_zone_entry": None,
            "first_zone_type": "NONE",
            "all_zones_failed": False,
            "sd_40": None,
            "sd_45": None,
            "bias_override": None,
        }

        n = len(df)
        if n < 5:
            return res

        high_swings = [sp for sp in swings if sp.is_high]
        low_swings = [sp for sp in swings if not sp.is_high]

        # Determine dealing range orientation:
        # 1. Structure bias or macro trend bias from HTF analysis
        if structure_bias == MarketBias.BULLISH or base_bias == MarketBias.BULLISH:
            is_bullish_dr = True
        elif structure_bias == MarketBias.BEARISH or base_bias == MarketBias.BEARISH:
            is_bullish_dr = False
        # 2. Sequential swing points analysis (if bias is neutral/unknown)
        elif high_swings and low_swings:
            is_bullish_dr = (high_swings[-1].index > low_swings[-1].index)
        elif high_swings:
            is_bullish_dr = True
        elif low_swings:
            is_bullish_dr = False
        # 3. Global high/low sequence in window
        else:
            high_idx = int(df['high'].argmax())
            low_idx = int(df['low'].argmin())
            is_bullish_dr = (high_idx >= low_idx)

        if is_bullish_dr:
            # Bullish Dealing Range: Anchor Low -> Peak High
            if high_swings:
                peak_sp = max(high_swings[-3:], key=lambda sp: sp.price)
                peak_high = peak_sp.price
                peak_idx = peak_sp.index
            else:
                peak_idx = int(df['high'].argmax())
                peak_high = float(df['high'].iloc[peak_idx])

            abs_high_idx = int(df['high'].argmax())
            if float(df['high'].iloc[abs_high_idx]) > peak_high:
                peak_high = float(df['high'].iloc[abs_high_idx])
                peak_idx = abs_high_idx

            pre_peak_lows = df['low'].iloc[:max(1, peak_idx)]
            if not pre_peak_lows.empty:
                anchor_low_idx = int(pre_peak_lows.values.argmin())
                anchor_low = float(pre_peak_lows.iloc[anchor_low_idx])
            elif low_swings:
                anchor_low = low_swings[0].price
                anchor_low_idx = low_swings[0].index
            else:
                anchor_low = float(df['low'].iloc[0])
                anchor_low_idx = 0

            impulse = max(peak_high - anchor_low, current_atr * 0.5)
            fib_50 = anchor_low + 0.50 * impulse
            is_disc = current_close < fib_50
            is_prem = current_close >= fib_50

            sd_40 = anchor_low - 0.50 * impulse
            sd_45 = anchor_low - 0.75 * impulse

            # Invalidation level: breakdown below Anchor Low & extreme discount extensions
            inval = anchor_low - max(current_atr * 0.5, 0.0005)

            # Find 1st Bullish FVG below 50%
            first_fvg_below = None
            start_fvg_idx = max(2, anchor_low_idx)
            for i in range(start_fvg_idx, n):
                c_curr = df.iloc[i]
                c_prev2 = df.iloc[i - 2]
                if c_curr['low'] > c_prev2['high']:
                    f_bot = float(c_prev2['high'])
                    f_top = float(c_curr['low'])
                    if f_top <= fib_50 or f_bot < fib_50:
                        if first_fvg_below is None or f_top > first_fvg_below[1]:
                            first_fvg_below = (f_bot, f_top)

            # Find 1st Liquidity area / swing low below 50%
            prior_lows_below = [sp.price for sp in low_swings if sp.price < fib_50]
            first_liq_below = max(prior_lows_below) if prior_lows_below else None

            # Rank candidate zones below 50%
            first_zone = None
            first_zone_type = "NONE"
            if first_fvg_below is not None:
                first_zone = first_fvg_below
                first_zone_type = "FIRST_FVG_<0.5"
            elif first_liq_below is not None:
                first_zone = (first_liq_below - current_atr * 0.25, first_liq_below + current_atr * 0.25)
                first_zone_type = "FIRST_LIQUIDITY_<0.5"
            elif sd_40 is not None:
                first_zone = (sd_45, sd_40)
                first_zone_type = "SD_4.0_4.5_EXHAUSTION"

            all_failed = current_close < inval

            res.update({
                "dealing_range_low": anchor_low,
                "dealing_range_high": peak_high,
                "fib_50": fib_50,
                "is_discount": is_disc,
                "is_premium": is_prem,
                "invalidation_level": inval,
                "first_zone_entry": first_zone,
                "first_zone_type": first_zone_type,
                "all_zones_failed": all_failed,
                "sd_40": sd_40,
                "sd_45": sd_45,
                "bias_override": MarketBias.BEARISH if all_failed else MarketBias.BULLISH,
            })
        else:
            # Bearish Dealing Range: Anchor High -> Trough Low
            if low_swings:
                trough_sp = min(low_swings[-3:], key=lambda sp: sp.price)
                trough_low = trough_sp.price
                trough_idx = trough_sp.index
            else:
                trough_idx = int(df['low'].argmin())
                trough_low = float(df['low'].iloc[trough_idx])

            abs_low_idx = int(df['low'].argmin())
            if float(df['low'].iloc[abs_low_idx]) < trough_low:
                trough_low = float(df['low'].iloc[abs_low_idx])
                trough_idx = abs_low_idx

            pre_trough_highs = df['high'].iloc[:max(1, trough_idx)]
            if not pre_trough_highs.empty:
                anchor_high_idx = int(pre_trough_highs.values.argmax())
                anchor_high = float(pre_trough_highs.iloc[anchor_high_idx])
            elif high_swings:
                anchor_high = high_swings[0].price
                anchor_high_idx = high_swings[0].index
            else:
                anchor_high = float(df['high'].iloc[0])
                anchor_high_idx = 0

            impulse = max(anchor_high - trough_low, current_atr * 0.5)
            fib_50 = trough_low + 0.50 * impulse
            is_disc = current_close < fib_50
            is_prem = current_close >= fib_50

            sd_40 = anchor_high + 0.50 * impulse
            sd_45 = anchor_high + 0.75 * impulse

            # Invalidation: breakout above Anchor High & extreme premium extensions
            inval = anchor_high + max(current_atr * 0.5, 0.0005)

            # Find 1st Bearish FVG above 50%
            first_fvg_above = None
            start_fvg_idx = max(2, anchor_high_idx)
            for i in range(start_fvg_idx, n):
                c_curr = df.iloc[i]
                c_prev2 = df.iloc[i - 2]
                if c_curr['high'] < c_prev2['low']:
                    f_bot = float(c_curr['high'])
                    f_top = float(c_prev2['low'])
                    if f_bot >= fib_50 or f_top > fib_50:
                        if first_fvg_above is None or f_bot < first_fvg_above[0]:
                            first_fvg_above = (f_bot, f_top)

            # Find 1st Liquidity area / swing high above 50%
            prior_highs_above = [sp.price for sp in high_swings if sp.price > fib_50]
            first_liq_above = min(prior_highs_above) if prior_highs_above else None

            first_zone = None
            first_zone_type = "NONE"
            if first_fvg_above is not None:
                first_zone = first_fvg_above
                first_zone_type = "FIRST_FVG_>0.5"
            elif first_liq_above is not None:
                first_zone = (first_liq_above - current_atr * 0.25, first_liq_above + current_atr * 0.25)
                first_zone_type = "FIRST_LIQUIDITY_>0.5"
            elif sd_40 is not None:
                first_zone = (sd_40, sd_45)
                first_zone_type = "SD_4.0_4.5_EXHAUSTION"

            all_failed = current_close > inval

            res.update({
                "dealing_range_low": trough_low,
                "dealing_range_high": anchor_high,
                "fib_50": fib_50,
                "is_discount": is_disc,
                "is_premium": is_prem,
                "invalidation_level": inval,
                "first_zone_entry": first_zone,
                "first_zone_type": first_zone_type,
                "all_zones_failed": all_failed,
                "sd_40": sd_40,
                "sd_45": sd_45,
                "bias_override": MarketBias.BULLISH if all_failed else MarketBias.BEARISH,
            })

        return res

    def analyze(self, df: pd.DataFrame) -> HTFAnalysis:
        """Run complete HTF macro trend & institutional liquidity analysis."""
        ema_series = self.compute_ema(df['close'])
        current_ema = float(ema_series.iloc[-1])
        current_close = float(df['close'].iloc[-1])

        swings = self.find_swing_points(df['high'], df['low'])
        structure_bias = self.detect_market_structure(swings)

        bias = MarketBias.NEUTRAL
        trend_clarity_score = 0.0

        is_above_ema = current_close > current_ema
        is_below_ema = current_close < current_ema

        if structure_bias == MarketBias.BULLISH and is_above_ema:
            bias = MarketBias.BULLISH
            trend_clarity_score = 30.0
        elif structure_bias == MarketBias.BEARISH and is_below_ema:
            bias = MarketBias.BEARISH
            trend_clarity_score = 30.0
        elif structure_bias == MarketBias.NEUTRAL and is_below_ema:
            # Check if EMA is rising (discount pullback in uptrend) vs falling (macro downtrend)
            ema_lookback = min(10, len(ema_series) - 1)
            ema_slope_rising = (current_ema >= float(ema_series.iloc[-1 - ema_lookback])) if ema_lookback > 0 else False
            if ema_slope_rising:
                bias = MarketBias.BULLISH
                trend_clarity_score = 20.0
            else:
                bias = MarketBias.BEARISH
                trend_clarity_score = 20.0
        elif structure_bias == MarketBias.NEUTRAL and is_above_ema:
            # Check if EMA is falling (premium pullback in downtrend) vs rising (macro uptrend)
            ema_lookback = min(10, len(ema_series) - 1)
            ema_slope_rising = (current_ema >= float(ema_series.iloc[-1 - ema_lookback])) if ema_lookback > 0 else True
            if not ema_slope_rising:
                bias = MarketBias.BEARISH
                trend_clarity_score = 20.0
            else:
                bias = MarketBias.BULLISH
                trend_clarity_score = 20.0
        elif structure_bias == MarketBias.BULLISH and is_below_ema:
            # Leading price action: Swing structure is making higher highs and higher lows (reversal/pullback rally)
            bias = MarketBias.BULLISH
            trend_clarity_score = 25.0
        elif structure_bias == MarketBias.BEARISH and is_above_ema:
            # Leading price action: Swing structure is making lower highs and lower lows (reversal/pullback decline)
            bias = MarketBias.BEARISH
            trend_clarity_score = 25.0
        else:
            bias = MarketBias.NEUTRAL
            trend_clarity_score = 10.0

        atr_series = compute_atr(df, 14)
        current_atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

        # Institutional Dealing Range & 0.5 Fib Equilibrium Rule
        dr_info = self.detect_dealing_range(
            df,
            swings,
            current_atr,
            current_close,
            structure_bias=structure_bias,
            base_bias=bias,
        )
        if dr_info["bias_override"] is not None:
            bias = dr_info["bias_override"]
            trend_clarity_score = max(trend_clarity_score, 25.0)

        liquidity_pools = self.find_liquidity_pools(swings, current_atr)
        sd_zones = self.find_supply_demand_zones(df, swings)

        last_swing_high = next((sp.price for sp in reversed(swings) if sp.is_high), None)
        last_swing_low = next((sp.price for sp in reversed(swings) if not sp.is_high), None)

        return HTFAnalysis(
            bias=bias,
            ema_value=current_ema,
            last_swing_high=last_swing_high,
            last_swing_low=last_swing_low,
            trend_clarity_score=trend_clarity_score,
            liquidity_pools=liquidity_pools,
            supply_demand_zones=sd_zones,
            dealing_range_low=dr_info["dealing_range_low"],
            dealing_range_high=dr_info["dealing_range_high"],
            fib_50=dr_info["fib_50"],
            is_discount=dr_info["is_discount"],
            is_premium=dr_info["is_premium"],
            invalidation_level=dr_info["invalidation_level"],
            first_zone_entry=dr_info["first_zone_entry"],
            first_zone_type=dr_info["first_zone_type"],
            all_zones_failed=dr_info["all_zones_failed"],
            sd_40=dr_info["sd_40"],
            sd_45=dr_info["sd_45"],
        )


# ─────────────────────────────────────────────
#  SMC Entry & Mitigation Detector (LTF)
# ─────────────────────────────────────────────

class SMCEntryDetector:
    """
    Implements Institutional Order Flow:
    1. Liquidity Sweep (Inducement)
    2. Displacement Candle
    3. Order Block / Fair Value Gap
    4. Mitigation Entry
    """

    def __init__(self, config: TimeframeConfig):
        self.config = config
        self.swing_lookback = config.swing_lookback
        self.ob_lookback = config.ob_lookback
        self.displacement_mult = config.displacement_atr_multiple
        self.fvg_min_gap = config.fvg_min_gap_atr
        self.sweep_buffer = config.sweep_buffer_atr

    def find_order_blocks(self, df: pd.DataFrame, displacement_idx: int, direction: Direction) -> list[OrderBlock]:
        """
        Scan back from displacement index to find the institutional Order Block:
        - Bullish OB: Last down-close candle prior to upward displacement.
        - Bearish OB: Last up-close candle prior to downward displacement.
        """
        obs: list[OrderBlock] = []
        start_scan = max(0, displacement_idx - self.ob_lookback)

        for i in range(displacement_idx - 1, start_scan - 1, -1):
            row = df.iloc[i]
            is_down_candle = row['close'] < row['open']
            is_up_candle = row['close'] > row['open']

            if direction == Direction.BUY and is_down_candle:
                obs.append(OrderBlock(
                    index=i,
                    direction=Direction.BUY,
                    low=float(row['low']),
                    high=float(row['high']),
                ))
                break  # Primary origin order block identified
            elif direction == Direction.SELL and is_up_candle:
                obs.append(OrderBlock(
                    index=i,
                    direction=Direction.SELL,
                    low=float(row['low']),
                    high=float(row['high']),
                ))
                break

        return obs

    def find_fair_value_gaps(self, df: pd.DataFrame, atr_val: float) -> list[FairValueGap]:
        """
        Detect Fair Value Gaps (FVGs) in the recent price history.
        - Bullish FVG: Low of candle i > High of candle i-2
        - Bearish FVG: High of candle i < Low of candle i-2
        """
        fvgs: list[FairValueGap] = []
        n = len(df)
        if n < 3:
            return fvgs

        min_gap_size = atr_val * self.fvg_min_gap

        # Scan recent 30 bars
        start_idx = max(2, n - 30)
        for i in range(start_idx, n):
            c_current = df.iloc[i]
            c_prev2 = df.iloc[i - 2]

            # Bullish FVG
            if c_current['low'] > c_prev2['high']:
                gap = float(c_current['low'] - c_prev2['high'])
                if gap >= min_gap_size:
                    fvgs.append(FairValueGap(
                        index=i,
                        direction=Direction.BUY,
                        bottom=float(c_prev2['high']),
                        top=float(c_current['low']),
                        size=gap,
                    ))

            # Bearish FVG
            elif c_current['high'] < c_prev2['low']:
                gap = float(c_prev2['low'] - c_current['high'])
                if gap >= min_gap_size:
                    fvgs.append(FairValueGap(
                        index=i,
                        direction=Direction.SELL,
                        bottom=float(c_current['high']),
                        top=float(c_prev2['low']),
                        size=gap,
                    ))

        return fvgs

    def detect_smc_entry(
        self,
        df: pd.DataFrame,
        htf_analysis: HTFAnalysis,
        instrument: InstrumentConfig,
        current_spread: float,
    ) -> dict | None:
        """
        Executes full SMC pipeline:
        1. Identify Liquidity Sweep of swing levels.
        2. Verify Displacement candle following the sweep.
        3. Extract Order Block and Fair Value Gap.
        4. Detect Mitigation retest at the current candle.
        """
        n = len(df)
        if n < 25:
            return None

        atr_series = compute_atr(df, 14)
        current_atr = float(atr_series.iloc[-1])
        if current_atr <= 0:
            return None

        swings = find_swing_points_logic(df['high'], df['low'], self.swing_lookback)
        current_candle = df.iloc[-1]
        current_close = float(current_candle['close'])
        current_low = float(current_candle['low'])
        current_high = float(current_candle['high'])
        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime('now', utc=True)

        bias = htf_analysis.bias
        sweep_detected = False
        swept_level = 0.0
        sweep_extreme = 0.0
        displacement_idx = -1
        conf_type = LTFConfirmation.NONE

        recent_window = min(15, n - 2)

        # ── Step 1 & 2: Sweep and Displacement Detection ──
        if bias == MarketBias.BULLISH:
            # Reversal / Pullback Guard: If LTF is in a strong established downtrend, do not buy into freefall
            if n >= 25:
                ltf_ema20 = df['close'].ewm(span=20, adjust=False).mean()
                ltf_ema50 = df['close'].ewm(span=50, adjust=False).mean()
                if float(current_close) < float(ltf_ema20.iloc[-1]) and float(ltf_ema20.iloc[-1]) < float(ltf_ema50.iloc[-1]) and (float(ltf_ema20.iloc[-1]) < float(ltf_ema20.iloc[-3])):
                    return None

            low_swings = [sp for sp in swings if not sp.is_high and sp.index < n - 1]
            if not low_swings:
                return None

            # Check if any recent candle swept a prior swing low
            for bar_offset in range(1, recent_window + 1):
                idx = n - 1 - bar_offset
                bar = df.iloc[idx]
                for sp in reversed(low_swings):
                    if sp.index >= idx:
                        continue
                    # Sweep condition: wick went below level, but closed above or immediately rejected
                    if bar['low'] < sp.price:
                        sweep_detected = True
                        swept_level = sp.price
                        sweep_extreme = float(bar['low'])
                        # Check displacement in bars following the sweep
                        for post_idx in range(idx, min(idx + 3, n)):
                            p_bar = df.iloc[post_idx]
                            body = p_bar['close'] - p_bar['open']
                            if body >= current_atr * self.displacement_mult:
                                displacement_idx = post_idx
                                break
                        break
                if sweep_detected:
                    break

            if not sweep_detected:
                return None

            # ── Step 3 & 4: Order Block, FVG, & Mitigation Retest ──
            obs = self.find_order_blocks(df, displacement_idx if displacement_idx > 0 else n - 1, Direction.BUY)
            fvgs = self.find_fair_value_gaps(df, current_atr)

            ob_touched = False
            fvg_touched = False
            target_ob = None

            if obs:
                target_ob = obs[0]
                # Price is currently mitigating the order block (current or previous candle)
                if current_low <= target_ob.high and current_close >= target_ob.low:
                    ob_touched = True
                elif n >= 2:
                    prev_c = df.iloc[-2]
                    if float(prev_c['low']) <= target_ob.high and float(prev_c['close']) >= target_ob.low:
                        ob_touched = True

            if fvgs:
                for fvg in reversed(fvgs):
                    if fvg.direction == Direction.BUY:
                        if current_low <= fvg.top and current_close >= fvg.bottom:
                            fvg_touched = True
                            break
                        elif n >= 2:
                            prev_c = df.iloc[-2]
                            if float(prev_c['low']) <= fvg.top and float(prev_c['close']) >= fvg.bottom:
                                fvg_touched = True
                                break

            if ob_touched and fvg_touched:
                conf_type = LTFConfirmation.OB_PLUS_FVG
            elif ob_touched:
                conf_type = LTFConfirmation.OB_MITIGATION
            elif fvg_touched:
                conf_type = LTFConfirmation.FVG_MITIGATION
            else:
                # Direct sweep reversal if strong reaction candle
                if current_close > swept_level and current_candle['close'] > current_candle['open']:
                    conf_type = LTFConfirmation.LIQUIDITY_SWEEP
                else:
                    return None

            # Directional Candle Confirmation: Must be GREEN candle with >= 50% body (<= 50% wick)
            min_body_ratio = getattr(self.config, 'min_candle_body_ratio', 0.50)
            is_confirmed, body_ratio, conf_desc = check_candle_body_confirmation(
                current_candle,
                Direction.BUY,
                min_body_ratio=min_body_ratio,
            )
            if not is_confirmed:
                return None

            # Stop-loss tight beyond invalidation point (sweep extreme) with minimum safety floor
            sl_buffer = max(1.0 * instrument.pip_size, current_atr * 0.1)
            stop_loss = min(sweep_extreme, swept_level) - sl_buffer
            min_sl_dist = max(current_atr * 0.8, 15.0 * instrument.pip_size if ("XAU" in instrument.symbol or "GOLD" in instrument.symbol) else 5.0 * instrument.pip_size)
            if abs(current_close - stop_loss) < min_sl_dist:
                stop_loss = current_close - min_sl_dist

            # Target liquidity pool or HTF supply zone (strictly clamped to [1.5, 3.0] R:R)
            take_profit = 0.0
            opposing_pools = [p.level for p in htf_analysis.liquidity_pools if p.is_high and p.level > current_close]
            opposing_zones = [z.bottom for z in htf_analysis.supply_demand_zones if z.is_supply and z.bottom > current_close]

            if opposing_pools:
                raw_tp = min(opposing_pools)
            elif opposing_zones:
                raw_tp = min(opposing_zones)
            elif htf_analysis.last_swing_high and htf_analysis.last_swing_high > current_close:
                raw_tp = htf_analysis.last_swing_high
            else:
                raw_tp = current_close + (current_close - stop_loss) * 2.0

            take_profit, _ = clamp_tp_to_rr(current_close, stop_loss, raw_tp, Direction.BUY, min_rr=1.5, max_rr=3.0)

            return {
                'direction': Direction.BUY,
                'entry': current_close,
                'sl': stop_loss,
                'tp': take_profit,
                'conf': conf_type,
                'timestamp': timestamp,
                'sweep_depth': swept_level - sweep_extreme,
                'candle_confirmation': conf_desc,
            }

        elif bias == MarketBias.BEARISH:
            # Reversal / Pullback Guard: If LTF is in a strong established uptrend, do not sell into active bull momentum
            if n >= 25:
                ltf_ema20 = df['close'].ewm(span=20, adjust=False).mean()
                ltf_ema50 = df['close'].ewm(span=50, adjust=False).mean()
                if float(current_close) > float(ltf_ema20.iloc[-1]) and float(ltf_ema20.iloc[-1]) > float(ltf_ema50.iloc[-1]) and (float(ltf_ema20.iloc[-1]) > float(ltf_ema20.iloc[-3])):
                    return None

            high_swings = [sp for sp in swings if sp.is_high and sp.index < n - 1]
            if not high_swings:
                return None

            for bar_offset in range(1, recent_window + 1):
                idx = n - 1 - bar_offset
                bar = df.iloc[idx]
                for sp in reversed(high_swings):
                    if sp.index >= idx:
                        continue
                    if bar['high'] > sp.price:
                        sweep_detected = True
                        swept_level = sp.price
                        sweep_extreme = float(bar['high'])
                        for post_idx in range(idx, min(idx + 3, n)):
                            p_bar = df.iloc[post_idx]
                            body = p_bar['open'] - p_bar['close']
                            if body >= current_atr * self.displacement_mult:
                                displacement_idx = post_idx
                                break
                        break
                if sweep_detected:
                    break

            if not sweep_detected:
                return None

            obs = self.find_order_blocks(df, displacement_idx if displacement_idx > 0 else n - 1, Direction.SELL)
            fvgs = self.find_fair_value_gaps(df, current_atr)

            ob_touched = False
            fvg_touched = False

            if obs:
                target_ob = obs[0]
                if current_high >= target_ob.low and current_close <= target_ob.high:
                    ob_touched = True
                elif n >= 2:
                    prev_c = df.iloc[-2]
                    if float(prev_c['high']) >= target_ob.low and float(prev_c['close']) <= target_ob.high:
                        ob_touched = True

            if fvgs:
                for fvg in reversed(fvgs):
                    if fvg.direction == Direction.SELL:
                        if current_high >= fvg.bottom and current_close <= fvg.top:
                            fvg_touched = True
                            break
                        elif n >= 2:
                            prev_c = df.iloc[-2]
                            if float(prev_c['high']) >= fvg.bottom and float(prev_c['close']) <= fvg.top:
                                fvg_touched = True
                                break

            if ob_touched and fvg_touched:
                conf_type = LTFConfirmation.OB_PLUS_FVG
            elif ob_touched:
                conf_type = LTFConfirmation.OB_MITIGATION
            elif fvg_touched:
                conf_type = LTFConfirmation.FVG_MITIGATION
            else:
                if current_close < swept_level and current_candle['close'] < current_candle['open']:
                    conf_type = LTFConfirmation.LIQUIDITY_SWEEP
                else:
                    return None

            # Directional Candle Confirmation: Must be RED candle with >= 50% body (<= 50% wick)
            min_body_ratio = getattr(self.config, 'min_candle_body_ratio', 0.50)
            is_confirmed, body_ratio, conf_desc = check_candle_body_confirmation(
                current_candle,
                Direction.SELL,
                min_body_ratio=min_body_ratio,
            )
            if not is_confirmed:
                return None

            # Stop-loss tight beyond invalidation point (sweep extreme) with minimum safety floor
            sl_buffer = max(1.0 * instrument.pip_size, current_atr * 0.1)
            stop_loss = max(sweep_extreme, swept_level) + sl_buffer
            min_sl_dist = max(current_atr * 0.8, 15.0 * instrument.pip_size if ("XAU" in instrument.symbol or "GOLD" in instrument.symbol) else 5.0 * instrument.pip_size)
            if abs(current_close - stop_loss) < min_sl_dist:
                stop_loss = current_close + min_sl_dist

            take_profit = 0.0
            opposing_pools = [p.level for p in htf_analysis.liquidity_pools if not p.is_high and p.level < current_close]
            opposing_zones = [z.top for z in htf_analysis.supply_demand_zones if not z.is_supply and z.top < current_close]

            if opposing_pools:
                raw_tp = max(opposing_pools)
            elif opposing_zones:
                raw_tp = max(opposing_zones)
            elif htf_analysis.last_swing_low and htf_analysis.last_swing_low < current_close:
                raw_tp = htf_analysis.last_swing_low
            else:
                raw_tp = current_close - (stop_loss - current_close) * 2.0

            take_profit, _ = clamp_tp_to_rr(current_close, stop_loss, raw_tp, Direction.SELL, min_rr=1.5, max_rr=3.0)

            return {
                'direction': Direction.SELL,
                'entry': current_close,
                'sl': stop_loss,
                'tp': take_profit,
                'conf': conf_type,
                'timestamp': timestamp,
                'sweep_depth': sweep_extreme - swept_level,
                'candle_confirmation': conf_desc,
            }

        return None


# ─────────────────────────────────────────────
#  5-Minute Order Block (OB) Scalp Engine (SMC)
# ─────────────────────────────────────────────

class SMCScalp5MEngine:
    """
    5-Minute Order Block (OB) Scalp Strategy (SMC).
    
    Setup: Establish directional bias on 1-Hour (1H) chart.
    BOS: Wait for a Break of Structure (BOS) on the 5-minute (5M) chart in that same direction.
    Order Block: The specific origin candle that caused this 5M BOS.
    Entry: Retest of this Order Block.
    Stop-Loss: Just beyond the wick of the OB candle (or user-defined fixed SL).
    Target: 1.5R reward for partial profit, with potential runner to next 5M liquidity pool.
    Session: London and New York AM sessions (07:00 - 16:00 UTC).
    """

    def __init__(
        self,
        swing_lookback: int = 3,
        session_start_utc: int = 7,
        session_end_utc: int = 16,
        target_rr: float = 1.5,
        min_candle_body_ratio: float = 0.50,
    ):
        self.swing_lookback = swing_lookback
        self.session_start_utc = session_start_utc
        self.session_end_utc = session_end_utc
        self.target_rr = target_rr
        self.min_candle_body_ratio = min_candle_body_ratio

    def check_trading_session(self, current_time: datetime) -> bool:
        """Check if timestamp is within London or NY AM high-volume sessions (07:00 - 16:00 UTC)."""
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        hour = current_time.hour
        return self.session_start_utc <= hour < self.session_end_utc

    def detect_scalp_entry(
        self,
        df: pd.DataFrame,
        htf_analysis: HTFAnalysis,
        instrument: InstrumentConfig,
        current_spread: float,
        fixed_sl_pips: float | None = None,
        enforce_session: bool = False,
    ) -> dict | None:
        """
        Scans 5M OHLCV data for:
        1. Trading session validation (07:00 - 16:00 UTC).
        2. Break of Structure (BOS) in the direction of HTF bias.
        3. Origin Order Block (OB) candle identification.
        4. Current bar retest of the OB price zone [low, high].
        5. Invalidation stop loss beyond the OB wick and 1.5R target.
        """
        n = len(df)
        if n < 10:
            return None

        bias = htf_analysis.bias
        if bias not in (MarketBias.BULLISH, MarketBias.BEARISH):
            return None

        current_bar = df.iloc[-1]
        timestamp = current_bar['time'] if 'time' in df.columns else datetime.now(timezone.utc)
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # 1. Trading Session Validation
        if enforce_session and not self.check_trading_session(timestamp):
            logger.debug(f"[SCALP_5M] Outside London/NY AM active session ({timestamp.strftime('%H:%M UTC')}). Skipping.")
            return None

        atr_series = compute_atr(df, period=14)
        current_atr = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 10 * instrument.pip_size

        swings = find_swing_points_logic(df['high'], df['low'], lookback=self.swing_lookback)
        if not swings:
            return None

        current_close = float(current_bar['close'])
        current_low = float(current_bar['low'])
        current_high = float(current_bar['high'])

        # Scan for the most recent 5M BOS within the last 15 bars
        lookback_window = min(15, n - 2)

        if bias == MarketBias.BULLISH:
            if n >= 20:
                ema9 = df['close'].ewm(span=9, adjust=False).mean()
                ema21 = df['close'].ewm(span=21, adjust=False).mean()
                if float(current_close) < float(ema9.iloc[-1]) and float(ema9.iloc[-1]) < float(ema21.iloc[-1]) and (float(ema9.iloc[-1]) < float(ema9.iloc[-2])):
                    return None

            swing_highs = [sp for sp in swings if sp.is_high and sp.index < n - 1]
            if not swing_highs:
                return None

            bos_detected = False
            bos_bar_idx = -1
            broken_swing_price = 0.0

            for offset in range(1, lookback_window + 1):
                idx = n - 1 - offset
                bar = df.iloc[idx]
                bar_close = float(bar['close'])
                for sp in reversed(swing_highs):
                    if sp.index < idx and bar_close > sp.price:
                        bos_detected = True
                        bos_bar_idx = idx
                        broken_swing_price = sp.price
                        break
                if bos_detected:
                    break

            if not bos_detected or bos_bar_idx <= 0:
                return None

            # Identify Bullish Order Block (OB):
            # The last down/bearish candle preceding the upward displacement that broke structure.
            ob_idx = -1
            search_start = max(0, bos_bar_idx - 8)
            for j in range(bos_bar_idx - 1, search_start - 1, -1):
                c = df.iloc[j]
                if float(c['close']) <= float(c['open']):
                    ob_idx = j
                    break

            if ob_idx == -1:
                search_slice = df.iloc[search_start:bos_bar_idx]
                ob_idx = search_start + int(np.argmin(search_slice['low'].to_numpy(dtype=float)))

            ob_candle = df.iloc[ob_idx]
            ob_high = float(ob_candle['high'])
            ob_low = float(ob_candle['low'])

            # Retest Check: Price pulls back into the OB zone [ob_low, ob_high]
            retest_confirmed = False
            if current_low <= ob_high and current_close >= ob_low:
                retest_confirmed = True
            else:
                prev_bar = df.iloc[-2]
                if float(prev_bar['low']) <= ob_high and current_close >= ob_low:
                    retest_confirmed = True

            if not retest_confirmed:
                return None

            # Directional Candle Confirmation: Must be Green candle with >= 50% body (<= 50% wick)
            min_body_ratio = getattr(self, 'min_candle_body_ratio', 0.50)
            is_confirmed, body_ratio, conf_desc = check_candle_body_confirmation(
                current_bar,
                Direction.BUY,
                min_body_ratio=min_body_ratio,
            )
            if not is_confirmed:
                return None

            # Invalidation / Stop Loss: Just beyond the wick of the OB candle
            sl_buffer = max(1.0 * instrument.pip_size, current_atr * 0.1)
            stop_loss = ob_low - sl_buffer
            entry_price = current_close

            if fixed_sl_pips is not None and fixed_sl_pips > 0:
                sl_distance = fixed_sl_pips * instrument.pip_size
                stop_loss = entry_price - sl_distance
            else:
                sl_distance = abs(entry_price - stop_loss)

            min_buffer = max(1.0 * instrument.pip_size, 12.0 * instrument.pip_size if ("XAU" in instrument.symbol or "GOLD" in instrument.symbol) else 4.0 * instrument.pip_size)
            if sl_distance < min_buffer:
                sl_distance = min_buffer
                stop_loss = entry_price - sl_distance

            # Take Profit: 1.5R target for partial profit
            take_profit = entry_price + (sl_distance * self.target_rr)
            take_profit, _ = clamp_tp_to_rr(entry_price, stop_loss, take_profit, Direction.BUY, min_rr=1.5, max_rr=3.0)

            # Potential runner target to the next 5M liquidity pool (clamped to max 3.0R)
            opposing_pools = [sp.price for sp in swing_highs if sp.price > take_profit]
            raw_runner = min(opposing_pools) if opposing_pools else take_profit
            runner_tp, _ = clamp_tp_to_rr(entry_price, stop_loss, raw_runner, Direction.BUY, min_rr=1.5, max_rr=3.0)

            return {
                'direction': Direction.BUY,
                'entry': entry_price,
                'sl': stop_loss,
                'tp': take_profit,
                'runner_tp': runner_tp,
                'conf': LTFConfirmation.OB_SCALP_5M,
                'timestamp': timestamp,
                'ob_high': ob_high,
                'ob_low': ob_low,
                'bos_price': broken_swing_price,
                'candle_confirmation': conf_desc,
            }

        elif bias == MarketBias.BEARISH:
            if n >= 20:
                ema9 = df['close'].ewm(span=9, adjust=False).mean()
                ema21 = df['close'].ewm(span=21, adjust=False).mean()
                if float(current_close) > float(ema9.iloc[-1]) and float(ema9.iloc[-1]) > float(ema21.iloc[-1]) and (float(ema9.iloc[-1]) > float(ema9.iloc[-2])):
                    return None

            swing_lows = [sp for sp in swings if not sp.is_high and sp.index < n - 1]
            if not swing_lows:
                return None

            bos_detected = False
            bos_bar_idx = -1
            broken_swing_price = 0.0

            for offset in range(1, lookback_window + 1):
                idx = n - 1 - offset
                bar = df.iloc[idx]
                bar_close = float(bar['close'])
                for sp in reversed(swing_lows):
                    if sp.index < idx and bar_close < sp.price:
                        bos_detected = True
                        bos_bar_idx = idx
                        broken_swing_price = sp.price
                        break
                if bos_detected:
                    break

            if not bos_detected or bos_bar_idx <= 0:
                return None

            # Identify Bearish Order Block (OB):
            # The last up/bullish candle preceding the downward displacement that broke structure.
            ob_idx = -1
            search_start = max(0, bos_bar_idx - 8)
            for j in range(bos_bar_idx - 1, search_start - 1, -1):
                c = df.iloc[j]
                if float(c['close']) >= float(c['open']):
                    ob_idx = j
                    break

            if ob_idx == -1:
                search_slice = df.iloc[search_start:bos_bar_idx]
                ob_idx = search_start + int(np.argmax(search_slice['high'].to_numpy(dtype=float)))

            ob_candle = df.iloc[ob_idx]
            ob_high = float(ob_candle['high'])
            ob_low = float(ob_candle['low'])

            # Retest Check: Price pulls back up into the OB zone [ob_low, ob_high]
            retest_confirmed = False
            if current_high >= ob_low and current_close <= ob_high:
                retest_confirmed = True
            else:
                prev_bar = df.iloc[-2]
                if float(prev_bar['high']) >= ob_low and current_close <= ob_high:
                    retest_confirmed = True

            if not retest_confirmed:
                return None

            # Directional Candle Confirmation: Must be Red candle with >= 50% body (<= 50% wick)
            min_body_ratio = getattr(self, 'min_candle_body_ratio', 0.50)
            is_confirmed, body_ratio, conf_desc = check_candle_body_confirmation(
                current_bar,
                Direction.SELL,
                min_body_ratio=min_body_ratio,
            )
            if not is_confirmed:
                return None

            # Invalidation / Stop Loss: Just beyond the wick of the OB candle
            sl_buffer = max(1.0 * instrument.pip_size, current_atr * 0.1)
            stop_loss = ob_high + sl_buffer
            entry_price = current_close

            if fixed_sl_pips is not None and fixed_sl_pips > 0:
                sl_distance = fixed_sl_pips * instrument.pip_size
                stop_loss = entry_price + sl_distance
            else:
                sl_distance = abs(stop_loss - entry_price)

            min_buffer = max(1.0 * instrument.pip_size, 12.0 * instrument.pip_size if ("XAU" in instrument.symbol or "GOLD" in instrument.symbol) else 4.0 * instrument.pip_size)
            if sl_distance < min_buffer:
                sl_distance = min_buffer
                stop_loss = entry_price + sl_distance

            # Take Profit: 1.5R target for partial profit
            take_profit = entry_price - (sl_distance * self.target_rr)
            take_profit, _ = clamp_tp_to_rr(entry_price, stop_loss, take_profit, Direction.SELL, min_rr=1.5, max_rr=3.0)

            # Potential runner target to the next 5M liquidity pool (clamped to max 3.0R)
            opposing_pools = [sp.price for sp in swing_lows if sp.price < take_profit]
            raw_runner = max(opposing_pools) if opposing_pools else take_profit
            runner_tp, _ = clamp_tp_to_rr(entry_price, stop_loss, raw_runner, Direction.SELL, min_rr=1.5, max_rr=3.0)

            return {
                'direction': Direction.SELL,
                'entry': entry_price,
                'sl': stop_loss,
                'tp': take_profit,
                'runner_tp': runner_tp,
                'conf': LTFConfirmation.OB_SCALP_5M,
                'timestamp': timestamp,
                'ob_high': ob_high,
                'ob_low': ob_low,
                'bos_price': broken_swing_price,
                'candle_confirmation': conf_desc,
            }

        return None


# ─────────────────────────────────────────────
#  ICT (Inner Circle Trader) Strategy Engine
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
#  ICT (Inner Circle Trader) Time & Price Theory & AOX Model
# ─────────────────────────────────────────────

def to_ny_time(ts: datetime | pd.Timestamp) -> datetime:
    """Convert any UTC, ISO, or naive datetime into New York Local Time (America/New_York - EST/EDT)."""
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ny_tz = ZoneInfo("America/New_York")
    return ts.astimezone(ny_tz)


@dataclass
class ICTOpeningRanges:
    date: date
    tdo: float | None = None                 # 00:00 EST True Day Open (NYMO)
    mor_high: float | None = None            # Highest High strictly between 00:01 and 00:29 EST
    mor_low: float | None = None             # Lowest Low strictly between 00:01 and 00:29 EST
    mor_mid: float | None = None             # Midline (Consequent Encroachment) of MOR
    ny_pre_market_open: float | None = None  # 08:30 EST Open
    nyse_open: float | None = None           # 09:30 EST Open
    nyse_close: float | None = None          # 16:00 EST Close
    fpfvg: dict | None = None                # First Presented FVG (00:00 - 00:30 EST)


@dataclass
class SDProjections:
    anchor_high: float
    anchor_low: float
    direction: Direction
    range_size: float
    ce_midline: float
    # Standard Deviation zones (-1.0/-1.5, -2.0/-2.5, -4.0/-4.5 SD)
    sd_1_0: float
    sd_1_5: float
    sd_2_0: float
    sd_2_5: float
    sd_4_0: float
    sd_4_5: float


@dataclass
class AOXFibLevels:
    manipulation_high: float
    manipulation_low: float
    direction: Direction
    range_size: float
    mean_threshold: float
    # Sniper Entry Zone (Stop-Hunt Extension): 0, -0.21, -0.255, -0.29
    sniper_zone_entry: float      # -0.21 level
    sniper_zone_mid: float        # -0.255 level
    sniper_zone_max: float        # -0.29 level
    # Partial Take-Profit (TP1 - Majority Book): 1.47, 1.55
    tp1_min: float
    tp1_max: float
    tp1_mid: float
    # Final Take-Profit (TP2 - Full Book): 2.56, 2.60, 2.64
    tp2_min: float
    tp2_mid: float
    tp2_max: float


class ICTKillZone(str, Enum):
    """ICT Defined Trading Kill Zones & Algorithmic Macros (Anchored to NY Local Time)."""
    # Classical / UTC sessions
    LONDON_OPEN = "LONDON_KILLZONE"
    NY_AM = "NY_AM_KILLZONE"
    SILVER_BULLET = "NY_SILVER_BULLET"
    LONDON_CLOSE = "LONDON_CLOSE_KILLZONE"
    ASIA = "ASIAN_SESSION"
    # Core NY Time Windows & Macros
    LONDON_MACRO = "LONDON_MACRO_0233_0300"             # 02:33 - 03:00 EST
    LONDON_HEAVY_DIST = "LONDON_DISTRIBUTION_0300_0330" # 03:00 - 03:30 EST
    NY_PRE_MARKET = "NY_PRE_MARKET_0830"               # 08:30 EST
    NYSE_OPEN = "NYSE_OPEN_0930"                       # 09:30 EST
    NY_AM_MACRO_1 = "NY_AM_MACRO_0850_0910"           # 08:50 - 09:10 EST
    NY_AM_MACRO_2 = "NY_AM_MACRO_0950_1010"           # 09:50 - 10:10 EST
    NY_AM_MACRO_3 = "NY_AM_MACRO_1050_1110"           # 10:50 - 11:10 EST
    NY_LUNCH_MACRO = "NY_LUNCH_MACRO_1150_1210"       # 11:50 - 12:10 EST
    HOURLY_MACRO = "HOURLY_MACRO_XX50_XX10"           # XX:50 - (XX+1):10
    NONE = "NONE"


class ICTEngine:
    """
    Institutional ICT (Inner Circle Trader) Time & Price Theory and AOX Model Engine.

    Pillars:
    1. Core Time Windows (New York Time - EST/EDT):
       - True Day Open (TDO): 00:00 EST.
       - Midnight Opening Range (MOR): 00:00 to 00:30 EST (High/Low strictly 00:01-00:29 EST).
       - NY Pre-Market Open: 08:30 EST.
       - NYSE Official Open: 09:30 EST (closes at 16:00 EST).
       - Algorithmic Macros:
         * London Macros: 02:33 - 03:00 EST
         * London Heavy Distribution: 03:00 - 03:30 EST
         * NY AM Macros: 08:50-09:10, 09:50-10:10, 10:50-11:10 EST
         * NY Lunch Macro: 11:50-12:10 EST
         * Hourly Macro: XX:50 to (XX+1):10
    2. Market Bias, Premium/Discount & First Presented FVG (FPFVG):
       - Standard Premium (> 00:00 EST Open) vs Discount (< 00:00 EST Open).
       - Deep Premium (> TDO, > 08:30, > 09:30) vs Deep Discount (< TDO, < 08:30, < 09:30).
       - FPFVG: First FVG in 00:00-00:30 EST with strict 3-consecutive displacement candle closure.
       - Magnet Rule: TDO and unmitigated FPFVG act as magnets for mean-reversion snapbacks.
    3. Standard Deviation (SD) Projections on MOR and 08:30/09:30 manipulation legs:
       - -1.0 to -1.5 SD (Silver Bullet Zone)
       - -2.0 to -2.5 SD (Minor Retracement Zone)
       - -4.0 to -4.5 SD (Major Reversal Zone - 90% daily expansion exhaustion)
    4. 08:30 & 09:30 EST PO3 Manipulation Logic (Accumulation -> Manipulation -> Distribution).
    5. AOX Model (Algorithmic Order Execution):
       - Plotted on manipulation legs into SD zones or HTF POIs (4H/1H OB, FVG, Liquidity Voids).
       - Custom Fib: Sniper Entry (-0.21, -0.255, -0.29), TP1 (1.47, 1.55), TP2 (2.56, 2.60, 2.64).
       - 3-Entry Sequence:
         * Entry 1: Sniper Tap at POI MT / AOX Fib -0.21 to -0.29 (25% Risk)
         * Entry 2: SMT Divergence + Turtle Soup into First FVG (50% Risk)
         * Entry 3: HIPPO / Final Mitigation Block (50% Risk)
    6. Position Sizing & Risk Management (25-25-50 Layered Model):
       - Layered risk allocation: 25% on tap, 25% on spike into MT/-0.29, 50% on confirmation.
       - Break-Even at 1:1 RR or after 80-90 pips on Gold.
       - Partials at 1:1 to 1:2 RR (or Fib 1.47-1.55) with runner riding to 2.56-2.64 Fib / -2.5 or -4.5 SD.
       - Strict daily quota (stop trading after 90+ pips).
    """

    def __init__(self, config: ICTConfig | None = None):
        self.config = config or ICTConfig()

    def identify_kill_zone(self, current_time: datetime, include_hourly_macro: bool = False) -> ICTKillZone:
        """
        Identify active ICT session Kill Zone or Algorithmic Macro window.
        Anchored to New York Local Time (EST/EDT) while remaining backward compatible with UTC callers.
        """
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        ny_dt = to_ny_time(current_time)
        ny_hour = ny_dt.hour
        ny_min = ny_dt.minute
        ny_time_val = ny_hour * 60 + ny_min

        # 1. High-Priority NY Macros
        if (2 * 60 + 33) <= ny_time_val < (3 * 60):
            return ICTKillZone.LONDON_MACRO
        if (3 * 60) <= ny_time_val < (3 * 60 + 30):
            return ICTKillZone.LONDON_HEAVY_DIST
        if (8 * 60 + 50) <= ny_time_val < (9 * 60 + 10):
            return ICTKillZone.NY_AM_MACRO_1
        if (9 * 60 + 50) <= ny_time_val < (10 * 60 + 10):
            return ICTKillZone.NY_AM_MACRO_2
        if (10 * 60 + 50) <= ny_time_val < (11 * 60 + 10):
            return ICTKillZone.NY_AM_MACRO_3
        if (11 * 60 + 50) <= ny_time_val < (12 * 60 + 10):
            return ICTKillZone.NY_LUNCH_MACRO

        # 2. Standard / Legacy Kill Zones (UTC or NY hours)
        utc_dt = current_time.astimezone(timezone.utc)
        utc_hour = utc_dt.hour
        if self.config.silver_bullet_start_utc <= utc_hour < self.config.silver_bullet_end_utc:
            return ICTKillZone.SILVER_BULLET
        elif self.config.london_kz_start_utc <= utc_hour < self.config.london_kz_end_utc:
            return ICTKillZone.LONDON_OPEN
        elif self.config.ny_am_kz_start_utc <= utc_hour < self.config.ny_am_kz_end_utc:
            return ICTKillZone.NY_AM
        elif self.config.london_close_start_utc <= utc_hour < self.config.london_close_end_utc:
            return ICTKillZone.LONDON_CLOSE
        elif 0 <= utc_hour < 6:
            return ICTKillZone.ASIA

        # 3. Hourly Macro: XX:50 to (XX+1):10
        if include_hourly_macro and (ny_min >= 50 or ny_min <= 10):
            return ICTKillZone.HOURLY_MACRO

        return ICTKillZone.NONE

    def get_active_killzone(self, current_time: datetime) -> str | None:
        """Identify which ICT Kill Zone / Macro is active (string name)."""
        kz = self.identify_kill_zone(current_time)
        if kz not in (ICTKillZone.NONE, ICTKillZone.ASIA):
            return kz.value
        return None

    def get_active_macro(self, current_time: datetime) -> tuple[bool, str]:
        """Check if current time is within an algorithmic macro or distribution window."""
        ny_dt = to_ny_time(current_time)
        ny_min = ny_dt.minute
        ny_time_val = ny_dt.hour * 60 + ny_min

        if (2 * 60 + 33) <= ny_time_val < (3 * 60):
            return True, "LONDON_MACRO_0233_0300"
        if (3 * 60) <= ny_time_val < (3 * 60 + 30):
            return True, "LONDON_DISTRIBUTION_0300_0330"
        if (8 * 60 + 30) <= ny_time_val < (8 * 60 + 50):
            return True, "NY_PRE_MARKET_0830"
        if (8 * 60 + 50) <= ny_time_val < (9 * 60 + 10):
            return True, "NY_AM_MACRO_0850_0910"
        if (9 * 60 + 30) <= ny_time_val < (9 * 60 + 50):
            return True, "NYSE_OPEN_0930"
        if (9 * 60 + 50) <= ny_time_val < (10 * 60 + 10):
            return True, "NY_AM_MACRO_0950_1010"
        if (10 * 60 + 50) <= ny_time_val < (11 * 60 + 10):
            return True, "NY_AM_MACRO_1050_1110"
        if (11 * 60 + 50) <= ny_time_val < (12 * 60 + 10):
            return True, "NY_LUNCH_MACRO_1150_1210"
        if ny_min >= 50 or ny_min <= 10:
            return True, f"HOURLY_MACRO_{ny_dt.hour:02d}50_{((ny_dt.hour+1)%24):02d}10"

        return False, "NONE"

    def calculate_opening_ranges(self, df: pd.DataFrame, target_date: date | None = None) -> ICTOpeningRanges:
        """
        Calculate key ICT reference openings anchored strictly to New York Local Time (EST/EDT):
        1. True Day Open (TDO): exact open price of 00:00 EST 1-min candle.
        2. Midnight Opening Range (MOR): 00:00 to 00:30 EST. High & Low strictly from 00:01 to 00:29 EST.
        3. NY Pre-Market Open: 08:30 EST open.
        4. NYSE Official Open: 09:30 EST open.
        5. First Presented FVG (FPFVG): First 3-candle displacement FVG in 00:00-00:30 EST.
        """
        if df.empty:
            return ICTOpeningRanges(date=target_date or date.today())

        # Determine timestamp series converted to NY timezone
        if 'time' in df.columns:
            times_ny = [to_ny_time(t) for t in df['time']]
        else:
            times_ny = [to_ny_time(t) if isinstance(t, (datetime, pd.Timestamp)) else datetime.now(ZoneInfo("America/New_York")) for t in df.index]

        if target_date is None:
            target_date = times_ny[-1].date()

        day_indices = [i for i, t in enumerate(times_ny) if t.date() == target_date]
        if not day_indices:
            # Fallback to recent bars
            day_indices = list(range(len(df)))

        df_day = df.iloc[day_indices]
        day_times = [times_ny[i] for i in day_indices]

        tdo = None
        mor_high = None
        mor_low = None
        p_0830 = None
        p_0930 = None
        p_1600 = None

        # 1. True Day Open (00:00 EST)
        for i, t in enumerate(day_times):
            if t.hour == 0 and t.minute == 0:
                tdo = float(df_day.iloc[i]['open'])
                break
        if tdo is None and len(df_day) > 0:
            # Fallback: first candle at or after 00:00 EST
            tdo = float(df_day.iloc[0]['open'])

        # 2. Midnight Opening Range (00:01 - 00:29 EST)
        mor_sub = [
            df_day.iloc[i] for i, t in enumerate(day_times)
            if t.hour == 0 and (self.config.mor_start_minute <= t.minute <= self.config.mor_end_minute)
        ]
        if mor_sub:
            mor_high = max(float(r['high']) for r in mor_sub)
            mor_low = min(float(r['low']) for r in mor_sub)
        else:
            # Fallback if 00:01-00:29 bars not explicitly segmented
            first_n = min(10, len(df_day))
            mor_high = float(df_day.iloc[:first_n]['high'].max())
            mor_low = float(df_day.iloc[:first_n]['low'].min())

        mor_mid = (mor_high + mor_low) / 2.0 if (mor_high is not None and mor_low is not None) else None

        # 3. NY Pre-Market Open (08:30 EST)
        for i, t in enumerate(day_times):
            if t.hour == 8 and 30 <= t.minute <= 35:
                p_0830 = float(df_day.iloc[i]['open'])
                break

        # 4. NYSE Official Open (09:30 EST)
        for i, t in enumerate(day_times):
            if t.hour == 9 and 30 <= t.minute <= 35:
                p_0930 = float(df_day.iloc[i]['open'])
                break

        # 5. NYSE Close (16:00 EST)
        for i, t in enumerate(day_times):
            if t.hour == 16 and t.minute <= 5:
                p_1600 = float(df_day.iloc[i]['close'])
                break

        # 6. First Presented FVG (00:00 - 00:30 EST)
        fpfvg = self.detect_fpfvg(df, target_date=target_date)

        return ICTOpeningRanges(
            date=target_date,
            tdo=tdo,
            mor_high=mor_high,
            mor_low=mor_low,
            mor_mid=mor_mid,
            ny_pre_market_open=p_0830,
            nyse_open=p_0930,
            nyse_close=p_1600,
            fpfvg=fpfvg,
        )

    def detect_fpfvg(self, df: pd.DataFrame, target_date: date | None = None) -> dict | None:
        """
        Scan 1-minute chart within 00:00 - 00:30 EST for the VERY FIRST Fair Value Gap (FPFVG).
        Strict Validation Rule: ONLY valid if formed by 3 consecutive displacement candles
        where candles close decisively beyond previous candle's high (for bullish) or low (for bearish).
        """
        n = len(df)
        if n < 3:
            return None

        if 'time' in df.columns:
            times_ny = [to_ny_time(t) for t in df['time']]
        else:
            times_ny = [to_ny_time(t) if isinstance(t, (datetime, pd.Timestamp)) else datetime.now(ZoneInfo("America/New_York")) for t in df.index]

        if target_date is None:
            target_date = times_ny[-1].date()

        mor_indices = [
            i for i, t in enumerate(times_ny)
            if t.date() == target_date and (t.hour == 0 and t.minute <= 30)
        ]
        if len(mor_indices) < 3:
            # Fallback to initial bars of dataframe if date-based filtering yields too few bars
            mor_indices = list(range(min(30, n)))

        # Scan for the very first valid FVG
        for idx_pos in range(2, len(mor_indices)):
            i = mor_indices[idx_pos]
            c0 = df.iloc[i - 2]
            c1 = df.iloc[i - 1]
            c2 = df.iloc[i]

            o0, c_close0, h0, l0 = float(c0['open']), float(c0['close']), float(c0['high']), float(c0['low'])
            o1, c_close1, h1, l1 = float(c1['open']), float(c1['close']), float(c1['high']), float(c1['low'])
            o2, c_close2, h2, l2 = float(c2['open']), float(c2['close']), float(c2['high']), float(c2['low'])

            # Bullish BISI: c2['low'] > c0['high'] with 3 consecutive displacement closes
            if l2 > h0:
                is_3_disp_bull = (
                    c_close0 > o0 and
                    c_close1 > o1 and c_close1 > h0 and
                    c_close2 > o2 and c_close2 > h1
                )
                if is_3_disp_bull:
                    top = l2
                    bottom = h0
                    # Check if rebalanced by any subsequent candle up to current
                    is_rebal = False
                    for future_idx in range(i + 1, n):
                        f_low = float(df.iloc[future_idx]['low'])
                        if f_low <= top:
                            is_rebal = True
                            break
                    return {
                        'type': 'BULLISH_FPFVG',
                        'direction': Direction.BUY,
                        'top': top,
                        'bottom': bottom,
                        'midpoint': (top + bottom) / 2.0,
                        'gap_size': top - bottom,
                        'index': i,
                        'time': times_ny[i],
                        'is_rebalanced': is_rebal,
                    }

            # Bearish SIBI: c2['high'] < c0['low'] with 3 consecutive displacement closes
            if h2 < l0:
                is_3_disp_bear = (
                    c_close0 < o0 and
                    c_close1 < o1 and c_close1 < l0 and
                    c_close2 < o2 and c_close2 < l1
                )
                if is_3_disp_bear:
                    top = l0
                    bottom = h2
                    is_rebal = False
                    for future_idx in range(i + 1, n):
                        f_high = float(df.iloc[future_idx]['high'])
                        if f_high >= bottom:
                            is_rebal = True
                            break
                    return {
                        'type': 'BEARISH_FPFVG',
                        'direction': Direction.SELL,
                        'top': top,
                        'bottom': bottom,
                        'midpoint': (top + bottom) / 2.0,
                        'gap_size': top - bottom,
                        'index': i,
                        'time': times_ny[i],
                        'is_rebalanced': is_rebal,
                    }

        return None

    def evaluate_premium_discount(
        self,
        current_price: float,
        openings: ICTOpeningRanges,
        htf_bias: MarketBias = MarketBias.NEUTRAL,
    ) -> dict[str, Any]:
        """
        Evaluate Premium & Discount zones according to Section 2.A:
        - Standard Premium: Price > 00:00 EST Midnight Open (TDO) -> Primarily SELL setups.
        - Standard Discount: Price < 00:00 EST Midnight Open (TDO) -> Primarily BUY setups.
        - Deep Premium (High-Probability Bearish Reversal): Price > TDO, > 08:30 Open, AND > 09:30 Open.
        - Deep Discount (High-Probability Bullish Reversal): Price < TDO, < 08:30 Open, AND < 09:30 Open.
        """
        tdo = openings.tdo
        p_0830 = openings.ny_pre_market_open
        p_0930 = openings.nyse_open

        is_premium = current_price > tdo if tdo is not None else False
        is_discount = current_price < tdo if tdo is not None else False

        is_deep_premium = False
        if tdo is not None and is_premium:
            c_0830 = (p_0830 is None) or (current_price > p_0830)
            c_0930 = (p_0930 is None) or (current_price > p_0930)
            is_deep_premium = c_0830 and c_0930 and (p_0830 is not None or p_0930 is not None)

        is_deep_discount = False
        if tdo is not None and is_discount:
            c_0830 = (p_0830 is None) or (current_price < p_0830)
            c_0930 = (p_0930 is None) or (current_price < p_0930)
            is_deep_discount = c_0830 and c_0930 and (p_0830 is not None or p_0930 is not None)

        favored_direction = None
        if is_deep_premium:
            favored_direction = Direction.SELL
        elif is_deep_discount:
            favored_direction = Direction.BUY
        elif is_premium and htf_bias != MarketBias.BULLISH:
            favored_direction = Direction.SELL
        elif is_discount and htf_bias != MarketBias.BEARISH:
            favored_direction = Direction.BUY

        return {
            'is_premium': is_premium,
            'is_discount': is_discount,
            'is_deep_premium': is_deep_premium,
            'is_deep_discount': is_deep_discount,
            'favored_direction': favored_direction,
            'tdo': tdo,
            'p_0830': p_0830,
            'p_0930': p_0930,
        }

    def calculate_sd_projections(
        self,
        anchor_high: float,
        anchor_low: float,
        direction: Direction,
    ) -> SDProjections:
        """
        Calculate ICT Standard Deviation projections on MOR or 08:30/09:30 manipulation legs:
        - Bearish: 0 anchored at Low, 1 at High -> project downwards.
        - Bullish: 0 anchored at High, 1 at Low -> project upwards.
        - Internal Quadrant Filter: 0.705 to 1.0 CE midline check.
        - Key zones:
          -1.0 to -1.5 SD (Silver Bullet Zone)
          -2.0 to -2.5 SD (Minor Retracement Zone)
          -4.0 to -4.5 SD (Major Reversal Zone - 90% daily expansion exhaustion)
        """
        rng = abs(anchor_high - anchor_low)
        ce_midline = (anchor_high + anchor_low) / 2.0

        if direction == Direction.SELL:
            # Bearish downward projection from Low
            return SDProjections(
                anchor_high=anchor_high,
                anchor_low=anchor_low,
                direction=direction,
                range_size=rng,
                ce_midline=ce_midline,
                sd_1_0=anchor_low - (1.0 * rng),
                sd_1_5=anchor_low - (1.5 * rng),
                sd_2_0=anchor_low - (2.0 * rng),
                sd_2_5=anchor_low - (2.5 * rng),
                sd_4_0=anchor_low - (4.0 * rng),
                sd_4_5=anchor_low - (4.5 * rng),
            )
        else:
            # Bullish upward projection from High
            return SDProjections(
                anchor_high=anchor_high,
                anchor_low=anchor_low,
                direction=direction,
                range_size=rng,
                ce_midline=ce_midline,
                sd_1_0=anchor_high + (1.0 * rng),
                sd_1_5=anchor_high + (1.5 * rng),
                sd_2_0=anchor_high + (2.0 * rng),
                sd_2_5=anchor_high + (2.5 * rng),
                sd_4_0=anchor_high + (4.0 * rng),
                sd_4_5=anchor_high + (4.5 * rng),
            )

    def calculate_aox_fib(
        self,
        manipulation_high: float,
        manipulation_low: float,
        direction: Direction,
    ) -> AOXFibLevels:
        """
        Calculate AOX Custom Fibonacci levels on the most recent manipulation leg into POI/SD zone:
        - Sniper Entry Zone (Stop-Hunt Extension): 0, -0.21, -0.255, -0.29
        - Partial TP1 (Majority Book): 1.47, 1.55
        - Final TP2 (Full Book): 2.56, 2.60, 2.64
        """
        rng = abs(manipulation_high - manipulation_low)
        mean_threshold = (manipulation_high + manipulation_low) / 2.0

        if direction == Direction.BUY:
            # Downward manipulation leg leading into discount / POI:
            # Anchor 1 at high, 0 at low. Extensions terminate below low:
            sniper_entry = manipulation_low - (0.21 * rng)
            sniper_mid = manipulation_low - (0.255 * rng)
            sniper_max = manipulation_low - (0.29 * rng)
            tp1_min = manipulation_low + (1.47 * rng)
            tp1_max = manipulation_low + (1.55 * rng)
            tp1_mid = (tp1_min + tp1_max) / 2.0
            tp2_min = manipulation_low + (2.56 * rng)
            tp2_mid = manipulation_low + (2.60 * rng)
            tp2_max = manipulation_low + (2.64 * rng)
        else:
            # Upward manipulation leg leading into premium / POI:
            # Anchor 1 at low, 0 at high. Extensions terminate above high:
            sniper_entry = manipulation_high + (0.21 * rng)
            sniper_mid = manipulation_high + (0.255 * rng)
            sniper_max = manipulation_high + (0.29 * rng)
            tp1_min = manipulation_high - (1.47 * rng)
            tp1_max = manipulation_high - (1.55 * rng)
            tp1_mid = (tp1_min + tp1_max) / 2.0
            tp2_min = manipulation_high - (2.56 * rng)
            tp2_mid = manipulation_high - (2.60 * rng)
            tp2_max = manipulation_high - (2.64 * rng)

        return AOXFibLevels(
            manipulation_high=manipulation_high,
            manipulation_low=manipulation_low,
            direction=direction,
            range_size=rng,
            mean_threshold=mean_threshold,
            sniper_zone_entry=sniper_entry,
            sniper_zone_mid=sniper_mid,
            sniper_zone_max=sniper_max,
            tp1_min=tp1_min,
            tp1_max=tp1_max,
            tp1_mid=tp1_mid,
            tp2_min=tp2_min,
            tp2_mid=tp2_mid,
            tp2_max=tp2_max,
        )

    def detect_smt_divergence(
        self,
        primary_df: pd.DataFrame,
        correlated_df: pd.DataFrame | None = None,
        lookback: int = 15,
    ) -> tuple[bool, str, str]:
        """
        Detect Smart Money Divergence (SMT) between primary asset and correlated asset:
        - Bearish SMT: Primary sweeps previous High (Higher High), Correlated FAILS to sweep (Lower High).
        - Bullish SMT: Primary sweeps previous Low (Lower Low), Correlated FAILS to sweep (Higher Low).
        - Confirms Protected High / Low.
        """
        if correlated_df is None or len(correlated_df) < 5 or len(primary_df) < 5:
            return False, "NONE", "Correlated data not provided"

        w_p = primary_df.iloc[-lookback:]
        w_c = correlated_df.iloc[-lookback:]

        p_swings = find_swing_points_logic(w_p['high'], w_p['low'], lookback=2)
        c_swings = find_swing_points_logic(w_c['high'], w_c['low'], lookback=2)

        p_highs = [s for s in p_swings if s.is_high]
        p_lows = [s for s in p_swings if not s.is_high]
        c_highs = [s for s in c_swings if s.is_high]
        c_lows = [s for s in c_swings if not s.is_high]

        # Check Bearish SMT
        if len(p_highs) >= 2 and len(c_highs) >= 2:
            p_hh = p_highs[-1].price > p_highs[-2].price
            c_lh = c_highs[-1].price < c_highs[-2].price
            if p_hh and c_lh:
                return True, "BEARISH_SMT", f"Primary HH ({p_highs[-1].price:.5f}) vs Correlated LH ({c_highs[-1].price:.5f})"
            c_hh = c_highs[-1].price > c_highs[-2].price
            p_lh = p_highs[-1].price < p_highs[-2].price
            if c_hh and p_lh:
                return True, "BEARISH_SMT", f"Correlated HH ({c_highs[-1].price:.5f}) vs Primary LH ({p_highs[-1].price:.5f})"

        # Check Bullish SMT
        if len(p_lows) >= 2 and len(c_lows) >= 2:
            p_ll = p_lows[-1].price < p_lows[-2].price
            c_hl = c_lows[-1].price > c_lows[-2].price
            if p_ll and c_hl:
                return True, "BULLISH_SMT", f"Primary LL ({p_lows[-1].price:.5f}) vs Correlated HL ({c_lows[-1].price:.5f})"
            c_ll = c_lows[-1].price < c_lows[-2].price
            p_hl = p_lows[-1].price > p_lows[-2].price
            if c_ll and p_hl:
                return True, "BULLISH_SMT", f"Correlated LL ({c_lows[-1].price:.5f}) vs Primary HL ({p_lows[-1].price:.5f})"

        return False, "NONE", "No SMT Divergence"

    def detect_fvg(self, df: pd.DataFrame, index: int = -1, direction: Direction = Direction.BUY) -> dict | None:
        """Detect a Fair Value Gap (FVG) at the specified bar index in a DataFrame (backward compatible)."""
        n = len(df)
        if index < 0:
            index = n + index
        if index < 2 or index >= n:
            return None

        c_curr = df.iloc[index]
        c_p2 = df.iloc[index - 2]

        if direction == Direction.BUY:
            if float(c_curr['low']) > float(c_p2['high']):
                top = float(c_curr['low'])
                bottom = float(c_p2['high'])
                return {
                    'type': 'BULLISH_FVG',
                    'top': top,
                    'bottom': bottom,
                    'midpoint': (top + bottom) / 2.0,
                    'gap_size': top - bottom,
                    'index': index,
                }
        elif direction == Direction.SELL:
            if float(c_curr['high']) < float(c_p2['low']):
                top = float(c_p2['low'])
                bottom = float(c_curr['high'])
                return {
                    'type': 'BEARISH_FVG',
                    'top': top,
                    'bottom': bottom,
                    'midpoint': (top + bottom) / 2.0,
                    'gap_size': top - bottom,
                    'index': index,
                }
        return None

    def calculate_ote_zone(self, low: float, high: float, direction: Direction = Direction.BUY) -> dict:
        """Calculate the Optimal Trade Entry (61.8% to 78.6% Fib) zone (backward compatible)."""
        rng = abs(high - low)
        if direction == Direction.BUY:
            return {
                'fib_618': high - (rng * 0.618),
                'fib_705': high - (rng * 0.705),
                'fib_786': high - (rng * 0.786),
                'direction': Direction.BUY,
            }
        else:
            return {
                'fib_618': low + (rng * 0.618),
                'fib_705': low + (rng * 0.705),
                'fib_786': low + (rng * 0.786),
                'direction': Direction.SELL,
            }

    def detect_aox_setup(
        self,
        df: pd.DataFrame,
        htf_analysis: HTFAnalysis,
        instrument: InstrumentConfig,
        current_spread: float,
        fixed_sl_pips: float | None = None,
        correlated_df: pd.DataFrame | None = None,
        openings: ICTOpeningRanges | None = None,
    ) -> list[dict]:
        """
        Scan for institutional ICT Time & Price + AOX Model setups:
        1. Anchors to NY Local Time and calculates Reference Openings (TDO, MOR, 08:30, 09:30).
        2. Premium / Discount filtering (Standard vs Deep Premium/Discount).
        3. Standard Deviation (SD) Projections on MOR and Manipulation Legs (-1.5, -2.5, -4.5 SD).
        4. First Presented FVG (FPFVG) Magnet Snapback detection.
        5. Smart Money Divergence (SMT) with correlated asset to confirm Protected High/Low.
        6. 3-Entry Execution Sequence:
           - Entry 1: Sniper Tap at POI MT / AOX Fib -0.21 to -0.29 (25% Risk)
           - Entry 2: SMT Divergence + Turtle Soup into First FVG of MSS (50% Risk)
           - Entry 3: HIPPO / Final Mitigation Block retest (50% Risk)
        7. Strict 25-25-50 Layered Risk Management.
        """
        n = len(df)
        if n < 5:
            return []

        current_bar = df.iloc[-1]
        timestamp = current_bar['time'] if 'time' in df.columns else datetime.now(timezone.utc)
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        ny_time = to_ny_time(timestamp)
        is_macro, macro_name = self.get_active_macro(timestamp)

        # Check macro gating if enforced
        if self.config.enforce_macro_windows and not is_macro:
            logger.debug(f"[ICT-AOX] Outside Macro Windows ({ny_time.strftime('%H:%M EST')}). Skipping.")
            return []

        # 1. Calculate Reference Openings & FPFVG
        if openings is None:
            openings = self.calculate_opening_ranges(df, target_date=ny_time.date())

        current_close = float(current_bar['close'])
        current_low = float(current_bar['low'])
        current_high = float(current_bar['high'])

        # 2. Premium / Discount Filter
        pd_eval = self.evaluate_premium_discount(current_close, openings, htf_analysis.bias)
        favored_dir = pd_eval.get('favored_direction')

        # 3. ATR & Swings
        atr_series = compute_atr(df, period=14)
        current_atr = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 10 * instrument.pip_size

        swings = find_swing_points_logic(df['high'], df['low'], lookback=2)
        if not swings:
            h_idx = int(df['high'].idxmax()) if hasattr(df['high'], 'idxmax') else 0
            l_idx = int(df['low'].idxmin()) if hasattr(df['low'], 'idxmin') else 0
            swings = [
                SwingPoint(index=h_idx, price=float(df['high'].max()), is_high=True),
                SwingPoint(index=l_idx, price=float(df['low'].min()), is_high=False),
            ]

        # 4. Check SMT Divergence
        has_smt, smt_type, smt_desc = self.detect_smt_divergence(df, correlated_df, lookback=15)

        # 5. SD Projections on MOR
        mor_h = openings.mor_high or float(df['high'].max())
        mor_l = openings.mor_low or float(df['low'].min())
        sd_bear = self.calculate_sd_projections(mor_h, mor_l, Direction.SELL)
        sd_bull = self.calculate_sd_projections(mor_h, mor_l, Direction.BUY)

        results: list[dict] = []

        # Determine target setups based on bias and premium/discount
        test_directions = []
        if favored_dir:
            test_directions.append(favored_dir)
        elif htf_analysis.bias == MarketBias.BULLISH:
            test_directions.append(Direction.BUY)
        elif htf_analysis.bias == MarketBias.BEARISH:
            test_directions.append(Direction.SELL)
        else:
            test_directions = [Direction.BUY, Direction.SELL]

        for direction in test_directions:
            sd_proj = sd_bull if direction == Direction.BUY else sd_bear

            # Find manipulation leg into the zone (most recent aggressive push against trend)
            lookback_win = min(25, n - 2)
            leg_high, leg_low = 0.0, 0.0
            sweep_found = False
            sweep_extreme = 0.0
            sweep_idx = -1

            if direction == Direction.BUY:
                low_swings = [sp for sp in swings if not sp.is_high and sp.index < n - 1]
                high_swings = [sp for sp in swings if sp.is_high and sp.index < n - 1]
                if not low_swings:
                    low_swings = [SwingPoint(index=0, price=float(df['low'].min()), is_high=False)]
                if not high_swings:
                    high_swings = [SwingPoint(index=0, price=float(df['high'].max()), is_high=True)]

                for offset in range(1, lookback_win + 1):
                    idx = n - 1 - offset
                    bar = df.iloc[idx]
                    for sp in reversed(low_swings):
                        if sp.index < idx and float(bar['low']) < sp.price:
                            sweep_found = True
                            sweep_extreme = float(bar['low'])
                            sweep_idx = idx
                            break
                    if sweep_found:
                        break

                if not sweep_found:
                    sweep_extreme = float(df.iloc[-lookback_win:]['low'].min())
                    sweep_idx = int(df.iloc[-lookback_win:]['low'].idxmin()) if hasattr(df.iloc[-lookback_win:]['low'], 'idxmin') else n - 5

                # Manipulation leg starts from recent swing high into sweep low
                prev_highs = [sp.price for sp in high_swings if sp.index <= sweep_idx]
                leg_high = max(prev_highs) if prev_highs else float(df.iloc[-lookback_win:]['high'].max())
                leg_low = sweep_extreme

            else: # Direction.SELL
                high_swings = [sp for sp in swings if sp.is_high and sp.index < n - 1]
                low_swings = [sp for sp in swings if not sp.is_high and sp.index < n - 1]
                if not high_swings:
                    high_swings = [SwingPoint(index=0, price=float(df['high'].max()), is_high=True)]
                if not low_swings:
                    low_swings = [SwingPoint(index=0, price=float(df['low'].min()), is_high=False)]

                for offset in range(1, lookback_win + 1):
                    idx = n - 1 - offset
                    bar = df.iloc[idx]
                    for sp in reversed(high_swings):
                        if sp.index < idx and float(bar['high']) > sp.price:
                            sweep_found = True
                            sweep_extreme = float(bar['high'])
                            sweep_idx = idx
                            break
                    if sweep_found:
                        break

                if not sweep_found:
                    sweep_extreme = float(df.iloc[-lookback_win:]['high'].max())
                    sweep_idx = int(df.iloc[-lookback_win:]['high'].idxmax()) if hasattr(df.iloc[-lookback_win:]['high'], 'idxmax') else n - 5

                prev_lows = [sp.price for sp in low_swings if sp.index <= sweep_idx]
                leg_low = min(prev_lows) if prev_lows else float(df.iloc[-lookback_win:]['low'].min())
                leg_high = sweep_extreme

            leg_range = abs(leg_high - leg_low)
            if leg_range <= 0:
                continue

            # Calculate AOX Custom Fibonacci levels on manipulation leg
            aox_fib = self.calculate_aox_fib(leg_high, leg_low, direction)

            # Determine SD zone classification
            sd_tag = "STANDARD_EXPANSION"
            if direction == Direction.BUY:
                # Downward projection into discount below MOR
                if current_low <= sd_bear.sd_4_0:
                    sd_tag = "-4.0_TO_-4.5_SD_MAJOR_REVERSAL"
                elif current_low <= sd_bear.sd_2_0:
                    sd_tag = "-2.0_TO_-2.5_SD_MINOR_RETRACEMENT"
                elif current_low <= sd_bear.sd_1_0:
                    sd_tag = "-1.0_TO_-1.5_SD_SILVER_BULLET"
            else:
                # Upward projection into premium above MOR
                if current_high >= sd_bull.sd_4_0:
                    sd_tag = "-4.0_TO_-4.5_SD_MAJOR_REVERSAL"
                elif current_high >= sd_bull.sd_2_0:
                    sd_tag = "-2.0_TO_-2.5_SD_MINOR_RETRACEMENT"
                elif current_high >= sd_bull.sd_1_0:
                    sd_tag = "-1.0_TO_-1.5_SD_SILVER_BULLET"

            in_sd_zone = sd_tag in (
                "-2.0_TO_-2.5_SD_MINOR_RETRACEMENT",
                "-4.0_TO_-4.5_SD_MAJOR_REVERSAL",
                "-1.0_TO_-1.5_SD_SILVER_BULLET",
            )

            # ── Check Entry 1: AOX Sniper Tap (-0.21 to -0.29 Fib / SD Zone Tap / Mean Threshold Tap) ──
            entry1_eligible = False
            if direction == Direction.BUY:
                if in_sd_zone or (current_low <= aox_fib.sniper_zone_entry and current_close >= aox_fib.sniper_zone_max) or (current_low <= leg_low):
                    entry1_eligible = True
            else:
                if in_sd_zone or (current_high >= aox_fib.sniper_zone_entry and current_close <= aox_fib.sniper_zone_max) or (current_high >= leg_high):
                    entry1_eligible = True

            # ── Check Entry 2: SMT Divergence + Turtle Soup into First FVG ──
            entry2_eligible = False
            turtle_soup_desc = ""
            # Check displacement MSS after sweep
            mss_confirmed = False
            mss_idx = -1
            if direction == Direction.BUY:
                for post_idx in range(sweep_idx, n):
                    p_bar = df.iloc[post_idx]
                    p_close = float(p_bar['close'])
                    for sp in high_swings:
                        if sweep_idx <= sp.index < post_idx and p_close > sp.price:
                            mss_confirmed = True
                            mss_idx = post_idx
                            break
                    if mss_confirmed:
                        break
            else:
                for post_idx in range(sweep_idx, n):
                    p_bar = df.iloc[post_idx]
                    p_close = float(p_bar['close'])
                    for sp in low_swings:
                        if sweep_idx <= sp.index < post_idx and p_close < sp.price:
                            mss_confirmed = True
                            mss_idx = post_idx
                            break
                    if mss_confirmed:
                        break

            # Find First Presented FVG of displacement leg
            disp_fvg = None
            if mss_confirmed and mss_idx >= 2:
                for fi in range(max(2, sweep_idx), min(mss_idx + 4, n)):
                    f = self.detect_fvg(df, index=fi, direction=direction)
                    if f:
                        disp_fvg = f
                        break

            # Turtle Soup: price sweeps minor phantom liquidity just outside FVG and taps FVG
            if disp_fvg is not None:
                fvg_top, fvg_bottom = disp_fvg['top'], disp_fvg['bottom']
                if direction == Direction.BUY:
                    if current_low <= fvg_top and current_close >= fvg_bottom:
                        entry2_eligible = True
                        turtle_soup_desc = f"Turtle Soup tap into BISI [{fvg_bottom:.5f} - {fvg_top:.5f}]"
                else:
                    if current_high >= fvg_bottom and current_close <= fvg_top:
                        entry2_eligible = True
                        turtle_soup_desc = f"Turtle Soup tap into SIBI [{fvg_bottom:.5f} - {fvg_top:.5f}]"

            if has_smt:
                turtle_soup_desc = f"{turtle_soup_desc} | SMT Confirmed ({smt_type})"

            # ── Check Entry 3: HIPPO / Final Mitigation Block Retest ──
            entry3_eligible = False
            hippo_desc = ""
            if mss_confirmed and not entry2_eligible:
                # Mitigation pullback clearing trap breaker liquidity
                if direction == Direction.BUY:
                    if current_low <= aox_fib.mean_threshold and current_close > aox_fib.mean_threshold:
                        entry3_eligible = True
                        hippo_desc = f"HIPPO Mitigation retest at Mean Threshold ({aox_fib.mean_threshold:.5f})"
                else:
                    if current_high >= aox_fib.mean_threshold and current_close < aox_fib.mean_threshold:
                        entry3_eligible = True
                        hippo_desc = f"HIPPO Mitigation retest at Mean Threshold ({aox_fib.mean_threshold:.5f})"

            # Check directional candlestick body confirmation (>= 50% body)
            min_body = getattr(self.config, 'min_candle_body_ratio', 0.50)
            is_confirmed_candle, body_ratio, conf_desc = check_candle_body_confirmation(
                current_bar, direction, min_body_ratio=min_body
            )

            # Invalidation Stop Loss & Targets
            # Rule: Stop Loss ~20 pips beyond -0.29 Fib level or protected swing extreme
            gold_20_pips = 20.0 * instrument.pip_size
            sl_buffer = max(gold_20_pips, current_atr * 0.2)

            if direction == Direction.BUY:
                stop_loss = min(aox_fib.sniper_zone_max - (2.0 * instrument.pip_size), sweep_extreme - sl_buffer)
                if fixed_sl_pips is not None and fixed_sl_pips > 0:
                    stop_loss = current_close - (fixed_sl_pips * instrument.pip_size)
                tp1 = aox_fib.tp1_mid
                tp2 = aox_fib.tp2_mid
            else:
                stop_loss = max(aox_fib.sniper_zone_max + (2.0 * instrument.pip_size), sweep_extreme + sl_buffer)
                if fixed_sl_pips is not None and fixed_sl_pips > 0:
                    stop_loss = current_close + (fixed_sl_pips * instrument.pip_size)
                tp1 = aox_fib.tp1_mid
                tp2 = aox_fib.tp2_mid

            sl_dist = abs(current_close - stop_loss)
            min_buf = 2.0 * instrument.pip_size
            if sl_dist < min_buf:
                sl_dist = min_buf
                stop_loss = current_close - sl_dist if direction == Direction.BUY else current_close + sl_dist

            # Magnet check: If FPFVG is unmitigated and price is snapping back
            if openings.fpfvg and not openings.fpfvg.get('is_rebalanced', False):
                fpfvg_mid = openings.fpfvg['midpoint']
                if direction == Direction.BUY and fpfvg_mid > current_close:
                    tp1 = fpfvg_mid
                elif direction == Direction.SELL and fpfvg_mid < current_close:
                    tp1 = fpfvg_mid

            # Build Layer 1 Setup (25% Risk)
            if entry1_eligible:
                results.append({
                    'layer': 1,
                    'risk_fraction': self.config.layer1_risk_pct,
                    'direction': direction,
                    'entry': current_close,
                    'sl': stop_loss,
                    'tp': tp1,
                    'tp1': tp1,
                    'tp2': tp2,
                    'conf': LTFConfirmation.ICT_AOX_SNIPER_TAP,
                    'timestamp': timestamp,
                    'kz': macro_name if is_macro else (self.get_active_killzone(timestamp) or "ALL_HOURS"),
                    'candle_confirmation': conf_desc,
                    'smt_confirmed': has_smt,
                    'sd_zone': sd_tag,
                    'description': f"AOX Layer 1 (25% Risk): Sniper Tap @ {current_close:.5f} ({sd_tag})",
                })

            # Build Layer 2 Setup (25% Risk - Deep Drawdown / MT Spike or SMT Turtle Soup)
            if entry2_eligible:
                results.append({
                    'layer': 2,
                    'risk_fraction': self.config.layer2_risk_pct,
                    'direction': direction,
                    'entry': current_close,
                    'sl': stop_loss,
                    'tp': tp1,
                    'tp1': tp1,
                    'tp2': tp2,
                    'conf': LTFConfirmation.ICT_AOX_SMT_TURTLE_SOUP,
                    'timestamp': timestamp,
                    'kz': macro_name if is_macro else (self.get_active_killzone(timestamp) or "ALL_HOURS"),
                    'candle_confirmation': conf_desc,
                    'smt_confirmed': has_smt,
                    'sd_zone': sd_tag,
                    'description': f"AOX Layer 2 (SMT Turtle Soup): {turtle_soup_desc}",
                })
            elif entry1_eligible and (in_sd_zone or (current_low <= aox_fib.sniper_zone_mid if direction == Direction.BUY else current_high >= aox_fib.sniper_zone_mid)):
                results.append({
                    'layer': 2,
                    'risk_fraction': self.config.layer2_risk_pct,
                    'direction': direction,
                    'entry': current_close,
                    'sl': stop_loss,
                    'tp': tp1,
                    'tp1': tp1,
                    'tp2': tp2,
                    'conf': LTFConfirmation.ICT_AOX_SNIPER_TAP,
                    'timestamp': timestamp,
                    'kz': macro_name if is_macro else (self.get_active_killzone(timestamp) or "ALL_HOURS"),
                    'candle_confirmation': conf_desc,
                    'smt_confirmed': has_smt,
                    'sd_zone': sd_tag,
                    'description': f"AOX Layer 2 (25% Risk): Drawdown Spike into MT/-0.29 @ {current_close:.5f}",
                })

            # Build Layer 3 Setup (50% Risk - Confirmation Close / Turtle Soup / HIPPO)
            if entry3_eligible:
                results.append({
                    'layer': 3,
                    'risk_fraction': self.config.layer3_risk_pct,
                    'direction': direction,
                    'entry': current_close,
                    'sl': stop_loss,
                    'tp': tp2,
                    'tp1': tp1,
                    'tp2': tp2,
                    'conf': LTFConfirmation.ICT_AOX_HIPPO_MITIGATION,
                    'timestamp': timestamp,
                    'kz': macro_name if is_macro else (self.get_active_killzone(timestamp) or "ALL_HOURS"),
                    'candle_confirmation': conf_desc,
                    'smt_confirmed': has_smt,
                    'sd_zone': sd_tag,
                    'description': f"AOX Layer 3 (HIPPO Mitigation): {hippo_desc}",
                })
            elif (entry1_eligible and is_confirmed_candle) or (entry2_eligible and is_confirmed_candle):
                results.append({
                    'layer': 3,
                    'risk_fraction': self.config.layer3_risk_pct,
                    'direction': direction,
                    'entry': current_close,
                    'sl': stop_loss,
                    'tp': tp1,
                    'tp1': tp1,
                    'tp2': tp2,
                    'conf': LTFConfirmation.ICT_AOX_SMT_TURTLE_SOUP if entry2_eligible else LTFConfirmation.ICT_SD_REVERSAL,
                    'timestamp': timestamp,
                    'kz': macro_name if is_macro else (self.get_active_killzone(timestamp) or "ALL_HOURS"),
                    'candle_confirmation': conf_desc,
                    'smt_confirmed': has_smt,
                    'sd_zone': sd_tag,
                    'description': f"AOX Layer 3 (50% Risk): Confirmation Close @ {current_close:.5f} ({sd_tag})",
                })

            # Backward compatibility fallback: if candle confirmed in active killzone/macro with FVG or OTE tap
            if not results and is_confirmed_candle and (entry1_eligible or sweep_found):
                results.append({
                    'layer': 1,
                    'risk_fraction': self.config.layer1_risk_pct,
                    'direction': direction,
                    'entry': current_close,
                    'sl': stop_loss,
                    'tp': tp1,
                    'tp1': tp1,
                    'tp2': tp2,
                    'conf': LTFConfirmation.ICT_SILVER_BULLET if "SILVER_BULLET" in str(macro_name) else LTFConfirmation.ICT_KILLZONE_FVG,
                    'timestamp': timestamp,
                    'kz': macro_name if is_macro else (self.get_active_killzone(timestamp) or "ALL_HOURS"),
                    'candle_confirmation': conf_desc,
                    'smt_confirmed': has_smt,
                    'sd_zone': sd_tag,
                    'description': f"ICT AOX Setup: {direction.value} @ {current_close:.5f} ({sd_tag})",
                })

        return results

    def detect_ict_entry(
        self,
        df: pd.DataFrame,
        htf_analysis: HTFAnalysis,
        instrument: InstrumentConfig,
        current_spread: float,
        fixed_sl_pips: float | None = None,
        correlated_df: pd.DataFrame | None = None,
    ) -> dict | None:
        """Backward-compatible single-entry detection returning the primary active AOX setup."""
        setups = self.detect_aox_setup(
            df=df,
            htf_analysis=htf_analysis,
            instrument=instrument,
            current_spread=current_spread,
            fixed_sl_pips=fixed_sl_pips,
            correlated_df=correlated_df,
        )
        if not setups:
            return None

        # Prefer Layer 2 (SMT Turtle Soup), else Layer 1, else Layer 3
        sorted_setups = sorted(setups, key=lambda s: (s.get('smt_confirmed', False), s.get('layer', 1) == 2, s.get('layer', 1)), reverse=True)
        return sorted_setups[0]


# ─────────────────────────────────────────────
#  Order Flow Microstructure Engine
# ─────────────────────────────────────────────

class OrderFlowEngine:
    """
    Order Flow Microstructure Strategy Engine.

    Analyzes real-time or aggregated candlestick volume data to detect:
    1. Intrabar / Tick Volume Delta and Cumulative Volume Delta (CVD).
    2. Institutional Absorption (OF_ABSORPTION) at key support/resistance levels.
    3. CVD Divergences (OF_DELTA_DIVERGENCE) signaling exhaustion.
    4. Liquidity Traps and Stop Sweeps (OF_LIQUIDITY_TRAP) with delta reversal.
    """

    def __init__(self, config: OrderFlowConfig | None = None):
        self.config = config or OrderFlowConfig()

    def compute_volume_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich dataframe with estimated Volume Delta and Cumulative Volume Delta (CVD).
        Works seamlessly with real_volume or tick_volume in MT5.
        """
        df = df.copy()

        # Determine volume series
        if 'real_volume' in df.columns and (df['real_volume'] > 0).any():
            vol = df['real_volume'].astype(float)
        elif 'tick_volume' in df.columns:
            vol = df['tick_volume'].astype(float)
        elif 'volume' in df.columns:
            vol = df['volume'].astype(float)
        else:
            vol = pd.Series(100.0, index=df.index)

        # Handle zero or missing volume gracefully
        vol = vol.replace(0, 1.0)

        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)

        hl = high - low
        hl = hl.replace(0, 1e-6)

        # Intrabar volume fraction based on price location within range
        buy_fraction = ((close - low) / hl).clip(0.0, 1.0)

        buy_vol = vol * buy_fraction
        sell_vol = vol * (1.0 - buy_fraction)

        bar_delta = buy_vol - sell_vol
        if 'delta' in df.columns:
            bar_delta = df['delta'].astype(float)

        df['vol_delta'] = bar_delta
        df['cvd'] = bar_delta.cumsum()
        df['vol_sma20'] = vol.rolling(window=min(20, len(df)), min_periods=1).mean()
        df['raw_vol'] = vol

        return df

    def detect_order_flow_entry(
        self,
        df: pd.DataFrame,
        htf_analysis: HTFAnalysis,
        instrument: InstrumentConfig,
        current_spread: float,
        fixed_sl_pips: float | None = None,
    ) -> dict | None:
        """
        Scan for Order Flow confirmations aligned with HTF bias:
        - Liquidity Trap / Stop Sweep (Highest conviction)
        - Institutional Absorption
        - CVD Divergence
        """
        n = len(df)
        if n < 15:
            return None

        bias = htf_analysis.bias
        if bias not in (MarketBias.BULLISH, MarketBias.BEARISH):
            return None

        df_of = self.compute_volume_delta(df)

        current_bar = df_of.iloc[-1]
        prev_bar = df_of.iloc[-2]

        timestamp = current_bar['time'] if 'time' in df_of.columns else datetime.now(timezone.utc)
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        atr_series = compute_atr(df, period=14)
        current_atr = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 10.0 * instrument.pip_size
        if current_atr <= 0:
            current_atr = 10.0 * instrument.pip_size

        entry_price = float(current_bar['close'])
        min_buffer = 1.0 * instrument.pip_size

        # Find recent swing points for structure reference
        swing_points = find_swing_points_logic(df['high'], df['low'], lookback=3)
        recent_swing_highs = [p.price for p in swing_points if p.is_high and p.index < n - 1]
        recent_swing_lows = [p.price for p in swing_points if not p.is_high and p.index < n - 1]

        # -------------------------------------------------------------
        # BULLISH SETUPS (HTF Bias == BULLISH)
        # -------------------------------------------------------------
        if bias == MarketBias.BULLISH:
            # Check 1: Liquidity Trap / Stop Sweep
            if recent_swing_lows:
                last_swing_low = recent_swing_lows[-1]
                swept_low = min(float(prev_bar['low']), float(current_bar['low']))
                if swept_low < last_swing_low:
                    if float(current_bar['close']) > last_swing_low and float(current_bar['vol_delta']) > 0:
                        is_conf, b_ratio, c_desc = check_candle_body_confirmation(
                            current_bar, Direction.BUY, min_body_ratio=getattr(self.config, 'min_candle_body_ratio', 0.50)
                        )
                        if is_conf:
                            stop_loss = swept_low - min_buffer
                            sl_distance = abs(entry_price - stop_loss)
                            if fixed_sl_pips is not None and fixed_sl_pips > 0:
                                sl_distance = fixed_sl_pips * instrument.pip_size
                                stop_loss = entry_price - sl_distance

                            target_dist = max(sl_distance * self.config.target_rr, 6.0 * current_spread)
                            erl_targets = [p.level for p in htf_analysis.liquidity_pools if p.is_high and p.level > entry_price + target_dist]
                            take_profit = min(erl_targets) if erl_targets else entry_price + target_dist

                            return {
                                'direction': Direction.BUY,
                                'entry': entry_price,
                                'sl': stop_loss,
                                'tp': take_profit,
                                'conf': LTFConfirmation.OF_LIQUIDITY_TRAP,
                                'timestamp': timestamp,
                                'candle_confirmation': c_desc,
                            }

            # Check 2: Institutional Absorption (Selling absorbed at support)
            for bar in [prev_bar, current_bar]:
                bar_range = float(bar['high'] - bar['low'])
                if bar_range > 0:
                    lower_wick = float(min(bar['open'], bar['close']) - bar['low'])
                    wick_ratio = lower_wick / bar_range
                    vol_factor = float(bar['raw_vol']) / max(1.0, float(bar['vol_sma20']))

                    if (wick_ratio >= self.config.wick_ratio_threshold and 
                        vol_factor >= self.config.absorption_volume_factor):

                        is_conf, b_ratio, c_desc = check_candle_body_confirmation(
                            current_bar, Direction.BUY, min_body_ratio=getattr(self.config, 'min_candle_body_ratio', 0.50)
                        )
                        if is_conf:
                            stop_loss = min(float(prev_bar['low']), float(current_bar['low'])) - min_buffer
                            sl_distance = abs(entry_price - stop_loss)
                            if fixed_sl_pips is not None and fixed_sl_pips > 0:
                                sl_distance = fixed_sl_pips * instrument.pip_size
                                stop_loss = entry_price - sl_distance

                            target_dist = max(sl_distance * self.config.target_rr, 6.0 * current_spread)
                            erl_targets = [p.level for p in htf_analysis.liquidity_pools if p.is_high and p.level > entry_price + target_dist]
                            take_profit = min(erl_targets) if erl_targets else entry_price + target_dist

                            return {
                                'direction': Direction.BUY,
                                'entry': entry_price,
                                'sl': stop_loss,
                                'tp': take_profit,
                                'conf': LTFConfirmation.OF_ABSORPTION,
                                'timestamp': timestamp,
                                'candle_confirmation': c_desc,
                            }

            # Check 3: Cumulative Volume Delta Divergence (Bullish)
            lookback = min(self.config.cvd_divergence_bars, n - 2)
            if lookback >= 5:
                window_df = df_of.iloc[-(lookback + 2):-1]
                min_price_idx = window_df['low'].idxmin()
                prior_lowest_price = float(window_df.loc[min_price_idx, 'low'])
                prior_cvd_at_low = float(window_df.loc[min_price_idx, 'cvd'])

                curr_low = float(current_bar['low'])
                curr_cvd = float(current_bar['cvd'])

                if curr_low <= prior_lowest_price + (0.1 * current_atr) and curr_cvd > prior_cvd_at_low:
                    if float(current_bar['vol_delta']) > 0:
                        is_conf, b_ratio, c_desc = check_candle_body_confirmation(
                            current_bar, Direction.BUY, min_body_ratio=getattr(self.config, 'min_candle_body_ratio', 0.50)
                        )
                        if is_conf:
                            stop_loss = curr_low - min_buffer
                            sl_distance = abs(entry_price - stop_loss)
                            if fixed_sl_pips is not None and fixed_sl_pips > 0:
                                sl_distance = fixed_sl_pips * instrument.pip_size
                                stop_loss = entry_price - sl_distance

                            target_dist = max(sl_distance * self.config.target_rr, 6.0 * current_spread)
                            erl_targets = [p.level for p in htf_analysis.liquidity_pools if p.is_high and p.level > entry_price + target_dist]
                            take_profit = min(erl_targets) if erl_targets else entry_price + target_dist

                            return {
                                'direction': Direction.BUY,
                                'entry': entry_price,
                                'sl': stop_loss,
                                'tp': take_profit,
                                'conf': LTFConfirmation.OF_DELTA_DIVERGENCE,
                                'timestamp': timestamp,
                                'candle_confirmation': c_desc,
                            }

        # -------------------------------------------------------------
        # BEARISH SETUPS (HTF Bias == BEARISH)
        # -------------------------------------------------------------
        elif bias == MarketBias.BEARISH:
            # Check 1: Liquidity Trap / Stop Sweep (Bearish)
            if recent_swing_highs:
                last_swing_high = recent_swing_highs[-1]
                swept_high = max(float(prev_bar['high']), float(current_bar['high']))
                if swept_high > last_swing_high:
                    if float(current_bar['close']) < last_swing_high and float(current_bar['vol_delta']) < 0:
                        is_conf, b_ratio, c_desc = check_candle_body_confirmation(
                            current_bar, Direction.SELL, min_body_ratio=getattr(self.config, 'min_candle_body_ratio', 0.50)
                        )
                        if is_conf:
                            stop_loss = swept_high + min_buffer
                            sl_distance = abs(entry_price - stop_loss)
                            if fixed_sl_pips is not None and fixed_sl_pips > 0:
                                sl_distance = fixed_sl_pips * instrument.pip_size
                                stop_loss = entry_price + sl_distance

                            target_dist = max(sl_distance * self.config.target_rr, 6.0 * current_spread)
                            erl_targets = [p.level for p in htf_analysis.liquidity_pools if not p.is_high and p.level < entry_price - target_dist]
                            take_profit = max(erl_targets) if erl_targets else entry_price - target_dist

                            return {
                                'direction': Direction.SELL,
                                'entry': entry_price,
                                'sl': stop_loss,
                                'tp': take_profit,
                                'conf': LTFConfirmation.OF_LIQUIDITY_TRAP,
                                'timestamp': timestamp,
                                'candle_confirmation': c_desc,
                            }

            # Check 2: Institutional Absorption (Buying absorbed at resistance)
            for bar in [prev_bar, current_bar]:
                bar_range = float(bar['high'] - bar['low'])
                if bar_range > 0:
                    upper_wick = float(bar['high'] - max(bar['open'], bar['close']))
                    wick_ratio = upper_wick / bar_range
                    vol_factor = float(bar['raw_vol']) / max(1.0, float(bar['vol_sma20']))

                    if (wick_ratio >= self.config.wick_ratio_threshold and 
                        vol_factor >= self.config.absorption_volume_factor):

                        is_conf, b_ratio, c_desc = check_candle_body_confirmation(
                            current_bar, Direction.SELL, min_body_ratio=getattr(self.config, 'min_candle_body_ratio', 0.50)
                        )
                        if is_conf:
                            stop_loss = max(float(prev_bar['high']), float(current_bar['high'])) + min_buffer
                            sl_distance = abs(entry_price - stop_loss)
                            if fixed_sl_pips is not None and fixed_sl_pips > 0:
                                sl_distance = fixed_sl_pips * instrument.pip_size
                                stop_loss = entry_price + sl_distance

                            target_dist = max(sl_distance * self.config.target_rr, 6.0 * current_spread)
                            erl_targets = [p.level for p in htf_analysis.liquidity_pools if not p.is_high and p.level < entry_price - target_dist]
                            take_profit = max(erl_targets) if erl_targets else entry_price - target_dist

                            return {
                                'direction': Direction.SELL,
                                'entry': entry_price,
                                'sl': stop_loss,
                                'tp': take_profit,
                                'conf': LTFConfirmation.OF_ABSORPTION,
                                'timestamp': timestamp,
                                'candle_confirmation': c_desc,
                            }

            # Check 3: Cumulative Volume Delta Divergence (Bearish)
            lookback = min(self.config.cvd_divergence_bars, n - 2)
            if lookback >= 5:
                window_df = df_of.iloc[-(lookback + 2):-1]
                max_price_idx = window_df['high'].idxmax()
                prior_highest_price = float(window_df.loc[max_price_idx, 'high'])
                prior_cvd_at_high = float(window_df.loc[max_price_idx, 'cvd'])

                curr_high = float(current_bar['high'])
                curr_cvd = float(current_bar['cvd'])

                if curr_high >= prior_highest_price - (0.1 * current_atr) and curr_cvd < prior_cvd_at_high:
                    if float(current_bar['vol_delta']) < 0:
                        is_conf, b_ratio, c_desc = check_candle_body_confirmation(
                            current_bar, Direction.SELL, min_body_ratio=getattr(self.config, 'min_candle_body_ratio', 0.50)
                        )
                        if is_conf:
                            stop_loss = curr_high + min_buffer
                            sl_distance = abs(entry_price - stop_loss)
                            if fixed_sl_pips is not None and fixed_sl_pips > 0:
                                sl_distance = fixed_sl_pips * instrument.pip_size
                                stop_loss = entry_price + sl_distance

                            target_dist = max(sl_distance * self.config.target_rr, 6.0 * current_spread)
                            erl_targets = [p.level for p in htf_analysis.liquidity_pools if not p.is_high and p.level < entry_price - target_dist]
                            take_profit = max(erl_targets) if erl_targets else entry_price - target_dist

                            return {
                                'direction': Direction.SELL,
                                'entry': entry_price,
                                'sl': stop_loss,
                                'tp': take_profit,
                                'conf': LTFConfirmation.OF_DELTA_DIVERGENCE,
                                'timestamp': timestamp,
                                'candle_confirmation': c_desc,
                            }

        return None


# ─────────────────────────────────────────────
#  Strategy Classes (Unified BaseStrategy Interface)
# ─────────────────────────────────────────────

class BaseStrategy(ABC):
    """Abstract Strategy Base Class for all trading algorithms."""
    id: str
    name: str
    enabled: bool = True
    magic_offset: int = 0

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        instrument: InstrumentConfig,
        current_spread: float,
        fixed_sl_pips: float | None = None,
        htf_analysis: HTFAnalysis | None = None,
    ) -> list[TradeSignal]:
        """Evaluate market data and return candidate TradeSignals."""
        pass

    @abstractmethod
    def default_sl(self, symbol: str, entry_price: float, direction: Direction, instrument: InstrumentConfig) -> float:
        """Calculate default Stop Loss price for this strategy."""
        pass

    @abstractmethod
    def default_tp(self, symbol: str, entry_price: float, sl: float, direction: Direction) -> float:
        """Calculate default Take Profit price for this strategy."""
        pass


class SMCSwingStrategy(BaseStrategy):
    """Strategy A: 15m Institutional Swing Liquidity Sweep."""
    id = "SMC"
    name = "SMC Swing (15m)"
    magic_offset = 1000

    def __init__(self, htf_analyzer_or_config: HTFAnalyzer | TimeframeConfig | TradingConfig, config_or_instrument: TimeframeConfig | InstrumentConfig | None = None):
        if isinstance(htf_analyzer_or_config, HTFAnalyzer):
            self.htf_analyzer = htf_analyzer_or_config
            self.config = config_or_instrument if isinstance(config_or_instrument, TimeframeConfig) else TimeframeConfig()
        elif hasattr(htf_analyzer_or_config, 'timeframes'):
            self.config = htf_analyzer_or_config.timeframes
            self.htf_analyzer = HTFAnalyzer(
                ema_period=self.config.htf_ema_period,
                swing_lookback=self.config.swing_lookback,
                equal_level_tolerance=self.config.equal_level_tolerance,
            )
        else:
            self.config = htf_analyzer_or_config
            self.htf_analyzer = HTFAnalyzer(
                ema_period=self.config.htf_ema_period,
                swing_lookback=self.config.swing_lookback,
                equal_level_tolerance=self.config.equal_level_tolerance,
            )
        self.detector = SMCEntryDetector(self.config)

    def default_sl(self, symbol: str, entry_price: float, direction: Direction, instrument: InstrumentConfig) -> float:
        dist = 25.0 * instrument.pip_size
        return entry_price - dist if direction == Direction.BUY else entry_price + dist

    def default_tp(self, symbol: str, entry_price: float, sl: float, direction: Direction) -> float:
        sl_dist = abs(entry_price - sl)
        return entry_price + (sl_dist * 3.0) if direction == Direction.BUY else entry_price - (sl_dist * 3.0)

    def evaluate(
        self,
        symbol: str,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        instrument: InstrumentConfig,
        current_spread: float,
        fixed_sl_pips: float | None = None,
        htf_analysis: HTFAnalysis | None = None,
    ) -> list[TradeSignal]:
        if htf_analysis is None:
            htf_analysis = self.htf_analyzer.analyze(htf_data)
        if htf_analysis.bias == MarketBias.NEUTRAL:
            return []

        raw = self.detector.detect_smc_entry(
            df=ltf_data,
            htf_analysis=htf_analysis,
            instrument=instrument,
            current_spread=current_spread,
        )
        if not raw:
            return []

        entry = raw['entry']
        direction = raw['direction']
        conf = raw['conf']

        if fixed_sl_pips is not None and fixed_sl_pips > 0:
            sl_dist = fixed_sl_pips * instrument.pip_size
            sl = entry - sl_dist if direction == Direction.BUY else entry + sl_dist
        else:
            sl = raw['sl']
            sl_dist = abs(entry - sl)

        tp = raw['tp']
        tp_dist = abs(entry - tp)
        min_buffer = max(1.0 * instrument.pip_size, 15.0 * instrument.pip_size if ("XAU" in instrument.symbol or "GOLD" in instrument.symbol) else 5.0 * instrument.pip_size)
        if sl_dist < min_buffer:
            sl_dist = min_buffer
            sl = entry - sl_dist if direction == Direction.BUY else entry + sl_dist

        # Strictly clamp swing target to [1.5, 3.0] R:R
        tp, rr_ratio = clamp_tp_to_rr(entry, sl, tp, direction, min_rr=1.5, max_rr=3.0)
        tp_dist = abs(entry - tp)

        # Quality scoring
        quality_score = min(25.0, htf_analysis.trend_clarity_score)
        if conf == LTFConfirmation.OB_PLUS_FVG:
            quality_score += 30.0
        elif conf == LTFConfirmation.OB_MITIGATION:
            quality_score += 25.0
        elif conf == LTFConfirmation.FVG_MITIGATION:
            quality_score += 20.0
        else:
            quality_score += 15.0
        quality_score += 15.0  # Invalidation quality
        quality_score += min(15.0, rr_ratio * 2.5)
        if current_spread > 0:
            quality_score += min(10.0, (tp_dist / current_spread) * 1.0)
        else:
            quality_score += 10.0
        quality_score = min(100.0, quality_score)

        return [TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            htf_bias=htf_analysis.bias,
            ltf_confirmation=conf,
            rr_ratio=rr_ratio,
            quality_score=quality_score,
            timestamp=raw['timestamp'],
            sl_distance=sl_dist,
            tp_distance=tp_dist,
            strategy_id=self.id,
            strategy_name=self.name,
            magic_number=123456 + self.magic_offset,
            candle_confirmation=raw.get('candle_confirmation', ''),
        )]


class SMCScalp5MStrategy(BaseStrategy):
    """Strategy B: 5M Break of Structure & Order Block Retest Scalp."""
    id = "SMC_SCALP_5M"
    name = "SMC Scalp (5m)"
    magic_offset = 2000

    def __init__(self, htf_analyzer_or_config: HTFAnalyzer | TimeframeConfig | TradingConfig, config_or_instrument: TimeframeConfig | InstrumentConfig | None = None):
        if isinstance(htf_analyzer_or_config, HTFAnalyzer):
            self.htf_analyzer = htf_analyzer_or_config
            self.config = config_or_instrument if isinstance(config_or_instrument, TimeframeConfig) else TimeframeConfig()
        elif hasattr(htf_analyzer_or_config, 'timeframes'):
            self.config = htf_analyzer_or_config.timeframes
            self.htf_analyzer = HTFAnalyzer(
                ema_period=self.config.htf_ema_period,
                swing_lookback=self.config.swing_lookback,
                equal_level_tolerance=self.config.equal_level_tolerance,
            )
        else:
            self.config = htf_analyzer_or_config
            self.htf_analyzer = HTFAnalyzer(
                ema_period=self.config.htf_ema_period,
                swing_lookback=self.config.swing_lookback,
                equal_level_tolerance=self.config.equal_level_tolerance,
            )
        self.engine = SMCScalp5MEngine()

    def default_sl(self, symbol: str, entry_price: float, direction: Direction, instrument: InstrumentConfig) -> float:
        dist = 15.0 * instrument.pip_size
        return entry_price - dist if direction == Direction.BUY else entry_price + dist

    def default_tp(self, symbol: str, entry_price: float, sl: float, direction: Direction) -> float:
        sl_dist = abs(entry_price - sl)
        return entry_price + (sl_dist * 1.5) if direction == Direction.BUY else entry_price - (sl_dist * 1.5)

    def evaluate(
        self,
        symbol: str,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        instrument: InstrumentConfig,
        current_spread: float,
        fixed_sl_pips: float | None = None,
        htf_analysis: HTFAnalysis | None = None,
    ) -> list[TradeSignal]:
        if htf_analysis is None:
            htf_analysis = self.htf_analyzer.analyze(htf_data)
        if htf_analysis.bias == MarketBias.NEUTRAL:
            return []

        raw = self.engine.detect_scalp_entry(
            df=ltf_data,
            htf_analysis=htf_analysis,
            instrument=instrument,
            current_spread=current_spread,
            fixed_sl_pips=fixed_sl_pips,
            enforce_session=getattr(self.config, 'scalp_enforce_session', False),
        )
        if not raw:
            return []

        entry = raw['entry']
        direction = raw['direction']
        conf = raw['conf']

        if fixed_sl_pips is not None and fixed_sl_pips > 0:
            sl_dist = fixed_sl_pips * instrument.pip_size
            sl = entry - sl_dist if direction == Direction.BUY else entry + sl_dist
        else:
            sl = raw['sl']
            sl_dist = abs(entry - sl)

        tp = raw['tp']
        tp, rr_ratio = clamp_tp_to_rr(entry, sl, tp, direction, min_rr=1.5, max_rr=3.0)
        tp_dist = abs(entry - tp)

        quality_score = min(25.0, htf_analysis.trend_clarity_score)
        quality_score += 35.0  # High-conviction BOS + OB retest
        quality_score += 15.0  # Invalidation quality
        quality_score += min(15.0, (rr_ratio / 1.5) * 15.0)
        if current_spread > 0:
            quality_score += min(10.0, (tp_dist / current_spread) * 1.0)
        else:
            quality_score += 10.0
        quality_score = min(100.0, quality_score)

        return [TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            htf_bias=htf_analysis.bias,
            ltf_confirmation=conf,
            rr_ratio=rr_ratio,
            quality_score=quality_score,
            timestamp=raw['timestamp'],
            sl_distance=sl_dist,
            tp_distance=tp_dist,
            strategy_id=self.id,
            strategy_name=self.name,
            magic_number=123456 + self.magic_offset,
            runner_tp=raw.get('runner_tp'),
            candle_confirmation=raw.get('candle_confirmation', ''),
        )]


class ICTStrategy(BaseStrategy):
    """Strategy C: Institutional ICT Time & Price Theory & AOX Model."""
    id = "ICT"
    name = "ICT"
    magic_offset = 3000

    def __init__(self, htf_analyzer_or_config: HTFAnalyzer | TimeframeConfig | TradingConfig, ict_config: ICTConfig | InstrumentConfig | None = None):
        if isinstance(htf_analyzer_or_config, HTFAnalyzer):
            self.htf_analyzer = htf_analyzer_or_config
            self.ict_config = ict_config if isinstance(ict_config, ICTConfig) else ICTConfig()
        elif hasattr(htf_analyzer_or_config, 'ict'):
            self.ict_config = htf_analyzer_or_config.ict
            self.htf_analyzer = HTFAnalyzer(
                ema_period=htf_analyzer_or_config.timeframes.htf_ema_period,
                swing_lookback=htf_analyzer_or_config.timeframes.swing_lookback,
                equal_level_tolerance=htf_analyzer_or_config.timeframes.equal_level_tolerance,
            )
        else:
            self.ict_config = ict_config if isinstance(ict_config, ICTConfig) else ICTConfig()
            self.htf_analyzer = HTFAnalyzer()
        self.engine = ICTEngine(self.ict_config)

    def default_sl(self, symbol: str, entry_price: float, direction: Direction, instrument: InstrumentConfig) -> float:
        dist = 20.0 * instrument.pip_size
        return entry_price - dist if direction == Direction.BUY else entry_price + dist

    def default_tp(self, symbol: str, entry_price: float, sl: float, direction: Direction) -> float:
        sl_dist = abs(entry_price - sl)
        return entry_price + (sl_dist * 2.0) if direction == Direction.BUY else entry_price - (sl_dist * 2.0)

    def evaluate(
        self,
        symbol: str,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        instrument: InstrumentConfig,
        current_spread: float,
        fixed_sl_pips: float | None = None,
        htf_analysis: HTFAnalysis | None = None,
        correlated_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> list[TradeSignal]:
        if htf_analysis is None:
            htf_analysis = self.htf_analyzer.analyze(htf_data)

        # Allow evaluation if bias is Bullish/Bearish or neutral with deep discount/premium
        aox_setups = self.engine.detect_aox_setup(
            df=ltf_data,
            htf_analysis=htf_analysis,
            instrument=instrument,
            current_spread=current_spread,
            fixed_sl_pips=fixed_sl_pips,
            correlated_df=correlated_data,
        )

        if not aox_setups:
            # Fallback to single detect_ict_entry for backward compatibility
            single = self.engine.detect_ict_entry(
                df=ltf_data,
                htf_analysis=htf_analysis,
                instrument=instrument,
                current_spread=current_spread,
                fixed_sl_pips=fixed_sl_pips,
                correlated_df=correlated_data,
            )
            if single:
                aox_setups = [single]
            else:
                return []

        signals: list[TradeSignal] = []
        for raw in aox_setups:
            entry = raw['entry']
            direction = raw['direction']
            conf = raw['conf']
            layer = raw.get('layer', 1)
            risk_frac = raw.get('risk_fraction', 1.0)

            if fixed_sl_pips is not None and fixed_sl_pips > 0:
                sl_dist = fixed_sl_pips * instrument.pip_size
                sl = entry - sl_dist if direction == Direction.BUY else entry + sl_dist
            else:
                sl = raw['sl']
                sl_dist = abs(entry - sl)

            tp = raw['tp']
            tp, rr_ratio = clamp_tp_to_rr(entry, sl, tp, direction, min_rr=1.5, max_rr=3.0)
            tp_dist = abs(entry - tp)

            runner_tp = raw.get('tp2')
            if runner_tp is not None:
                runner_tp, _ = clamp_tp_to_rr(entry, sl, runner_tp, direction, min_rr=1.5, max_rr=4.5)

            # Quality scoring
            quality_score = min(25.0, htf_analysis.trend_clarity_score) if htf_analysis else 20.0
            if conf == LTFConfirmation.ICT_AOX_SMT_TURTLE_SOUP or raw.get('smt_confirmed', False):
                quality_score += 40.0
            elif conf == LTFConfirmation.ICT_AOX_SNIPER_TAP:
                quality_score += 35.0
            elif conf == LTFConfirmation.ICT_AOX_HIPPO_MITIGATION:
                quality_score += 30.0
            elif conf in (LTFConfirmation.ICT_SILVER_BULLET, LTFConfirmation.ICT_KILLZONE_FVG):
                quality_score += 30.0
            else:
                quality_score += 25.0

            quality_score += 15.0  # Invalidation quality
            quality_score += min(15.0, (rr_ratio / 2.0) * 15.0)
            if current_spread > 0:
                quality_score += min(10.0, (tp_dist / current_spread) * 1.0)
            else:
                quality_score += 10.0
            quality_score = min(100.0, quality_score)

            # Magic number offset: 3000 base + layer offset (layer 1 -> 3000, layer 2 -> 3001, layer 3 -> 3002)
            layer_offset = (layer - 1) if layer in (1, 2, 3) else 0

            signals.append(TradeSignal(
                symbol=symbol,
                direction=direction,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                htf_bias=htf_analysis.bias if htf_analysis else MarketBias.NEUTRAL,
                ltf_confirmation=conf,
                rr_ratio=rr_ratio,
                quality_score=quality_score,
                timestamp=raw['timestamp'],
                sl_distance=sl_dist,
                tp_distance=tp_dist,
                strategy_id=self.id,
                strategy_name=self.name,
                magic_number=123456 + self.magic_offset + layer_offset,
                runner_tp=runner_tp,
                candle_confirmation=raw.get('candle_confirmation', ''),
                layer=layer,
                risk_fraction=risk_frac,
                aox_tp1=raw.get('tp1'),
                aox_tp2=raw.get('tp2'),
                session_killzone=raw.get('kz', ''),
                smt_confirmed=raw.get('smt_confirmed', False),
                sd_zone=raw.get('sd_zone', ''),
            ))

        return signals


class OrderFlowStrategy(BaseStrategy):
    """Strategy D: Order Flow Volume Delta, CVD Divergence & Absorption Model."""
    id = "ORDER_FLOW"
    name = "Order Flow (Delta & Absorption)"
    magic_offset = 4000

    def __init__(self, htf_analyzer_or_config: HTFAnalyzer | TimeframeConfig | TradingConfig, of_config: OrderFlowConfig | InstrumentConfig | None = None):
        if isinstance(htf_analyzer_or_config, HTFAnalyzer):
            self.htf_analyzer = htf_analyzer_or_config
            self.of_config = of_config if isinstance(of_config, OrderFlowConfig) else OrderFlowConfig()
        elif hasattr(htf_analyzer_or_config, 'order_flow'):
            self.of_config = htf_analyzer_or_config.order_flow
            self.htf_analyzer = HTFAnalyzer(
                ema_period=htf_analyzer_or_config.timeframes.htf_ema_period,
                swing_lookback=htf_analyzer_or_config.timeframes.swing_lookback,
                equal_level_tolerance=htf_analyzer_or_config.timeframes.equal_level_tolerance,
            )
        else:
            self.of_config = of_config if isinstance(of_config, OrderFlowConfig) else OrderFlowConfig()
            self.htf_analyzer = HTFAnalyzer()
        self.engine = OrderFlowEngine(self.of_config)

    def default_sl(self, symbol: str, entry_price: float, direction: Direction, instrument: InstrumentConfig) -> float:
        dist = 20.0 * instrument.pip_size
        return entry_price - dist if direction == Direction.BUY else entry_price + dist

    def default_tp(self, symbol: str, entry_price: float, sl: float, direction: Direction) -> float:
        sl_dist = abs(entry_price - sl)
        return entry_price + (sl_dist * 2.2) if direction == Direction.BUY else entry_price - (sl_dist * 2.2)

    def evaluate(
        self,
        symbol: str,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        instrument: InstrumentConfig,
        current_spread: float,
        fixed_sl_pips: float | None = None,
        htf_analysis: HTFAnalysis | None = None,
    ) -> list[TradeSignal]:
        if htf_analysis is None:
            htf_analysis = self.htf_analyzer.analyze(htf_data)
        if htf_analysis.bias == MarketBias.NEUTRAL and getattr(self.of_config, 'enforce_htf_alignment', True):
            return []

        raw = self.engine.detect_order_flow_entry(
            df=ltf_data,
            htf_analysis=htf_analysis,
            instrument=instrument,
            current_spread=current_spread,
            fixed_sl_pips=fixed_sl_pips,
        )
        if not raw:
            return []

        entry = raw['entry']
        direction = raw['direction']
        conf = raw['conf']

        if fixed_sl_pips is not None and fixed_sl_pips > 0:
            sl_dist = fixed_sl_pips * instrument.pip_size
            sl = entry - sl_dist if direction == Direction.BUY else entry + sl_dist
        else:
            sl = raw['sl']
            sl_dist = abs(entry - sl)

        tp = raw['tp']
        tp, rr_ratio = clamp_tp_to_rr(entry, sl, tp, direction, min_rr=1.5, max_rr=3.0)
        tp_dist = abs(entry - tp)

        # Quality scoring
        quality_score = min(25.0, htf_analysis.trend_clarity_score)
        if conf == LTFConfirmation.OF_LIQUIDITY_TRAP:
            quality_score += 35.0
        elif conf == LTFConfirmation.OF_ABSORPTION:
            quality_score += 30.0
        elif conf == LTFConfirmation.OF_DELTA_DIVERGENCE:
            quality_score += 28.0
        else:
            quality_score += 25.0

        quality_score += 15.0  # Invalidation quality
        quality_score += min(15.0, (rr_ratio / 2.0) * 15.0)
        if current_spread > 0:
            quality_score += min(10.0, (tp_dist / current_spread) * 1.0)
        else:
            quality_score += 10.0
        quality_score = min(100.0, quality_score)

        return [TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            htf_bias=htf_analysis.bias,
            ltf_confirmation=conf,
            rr_ratio=rr_ratio,
            quality_score=quality_score,
            timestamp=raw['timestamp'],
            sl_distance=sl_dist,
            tp_distance=tp_dist,
            strategy_id=self.id,
            strategy_name=self.name,
            magic_number=123456 + self.magic_offset,
            candle_confirmation=raw.get('candle_confirmation', ''),
        )]


class TrendReversalStrategy(BaseStrategy):
    """
    Trend Reversal Strategy (Multi-TF Reversal Zones: SD, Fib 0.5-0.6, FVG in 4H/1H/Daily).

    Executes trades at the biased direction of identified institutional reversal zones:
    1. ICT Standard Deviation Projections (-2.0, -2.5, -4.0 SD).
    2. Fibonacci 0.5 to 0.6 Retracement / Equilibrium Zone.
    3. Fair Value Gaps (FVG) across 4H, 1H, and Daily.
    4. "Below 0.5 Level" Rule:
       - Any key level (FVG, liquidity sweep, SD extension) below 0.5 Fibonacci represents
         high-probability Discount accumulation -> Bullish Reversal (BUY).
       - Any key level above 0.5 Fibonacci represents Premium distribution -> Bearish Reversal (SELL).
    5. Sets structural stop loss; position manager locks SL at cost-to-cost (+0.3R).
    """

    def __init__(self, htf_analyzer: HTFAnalyzer, detector: Any | None = None):
        self.htf_analyzer = htf_analyzer
        from strategies.trend_reversal import TrendReversalDetector
        self.detector = detector or TrendReversalDetector(swing_lookback=3)
        self.magic_offset = 5000  # Magic number: 123456 + 5000 = 128456
        self.id = "TREND_REVERSAL"
        self.name = "Trend Reversal Strategy"

    def default_sl(self, symbol: str, entry_price: float, direction: Direction, instrument: InstrumentConfig) -> float:
        dist = 20.0 * (instrument.pip_size if hasattr(instrument, 'pip_size') else 0.0001)
        return entry_price - dist if direction == Direction.BUY else entry_price + dist

    def default_tp(self, symbol: str, entry_price: float, sl: float, direction: Direction) -> float:
        sl_dist = abs(entry_price - sl)
        return entry_price + (sl_dist * 2.5) if direction == Direction.BUY else entry_price - (sl_dist * 2.5)

    def evaluate(
        self,
        symbol: str,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        instrument: InstrumentConfig,
        current_spread: float,
        fixed_sl_pips: float | None = None,
        htf_analysis: HTFAnalysis | None = None,
        dfs_by_tf: dict[str, pd.DataFrame] | None = None,
    ) -> list[TradeSignal]:
        if htf_data is None or len(htf_data) < 25 or ltf_data is None or len(ltf_data) < 5:
            return []

        curr_price = float(ltf_data['close'].iloc[-1])
        htf_an = htf_analysis or self.htf_analyzer.analyze(htf_data)
        bias = htf_an.bias if htf_an else None

        # Analyze reversal zones (multi-TF if dictionary provided, else HTF)
        if dfs_by_tf and len(dfs_by_tf) > 1:
            analysis = self.detector.analyze_multitf(
                dfs=dfs_by_tf,
                trend=bias,
                symbol=symbol,
                htf_analysis=htf_an,
                current_price=curr_price,
            )
        else:
            analysis = self.detector.analyze(
                df=htf_data,
                trend=bias,
                symbol=symbol,
                htf_analysis=htf_an,
                timeframe="1H",
                current_price=curr_price,
            )

        if not analysis.choch_detected and not getattr(analysis, "reversal_detected", False):
            return []

        if analysis.reversal_probability < 40.0:
            return []

        # Determine biased direction of reversal or dealing range retracement
        rev_type = getattr(analysis, "reversal_type", None) or analysis.choch_type
        from strategies.trend_reversal import CHoCHType, ReversalStage
        if bias == MarketBias.BULLISH and getattr(htf_an, 'is_discount', False):
            direction = Direction.BUY
            dr_entry_zone = getattr(htf_an, 'first_zone_entry', None)
        elif bias == MarketBias.BEARISH and getattr(htf_an, 'is_premium', False):
            direction = Direction.SELL
            dr_entry_zone = getattr(htf_an, 'first_zone_entry', None)
        elif rev_type == CHoCHType.BULLISH:
            direction = Direction.BUY
            dr_entry_zone = None
        elif rev_type == CHoCHType.BEARISH:
            direction = Direction.SELL
            dr_entry_zone = None
        else:
            return []

        # ── Dealing Range & 0.5 Fib Equilibrium Rule ──
        # 1. Buy trades ONLY executed in Discount (below 0.5 level)
        # 2. Sell trades ONLY executed in Premium (above 0.5 level)
        # 3. Setup invalidated if all levels fail
        if direction == Direction.BUY:
            if getattr(htf_an, 'fib_50', None) is not None and curr_price > htf_an.fib_50 + 1e-5:
                return []
            if getattr(htf_an, 'all_zones_failed', False):
                return []
        elif direction == Direction.SELL:
            if getattr(htf_an, 'fib_50', None) is not None and curr_price < htf_an.fib_50 - 1e-5:
                return []
            if getattr(htf_an, 'all_zones_failed', False):
                return []

        # Sizing and price calculation
        atr_series = compute_atr(ltf_data, 14)
        ltf_atr = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 0.001

        # ── Zone Mitigation Check: Price must reach the 1st FVG, liquidity area, or SD zone ──
        is_stage_confirmed = analysis.stage in (
            ReversalStage.RETRACEMENT_IN_ZONE,
            ReversalStage.REVERSAL_ZONE_TESTED,
            ReversalStage.CONFIRMED_MSS,
            ReversalStage.CHOCH_DISPLACEMENT,
        ) or getattr(analysis.confluence, 'in_retracement_zone', False)

        entry_zone = dr_entry_zone or analysis.entry_zone or getattr(htf_an, 'first_zone_entry', None)
        if not is_stage_confirmed and entry_zone is not None:
            z_low, z_high = min(entry_zone), max(entry_zone)
            atr_buf = ltf_atr * 0.35
            candle_low = float(ltf_data['low'].iloc[-1])
            candle_high = float(ltf_data['high'].iloc[-1])
            prev_low = float(ltf_data['low'].iloc[-2]) if len(ltf_data) >= 2 else candle_low
            prev_high = float(ltf_data['high'].iloc[-2]) if len(ltf_data) >= 2 else candle_high

            if direction == Direction.BUY:
                tested_zone = (min(candle_low, prev_low) <= z_high + atr_buf) and (curr_price >= z_low - atr_buf)
                if not tested_zone and not getattr(analysis.confluence, 'is_first_zone_below_50', False):
                    return []
            elif direction == Direction.SELL:
                tested_zone = (max(candle_high, prev_high) >= z_low - atr_buf) and (curr_price <= z_high + atr_buf)
                if not tested_zone and not getattr(analysis.confluence, 'is_first_zone_above_50', False):
                    return []

        # Directional Candlestick Confirmation on current LTF candle
        is_confirmed, body_ratio, conf_desc = check_candle_body_confirmation(
            ltf_data.iloc[-1], direction, min_body_ratio=0.50
        )
        if not is_confirmed:
            return []
        point_size = 10 ** -instrument.digits

        entry = curr_price

        # Structural Stop Loss
        if fixed_sl_pips is not None and fixed_sl_pips > 0:
            pip_unit = 10 * point_size if instrument.digits in (3, 5) else point_size
            sl_dist = fixed_sl_pips * pip_unit
            sl = entry - sl_dist if direction == Direction.BUY else entry + sl_dist
        elif analysis.suggested_sl and (
            (direction == Direction.BUY and analysis.suggested_sl < entry) or
            (direction == Direction.SELL and analysis.suggested_sl > entry)
        ):
            sl = analysis.suggested_sl
            sl_dist = abs(entry - sl)
        else:
            # Use dealing range invalidation level or ATR buffer
            if direction == Direction.BUY and getattr(htf_an, 'invalidation_level', None) and htf_an.invalidation_level < entry:
                sl = htf_an.invalidation_level
            elif direction == Direction.SELL and getattr(htf_an, 'invalidation_level', None) and htf_an.invalidation_level > entry:
                sl = htf_an.invalidation_level
            else:
                sl_dist = max(2.0 * ltf_atr, 15.0 * point_size)
                sl = entry - sl_dist if direction == Direction.BUY else entry + sl_dist

        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            return []

        # Take Profit targeting equilibrium or opposing liquidity pool (strictly clamped to [1.5, 3.0] R:R)
        if analysis.suggested_tp and (
            (direction == Direction.BUY and analysis.suggested_tp > entry and (analysis.suggested_tp - entry) >= 1.5 * sl_dist) or
            (direction == Direction.SELL and analysis.suggested_tp < entry and (entry - analysis.suggested_tp) >= 1.5 * sl_dist)
        ):
            raw_tp = analysis.suggested_tp
        else:
            if direction == Direction.BUY and getattr(htf_an, 'dealing_range_high', None) and htf_an.dealing_range_high > entry + 1.5 * sl_dist:
                raw_tp = htf_an.dealing_range_high
            elif direction == Direction.SELL and getattr(htf_an, 'dealing_range_low', None) and htf_an.dealing_range_low < entry - 1.5 * sl_dist:
                raw_tp = htf_an.dealing_range_low
            else:
                raw_tp = entry + 2.0 * sl_dist if direction == Direction.BUY else entry - 2.0 * sl_dist

        tp, rr_ratio = clamp_tp_to_rr(entry, sl, raw_tp, direction, min_rr=1.5, max_rr=3.0)
        tp_dist = abs(tp - entry)
        if rr_ratio < 1.5:
            return []

        # Determine LTF Confirmation enum
        z_type = getattr(analysis, "reversal_zone_type", "") or ""
        if "SD" in z_type or getattr(analysis.confluence, "sd_4_to_4_5_hit", False):
            conf = LTFConfirmation.REVERSAL_ZONE_SD
        elif "FIB" in z_type or getattr(analysis.confluence, "in_fib_50_60_zone", False):
            conf = LTFConfirmation.REVERSAL_ZONE_FIB
        elif "FVG" in z_type or getattr(analysis.confluence, "fvg_present", False):
            conf = LTFConfirmation.REVERSAL_ZONE_FVG
        else:
            conf = LTFConfirmation.REVERSAL_CONFLUENCE

        # Priority metadata enrichment
        p_rank = getattr(analysis, "priority_rank", 1)
        p_type = getattr(analysis, "priority_zone_type", "") or ""
        enhanced_conf_desc = conf_desc
        if p_type and p_type != "NONE":
            enhanced_conf_desc = f"[P{p_rank}:{p_type}] {conf_desc}"

        quality_sc = analysis.reversal_probability
        if getattr(analysis.confluence, "is_first_zone_below_50", False) or getattr(analysis.confluence, "is_first_zone_above_50", False):
            quality_sc = max(quality_sc, 75.0)
        if getattr(analysis.confluence, "sd_4_to_4_5_hit", False):
            quality_sc = min(98.0, quality_sc + 5.0)

        # Timestamp
        raw_time = ltf_data['time'].iloc[-1] if 'time' in ltf_data.columns else datetime.now(timezone.utc)
        if isinstance(raw_time, str):
            ts = pd.to_datetime(raw_time, utc=True).to_pydatetime()
        elif hasattr(raw_time, 'to_pydatetime'):
            ts = raw_time.to_pydatetime()
        else:
            ts = datetime.now(timezone.utc)

        return [TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            htf_bias=MarketBias.BULLISH if direction == Direction.BUY else MarketBias.BEARISH,
            ltf_confirmation=conf,
            rr_ratio=rr_ratio,
            quality_score=min(98.0, quality_sc),
            timestamp=ts,
            sl_distance=sl_dist,
            tp_distance=tp_dist,
            strategy_id=self.id,
            strategy_name=self.name,
            magic_number=123456 + self.magic_offset,
            candle_confirmation=enhanced_conf_desc,
        )]


# ─────────────────────────────────────────────
#  Strategy Orchestrator Engine
# ─────────────────────────────────────────────

class StrategyEngine:
    """Orchestrates HTF macro structure and executes registered strategies."""

    def __init__(self, config: TradingConfig | TimeframeConfig | None = None, instrument: InstrumentConfig | None = None, ict_config: ICTConfig | None = None, of_config: OrderFlowConfig | None = None):
        if config is None:
            from config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG

        if hasattr(config, 'timeframes'):
            self.tf_config = config.timeframes
            self.ict_config = ict_config or getattr(config, 'ict', ICTConfig())
            self.of_config = of_config or getattr(config, 'order_flow', OrderFlowConfig())
            self.trading_config = config
        else:
            self.tf_config = config
            self.ict_config = ict_config or ICTConfig()
            self.of_config = of_config or OrderFlowConfig()
            self.trading_config = None

        self.instrument = instrument
        self.htf_analyzer = HTFAnalyzer(
            ema_period=self.tf_config.htf_ema_period,
            swing_lookback=self.tf_config.swing_lookback,
            equal_level_tolerance=self.tf_config.equal_level_tolerance,
        )

        # Strategy registry
        self.strategies: dict[str, BaseStrategy] = {
            "SMC": SMCSwingStrategy(self.htf_analyzer, self.tf_config),
            "SMC_SCALP_5M": SMCScalp5MStrategy(self.htf_analyzer, self.tf_config),
            "ICT": ICTStrategy(self.htf_analyzer, self.ict_config),
            "ORDER_FLOW": OrderFlowStrategy(self.htf_analyzer, self.of_config),
            "TREND_REVERSAL": TrendReversalStrategy(self.htf_analyzer),
        }
        if self.trading_config and hasattr(self.trading_config, 'enabled_strategies'):
            self.enabled_strategies: list[str] = [s for s in self.trading_config.enabled_strategies if s in self.strategies]
        else:
            self.enabled_strategies: list[str] = ["SMC", "SMC_SCALP_5M", "ICT", "ORDER_FLOW", "TREND_REVERSAL"]

    @property
    def active_strategies(self) -> list[BaseStrategy]:
        return [self.strategies[k] for k in self.enabled_strategies if k in self.strategies]

    def set_enabled_strategies(self, strategy_types: list[str | StrategyType]):
        """Update active strategies list."""
        self.enabled_strategies = [s.value if hasattr(s, 'value') else str(s) for s in strategy_types]

    def evaluate_all(
        self,
        symbol: str | None = None,
        htf_data: pd.DataFrame | None = None,
        ltf_data: pd.DataFrame | None = None,
        instrument: InstrumentConfig | None = None,
        current_spread: float = 0.0,
        fixed_sl_pips: float | None = None,
        enabled_strategies: list[str] | None = None,
        bars_15m: pd.DataFrame | None = None,
        bars_1h: pd.DataFrame | None = None,
        bars_5m: pd.DataFrame | None = None,
        current_spread_points: float | None = None,
        account_balance: float | None = None,
        htf_analysis: HTFAnalysis | None = None,
        dfs_by_tf: dict[str, pd.DataFrame] | None = None,
        correlated_data: pd.DataFrame | None = None,
    ) -> list[TradeSignal]:
        """Runs all enabled strategies concurrently and returns aggregate signals."""
        inst = instrument or self.instrument
        sym = symbol or (inst.symbol if inst else "XAUUSD")
        htf = htf_data if htf_data is not None else bars_1h
        ltf = ltf_data if ltf_data is not None else (bars_15m if bars_15m is not None else bars_5m)
        spread = current_spread if current_spread > 0 else (current_spread_points or 0.0)

        targets = enabled_strategies if enabled_strategies else self.enabled_strategies
        all_signals: list[TradeSignal] = []

        if htf is None or ltf is None:
            return []

        # Precompute HTF analysis once if not provided, and share across strategies
        shared_htf_analysis = htf_analysis if htf_analysis is not None else self.htf_analyzer.analyze(htf)

        for strat_id in targets:
            strat = self.strategies.get(strat_id)
            if strat is None:
                continue
            try:
                # If scalping or order flow strategy and 5m bars available, prefer 5m bars as LTF
                active_ltf = bars_5m if (strat_id in ("SMC_SCALP_5M", "ORDER_FLOW") and bars_5m is not None) else ltf
                eval_kwargs = {
                    "symbol": sym,
                    "htf_data": htf,
                    "ltf_data": active_ltf,
                    "instrument": inst,
                    "current_spread": spread,
                    "fixed_sl_pips": fixed_sl_pips,
                    "htf_analysis": shared_htf_analysis,
                }
                if strat_id == "TREND_REVERSAL" and dfs_by_tf:
                    eval_kwargs["dfs_by_tf"] = dfs_by_tf
                if strat_id == "ICT" and correlated_data is not None:
                    eval_kwargs["correlated_data"] = correlated_data
                sigs = strat.evaluate(**eval_kwargs)
                all_signals.extend(sigs)
            except Exception as e:
                logger.error(f"Error evaluating strategy {strat_id} on {sym}: {e}")

        # Master post-processor: strictly enforce [1.5, 3.0] R:R bounds on all signals
        clamped_signals: list[TradeSignal] = []
        for sig in all_signals:
            if sig.strategy_id == "ICT":
                # Preserve AOX Model exact TP1 / TP2 & SD projection targets
                clamped_signals.append(sig)
                continue
            clamped_tp, eff_rr = clamp_tp_to_rr(
                sig.entry_price, sig.stop_loss, sig.take_profit, sig.direction, min_rr=1.5, max_rr=3.0
            )
            sig.take_profit = clamped_tp
            sig.tp_distance = abs(clamped_tp - sig.entry_price)
            sig.rr_ratio = eff_rr
            if sig.runner_tp is not None:
                clamped_runner, _ = clamp_tp_to_rr(
                    sig.entry_price, sig.stop_loss, sig.runner_tp, sig.direction, min_rr=1.5, max_rr=3.0
                )
                sig.runner_tp = clamped_runner
            clamped_signals.append(sig)

        return clamped_signals

    def generate_signals(
        self,
        symbol: str,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        instrument: InstrumentConfig,
        current_spread: float,
        fixed_sl_pips: float | None = None,
        strategy_type: str = "SMC",
        enabled_strategies: list[str] | None = None,
    ) -> list[TradeSignal]:
        """Backward-compatible signal generation method."""
        if enabled_strategies:
            return self.evaluate_all(
                symbol=symbol,
                htf_data=htf_data,
                ltf_data=ltf_data,
                instrument=instrument,
                current_spread=current_spread,
                fixed_sl_pips=fixed_sl_pips,
                enabled_strategies=enabled_strategies,
            )
        else:
            return self.evaluate_all(
                symbol=symbol,
                htf_data=htf_data,
                ltf_data=ltf_data,
                instrument=instrument,
                current_spread=current_spread,
                fixed_sl_pips=fixed_sl_pips,
                enabled_strategies=[strategy_type],
            )



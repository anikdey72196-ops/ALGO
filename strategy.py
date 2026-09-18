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
from datetime import datetime, timezone
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger

from config import MarketBias, Direction, TimeframeConfig, InstrumentConfig, StrategyType, ICTConfig



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
    PULLBACK = "PULLBACK"                 # Compatibility fallback
    STRUCTURAL_BREAK = "STRUCTURAL_BREAK" # Compatibility fallback
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



# ─────────────────────────────────────────────
#  Mathematical & Indicator Helpers
# ─────────────────────────────────────────────

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
        elif structure_bias != MarketBias.NEUTRAL or is_above_ema or is_below_ema:
            trend_clarity_score = 15.0

        atr_series = compute_atr(df, 14)
        current_atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

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
                # Price is currently mitigating the order block
                if current_low <= target_ob.high and current_close >= target_ob.low:
                    ob_touched = True

            if fvgs:
                for fvg in reversed(fvgs):
                    if fvg.direction == Direction.BUY:
                        if current_low <= fvg.top and current_close >= fvg.bottom:
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

            # Stop-loss tight beyond invalidation point (sweep extreme)
            sl_buffer = max(1.0 * instrument.pip_size, current_atr * 0.1)
            stop_loss = min(sweep_extreme, swept_level) - sl_buffer

            # Target liquidity pool or HTF supply zone
            take_profit = 0.0
            opposing_pools = [p.level for p in htf_analysis.liquidity_pools if p.is_high and p.level > current_close]
            opposing_zones = [z.bottom for z in htf_analysis.supply_demand_zones if z.is_supply and z.bottom > current_close]

            if opposing_pools:
                take_profit = min(opposing_pools)
            elif opposing_zones:
                take_profit = min(opposing_zones)
            elif htf_analysis.last_swing_high and htf_analysis.last_swing_high > current_close:
                take_profit = htf_analysis.last_swing_high
            else:
                take_profit = current_close + (current_close - stop_loss) * 3.5

            return {
                'direction': Direction.BUY,
                'entry': current_close,
                'sl': stop_loss,
                'tp': take_profit,
                'conf': conf_type,
                'timestamp': timestamp,
                'sweep_depth': swept_level - sweep_extreme,
            }

        elif bias == MarketBias.BEARISH:
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

            if fvgs:
                for fvg in reversed(fvgs):
                    if fvg.direction == Direction.SELL:
                        if current_high >= fvg.bottom and current_close <= fvg.top:
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

            sl_buffer = max(1.0 * instrument.pip_size, current_atr * 0.1)
            stop_loss = max(sweep_extreme, swept_level) + sl_buffer

            take_profit = 0.0
            opposing_pools = [p.level for p in htf_analysis.liquidity_pools if not p.is_high and p.level < current_close]
            opposing_zones = [z.top for z in htf_analysis.supply_demand_zones if not z.is_supply and z.top < current_close]

            if opposing_pools:
                take_profit = max(opposing_pools)
            elif opposing_zones:
                take_profit = max(opposing_zones)
            elif htf_analysis.last_swing_low and htf_analysis.last_swing_low < current_close:
                take_profit = htf_analysis.last_swing_low
            else:
                take_profit = current_close - (stop_loss - current_close) * 3.5

            return {
                'direction': Direction.SELL,
                'entry': current_close,
                'sl': stop_loss,
                'tp': take_profit,
                'conf': conf_type,
                'timestamp': timestamp,
                'sweep_depth': sweep_extreme - swept_level,
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
    ):
        self.swing_lookback = swing_lookback
        self.session_start_utc = session_start_utc
        self.session_end_utc = session_end_utc
        self.target_rr = target_rr

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
                ob_idx = int(df.iloc[search_start:bos_bar_idx]['low'].astype(float).idxmin())

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

            # Invalidation / Stop Loss: Just beyond the wick of the OB candle
            sl_buffer = max(1.0 * instrument.pip_size, current_atr * 0.1)
            stop_loss = ob_low - sl_buffer
            entry_price = current_close

            if fixed_sl_pips is not None and fixed_sl_pips > 0:
                sl_distance = fixed_sl_pips * instrument.pip_size
                stop_loss = entry_price - sl_distance
            else:
                sl_distance = abs(entry_price - stop_loss)

            min_buffer = 1.0 * instrument.pip_size
            if sl_distance < min_buffer:
                sl_distance = min_buffer
                stop_loss = entry_price - sl_distance

            # Take Profit: 1.5R target for partial profit
            take_profit = entry_price + (sl_distance * self.target_rr)

            # Potential runner target to the next 5M liquidity pool
            opposing_pools = [sp.price for sp in swing_highs if sp.price > take_profit]
            runner_tp = min(opposing_pools) if opposing_pools else take_profit

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
            }

        elif bias == MarketBias.BEARISH:
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
                ob_idx = int(df.iloc[search_start:bos_bar_idx]['high'].astype(float).idxmax())

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

            # Invalidation / Stop Loss: Just beyond the wick of the OB candle
            sl_buffer = max(1.0 * instrument.pip_size, current_atr * 0.1)
            stop_loss = ob_high + sl_buffer
            entry_price = current_close

            if fixed_sl_pips is not None and fixed_sl_pips > 0:
                sl_distance = fixed_sl_pips * instrument.pip_size
                stop_loss = entry_price + sl_distance
            else:
                sl_distance = abs(stop_loss - entry_price)

            min_buffer = 1.0 * instrument.pip_size
            if sl_distance < min_buffer:
                sl_distance = min_buffer
                stop_loss = entry_price + sl_distance

            # Take Profit: 1.5R target for partial profit
            take_profit = entry_price - (sl_distance * self.target_rr)

            # Potential runner target to the next 5M liquidity pool
            opposing_pools = [sp.price for sp in swing_lows if sp.price < take_profit]
            runner_tp = max(opposing_pools) if opposing_pools else take_profit

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
            }

        return None


# ─────────────────────────────────────────────
#  ICT (Inner Circle Trader) Strategy Engine
# ─────────────────────────────────────────────

class ICTKillZone(str, Enum):
    """ICT Defined Trading Kill Zones."""
    LONDON_OPEN = "LONDON_KILLZONE"
    NY_AM = "NY_AM_KILLZONE"
    SILVER_BULLET = "NY_SILVER_BULLET"
    LONDON_CLOSE = "LONDON_CLOSE_KILLZONE"
    ASIA = "ASIAN_SESSION"
    NONE = "NONE"


class ICTEngine:
    """
    Inner Circle Trader (ICT) Methodology Engine.
    
    Core ICT Pillars:
    1. Kill Zones (Time-Based Institutional Order Flow Delivery):
       - London Kill Zone: 07:00 - 10:00 UTC
       - New York AM Kill Zone: 12:00 - 15:00 UTC
       - Silver Bullet Hour: 14:00 - 15:00 UTC
       - London Close: 15:00 - 17:00 UTC
    2. Liquidity Runs & Judas Swing (Stop hunts above/below session extremes).
    3. Market Structure Shift (MSS): Impulsive displacement break of swing structure.
    4. Fair Value Gaps (FVG) & Optimal Trade Entry (OTE - 61.8% to 78.6% Fib).
    5. External Range Liquidity (ERL) vs. Internal Range Liquidity (IRL) delivery.
    """

    def __init__(self, config: ICTConfig | None = None):
        self.config = config or ICTConfig()

    def identify_kill_zone(self, current_time: datetime) -> ICTKillZone:
        """Identify which ICT Kill Zone enum is active at current_time UTC."""
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        hour = current_time.hour

        if self.config.silver_bullet_start_utc <= hour < self.config.silver_bullet_end_utc:
            return ICTKillZone.SILVER_BULLET
        elif self.config.london_kz_start_utc <= hour < self.config.london_kz_end_utc:
            return ICTKillZone.LONDON_OPEN
        elif self.config.ny_am_kz_start_utc <= hour < self.config.ny_am_kz_end_utc:
            return ICTKillZone.NY_AM
        elif self.config.london_close_start_utc <= hour < self.config.london_close_end_utc:
            return ICTKillZone.LONDON_CLOSE
        elif 0 <= hour < 6:
            return ICTKillZone.ASIA
        return ICTKillZone.NONE

    def get_active_killzone(self, current_time: datetime) -> str | None:
        """Identify which ICT Kill Zone is active at current_time UTC (string name)."""
        kz = self.identify_kill_zone(current_time)
        return kz.value if kz != ICTKillZone.NONE and kz != ICTKillZone.ASIA else None

    def detect_fvg(self, df: pd.DataFrame, index: int = -1, direction: Direction = Direction.BUY) -> dict | None:
        """Detect a Fair Value Gap (FVG) at the specified bar index in a DataFrame."""
        n = len(df)
        if index < 0:
            index = n + index
        if index < 2 or index >= n:
            return None

        c_curr = df.iloc[index]
        c_p2 = df.iloc[index - 2]

        if direction == Direction.BUY:
            # Bullish FVG: Low of candle[i] > High of candle[i-2]
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
            # Bearish FVG: High of candle[i] < Low of candle[i-2]
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
        """Calculate the Optimal Trade Entry (61.8% to 78.6% Fib) zone."""
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

    def detect_ict_entry(
        self,
        df: pd.DataFrame,
        htf_analysis: HTFAnalysis,
        instrument: InstrumentConfig,
        current_spread: float,
        fixed_sl_pips: float | None = None,
    ) -> dict | None:
        """
        Scan for high-probability ICT Setup:
        1. Check active Kill Zone (or 24/7 if enforce_killzones=False).
        2. Detect Liquidity Sweep / Judas Swing against the HTF trend.
        3. Confirm Market Structure Shift (MSS) with displacement.
        4. Detect FVG tap or OTE (61.8% - 78.6% Fib) mitigation entry.
        5. Target opposing External Range Liquidity (ERL).
        """
        n = len(df)
        if n < 15:
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

        active_kz = self.get_active_killzone(timestamp)
        if self.config.enforce_killzones and not active_kz:
            logger.debug(f"[ICT] Outside Kill Zones ({timestamp.strftime('%H:%M UTC')}). Skipping.")
            return None

        atr_series = compute_atr(df, period=14)
        current_atr = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else 10 * instrument.pip_size

        swings = find_swing_points_logic(df['high'], df['low'], lookback=3)
        if not swings:
            return None

        current_close = float(current_bar['close'])
        current_low = float(current_bar['low'])
        current_high = float(current_bar['high'])

        lookback_window = min(20, n - 2)

        if bias == MarketBias.BULLISH:
            low_swings = [sp for sp in swings if not sp.is_high and sp.index < n - 1]
            high_swings = [sp for sp in swings if sp.is_high and sp.index < n - 1]
            if not low_swings or not high_swings:
                return None

            # 1. Detect Judas Swing / Liquidity Sweep below a recent swing low
            sweep_found = False
            swept_level = 0.0
            sweep_extreme = 0.0
            sweep_idx = -1

            for offset in range(2, lookback_window + 1):
                idx = n - 1 - offset
                bar = df.iloc[idx]
                for sp in reversed(low_swings):
                    if sp.index < idx and float(bar['low']) < sp.price:
                        sweep_found = True
                        swept_level = sp.price
                        sweep_extreme = float(bar['low'])
                        sweep_idx = idx
                        break
                if sweep_found:
                    break

            if not sweep_found or sweep_idx <= 0:
                return None

            # 2. Detect Bullish Market Structure Shift (MSS) displacement breaking recent swing high
            mss_confirmed = False
            mss_idx = -1
            broken_high = 0.0
            for post_idx in range(sweep_idx, min(sweep_idx + 8, n)):
                p_bar = df.iloc[post_idx]
                p_close = float(p_bar['close'])
                for sp in high_swings:
                    if sweep_idx <= sp.index < post_idx and p_close > sp.price:
                        # Displacement check: candle body >= 1.2x ATR
                        body = p_close - float(p_bar['open'])
                        if body >= current_atr * 1.0:
                            mss_confirmed = True
                            mss_idx = post_idx
                            broken_high = sp.price
                            break
                if mss_confirmed:
                    break

            if not mss_confirmed or mss_idx <= 0:
                return None

            # 3. Calculate Optimal Trade Entry (OTE) & FVG in the displacement leg
            leg_low = sweep_extreme
            leg_high = float(df.iloc[sweep_idx:n]['high'].max())
            leg_range = leg_high - leg_low

            if leg_range <= 0:
                return None

            # OTE zone: 61.8% to 78.6% retracement down from leg_high
            ote_top = leg_high - (leg_range * self.config.ote_fib_min)
            ote_bottom = leg_high - (leg_range * self.config.ote_fib_max)

            # Check FVG in the displacement impulse
            fvg_touched = False
            fvg_top, fvg_bottom = 0.0, 0.0
            for i in range(max(2, sweep_idx), min(mss_idx + 3, n)):
                c_curr = df.iloc[i]
                c_p2 = df.iloc[i - 2]
                if float(c_curr['low']) > float(c_p2['high']):
                    gap = float(c_curr['low']) - float(c_p2['high'])
                    if gap >= current_atr * 0.3:
                        fvg_top = float(c_curr['low'])
                        fvg_bottom = float(c_p2['high'])
                        if current_low <= fvg_top and current_close >= fvg_bottom:
                            fvg_touched = True
                            break

            ote_touched = (current_low <= ote_top and current_close >= ote_bottom)

            if not (fvg_touched or ote_touched or (current_close > broken_high and current_low <= broken_high)):
                return None

            # Confirmation classification
            if active_kz == "NY_SILVER_BULLET":
                conf = LTFConfirmation.ICT_SILVER_BULLET
            elif fvg_touched:
                conf = LTFConfirmation.ICT_KILLZONE_FVG
            elif ote_touched:
                conf = LTFConfirmation.ICT_OTE_RETEST
            else:
                conf = LTFConfirmation.ICT_JUDAS_SWING

            # Stop Loss & Take Profit
            sl_buffer = max(1.0 * instrument.pip_size, current_atr * 0.1)
            stop_loss = sweep_extreme - sl_buffer
            entry_price = current_close

            if fixed_sl_pips is not None and fixed_sl_pips > 0:
                sl_distance = fixed_sl_pips * instrument.pip_size
                stop_loss = entry_price - sl_distance
            else:
                sl_distance = abs(entry_price - stop_loss)

            min_buffer = 1.0 * instrument.pip_size
            if sl_distance < min_buffer:
                sl_distance = min_buffer
                stop_loss = entry_price - sl_distance

            # Target Opposing External Range Liquidity (ERL)
            erl_targets = [p.level for p in htf_analysis.liquidity_pools if p.is_high and p.level > entry_price + (sl_distance * self.config.target_rr)]
            take_profit = min(erl_targets) if erl_targets else entry_price + (sl_distance * self.config.target_rr)

            return {
                'direction': Direction.BUY,
                'entry': entry_price,
                'sl': stop_loss,
                'tp': take_profit,
                'conf': conf,
                'timestamp': timestamp,
                'kz': active_kz or "ALL_HOURS",
            }

        elif bias == MarketBias.BEARISH:
            high_swings = [sp for sp in swings if sp.is_high and sp.index < n - 1]
            low_swings = [sp for sp in swings if not sp.is_high and sp.index < n - 1]
            if not high_swings or not low_swings:
                return None

            # 1. Detect Judas Swing / Liquidity Sweep above a recent swing high
            sweep_found = False
            swept_level = 0.0
            sweep_extreme = 0.0
            sweep_idx = -1

            for offset in range(2, lookback_window + 1):
                idx = n - 1 - offset
                bar = df.iloc[idx]
                for sp in reversed(high_swings):
                    if sp.index < idx and float(bar['high']) > sp.price:
                        sweep_found = True
                        swept_level = sp.price
                        sweep_extreme = float(bar['high'])
                        sweep_idx = idx
                        break
                if sweep_found:
                    break

            if not sweep_found or sweep_idx <= 0:
                return None

            # 2. Detect Bearish Market Structure Shift (MSS) displacement breaking recent swing low
            mss_confirmed = False
            mss_idx = -1
            broken_low = 0.0
            for post_idx in range(sweep_idx, min(sweep_idx + 8, n)):
                p_bar = df.iloc[post_idx]
                p_close = float(p_bar['close'])
                for sp in low_swings:
                    if sweep_idx <= sp.index < post_idx and p_close < sp.price:
                        body = float(p_bar['open']) - p_close
                        if body >= current_atr * 1.0:
                            mss_confirmed = True
                            mss_idx = post_idx
                            broken_low = sp.price
                            break
                if mss_confirmed:
                    break

            if not mss_confirmed or mss_idx <= 0:
                return None

            # 3. Calculate OTE & FVG in the displacement leg
            leg_high = sweep_extreme
            leg_low = float(df.iloc[sweep_idx:n]['low'].min())
            leg_range = leg_high - leg_low

            if leg_range <= 0:
                return None

            # OTE zone: 61.8% to 78.6% retracement up from leg_low
            ote_bottom = leg_low + (leg_range * self.config.ote_fib_min)
            ote_top = leg_low + (leg_range * self.config.ote_fib_max)

            # Check Bearish FVG
            fvg_touched = False
            for i in range(max(2, sweep_idx), min(mss_idx + 3, n)):
                c_curr = df.iloc[i]
                c_p2 = df.iloc[i - 2]
                if float(c_curr['high']) < float(c_p2['low']):
                    gap = float(c_p2['low']) - float(c_curr['high'])
                    if gap >= current_atr * 0.3:
                        fvg_top = float(c_p2['low'])
                        fvg_bottom = float(c_curr['high'])
                        if current_high >= fvg_bottom and current_close <= fvg_top:
                            fvg_touched = True
                            break

            ote_touched = (current_high >= ote_bottom and current_close <= ote_top)

            if not (fvg_touched or ote_touched or (current_close < broken_low and current_high >= broken_low)):
                return None

            if active_kz == "NY_SILVER_BULLET":
                conf = LTFConfirmation.ICT_SILVER_BULLET
            elif fvg_touched:
                conf = LTFConfirmation.ICT_KILLZONE_FVG
            elif ote_touched:
                conf = LTFConfirmation.ICT_OTE_RETEST
            else:
                conf = LTFConfirmation.ICT_JUDAS_SWING

            sl_buffer = max(1.0 * instrument.pip_size, current_atr * 0.1)
            stop_loss = sweep_extreme + sl_buffer
            entry_price = current_close

            if fixed_sl_pips is not None and fixed_sl_pips > 0:
                sl_distance = fixed_sl_pips * instrument.pip_size
                stop_loss = entry_price + sl_distance
            else:
                sl_distance = abs(stop_loss - entry_price)

            min_buffer = 1.0 * instrument.pip_size
            if sl_distance < min_buffer:
                sl_distance = min_buffer
                stop_loss = entry_price + sl_distance

            erl_targets = [p.level for p in htf_analysis.liquidity_pools if not p.is_high and p.level < entry_price - (sl_distance * self.config.target_rr)]
            take_profit = max(erl_targets) if erl_targets else entry_price - (sl_distance * self.config.target_rr)

            return {
                'direction': Direction.SELL,
                'entry': entry_price,
                'sl': stop_loss,
                'tp': take_profit,
                'conf': conf,
                'timestamp': timestamp,
                'kz': active_kz or "ALL_HOURS",
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
        min_buffer = 1.0 * instrument.pip_size
        if sl_dist < min_buffer:
            sl_dist = min_buffer
            sl = entry - sl_dist if direction == Direction.BUY else entry + sl_dist

        rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0.0

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
        tp_dist = abs(entry - tp)
        min_buffer = 1.0 * instrument.pip_size
        if sl_dist < min_buffer:
            sl_dist = min_buffer
            sl = entry - sl_dist if direction == Direction.BUY else entry + sl_dist

        rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0.0

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
        )]


class ICTStrategy(BaseStrategy):
    """Strategy C: ICT KillZone, Judas Swing, MSS & FVG/OTE Model."""
    id = "ICT"
    name = "ICT KillZone / Silver Bullet"
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
    ) -> list[TradeSignal]:
        if htf_analysis is None:
            htf_analysis = self.htf_analyzer.analyze(htf_data)
        if htf_analysis.bias == MarketBias.NEUTRAL:
            return []

        raw = self.engine.detect_ict_entry(
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
        tp_dist = abs(entry - tp)
        min_buffer = 1.0 * instrument.pip_size
        if sl_dist < min_buffer:
            sl_dist = min_buffer
            sl = entry - sl_dist if direction == Direction.BUY else entry + sl_dist

        rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0.0

        # Quality scoring
        quality_score = min(25.0, htf_analysis.trend_clarity_score)
        if conf == LTFConfirmation.ICT_SILVER_BULLET:
            quality_score += 35.0
        elif conf == LTFConfirmation.ICT_KILLZONE_FVG:
            quality_score += 30.0
        elif conf == LTFConfirmation.ICT_OTE_RETEST:
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
        )]


# ─────────────────────────────────────────────
#  Strategy Orchestrator Engine
# ─────────────────────────────────────────────

class StrategyEngine:
    """Orchestrates HTF macro structure and executes registered strategies."""

    def __init__(self, config: TradingConfig | TimeframeConfig | None = None, instrument: InstrumentConfig | None = None, ict_config: ICTConfig | None = None):
        if config is None:
            from config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG

        if hasattr(config, 'timeframes'):
            self.tf_config = config.timeframes
            self.ict_config = ict_config or getattr(config, 'ict', ICTConfig())
            self.trading_config = config
        else:
            self.tf_config = config
            self.ict_config = ict_config or ICTConfig()
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
        }
        self.enabled_strategies: list[str] = ["SMC", "SMC_SCALP_5M", "ICT"]

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
                # If scalping strategy and 5m bars available, prefer 5m bars as LTF
                active_ltf = bars_5m if (strat_id == "SMC_SCALP_5M" and bars_5m is not None) else ltf
                sigs = strat.evaluate(
                    symbol=sym,
                    htf_data=htf,
                    ltf_data=active_ltf,
                    instrument=inst,
                    current_spread=spread,
                    fixed_sl_pips=fixed_sl_pips,
                    htf_analysis=shared_htf_analysis,
                )
                all_signals.extend(sigs)
            except Exception as e:
                logger.error(f"Error evaluating strategy {strat_id} on {sym}: {e}")

        return all_signals

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



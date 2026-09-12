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

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger

from config import MarketBias, Direction, TimeframeConfig, InstrumentConfig, StrategyType


# ─────────────────────────────────────────────
#  Enums & Data Structures
# ─────────────────────────────────────────────

class LTFConfirmation(str, Enum):
    OB_MITIGATION = "OB_MITIGATION"       # Retest of Order Block
    FVG_MITIGATION = "FVG_MITIGATION"     # Retest of Fair Value Gap
    OB_PLUS_FVG = "OB_PLUS_FVG"           # Overlapping OB + FVG (Prime Confluence)
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"   # Direct sweep & reversal confirmation
    OB_SCALP_5M = "OB_SCALP_5M"           # 5M Break of Structure & Order Block Retest Scalp
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


# ─────────────────────────────────────────────
#  Mathematical & Indicator Helpers
# ─────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)."""
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def find_swing_points_logic(highs: pd.Series, lows: pd.Series, lookback: int) -> list[SwingPoint]:
    """Identify swing highs and swing lows using N-bar pivot logic."""
    swings = []
    n = len(highs)
    if n < 2 * lookback + 1:
        return swings

    for i in range(lookback, n - lookback):
        window_highs = highs.iloc[i - lookback : i + lookback + 1]
        if highs.iloc[i] == window_highs.max():
            swings.append(SwingPoint(index=i, price=float(highs.iloc[i]), is_high=True))

        window_lows = lows.iloc[i - lookback : i + lookback + 1]
        if lows.iloc[i] == window_lows.min():
            swings.append(SwingPoint(index=i, price=float(lows.iloc[i]), is_high=False))

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
        enforce_session: bool = True,
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
#  Strategy Orchestrator Engine
# ─────────────────────────────────────────────

class StrategyEngine:
    """Orchestrates HTF macro structure and LTF Smart Money Concepts (SMC) entry engine."""

    def __init__(self, config: TimeframeConfig):
        self.htf_analyzer = HTFAnalyzer(
            ema_period=config.htf_ema_period,
            swing_lookback=config.swing_lookback,
            equal_level_tolerance=config.equal_level_tolerance,
        )
        self.smc_detector = SMCEntryDetector(config)
        self.scalp_engine = SMCScalp5MEngine()
        self.config = config

    def generate_signals(
        self,
        symbol: str,
        htf_data: pd.DataFrame,
        ltf_data: pd.DataFrame,
        instrument: InstrumentConfig,
        current_spread: float,
        fixed_sl_pips: float | None = None,
        strategy_type: str = "SMC",
    ) -> list[TradeSignal]:
        """
        Generates trade signals based on strategy_type:
        - "SMC": 15m Institutional Swing Liquidity Sweeps
        - "SMC_SCALP_5M": 5m Order Block (OB) Retest Scalps
        """
        htf_analysis = self.htf_analyzer.analyze(htf_data)

        if htf_analysis.bias == MarketBias.NEUTRAL:
            return []

        if strategy_type == StrategyType.SMC_SCALP_5M.value or strategy_type == "SMC_SCALP_5M":
            raw_signal = self.scalp_engine.detect_scalp_entry(
                df=ltf_data,
                htf_analysis=htf_analysis,
                instrument=instrument,
                current_spread=current_spread,
                fixed_sl_pips=fixed_sl_pips,
            )
        else:
            raw_signal = self.smc_detector.detect_smc_entry(
                df=ltf_data,
                htf_analysis=htf_analysis,
                instrument=instrument,
                current_spread=current_spread,
            )

        if not raw_signal:
            return []

        entry = raw_signal['entry']
        direction = raw_signal['direction']
        conf = raw_signal['conf']

        # Check if user specified a manual fixed stop loss in pips
        if fixed_sl_pips is not None and fixed_sl_pips > 0:
            sl_distance = fixed_sl_pips * instrument.pip_size
            if direction == Direction.BUY:
                sl = entry - sl_distance
            else:
                sl = entry + sl_distance
        else:
            sl = raw_signal['sl']
            sl_distance = abs(entry - sl)

        tp = raw_signal['tp']
        tp_distance = abs(entry - tp)

        min_buffer = 1.0 * instrument.pip_size
        if sl_distance < min_buffer:
            sl_distance = min_buffer
            sl = entry - sl_distance if direction == Direction.BUY else entry + sl_distance

        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0.0

        # Quality scoring (max 100)
        # 1. HTF Trend Clarity: up to 25
        quality_score = min(25.0, htf_analysis.trend_clarity_score)

        # 2. Confirmation Type Score (up to 35)
        if conf == LTFConfirmation.OB_PLUS_FVG:
            quality_score += 30.0
        elif conf == LTFConfirmation.OB_SCALP_5M:
            quality_score += 35.0  # High-conviction BOS + OB retest
        elif conf == LTFConfirmation.OB_MITIGATION:
            quality_score += 25.0
        elif conf == LTFConfirmation.FVG_MITIGATION:
            quality_score += 20.0
        elif conf == LTFConfirmation.LIQUIDITY_SWEEP:
            quality_score += 15.0

        # 3. Invalidation Quality (up to 15)
        quality_score += 15.0

        # 4. R:R Bonus (up to 15 points)
        if conf == LTFConfirmation.OB_SCALP_5M:
            rr_bonus = min(15.0, (rr_ratio / 1.5) * 15.0)
        else:
            rr_bonus = min(15.0, rr_ratio * 2.5)
        quality_score += rr_bonus

        # 5. Spread Viability (up to 10 points)
        if current_spread > 0:
            spread_ratio = tp_distance / current_spread
            quality_score += min(10.0, spread_ratio * 1.0)
        else:
            quality_score += 10.0

        quality_score = min(100.0, quality_score)

        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            htf_bias=htf_analysis.bias,
            ltf_confirmation=conf,
            rr_ratio=rr_ratio,
            quality_score=quality_score,
            timestamp=raw_signal['timestamp'],
            sl_distance=sl_distance,
            tp_distance=tp_distance,
        )

        return [signal]


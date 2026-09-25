"""
config.py — Strongly-typed configuration engine for the trading bot.

All parameters are validated via pydantic with explicit bounds.
To adjust strategy parameters, edit DEFAULT_CONFIG at the bottom of this file
or load from a JSON/YAML file via TradingConfig.model_validate().

Deployment Notes:
  1. Copy this file to your VPS alongside all other modules.
  2. Adjust DEFAULT_CONFIG values to match your broker account:
     - equity: your account balance in the deposit currency.
     - risk_pct: fraction of equity risked per trade (0.005 = 0.5%).
     - instruments: add/remove pairs; set point_value per your broker spec.
  3. For MetaTrader 5 connections, ensure MT5 terminal is running on the VPS
     and the account is logged in before starting the bot.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from enum import Enum
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


# ─────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────

class Direction(str, Enum):
    """Trade direction."""
    BUY = "BUY"
    SELL = "SELL"


class MarketBias(str, Enum):
    """Higher-timeframe directional bias."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class Impact(str, Enum):
    """Economic news impact level."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class StrategyType(str, Enum):
    """Supported trading strategies."""
    SMC = "SMC"                      # 15m Institutional Swing Liquidity Sweep
    SMC_SCALP_5M = "SMC_SCALP_5M"    # 5m Order Block (OB) Scalp
    ICT = "ICT"                      # ICT KillZone, Judas Swing, MSS & FVG/OTE Model
    ORDER_FLOW = "ORDER_FLOW"        # Order Flow Volume Delta, CVD & Absorption Model


def normalize_strategy_key(strat_name: str | None, magic: int | None = None) -> str:
    """
    Map various strategy name formats or magic numbers to canonical StrategyType IDs:
    - 'SMC'
    - 'SMC_SCALP_5M'
    - 'ICT'
    - 'ORDER_FLOW'
    """
    if magic == 124456:
        return "SMC"
    elif magic == 125456:
        return "SMC_SCALP_5M"
    elif magic == 126456:
        return "ICT"
    elif magic == 127456:
        return "ORDER_FLOW"

    if not strat_name:
        return "SMC"
    s = str(strat_name).upper().strip()
    if "FLOW" in s or "DELTA" in s or "ABSORPTION" in s or s == "ORDER_FLOW":
        return "ORDER_FLOW"
    elif "SCALP" in s or "5M" in s:
        return "SMC_SCALP_5M"
    elif "ICT" in s:
        return "ICT"
    elif "SWING" in s or "SMC" in s:
        return "SMC"
    return s


class PairSettings(BaseModel):
    """Independent settings for a single traded pair."""
    symbol: str = Field(default="XAUUSD", description="Instrument symbol (e.g. XAUUSD, EURUSD)")
    fixed_lot_size: float | None = Field(default=None, gt=0.0, description="Fixed lot size override. None uses risk % dynamic sizing.")
    fixed_sl_pips: float | None = Field(default=None, gt=0.0, description="Fixed Stop Loss in pips override. None uses dynamic structural SL.")
    enabled: bool = Field(default=True, description="Whether this pair is active for analysis and execution.")


class ICTConfig(BaseModel):
    """Parameters for Inner Circle Trader (ICT) methodology."""
    london_kz_start_utc: int = Field(default=7, ge=0, le=23, description="London Kill Zone Start (07:00 UTC)")
    london_kz_end_utc: int = Field(default=10, ge=0, le=23, description="London Kill Zone End (10:00 UTC)")
    ny_am_kz_start_utc: int = Field(default=12, ge=0, le=23, description="New York AM Kill Zone Start (12:00 UTC)")
    ny_am_kz_end_utc: int = Field(default=15, ge=0, le=23, description="New York AM Kill Zone End (15:00 UTC)")
    silver_bullet_start_utc: int = Field(default=14, ge=0, le=23, description="Silver Bullet Hour Start (14:00 UTC)")
    silver_bullet_end_utc: int = Field(default=15, ge=0, le=23, description="Silver Bullet Hour End (15:00 UTC)")
    london_close_start_utc: int = Field(default=15, ge=0, le=23, description="London Close Kill Zone Start (15:00 UTC)")
    london_close_end_utc: int = Field(default=17, ge=0, le=23, description="London Close Kill Zone End (17:00 UTC)")
    enforce_killzones: bool = Field(default=False, description="Restrict ICT trading exclusively to Kill Zones.")
    ote_fib_min: float = Field(default=0.618, description="Optimal Trade Entry minimum Fibonacci retracement.")
    ote_fib_max: float = Field(default=0.786, description="Optimal Trade Entry maximum Fibonacci retracement.")
    target_rr: float = Field(default=2.0, ge=1.0, description="Default Target Risk-to-Reward ratio for ICT setups.")


class OrderFlowConfig(BaseModel):
    """Parameters for Order Flow & Volume Delta analysis."""
    delta_lookback_bars: int = Field(default=20, ge=5, le=100, description="Lookback window for volume and CVD calculations.")
    absorption_volume_factor: float = Field(default=1.8, ge=1.1, le=5.0, description="Volume multiplier vs 20-period SMA to flag absorption candidate.")
    wick_ratio_threshold: float = Field(default=0.4, ge=0.2, le=0.9, description="Minimum wick-to-total-range ratio to qualify as absorption.")
    cvd_divergence_bars: int = Field(default=14, ge=5, le=50, description="Lookback period to check CVD divergence against price swings.")
    min_rr: float = Field(default=1.8, ge=1.0, description="Minimum acceptable R:R ratio for Order Flow setups.")
    target_rr: float = Field(default=2.2, ge=1.0, description="Default target Risk-to-Reward ratio for Order Flow setups.")
class PositionManagementRuleConfig(BaseModel):
    """Configuration for dynamic position management on open trades."""
    # 1. Breakeven after +1R
    breakeven_enabled: bool = Field(default=True, description="Move SL to breakeven once price moves in favor.")
    breakeven_trigger_r: float = Field(default=1.0, ge=0.5, le=5.0, description="R multiple to trigger breakeven (e.g. 1.0 = +1R).")
    breakeven_offset_r: float = Field(default=0.1, ge=0.0, le=1.0, description="Buffer above/below entry in R to cover costs/spread (e.g. +0.1R).")
    breakeven_spread_buffer: bool = Field(default=True, description="Add broker live spread buffer to breakeven SL.")

    # 2. Partial Take-Profit at 1R and/or 2R
    partial_tp_enabled: bool = Field(default=True, description="Enable partial position close at target R multiples.")
    partial_tp_stages: list[tuple[float, float]] = Field(
        default_factory=lambda: [(1.0, 0.50), (2.0, 0.25)],
        description="Stages of (R_trigger, pct_of_current_lot_to_close). e.g. [(1.0, 0.50), (2.0, 0.25)].",
    )

    # 3 & 4. Trailing Stop (ATR and Structure)
    trailing_stop_mode: str = Field(default="STRUCTURE", description="Trailing stop mode: 'NONE', 'ATR', 'STRUCTURE', 'HYBRID'.")
    trailing_activation_r: float = Field(default=1.0, ge=0.0, description="Minimum R profit before trailing stop starts ratcheting SL.")
    # ATR-Based Trailing
    trailing_atr_multiplier: float = Field(default=1.5, ge=0.5, le=5.0, description="ATR multiplier for trailing stop distance.")
    trailing_atr_period: int = Field(default=14, ge=5, le=50, description="ATR period for trailing calculation.")
    trailing_recalc_on: str = Field(default="BAR", description="Recalculate trailing stop on 'BAR' or 'TICK'.")
    # Structure-Based Trailing
    trailing_structure_lookback: int = Field(default=5, ge=2, le=20, description="Swing point lookback bars for LTF structure.")
    trailing_structure_confirm_bars: int = Field(default=2, ge=1, le=5, description="Consecutive closes beyond swing level required for confirmation.")
    trailing_structure_timeframe: str = Field(default="5m", description="Timeframe used for structure trailing (e.g. '5m' or '15m').")

    # 5. Time Stop
    time_stop_enabled: bool = Field(default=False, description="Enable time-based exit for stagnant trades.")
    time_stop_bars: int = Field(default=24, ge=5, le=200, description="Maximum bars to hold without reaching min profit.")
    time_stop_min_r: float = Field(default=0.5, ge=0.0, description="Minimum R profit required within time_stop_bars.")
    stagnant_exit_bars: int = Field(default=15, ge=5, le=100, description="Exit if no new high/low made in N bars.")

    # 6. Session-Close Exit
    session_close_enabled: bool = Field(default=False, description="Force close intraday position before session end.")
    session_close_minutes_before: int = Field(default=15, ge=1, le=60, description="Minutes before session close to exit position.")
    session_close_hour_utc: int = Field(default=16, ge=0, le=23, description="Target session close hour in UTC (e.g. 16:00 UTC London close, 21:00 UTC NY close).")


class PositionManagementConfig(BaseModel):
    """Aggregate configuration for position management with strategy and instrument overrides."""
    default_rules: PositionManagementRuleConfig = Field(default_factory=PositionManagementRuleConfig)
    strategy_overrides: dict[str, PositionManagementRuleConfig] = Field(default_factory=dict)
    instrument_overrides: dict[str, PositionManagementRuleConfig] = Field(default_factory=dict)

    def get_rules(self, strategy_name: str | None = None, symbol: str | None = None) -> PositionManagementRuleConfig:
        """Lookup hierarchy: Instrument override -> Strategy override -> Default."""
        strat_key = normalize_strategy_key(strategy_name) if strategy_name else None
        sym_key = symbol.strip().upper() if symbol else None

        # Check instrument override first if available
        if sym_key and sym_key in self.instrument_overrides:
            return self.instrument_overrides[sym_key]

        # Check strategy override
        if strat_key and strat_key in self.strategy_overrides:
            return self.strategy_overrides[strat_key]

        return self.default_rules


class AccountConfig(BaseModel):
    """Broker account parameters."""

    equity: float = Field(
        default=10_000.0,
        ge=100.0,
        description="Account equity in deposit currency.",
    )
    risk_pct: float = Field(
        default=0.005,
        ge=0.001,
        le=0.02,
        description="Fraction of equity to risk per trade (0.005 = 0.5%).",
    )
    max_daily_drawdown_pct: float = Field(
        default=0.02,
        ge=0.005,
        le=0.10,
        description="Maximum daily drawdown as fraction of equity (0.02 = 2%).",
    )


class InstrumentConfig(BaseModel):
    """Per-instrument trading parameters."""

    symbol: str = Field(
        description="Broker symbol name, e.g. 'EURUSD'.",
    )
    point_value: float = Field(
        default=10.0,
        gt=0,
        description="Monetary value of 1 point move per 1.0 standard lot.",
    )
    pip_size: float = Field(
        default=0.0001,
        gt=0,
        description="Size of one pip (e.g. 0.0001 for major FX, 0.01 for JPY pairs).",
    )
    avg_spread_points: float = Field(
        default=1.5,
        ge=0,
        description="Rolling 30-day average spread in points.",
    )
    digits: int = Field(
        default=5,
        ge=0,
        le=8,
        description="Price decimal digits (5 for most FX pairs).",
    )
    min_lot: float = Field(
        default=0.01,
        gt=0,
        description="Minimum lot size allowed by broker.",
    )
    max_lot: float = Field(
        default=100.0,
        gt=0,
        description="Maximum lot size allowed by broker.",
    )
    lot_step: float = Field(
        default=0.01,
        gt=0,
        description="Lot size increment (granularity).",
    )


class RiskConfig(BaseModel):
    """Quality and risk filter parameters."""

    min_rr_ratio: float = Field(
        default=2.5,
        ge=1.0,
        description="Minimum risk-to-reward ratio to accept a trade.",
    )
    max_open_positions: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum concurrent open positions allowed across all pairs/strategies.",
    )
    max_open_per_symbol: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent open positions allowed per symbol (1 per strategy).",
    )
    min_tp_spread_multiple: float = Field(
        default=6.0,
        ge=1.0,
        description="TP distance must be >= this multiple of current spread.",
    )
    max_spread_multiple: float = Field(
        default=2.0,
        ge=1.0,
        description="Reject trade if live spread > avg_spread * this value.",
    )
    # Automatic Night Trade Limit (11 PM - 8 AM)
    night_limit_enabled: bool = Field(
        default=True,
        description="Enable automatic night trade limit between start and end hours.",
    )
    night_start_hour: int = Field(
        default=23,
        ge=0,
        le=23,
        description="Night trading window start hour (23 = 11:00 PM).",
    )
    night_end_hour: int = Field(
        default=8,
        ge=0,
        le=23,
        description="Night trading window end hour (8 = 8:00 AM).",
    )
    night_max_open_positions: int = Field(
        default=2,
        ge=1,
        le=50,
        description="Maximum concurrent open positions allowed during night window (11 PM - 8 AM).",
    )
    night_timezone_mode: str = Field(
        default="LOCAL",
        description="Timezone mode for night window: 'LOCAL' (system clock) or 'UTC'.",
    )

    def is_night_window(self, current_dt: datetime | None = None) -> bool:
        """Check if current time is within the night trading limit window (e.g. 11 PM to 8 AM)."""
        if not self.night_limit_enabled:
            return False

        if current_dt is None:
            if self.night_timezone_mode.upper() == "UTC":
                current_dt = datetime.now(timezone.utc)
            else:
                current_dt = datetime.now().astimezone()
        else:
            if self.night_timezone_mode.upper() == "UTC":
                if current_dt.tzinfo is None:
                    current_dt = current_dt.replace(tzinfo=timezone.utc)
                else:
                    current_dt = current_dt.astimezone(timezone.utc)
            else:
                if current_dt.tzinfo is not None:
                    current_dt = current_dt.astimezone()

        hour = current_dt.hour
        if self.night_start_hour > self.night_end_hour:
            return hour >= self.night_start_hour or hour < self.night_end_hour
        elif self.night_start_hour < self.night_end_hour:
            return self.night_start_hour <= hour < self.night_end_hour
        else:
            return False

    def get_effective_max_open_positions(self, current_dt: datetime | None = None) -> int:
        """
        Return effective maximum open positions.
        If night limit is enabled and current time falls between night_start_hour (11 PM / 23:00)
        and night_end_hour (8 AM / 08:00), return night_max_open_positions (2).
        Otherwise return max_open_positions (e.g. 15).
        """
        if self.is_night_window(current_dt):
            return self.night_max_open_positions
        return self.max_open_positions


class TimeframeConfig(BaseModel):
    """Timeframe settings for dual-TF analysis."""

    htf: str = Field(
        default="1H",
        description="Higher-timeframe bar period (e.g. '4H', '1H').",
    )
    ltf: str = Field(
        default="15m",
        description="Lower-timeframe bar period (e.g. '15m', '5m').",
    )
    htf_ema_period: int = Field(
        default=200,
        ge=10,
        description="EMA period for HTF trend detection.",
    )
    swing_lookback: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of bars on each side to identify swing highs/lows.",
    )
    ob_lookback: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Bars to scan back for Order Blocks.",
    )
    fvg_min_gap_atr: float = Field(
        default=0.5,
        ge=0.1,
        description="Minimum gap size as fraction of ATR to qualify as Fair Value Gap.",
    )
    displacement_atr_multiple: float = Field(
        default=1.5,
        ge=0.5,
        description="Displacement candle body must be >= this multiple of ATR.",
    )
    sweep_buffer_atr: float = Field(
        default=0.3,
        ge=0.0,
        description="Sweep must exceed liquidity level by at least this multiple of ATR.",
    )
    equal_level_tolerance: float = Field(
        default=0.15,
        ge=0.01,
        description="ATR fraction to group swing points as equal highs/lows.",
    )
    breakeven_at_r: float = Field(
        default=2.0,
        ge=1.0,
        description="Move SL to breakeven after reaching this R multiple.",
    )


class TradingConfig(BaseModel):
    """Top-level aggregated configuration."""

    account: AccountConfig = Field(default_factory=AccountConfig)
    instruments: List[InstrumentConfig] = Field(
        default_factory=lambda: [
            InstrumentConfig(
                symbol="XAUUSD",
                point_value=1.0,      # 100 oz × $0.01 point = $1.00 per lot per point
                pip_size=0.01,        # Gold pip = $0.01
                avg_spread_points=25.0,  # ~$0.25 typical spread
                digits=2,
                min_lot=0.01,
                max_lot=100.0,
                lot_step=0.01,
            ),
            InstrumentConfig(
                symbol="EURUSD",
                point_value=1.0,      # 100,000 EUR × 0.00001 point = $1.00 per lot per point ($10/pip)
                pip_size=0.0001,
                avg_spread_points=1.2,
                digits=5,
                min_lot=0.01,
                max_lot=100.0,
                lot_step=0.01,
            ),
            InstrumentConfig(
                symbol="GBPUSD",
                point_value=1.0,      # 100,000 GBP × 0.00001 point = $1.00 per lot per point ($10/pip)
                pip_size=0.0001,
                avg_spread_points=1.5,
                digits=5,
                min_lot=0.01,
                max_lot=100.0,
                lot_step=0.01,
            ),
            InstrumentConfig(
                symbol="BTCUSD",
                point_value=1.0,
                pip_size=0.01,
                avg_spread_points=50.0,
                digits=2,
                min_lot=0.01,
                max_lot=50.0,
                lot_step=0.01,
            ),
            InstrumentConfig(
                symbol="ETHUSD",
                point_value=1.0,
                pip_size=0.01,
                avg_spread_points=30.0,
                digits=2,
                min_lot=0.01,
                max_lot=100.0,
                lot_step=0.01,
            ),
        ],
    )
    risk: RiskConfig = Field(default_factory=RiskConfig)
    timeframes: TimeframeConfig = Field(default_factory=TimeframeConfig)
    ict: ICTConfig = Field(default_factory=ICTConfig)
    order_flow: OrderFlowConfig = Field(default_factory=OrderFlowConfig)
    position_management: PositionManagementConfig = Field(
        default_factory=lambda: PositionManagementConfig(
            default_rules=PositionManagementRuleConfig(
                breakeven_enabled=True,
                breakeven_trigger_r=1.0,
                breakeven_offset_r=0.1,
                partial_tp_enabled=True,
                partial_tp_stages=[(1.0, 0.50), (2.0, 0.25)],
                trailing_stop_mode="STRUCTURE",
                trailing_activation_r=1.0,
                trailing_structure_lookback=5,
                trailing_structure_confirm_bars=2,
            ),
            strategy_overrides={
                "SMC": PositionManagementRuleConfig(
                    breakeven_enabled=True,
                    breakeven_trigger_r=1.0,
                    breakeven_offset_r=0.1,
                    partial_tp_enabled=True,
                    partial_tp_stages=[(2.0, 0.50)],
                    trailing_stop_mode="STRUCTURE",
                    trailing_activation_r=1.5,
                    trailing_structure_lookback=5,
                    session_close_enabled=False,
                    time_stop_enabled=False,
                ),
                "SMC_SCALP_5M": PositionManagementRuleConfig(
                    breakeven_enabled=True,
                    breakeven_trigger_r=1.0,
                    breakeven_offset_r=0.1,
                    partial_tp_enabled=True,
                    partial_tp_stages=[(1.0, 0.50)],
                    trailing_stop_mode="ATR",
                    trailing_activation_r=1.0,
                    trailing_atr_multiplier=1.5,
                    trailing_atr_period=14,
                    session_close_enabled=True,
                    session_close_minutes_before=15,
                    session_close_hour_utc=16,
                    time_stop_enabled=True,
                    time_stop_bars=24,
                    stagnant_exit_bars=12,
                ),
                "ICT": PositionManagementRuleConfig(
                    breakeven_enabled=True,
                    breakeven_trigger_r=1.0,
                    breakeven_offset_r=0.1,
                    partial_tp_enabled=True,
                    partial_tp_stages=[(1.0, 0.50), (2.0, 0.25)],
                    trailing_stop_mode="STRUCTURE",
                    trailing_activation_r=1.0,
                    trailing_structure_lookback=3,
                    trailing_structure_confirm_bars=2,
                    session_close_enabled=True,
                    session_close_minutes_before=15,
                    session_close_hour_utc=21,
                    time_stop_enabled=False,
                ),
                "ORDER_FLOW": PositionManagementRuleConfig(
                    breakeven_enabled=True,
                    breakeven_trigger_r=1.0,
                    breakeven_offset_r=0.1,
                    partial_tp_enabled=True,
                    partial_tp_stages=[(1.5, 0.50)],
                    trailing_stop_mode="ATR",
                    trailing_activation_r=1.0,
                    trailing_atr_multiplier=1.5,
                    session_close_enabled=False,
                    time_stop_enabled=True,
                    time_stop_bars=30,
                ),
            },
        ),
        description="Dynamic position management configuration with strategy-specific rules."
    )

    
    # Multi-strategy concurrent execution
    enabled_strategies: List[str] = Field(
        default_factory=lambda: ["SMC", "SMC_SCALP_5M", "ICT", "ORDER_FLOW"],
        description="List of strategies running concurrently (SMC, SMC_SCALP_5M, ICT, ORDER_FLOW).",
    )
    
    # Independent Primary Pair Configurations (3 Pairs)
    pair1: PairSettings = Field(
        default_factory=lambda: PairSettings(symbol="XAUUSD", fixed_lot_size=0.05, fixed_sl_pips=25.0, enabled=True),
        description="Pair 1 configuration with independent lot size and SL.",
    )
    pair2: PairSettings = Field(
        default_factory=lambda: PairSettings(symbol="EURUSD", fixed_lot_size=0.10, fixed_sl_pips=15.0, enabled=True),
        description="Pair 2 configuration with independent lot size and SL.",
    )
    pair3: PairSettings = Field(
        default_factory=lambda: PairSettings(symbol="GBPUSD", fixed_lot_size=0.12, fixed_sl_pips=20.0, enabled=True),
        description="Pair 3 configuration with independent lot size and SL.",
    )

    # Legacy compatibility fields
    fixed_lot_size: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional global fixed lot size fallback.",
    )
    fixed_sl_pips: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional global fixed Stop Loss in pips fallback.",
    )
    strategy_type: str = Field(
        default=StrategyType.SMC.value,
        description="Legacy single strategy selector for backward compatibility.",
    )
    scalp_session_start_utc: int = Field(
        default=7,
        ge=0,
        le=23,
        description="UTC hour when London session opens for scalping (default 07:00 UTC).",
    )
    scalp_session_end_utc: int = Field(
        default=16,
        ge=0,
        le=23,
        description="UTC hour when New York AM session closes for scalping (default 16:00 UTC).",
    )
    scalp_target_rr: float = Field(
        default=1.5,
        ge=1.0,
        description="Target Risk-to-Reward ratio for 5M scalping strategy.",
    )
    scalp_enforce_session: bool = Field(
        default=False,
        description="Whether to restrict scalping to London/NY sessions (False = active 24/7 in all sessions).",
    )
    selected_symbols: List[str] = Field(
        default_factory=lambda: ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD", "ETHUSD"],
        description="List of active symbols to analyze concurrently.",
    )


    news_blackout_minutes: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Minutes before and after HIGH-impact news to block trading.",
    )
    db_path: str = Field(
        default="trading_state.db",
        description="Path to the SQLite state database.",
    )
    news_calendar_path: str = Field(
        default="news_calendar.json",
        description="Path to the economic calendar JSON file.",
    )
    ai_confirmation_enabled: bool = Field(
        default=True,
        description="Enable AI (Gemini) second-opinion confirmation before trade execution.",
    )
    # Machine Learning Trap & SL Gate Settings
    ml_gating_enabled: bool = Field(
        default=True,
        description="When enabled, ML model predicts probability of hitting SL; skips trade if P(SL) is high.",
    )
    ml_max_sl_probability: float = Field(
        default=0.50,
        ge=0.10,
        le=0.90,
        description="Maximum allowed probability of hitting Stop Loss before ML vetoes the trade (e.g. 0.50 = 50%).",
    )
    ml_shadow_until_samples: int = Field(
        default=0,
        ge=0,
        description="Number of samples before active gating begins (0 = immediate active protection).",
    )
    gemini_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key. If None, checks GEMINI_API_KEY environment variable.",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model name for ultra-low latency trade analysis.",
    )
    ai_confidence_threshold: float = Field(
        default=75.0,
        ge=50.0,
        le=99.0,
        description="Minimum AI confidence percentage (e.g. 75%) required to approve a trade.",
    )
    use_mock_broker: bool = Field(
        default_factory=lambda: os.getenv("USE_MOCK_BROKER", "false").lower() in ("true", "1", "yes"),
        description="If True, use MockBrokerAdapter instead of live MT5.",
    )
    mt5_path: str = Field(
        default_factory=lambda: os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe"),
        description="Path to MT5 terminal executable.",
    )
    mt5_login: int | None = Field(
        default_factory=lambda: int(os.getenv("MT5_LOGIN", "0")) if os.getenv("MT5_LOGIN", "0").isdigit() and int(os.getenv("MT5_LOGIN", "0")) > 0 else None,
        description="MT5 broker account login ID.",
    )
    mt5_password: str | None = Field(
        default_factory=lambda: os.getenv("MT5_PASSWORD") or None,
        description="MT5 broker account password.",
    )
    mt5_server: str | None = Field(
        default_factory=lambda: os.getenv("MT5_SERVER") or None,
        description="MT5 broker server name.",
    )
    log_path: str = Field(
        default="logs/trading_bot.log",
        description="Path for structured log output.",
    )


# ─────────────────────────────────────────────
#  Default Configuration Instance
# ─────────────────────────────────────────────

DEFAULT_CONFIG = TradingConfig()


def get_instrument(config: TradingConfig, symbol: str) -> InstrumentConfig:
    """Look up an instrument by symbol, raising ValueError if not found."""
    for inst in config.instruments:
        if inst.symbol == symbol:
            return inst
    raise ValueError(
        f"Instrument '{symbol}' not found in config. "
        f"Available: {[i.symbol for i in config.instruments]}"
    )

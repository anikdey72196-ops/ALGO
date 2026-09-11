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


# ─────────────────────────────────────────────
#  Configuration Models
# ─────────────────────────────────────────────

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
    max_daily_trades: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Maximum number of trades per UTC day.",
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
                point_value=10.0,
                pip_size=0.0001,
                avg_spread_points=1.2,
                digits=5,
                min_lot=0.01,
                max_lot=100.0,
                lot_step=0.01,
            ),
            InstrumentConfig(
                symbol="GBPUSD",
                point_value=10.0,
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
    fixed_lot_size: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional fixed lot size. If set, overrides dynamic risk percentage sizing.",
    )
    fixed_sl_pips: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional fixed Stop Loss in pips. If set, overrides dynamic SMC structure stop loss.",
    )
    selected_symbols: List[str] = Field(
        default_factory=lambda: ["XAUUSD", "EURUSD"],
        description="List of up to 2 active symbols to analyze concurrently.",
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

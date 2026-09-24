"""
market_regime.py — Detects Trending vs. Sideways/Ranging Market Conditions.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class MarketRegime(Enum):
    TRENDING = "TRENDING"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE_CHOP = "VOLATILE_CHOP"


@dataclass(frozen=True)
class RegimeAnalysis:
    regime: MarketRegime
    adx: float
    choppiness: float
    bb_width_pct: float
    is_sideways: bool
    risk_multiplier: float
    recommended_rr_cap: float


class MarketRegimeDetector:
    """Calculates ADX, Choppiness Index, and Bollinger Band Width to identify sideways regimes."""

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
        if df is None or len(df) < period * 2:
            return 25.0
        high = df['high']
        low = df['low']
        close = df['close']

        up_move = high.diff()
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / (atr + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / (atr + 1e-9))

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        adx = dx.ewm(span=period, adjust=False).mean().iloc[-1]
        return float(adx) if not np.isnan(adx) else 25.0

    @staticmethod
    def calculate_choppiness_index(df: pd.DataFrame, period: int = 14) -> float:
        """Choppiness Index: Values > 61.8 indicate consolidation/sideways chop."""
        if df is None or len(df) < period + 1:
            return 50.0
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        sum_tr = tr.rolling(window=period).sum()

        max_high = high.rolling(window=period).max()
        min_low = low.rolling(window=period).min()
        range_hl = max_high - min_low

        chop = 100 * (np.log10((sum_tr + 1e-9) / (range_hl + 1e-9)) / np.log10(period))
        val = chop.iloc[-1]
        return float(val) if not np.isnan(val) else 50.0

    @staticmethod
    def analyze(df: pd.DataFrame, adx_threshold: float = 20.0, chop_threshold: float = 61.8,
                sideways_risk_multiplier: float = 0.5) -> RegimeAnalysis:
        if df is None or len(df) < 30:
            return RegimeAnalysis(
                regime=MarketRegime.TRENDING, adx=30.0, choppiness=45.0,
                bb_width_pct=1.0, is_sideways=False, risk_multiplier=1.0,
                recommended_rr_cap=3.0
            )

        adx = MarketRegimeDetector.calculate_adx(df)
        chop = MarketRegimeDetector.calculate_choppiness_index(df)

        # Bollinger Band Width
        close = df['close']
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_width = float(((2 * std20 * 2) / (sma20 + 1e-9)).iloc[-1]) if len(close) >= 20 else 1.0

        is_sideways = (adx < adx_threshold) or (chop >= chop_threshold)
        regime = MarketRegime.SIDEWAYS if is_sideways else MarketRegime.TRENDING
        risk_mult = sideways_risk_multiplier if is_sideways else 1.0
        rr_cap = 1.8 if is_sideways else 3.5

        return RegimeAnalysis(
            regime=regime,
            adx=round(adx, 2),
            choppiness=round(chop, 2),
            bb_width_pct=round(bb_width * 100.0, 2),
            is_sideways=is_sideways,
            risk_multiplier=risk_mult,
            recommended_rr_cap=rr_cap,
        )

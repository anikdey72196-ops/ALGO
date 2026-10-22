"""
tests/test_market_regime.py
Comprehensive unit tests for core/market_regime.py (MarketRegimeDetector, calculate_adx,
calculate_choppiness_index, and analyze).
"""

from dataclasses import FrozenInstanceError
import numpy as np
import pandas as pd
import pytest

from core.market_regime import MarketRegime, RegimeAnalysis, MarketRegimeDetector


def _generate_synthetic_ohlcv(
    count: int = 100,
    regime: str = "trend",
    base_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic OHLCV dataframe for testing."""
    np.random.seed(seed)
    timestamps = pd.date_range(start="2026-10-01 00:00:00", periods=count, freq="5min", tz="UTC")

    if regime == "trend":
        # Consistent upward trend with directional expansion
        drift = np.linspace(0, 30.0, count)
        noise = np.random.normal(0, 0.1, count)
        close = base_price + drift + noise
        high = close + np.random.uniform(0.1, 0.5, count)
        low = close - np.random.uniform(0.1, 0.3, count)
        open_p = close - np.random.uniform(-0.1, 0.2, count)
    elif regime == "chop":
        # Stationary mean-reverting consolidation with tight range
        base = np.full(count, base_price)
        noise = np.random.uniform(-0.1, 0.1, count)
        close = base + noise
        high = close + np.random.uniform(0.02, 0.05, count)
        low = close - np.random.uniform(0.02, 0.05, count)
        open_p = close - np.random.uniform(-0.02, 0.02, count)
    elif regime == "flat":
        # Completely dead market with identical bars
        close = np.full(count, base_price)
        open_p = np.full(count, base_price)
        high = np.full(count, base_price)
        low = np.full(count, base_price)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    volume = np.random.randint(50, 500, count)

    return pd.DataFrame({
        "time": timestamps,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1. MarketRegime Enum & Dataclass Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_market_regime_enum_values():
    """Verify MarketRegime enum constants and string values."""
    assert MarketRegime.TRENDING.value == "TRENDING"
    assert MarketRegime.SIDEWAYS.value == "SIDEWAYS"
    assert MarketRegime.VOLATILE_CHOP.value == "VOLATILE_CHOP"


def test_regime_analysis_dataclass_frozen():
    """Verify RegimeAnalysis dataclass is frozen and immutable."""
    analysis = RegimeAnalysis(
        regime=MarketRegime.TRENDING,
        adx=35.0,
        choppiness=40.0,
        bb_width_pct=2.5,
        is_sideways=False,
        risk_multiplier=1.0,
        recommended_rr_cap=3.5,
    )
    with pytest.raises(FrozenInstanceError):
        analysis.is_sideways = True


# ─────────────────────────────────────────────────────────────────────────────
# 2. ADX (Average Directional Index) Calculation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_calculate_adx_none_or_short_dataframe_returns_fallback():
    """Verify calculate_adx returns default 25.0 when df is None or too short (< period * 2)."""
    assert MarketRegimeDetector.calculate_adx(None) == 25.0

    empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert MarketRegimeDetector.calculate_adx(empty_df, period=14) == 25.0

    short_df = _generate_synthetic_ohlcv(count=20, regime="trend")
    # For period=14, minimum required is 14 * 2 = 28
    assert MarketRegimeDetector.calculate_adx(short_df, period=14) == 25.0


def test_calculate_adx_strong_trend_produces_high_value():
    """Verify strong directional trending candles produce elevated ADX (> 30)."""
    df = _generate_synthetic_ohlcv(count=80, regime="trend")
    adx = MarketRegimeDetector.calculate_adx(df, period=14)

    assert isinstance(adx, float)
    assert not np.isnan(adx)
    assert adx > 30.0, f"Expected strong trend ADX > 30.0, got {adx}"


def test_calculate_adx_flat_market_epsilon_protection():
    """Verify perfectly flat bars do not produce ZeroDivisionError or NaN."""
    flat_df = _generate_synthetic_ohlcv(count=60, regime="flat")
    adx = MarketRegimeDetector.calculate_adx(flat_df, period=14)

    assert isinstance(adx, float)
    assert not np.isnan(adx)
    assert adx >= 0.0


def test_calculate_adx_custom_period():
    """Verify calculate_adx respects custom period lengths."""
    df = _generate_synthetic_ohlcv(count=50, regime="trend")
    adx_short = MarketRegimeDetector.calculate_adx(df, period=7)
    adx_standard = MarketRegimeDetector.calculate_adx(df, period=14)

    assert isinstance(adx_short, float)
    assert isinstance(adx_standard, float)
    assert not np.isnan(adx_short)
    assert not np.isnan(adx_standard)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Choppiness Index Calculation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_calculate_choppiness_none_or_short_dataframe_returns_fallback():
    """Verify calculate_choppiness_index returns default 50.0 when df is None or too short (< period + 1)."""
    assert MarketRegimeDetector.calculate_choppiness_index(None) == 50.0

    empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    assert MarketRegimeDetector.calculate_choppiness_index(empty_df, period=14) == 50.0

    # For period=14, minimum required is 14 + 1 = 15
    short_df = _generate_synthetic_ohlcv(count=10, regime="chop")
    assert MarketRegimeDetector.calculate_choppiness_index(short_df, period=14) == 50.0


def test_calculate_choppiness_trending_market_produces_low_value():
    """Verify strong unidirectional trend produces low choppiness (< 50.0)."""
    df = _generate_synthetic_ohlcv(count=80, regime="trend")
    chop = MarketRegimeDetector.calculate_choppiness_index(df, period=14)

    assert isinstance(chop, float)
    assert not np.isnan(chop)
    assert chop < 50.0, f"Expected trending choppiness < 50.0, got {chop}"


def test_calculate_choppiness_oscillating_market_produces_high_value():
    """Verify oscillating choppy consolidation produces high choppiness (> 60.0)."""
    df = _generate_synthetic_ohlcv(count=80, regime="chop")
    chop = MarketRegimeDetector.calculate_choppiness_index(df, period=14)

    assert isinstance(chop, float)
    assert not np.isnan(chop)
    assert chop > 60.0, f"Expected choppy market choppiness > 60.0, got {chop}"


def test_calculate_choppiness_flat_market_epsilon_protection():
    """Verify flat market (zero range) handles division by zero safely via epsilon."""
    flat_df = _generate_synthetic_ohlcv(count=40, regime="flat")
    chop = MarketRegimeDetector.calculate_choppiness_index(flat_df, period=14)

    assert isinstance(chop, float)
    assert not np.isnan(chop)


def test_calculate_choppiness_custom_period():
    """Verify calculate_choppiness_index respects custom period argument."""
    df = _generate_synthetic_ohlcv(count=60, regime="chop")
    chop_7 = MarketRegimeDetector.calculate_choppiness_index(df, period=7)
    chop_14 = MarketRegimeDetector.calculate_choppiness_index(df, period=14)

    assert isinstance(chop_7, float)
    assert isinstance(chop_14, float)
    assert not np.isnan(chop_7)
    assert not np.isnan(chop_14)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MarketRegimeDetector.analyze Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_analyze_none_or_insufficient_data_returns_safe_defaults():
    """Verify analyze returns safe fallback RegimeAnalysis when df is None or len < 30."""
    res_none = MarketRegimeDetector.analyze(None)
    assert res_none.regime == MarketRegime.TRENDING
    assert res_none.adx == 30.0
    assert res_none.choppiness == 45.0
    assert res_none.bb_width_pct == 1.0
    assert res_none.is_sideways is False
    assert res_none.risk_multiplier == 1.0
    assert res_none.recommended_rr_cap == 3.0

    short_df = _generate_synthetic_ohlcv(count=25, regime="trend")
    res_short = MarketRegimeDetector.analyze(short_df)
    assert res_short.regime == MarketRegime.TRENDING
    assert res_short.is_sideways is False
    assert res_short.risk_multiplier == 1.0


def test_analyze_strong_trending_market():
    """Verify trending setup triggers MarketRegime.TRENDING with full risk multiplier."""
    trend_df = _generate_synthetic_ohlcv(count=80, regime="trend")
    analysis = MarketRegimeDetector.analyze(
        trend_df,
        adx_threshold=20.0,
        chop_threshold=61.8,
        sideways_risk_multiplier=0.5,
    )

    assert analysis.regime == MarketRegime.TRENDING
    assert analysis.is_sideways is False
    assert analysis.risk_multiplier == 1.0
    assert analysis.recommended_rr_cap == 3.5
    assert analysis.adx >= 20.0
    assert analysis.choppiness < 61.8
    assert analysis.bb_width_pct > 0.0


def test_analyze_sideways_via_high_choppiness():
    """Verify choppy consolidation triggers MarketRegime.SIDEWAYS specifically via high choppiness."""
    chop_df = _generate_synthetic_ohlcv(count=80, regime="chop")
    # adx is ~10.6, so adx_threshold=5.0 means (adx < adx_threshold) is False
    # chop is ~71.8, so chop_threshold=61.8 means (chop >= chop_threshold) is True
    analysis = MarketRegimeDetector.analyze(
        chop_df,
        adx_threshold=5.0,
        chop_threshold=61.8,
        sideways_risk_multiplier=0.5,
    )

    assert analysis.regime == MarketRegime.SIDEWAYS
    assert analysis.is_sideways is True
    assert analysis.risk_multiplier == 0.5
    assert analysis.recommended_rr_cap == 1.8
    assert analysis.choppiness >= 61.8


def test_analyze_sideways_via_low_adx_threshold():
    """Verify setup with ADX below threshold triggers SIDEWAYS specifically via low ADX."""
    chop_df = _generate_synthetic_ohlcv(count=80, regime="chop")
    # adx is ~10.6, so adx_threshold=20.0 means (adx < adx_threshold) is True
    # chop is ~71.8, so chop_threshold=85.0 means (chop >= chop_threshold) is False
    analysis = MarketRegimeDetector.analyze(
        chop_df,
        adx_threshold=20.0,
        chop_threshold=85.0,
        sideways_risk_multiplier=0.4,
    )

    assert analysis.is_sideways is True
    assert analysis.regime == MarketRegime.SIDEWAYS
    assert analysis.risk_multiplier == 0.4
    assert analysis.recommended_rr_cap == 1.8
    assert analysis.adx < 20.0


def test_analyze_custom_sideways_risk_multiplier():
    """Verify custom sideways_risk_multiplier is accurately reflected in RegimeAnalysis."""
    chop_df = _generate_synthetic_ohlcv(count=80, regime="chop")
    analysis = MarketRegimeDetector.analyze(
        chop_df,
        adx_threshold=20.0,
        chop_threshold=61.8,
        sideways_risk_multiplier=0.25,
    )

    assert analysis.is_sideways is True
    assert analysis.risk_multiplier == 0.25

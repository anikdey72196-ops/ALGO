"""
tests/test_ai_analyst_heuristics.py
Unit tests for AIAnalyst._heuristic_evaluation and offline fallback in core/ai_analyst.py.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

from core.ai_analyst import AIAnalyst, AIDecision
from core.config import TradingConfig, Direction, MarketBias
from strategies.strategy import TradeSignal, HTFAnalysis, LTFConfirmation


def _create_sample_signal(
    direction: Direction = Direction.BUY,
    htf_bias: MarketBias = MarketBias.BULLISH,
    ltf_confirmation: LTFConfirmation = LTFConfirmation.OB_SCALP_5M,
    rr_ratio: float = 1.5,
    sl_distance: float = 2.0,
    strategy_id: str = "SCALP",
) -> TradeSignal:
    """Helper to construct deterministic TradeSignal instances."""
    signal = TradeSignal(
        symbol="EURUSD",
        direction=direction,
        entry_price=1.08500,
        stop_loss=1.08300 if direction == Direction.BUY else 1.08700,
        take_profit=1.08800 if direction == Direction.BUY else 1.08200,
        htf_bias=htf_bias,
        ltf_confirmation=ltf_confirmation,
        rr_ratio=rr_ratio,
        quality_score=80.0,
        timestamp=datetime.now(timezone.utc),
        sl_distance=sl_distance,
        tp_distance=sl_distance * rr_ratio,
    )
    signal.strategy_id = strategy_id
    return signal


def _create_sample_htf(bias: MarketBias = MarketBias.BULLISH) -> HTFAnalysis:
    """Helper to construct deterministic HTFAnalysis instances."""
    return HTFAnalysis(
        bias=bias,
        ema_value=1.08000,
        last_swing_high=1.09000,
        last_swing_low=1.07500,
        trend_clarity_score=25.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Macro Trend Confluence & Rejection Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_heuristic_opposing_macro_bullish_bias_sell_signal():
    """Verify setup opposing macro trend (SELL in BULLISH macro) is rejected immediately."""
    analyst = AIAnalyst(TradingConfig(ai_confirmation_enabled=True, ai_confidence_threshold=75.0))
    signal = _create_sample_signal(direction=Direction.SELL, htf_bias=MarketBias.BULLISH)
    htf = _create_sample_htf(bias=MarketBias.BULLISH)

    decision = analyst._heuristic_evaluation(signal, htf, current_spread=0.0001)

    assert decision.confirmed is False
    assert decision.confidence == 30.0
    assert "AI Rejected: Setup opposes macro institutional order flow." in decision.reason


def test_heuristic_opposing_macro_bearish_bias_buy_signal():
    """Verify setup opposing macro trend (BUY in BEARISH macro) is rejected immediately."""
    analyst = AIAnalyst(TradingConfig(ai_confirmation_enabled=True, ai_confidence_threshold=75.0))
    signal = _create_sample_signal(direction=Direction.BUY, htf_bias=MarketBias.BEARISH)
    htf = _create_sample_htf(bias=MarketBias.BEARISH)

    decision = analyst._heuristic_evaluation(signal, htf, current_spread=0.0001)

    assert decision.confirmed is False
    assert decision.confidence == 30.0
    assert "AI Rejected: Setup opposes macro institutional order flow." in decision.reason


def test_heuristic_neutral_macro_bias_rejected():
    """Verify neutral macro trend bias cannot provide institutional confluence and is rejected."""
    analyst = AIAnalyst(TradingConfig(ai_confirmation_enabled=True, ai_confidence_threshold=75.0))
    signal = _create_sample_signal(direction=Direction.BUY, htf_bias=MarketBias.NEUTRAL)
    htf = _create_sample_htf(bias=MarketBias.NEUTRAL)

    decision = analyst._heuristic_evaluation(signal, htf, current_spread=0.0001)

    assert decision.confirmed is False
    assert decision.confidence == 30.0
    assert "AI Rejected: Setup opposes macro institutional order flow." in decision.reason


def test_heuristic_macro_alignment_bullish_and_bearish():
    """Verify aligned BUY/BULLISH and SELL/BEARISH award +35 pts macro confluence."""
    analyst = AIAnalyst(TradingConfig(ai_confirmation_enabled=True, ai_confidence_threshold=70.0))

    # Bullish Alignment
    sig_buy = _create_sample_signal(direction=Direction.BUY, htf_bias=MarketBias.BULLISH)
    htf_buy = _create_sample_htf(bias=MarketBias.BULLISH)
    dec_buy = analyst._heuristic_evaluation(sig_buy, htf_buy, current_spread=0.0001)
    assert dec_buy.confirmed is True
    assert "Macro bullish alignment above 200 EMA" in dec_buy.reason

    # Bearish Alignment
    sig_sell = _create_sample_signal(direction=Direction.SELL, htf_bias=MarketBias.BEARISH)
    htf_sell = _create_sample_htf(bias=MarketBias.BEARISH)
    dec_sell = analyst._heuristic_evaluation(sig_sell, htf_sell, current_spread=0.0001)
    assert dec_sell.confirmed is True
    assert "Macro bearish alignment below 200 EMA" in dec_sell.reason


# ─────────────────────────────────────────────────────────────────────────────
# 2. SMC / ICT Confirmation Scoring Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "confirmation, expected_pts, expected_reason",
    [
        (LTFConfirmation.OB_SCALP_5M, 35.0, "5M Break of Structure & Order Block retest scalp"),
        (LTFConfirmation.ICT_SILVER_BULLET, 35.0, "ICT Silver Bullet institutional liquidity sweep & FVG model"),
        (LTFConfirmation.ICT_KILLZONE_FVG, 35.0, "ICT Kill Zone high-volume Fair Value Gap retest"),
        (LTFConfirmation.ICT_JUDAS_SWING, 35.0, "ICT Judas Swing liquidity purge reversal"),
        (LTFConfirmation.ICT_OTE_RETEST, 30.0, "ICT Optimal Trade Entry (OTE) Fib confluence retest"),
        (LTFConfirmation.OB_PLUS_FVG, 35.0, "Order Block + Fair Value Gap confluence retest"),
        (LTFConfirmation.OF_LIQUIDITY_TRAP, 35.0, "Order Flow Liquidity Trap stop-run & delta reversal"),
        (LTFConfirmation.OF_ABSORPTION, 35.0, "Order Flow institutional passive limit absorption"),
        (LTFConfirmation.OF_DELTA_DIVERGENCE, 30.0, "Order Flow Cumulative Volume Delta (CVD) divergence"),
        (LTFConfirmation.OB_MITIGATION, 30.0, "Clean Order Block mitigation"),
        (LTFConfirmation.FVG_MITIGATION, 25.0, "Fair Value Gap fill"),
        (LTFConfirmation.LIQUIDITY_SWEEP, 20.0, "Liquidity sweep confirmed"),
        (LTFConfirmation.PULLBACK, 20.0, "Structural trend continuation"),
        (LTFConfirmation.STRUCTURAL_BREAK, 20.0, "Structural trend continuation"),
    ],
)
def test_heuristic_confirmation_type_scoring(confirmation, expected_pts, expected_reason):
    """Verify each SMC/ICT/OF confirmation type awards correct points and adds appropriate reason."""
    analyst = AIAnalyst(TradingConfig(ai_confirmation_enabled=True, ai_confidence_threshold=50.0))
    # Use rr_ratio=1.0 and sl_distance=0.0 to isolate confirmation scoring:
    # Macro (+35.0) + Confirmation (+expected_pts) + RR (+0.0) + SL (+0.0) = 35.0 + expected_pts
    signal = _create_sample_signal(
        direction=Direction.BUY,
        htf_bias=MarketBias.BULLISH,
        ltf_confirmation=confirmation,
        rr_ratio=1.0,
        sl_distance=0.0,
    )
    htf = _create_sample_htf(bias=MarketBias.BULLISH)

    decision = analyst._heuristic_evaluation(signal, htf, current_spread=0.0001)

    expected_score = min(98.0, 35.0 + expected_pts)
    assert decision.confidence == expected_score
    assert expected_reason in decision.reason


# ─────────────────────────────────────────────────────────────────────────────
# 3. Risk-Reward (R:R) Asymmetry Scoring Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_heuristic_rr_scoring_ob_scalp_5m():
    """Verify R:R scoring tiers for 5M OB scalps (>= 1.5 -> +20, >= 1.4 -> +15, < 1.4 -> +0)."""
    analyst = AIAnalyst(TradingConfig(ai_confirmation_enabled=True, ai_confidence_threshold=50.0))
    htf = _create_sample_htf(bias=MarketBias.BULLISH)

    # Base: Macro 35 + Model 35 + SL 10 = 80 pts
    # Case 1: R:R = 1.5 -> +20 pts -> total 100 capped at 98.0
    sig_optimal = _create_sample_signal(ltf_confirmation=LTFConfirmation.OB_SCALP_5M, rr_ratio=1.5, sl_distance=2.0)
    dec_optimal = analyst._heuristic_evaluation(sig_optimal, htf, current_spread=0.0001)
    assert dec_optimal.confidence == 98.0
    assert "Optimal 5M Scalp R:R (1.50:1)" in dec_optimal.reason

    # Case 2: R:R = 1.4 -> +15 pts -> total 95.0
    sig_acc = _create_sample_signal(ltf_confirmation=LTFConfirmation.OB_SCALP_5M, rr_ratio=1.4, sl_distance=2.0)
    dec_acc = analyst._heuristic_evaluation(sig_acc, htf, current_spread=0.0001)
    assert dec_acc.confidence == 95.0
    assert "Acceptable 5M Scalp R:R (1.40:1)" in dec_acc.reason

    # Case 3: R:R = 1.3 -> +0 pts -> total 80.0
    sig_low = _create_sample_signal(ltf_confirmation=LTFConfirmation.OB_SCALP_5M, rr_ratio=1.3, sl_distance=2.0)
    dec_low = analyst._heuristic_evaluation(sig_low, htf, current_spread=0.0001)
    assert dec_low.confidence == 80.0
    assert "Scalp R:R" not in dec_low.reason


def test_heuristic_rr_scoring_ict():
    """Verify R:R scoring tiers for ICT models (>= 2.0 -> +20, >= 1.5 -> +15, < 1.5 -> +0)."""
    analyst = AIAnalyst(TradingConfig(ai_confirmation_enabled=True, ai_confidence_threshold=50.0))
    htf = _create_sample_htf(bias=MarketBias.BULLISH)

    # Base: Macro 35 + Model 35 (Silver Bullet) + SL 10 = 80 pts
    # Case 1: R:R = 2.0 -> +20 pts -> total 100 capped at 98.0
    sig_optimal = _create_sample_signal(ltf_confirmation=LTFConfirmation.ICT_SILVER_BULLET, rr_ratio=2.0, sl_distance=2.0)
    dec_optimal = analyst._heuristic_evaluation(sig_optimal, htf, current_spread=0.0001)
    assert dec_optimal.confidence == 98.0
    assert "Optimal ICT Model R:R (2.00:1)" in dec_optimal.reason

    # Case 2: R:R = 1.6 -> +15 pts -> total 95.0
    sig_acc = _create_sample_signal(ltf_confirmation=LTFConfirmation.ICT_SILVER_BULLET, rr_ratio=1.6, sl_distance=2.0)
    dec_acc = analyst._heuristic_evaluation(sig_acc, htf, current_spread=0.0001)
    assert dec_acc.confidence == 95.0
    assert "Acceptable ICT Model R:R (1.60:1)" in dec_acc.reason

    # Case 3: R:R = 1.2 -> +0 pts -> total 80.0
    sig_low = _create_sample_signal(ltf_confirmation=LTFConfirmation.ICT_SILVER_BULLET, rr_ratio=1.2, sl_distance=2.0)
    dec_low = analyst._heuristic_evaluation(sig_low, htf, current_spread=0.0001)
    assert dec_low.confidence == 80.0
    assert "ICT Model R:R" not in dec_low.reason


def test_heuristic_rr_scoring_order_flow():
    """Verify R:R scoring tiers for Order Flow setups (>= 2.0 -> +20, >= 1.5 -> +15, < 1.5 -> +0)."""
    analyst = AIAnalyst(TradingConfig(ai_confirmation_enabled=True, ai_confidence_threshold=50.0))
    htf = _create_sample_htf(bias=MarketBias.BULLISH)

    # Tested via OF_ in confirmation name
    sig_optimal = _create_sample_signal(ltf_confirmation=LTFConfirmation.OF_ABSORPTION, rr_ratio=2.5, sl_distance=2.0)
    dec_optimal = analyst._heuristic_evaluation(sig_optimal, htf, current_spread=0.0001)
    assert "Optimal Order Flow R:R (2.50:1)" in dec_optimal.reason

    sig_acc = _create_sample_signal(ltf_confirmation=LTFConfirmation.OF_ABSORPTION, rr_ratio=1.5, sl_distance=2.0)
    dec_acc = analyst._heuristic_evaluation(sig_acc, htf, current_spread=0.0001)
    assert "Acceptable Order Flow R:R (1.50:1)" in dec_acc.reason

    # Tested via strategy_id containing ORDER_FLOW
    sig_by_id = _create_sample_signal(
        ltf_confirmation=LTFConfirmation.OB_MITIGATION,
        rr_ratio=2.2,
        sl_distance=2.0,
        strategy_id="ORDER_FLOW_REVERSAL",
    )
    dec_by_id = analyst._heuristic_evaluation(sig_by_id, htf, current_spread=0.0001)
    assert "Optimal Order Flow R:R (2.20:1)" in dec_by_id.reason


def test_heuristic_rr_scoring_default():
    """Verify default R:R scoring (>= 3.0 -> +20, >= 2.5 -> +15, < 2.5 -> +0)."""
    analyst = AIAnalyst(TradingConfig(ai_confirmation_enabled=True, ai_confidence_threshold=50.0))
    htf = _create_sample_htf(bias=MarketBias.BULLISH)

    # Base: Macro 35 + Model 30 (OB_MITIGATION) + SL 10 = 75 pts
    # Case 1: R:R = 3.2 -> +20 pts -> total 95.0
    sig_strong = _create_sample_signal(ltf_confirmation=LTFConfirmation.OB_MITIGATION, rr_ratio=3.2, sl_distance=2.0)
    dec_strong = analyst._heuristic_evaluation(sig_strong, htf, current_spread=0.0001)
    assert dec_strong.confidence == 95.0
    assert "Strong asymmetric R:R (3.20:1)" in dec_strong.reason

    # Case 2: R:R = 2.6 -> +15 pts -> total 90.0
    sig_acc = _create_sample_signal(ltf_confirmation=LTFConfirmation.OB_MITIGATION, rr_ratio=2.6, sl_distance=2.0)
    dec_acc = analyst._heuristic_evaluation(sig_acc, htf, current_spread=0.0001)
    assert dec_acc.confidence == 90.0
    assert "Acceptable R:R (2.60:1)" in dec_acc.reason

    # Case 3: R:R = 2.0 -> +0 pts -> total 75.0
    sig_low = _create_sample_signal(ltf_confirmation=LTFConfirmation.OB_MITIGATION, rr_ratio=2.0, sl_distance=2.0)
    dec_low = analyst._heuristic_evaluation(sig_low, htf, current_spread=0.0001)
    assert dec_low.confidence == 75.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Invalidation & Threshold Confirmation / Rejection Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_heuristic_invalidation_quality():
    """Verify sl_distance > 0 awards +10 pts, whereas sl_distance <= 0 awards 0 pts."""
    analyst = AIAnalyst(TradingConfig(ai_confirmation_enabled=True, ai_confidence_threshold=50.0))
    htf = _create_sample_htf(bias=MarketBias.BULLISH)

    # With sl_distance > 0: Macro 35 + Model 35 (OB_PLUS_FVG) + RR 0 + SL 10 = 80.0
    sig_with_sl = _create_sample_signal(ltf_confirmation=LTFConfirmation.OB_PLUS_FVG, rr_ratio=1.0, sl_distance=1.5)
    dec_with_sl = analyst._heuristic_evaluation(sig_with_sl, htf, current_spread=0.0001)
    assert dec_with_sl.confidence == 80.0

    # With sl_distance == 0: Macro 35 + Model 35 + RR 0 + SL 0 = 70.0
    sig_no_sl = _create_sample_signal(ltf_confirmation=LTFConfirmation.OB_PLUS_FVG, rr_ratio=1.0, sl_distance=0.0)
    dec_no_sl = analyst._heuristic_evaluation(sig_no_sl, htf, current_spread=0.0001)
    assert dec_no_sl.confidence == 70.0


def test_heuristic_threshold_rejection_insufficient_confluence():
    """Verify setups with confidence below threshold are rejected with an informative reason."""
    # Set high threshold: 85.0%
    analyst = AIAnalyst(TradingConfig(ai_confirmation_enabled=True, ai_confidence_threshold=85.0))
    htf = _create_sample_htf(bias=MarketBias.BULLISH)

    # Macro 35 + LIQUIDITY_SWEEP 20 + RR 0 + SL 10 = 65.0%
    signal = _create_sample_signal(ltf_confirmation=LTFConfirmation.LIQUIDITY_SWEEP, rr_ratio=1.0, sl_distance=1.0)

    decision = analyst._heuristic_evaluation(signal, htf, current_spread=0.0001)

    assert decision.confirmed is False
    assert decision.confidence == 65.0
    assert "Confidence (65.0%) below threshold (85.0%). Insufficient confluence." in decision.reason
    assert decision.warnings == "Evaluated via internal heuristic AI analyzer."


def test_heuristic_confidence_cap_at_98():
    """Verify maximum possible score (100+) is capped at 98.0% confidence."""
    analyst = AIAnalyst(TradingConfig(ai_confirmation_enabled=True, ai_confidence_threshold=75.0))
    htf = _create_sample_htf(bias=MarketBias.BULLISH)

    # Macro 35 + Model 35 + RR 20 + SL 10 = 100.0 pts
    signal = _create_sample_signal(
        ltf_confirmation=LTFConfirmation.ICT_KILLZONE_FVG,
        rr_ratio=2.5,
        sl_distance=2.0,
    )

    decision = analyst._heuristic_evaluation(signal, htf, current_spread=0.0001)

    assert decision.confidence == 98.0
    assert decision.confirmed is True


# ─────────────────────────────────────────────────────────────────────────────
# 5. End-to-End Fallback & Routing in evaluate_setup
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluate_setup_disabled_bypass():
    """Verify evaluate_setup immediately confirms with 100% when ai_confirmation_enabled is False."""
    config = TradingConfig(ai_confirmation_enabled=False)
    analyst = AIAnalyst(config)
    signal = _create_sample_signal(direction=Direction.SELL, htf_bias=MarketBias.BULLISH)  # Opposes macro
    htf = _create_sample_htf(bias=MarketBias.BULLISH)

    decision = analyst.evaluate_setup(signal, htf, current_spread=0.0001)

    assert decision.confirmed is True
    assert decision.confidence == 100.0
    assert "AI Confirmation Gate is bypassed in config." in decision.reason


def test_evaluate_setup_offline_routes_to_heuristic(monkeypatch):
    """Verify evaluate_setup without API key routes to heuristic evaluation."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config = TradingConfig(ai_confirmation_enabled=True, gemini_api_key="")
    analyst = AIAnalyst(config)
    assert analyst._client is None

    signal = _create_sample_signal(
        direction=Direction.BUY,
        htf_bias=MarketBias.BULLISH,
        ltf_confirmation=LTFConfirmation.OB_SCALP_5M,
        rr_ratio=1.5,
    )
    htf = _create_sample_htf(bias=MarketBias.BULLISH)

    decision = analyst.evaluate_setup(signal, htf, current_spread=0.0001)

    assert decision.confirmed is True
    assert decision.confidence == 98.0
    assert decision.warnings == "Evaluated via internal heuristic AI analyzer."


def test_evaluate_setup_gemini_api_exception_fallback(monkeypatch):
    """Verify evaluate_setup gracefully falls back to heuristic evaluation if Gemini API throws."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config = TradingConfig(ai_confirmation_enabled=True, gemini_api_key="")
    analyst = AIAnalyst(config)
    # Inject mock client to simulate API failure
    analyst._client = MagicMock()

    signal = _create_sample_signal(
        direction=Direction.BUY,
        htf_bias=MarketBias.BULLISH,
        ltf_confirmation=LTFConfirmation.ICT_SILVER_BULLET,
        rr_ratio=2.0,
    )
    htf = _create_sample_htf(bias=MarketBias.BULLISH)

    with patch.object(analyst, "_call_gemini_api", side_effect=RuntimeError("Connection timeout")):
        decision = analyst.evaluate_setup(signal, htf, current_spread=0.0001)

    # Fallback to heuristic evaluation succeeded
    assert decision.confirmed is True
    assert decision.confidence == 98.0
    assert decision.warnings == "Evaluated via internal heuristic AI analyzer."

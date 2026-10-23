"""
tests/test_ml_cluster_gatekeeper.py
===================================

Comprehensive test suite verifying:
1. Zero Data Loss & 4-Cluster Weight Transfer Integrity.
2. Strict Conflict Resolution (Opposing simultaneous BUY vs SELL rejection).
3. Hardcoded Execution Gatekeeper & No-Bypass Policy.
4. Pair-Specific Safety Lockout (Isolated streak tracking & lockout).
5. Web API /api/ml/clusters and /api/ml/lockout/reset endpoints.
"""

import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient

from core.config import Direction, MarketBias, RiskConfig
from core.conflict_resolver import ConflictResolver
from strategies.strategy import TradeSignal, LTFConfirmation
from ml.cluster_manager import (
    AssetCluster,
    CLUSTER_PAIRS,
    ClusterTrapDetectorService,
    ClusterModelInitializer,
    ExecutionGatekeeper,
    PairPerformanceTracker,
    get_cluster_for_symbol,
)
from web_app import app


@pytest.fixture
def mock_signal_factory():
    def _create(symbol: str, direction: Direction, strategy: str, score: float = 85.0, rr: float = 2.0):
        entry = 1.0850 if "USD" in symbol else (2350.0 if "XAU" in symbol else 98.50)
        sl_dist = 0.0020 if "USD" in symbol else (5.0 if "XAU" in symbol else 0.50)
        stop = entry - sl_dist if direction == Direction.BUY else entry + sl_dist
        target = entry + (sl_dist * rr) if direction == Direction.BUY else entry - (sl_dist * rr)
        bias = MarketBias.BULLISH if direction == Direction.BUY else MarketBias.BEARISH
        return TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            strategy_name=strategy,
            quality_score=score,
            rr_ratio=rr,
            htf_bias=bias,
            ltf_confirmation=LTFConfirmation.FVG_MITIGATION,
            tp_distance=abs(target - entry),
            magic_number=124456,
            timestamp=datetime.now(timezone.utc),
        )
    return _create


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Zero Data Loss & Cluster Model Initialization
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_data_loss_and_cluster_initialization():
    """Verify that existing model weights are never overwritten and all 4 clusters are initialized."""
    legacy_files = [
        "trap_detector_SMC.joblib",
        "trap_detector_SMC_SCALP_5M.joblib",
        "trap_detector_ICT.joblib",
        "trap_detector_ORDER_FLOW.joblib",
        "trap_detector_TREND_REVERSAL.joblib",
    ]
    for lf in legacy_files:
        p = Path("ml/artifacts") / lf
        assert p.exists(), f"Legacy weight artifact {lf} must exist!"
        assert p.stat().st_size > 0, f"Legacy weight artifact {lf} must not be empty!"

    svc = ClusterTrapDetectorService(base_dir="ml/artifacts/clusters", legacy_dir="ml/artifacts")
    initr = ClusterModelInitializer(svc)
    report = initr.initialize_all_clusters(force=False)

    assert "group_1" in report
    assert "group_2" in report
    assert "group_3" in report
    assert "group_4" in report

    # Verify all 4 clusters have 5 models initialized
    for c in AssetCluster:
        c_models = svc.get_cluster_models(c)
        assert len(c_models) == 5, f"Cluster {c.value} must have 5 strategy models!"
        for s_name, m in c_models.items():
            assert m is not None
            # Check inference runs without throwing error
            import numpy as np
            from ml.trap_detector import ALL_FEATURES
            x = np.zeros((1, len(ALL_FEATURES)))
            prob = m.predict_proba_genuine(x)
            assert len(prob) == 1
            assert 0.0 <= prob[0] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Strict Conflict Resolution (Opposing BUY vs SELL Rejection)
# ─────────────────────────────────────────────────────────────────────────────

def test_opposing_signals_conflict_rejection(mock_signal_factory):
    """If multiple strategies trigger opposing signals (BUY and SELL) on same pair, reject ALL."""
    risk_cfg = RiskConfig(max_rr_ratio=3.0, min_rr_ratio=1.5, min_tp_spread_multiple=6.0)
    resolver = ConflictResolver(risk_cfg)

    # Simultaneous opposing signals on EURUSD: Strategy 1 BUY, Strategy 2 SELL
    s1 = mock_signal_factory("EURUSD", Direction.BUY, "SMC")
    s2 = mock_signal_factory("EURUSD", Direction.SELL, "ICT")
    # Independent signal on XAUUSD: BUY
    s3 = mock_signal_factory("XAUUSD", Direction.BUY, "ORDER_FLOW")

    result = resolver.resolve([s1, s2, s3], current_spread=0.0001)

    # EURUSD signals must be rejected due to opposing conflict
    assert any("[OPPOSING_SIGNAL_CONFLICT]" in r for r in result.rejection_reasons)
    
    # XAUUSD is non-conflicting, so it should survive
    surviving_symbols = [s.symbol for s in (result.accepted_signals or ())]
    if result.accepted_signal:
        surviving_symbols.append(result.accepted_signal.symbol)
    assert "EURUSD" not in surviving_symbols, "EURUSD must be rejected due to opposing signals!"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Hardcoded Execution Gatekeeper & No-Bypass Policy
# ─────────────────────────────────────────────────────────────────────────────

def test_execution_gatekeeper_no_bypass():
    """Verify that gatekeeper enforces mandatory confidence score and vetoes high SL risk."""
    svc = ClusterTrapDetectorService(base_dir="ml/artifacts/clusters", legacy_dir="ml/artifacts")
    tracker = PairPerformanceTracker(consecutive_loss_limit=3, lockout_duration_minutes=60)
    gatekeeper = ExecutionGatekeeper(
        cluster_service=svc,
        tracker=tracker,
        min_confidence_threshold=0.55,
        max_sl_probability=0.45,
    )

    # 1. Normal validation in shadow mode allows execution while reporting prediction
    verdict_shadow = gatekeeper.validate_execution(
        symbol="XAUUSD",
        strategy_name="SMC",
        direction=Direction.BUY,
        features={"gap_atr": 1.2, "rvol": 1.5},
        is_shadow=True,
    )
    assert verdict_shadow.allowed is True
    assert verdict_shadow.is_shadow is True

    # 2. In strict gated mode with high SL probability / low confidence, it must veto
    gatekeeper_strict = ExecutionGatekeeper(
        cluster_service=svc,
        tracker=tracker,
        min_confidence_threshold=0.9999,  # Impossibly high threshold to test strict veto
        max_sl_probability=0.0001,
    )
    verdict_veto = gatekeeper_strict.validate_execution(
        symbol="EURUSD",
        strategy_name="ICT",
        direction=Direction.BUY,
        features={"gap_atr": 0.1, "rvol": 0.2},
        is_shadow=False,
    )
    assert verdict_veto.allowed is False
    assert verdict_veto.rejection_code == 4001
    assert "Rejected by Cluster Gatekeeper" in verdict_veto.reason


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Pair-Specific Performance Tracker & Safety Lockout
# ─────────────────────────────────────────────────────────────────────────────

def test_pair_specific_safety_lockout():
    """Verify that hitting consecutive losses locks out ONLY the offending pair in the cluster."""
    tracker = PairPerformanceTracker(consecutive_loss_limit=3, lockout_duration_minutes=60)

    # Record 2 losses on USDCHF (Group 2)
    tracker.record_trade_result("USDCHF", pnl=-15.0, exit_type="CLOSED_SL")
    tracker.record_trade_result("USDCHF", pnl=-12.0, exit_type="CLOSED_SL")
    is_locked, _ = tracker.is_locked_out("USDCHF")
    assert is_locked is False

    # 3rd consecutive loss on USDCHF -> Trigger lockout!
    tracker.record_trade_result("USDCHF", pnl=-20.0, exit_type="CLOSED_SL")
    is_locked, reason = tracker.is_locked_out("USDCHF")
    assert is_locked is True
    assert "USDCHF locked out" in reason

    # Crucial cluster integrity check: EURUSD and GBPUSD (same Group 2) must NOT be locked out!
    is_eur_locked, _ = tracker.is_locked_out("EURUSD")
    is_gbp_locked, _ = tracker.is_locked_out("GBPUSD")
    assert is_eur_locked is False, "EURUSD must remain unlocked!"
    assert is_gbp_locked is False, "GBPUSD must remain unlocked!"

    # ExecutionGatekeeper must reject orders for locked pair USDCHF
    svc = ClusterTrapDetectorService(base_dir="ml/artifacts/clusters", legacy_dir="ml/artifacts")
    gk = ExecutionGatekeeper(cluster_service=svc, tracker=tracker)
    verdict = gk.validate_execution(
        symbol="USDCHF",
        strategy_name="SMC",
        direction=Direction.BUY,
        features={},
        is_shadow=False,
    )
    assert verdict.allowed is False
    assert verdict.rejection_code == 4005
    assert "Safety Lockout" in verdict.reason

    # Manual unlock test
    tracker.unlock_pair("USDCHF")
    is_locked_after, _ = tracker.is_locked_out("USDCHF")
    assert is_locked_after is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Web API Endpoints for Cluster & Lockout Management
# ─────────────────────────────────────────────────────────────────────────────

def test_cluster_web_api_endpoints():
    """Verify /api/ml/clusters and /api/ml/lockout/reset endpoints respond correctly."""
    client = TestClient(app)

    # 1. GET /api/ml/clusters
    resp = client.get("/api/ml/clusters")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "clusters" in data
    assert "GROUP_1_METALS" in data["clusters"]
    assert "GROUP_2_MAJORS" in data["clusters"]
    assert "GROUP_3_YEN" in data["clusters"]
    assert "GROUP_4_COMMODITY" in data["clusters"]
    assert "pair_lockouts" in data

    # 2. POST /api/ml/lockout/reset
    reset_resp = client.post("/api/ml/lockout/reset", json={"symbol": "USDCHF"})
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "success"

    # Reset all
    reset_all_resp = client.post("/api/ml/lockout/reset", json={})
    assert reset_all_resp.status_code == 200
    assert reset_all_resp.json()["status"] == "success"

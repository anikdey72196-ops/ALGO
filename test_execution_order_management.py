"""
test_execution_order_management.py — Unit Tests for Execution & Order Management Subsystem.
"""

import time
import os
import sqlite3
import pandas as pd
import numpy as np
from config import Direction, InstrumentConfig, DEFAULT_CONFIG
from execution import MockBrokerAdapter, BracketOrder, OrderResult
from market_regime import MarketRegimeDetector, MarketRegime
from spread_guard import DynamicSpreadGuard
from order_manager import (
    OrderManager, OrderState, classify_error, ErrorClassification, generate_idempotency_key
)
from reconciliation import ReconciliationEngine
from position_manager import PositionManager
from state import StateManager, TradeRecord


def create_synthetic_candles(count: int = 100, trend: bool = True) -> pd.DataFrame:
    """Helper to create trending vs sideways OHLCV DataFrames."""
    timestamps = pd.date_range(start="2026-09-01 00:00:00", periods=count, freq="5min", tz="UTC")
    if trend:
        base = np.linspace(100, 150, count)
        noise = np.random.normal(0, 0.2, count)
        close = base + noise
        high = close + np.random.uniform(0.1, 0.5, count)
        low = close - np.random.uniform(0.1, 0.5, count)
        open_p = close - np.random.normal(0, 0.2, count)
    else:
        base = np.full(count, 100.0)
        noise = np.random.uniform(-0.1, 0.1, count)
        close = base + noise
        high = close + np.random.uniform(0.02, 0.05, count)
        low = close - np.random.uniform(0.02, 0.05, count)
        open_p = close - np.random.uniform(-0.02, 0.02, count)

    return pd.DataFrame({
        'time': timestamps,
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(100, 500, count),
    })


def test_market_regime_detection():
    print("Testing Market Regime Detector...")
    # Trending test
    trend_df = create_synthetic_candles(count=80, trend=True)
    analysis_trend = MarketRegimeDetector.analyze(trend_df, adx_threshold=20.0, chop_threshold=61.8)
    assert not analysis_trend.is_sideways, "Trending market incorrectly classified as sideways"
    assert analysis_trend.risk_multiplier == 1.0

    # Sideways test
    side_df = create_synthetic_candles(count=80, trend=False)
    analysis_side = MarketRegimeDetector.analyze(side_df, adx_threshold=25.0, chop_threshold=55.0, sideways_risk_multiplier=0.5)
    assert analysis_side.is_sideways, "Sideways market incorrectly classified as trending"
    assert analysis_side.risk_multiplier == 0.5
    print("PASS: Market Regime Detection")


def test_dynamic_spread_guard():
    print("Testing Dynamic Spread Guard...")
    guard = DynamicSpreadGuard(window_size=100, percentile_cutoff=95.0)

    # Feed normal spreads around 0.20
    for _ in range(50):
        guard.record_tick("XAUUSD", 0.20 + np.random.uniform(-0.02, 0.02))

    # Evaluate normal spread
    ok, msg, p_val = guard.evaluate_spread("XAUUSD", 0.21, session_killzone="LONDON_OPEN")
    assert ok, f"Normal spread rejected: {msg}"

    # Feed anomalous spike (e.g. 0.85)
    ok_spike, msg_spike, _ = guard.evaluate_spread("XAUUSD", 0.85, session_killzone="LONDON_OPEN")
    assert not ok_spike, "Anomalous spread spike was not blocked"
    print("PASS: Dynamic Spread Guard")


def test_order_idempotency_and_state_machine():
    print("Testing OrderManager Idempotency & Error Classification...")
    broker = MockBrokerAdapter()
    broker.connect()
    om = OrderManager(broker=broker, max_retries=2, base_backoff_sec=0.01)

    bracket = BracketOrder(
        symbol="EURUSD",
        direction=Direction.BUY,
        lot_size=0.1,
        entry_price=1.0850,
        stop_loss=1.0830,
        take_profit=1.0890,
    )
    key = generate_idempotency_key("EURUSD", "SMC", "BUY", 1.0850, "2026-09-24T12:00", 123456)

    # 1. First submission -> FILLED
    state1, res1 = om.submit_bracket_order(bracket, key)
    assert state1 == OrderState.FILLED
    assert res1.success

    # 2. Duplicate submission with exact same idempotency key -> Ignored & returns existing
    state2, res2 = om.submit_bracket_order(bracket, key)
    assert state2 == OrderState.FILLED
    assert res2.order_id == res1.order_id

    # 3. Permanent error classification
    perm_class = classify_error(10014, "Invalid Volume")
    assert perm_class == ErrorClassification.PERMANENT

    # 4. Transient error classification
    trans_class = classify_error(10004, "Requote")
    assert trans_class == ErrorClassification.TRANSIENT
    print("PASS: OrderManager Idempotency & State Machine")


def test_reconciliation_engine(tmp_path):
    print("Testing Reconciliation Engine...")
    db_file = str(tmp_path / "test_recon.db")
    state = StateManager(db_path=db_file)
    broker = MockBrokerAdapter()
    broker.connect()

    # Create a trade in local state
    trade = TradeRecord(
        id=999,
        timestamp=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        symbol="GBPUSD",
        direction=Direction.BUY,
        entry_price=1.2850,
        stop_loss=1.2800,
        take_profit=1.2950,
        lot_size=0.1,
        realized_pnl=0.0,
        status="OPEN",
    )
    state.record_trade(trade)
    assert len(state.get_open_positions()) == 1

    # Broker has NO position #999 (it closed) -> Reconciler should sync it to CLOSED
    reconciler = ReconciliationEngine(broker=broker, state=state, sync_interval_sec=100.0)
    report = reconciler.reconcile_now()
    assert report["closed_synced"] == 1
    assert len(state.get_open_positions()) == 0

    reconciler.shutdown()
    state.close()
    print("PASS: Reconciliation Engine")


def test_position_manager_breakeven(tmp_path):
    print("Testing PositionManager +1R Breakeven...")
    db_file = str(tmp_path / "test_pm.db")
    state = StateManager(db_path=db_file)
    broker = MockBrokerAdapter()
    broker.connect()

    # Entry = 2580.00, SL = 2570.00 (Risk = 10.00 pts)
    trade = TradeRecord(
        id=777,
        timestamp=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        symbol="XAUUSD",
        direction=Direction.BUY,
        entry_price=2580.00,
        stop_loss=2570.00,
        take_profit=2610.00,
        lot_size=0.1,
        realized_pnl=0.0,
        status="OPEN",
    )
    state.record_trade(trade)

    pm = PositionManager(broker=broker, state=state, poll_interval_sec=100.0)

    # Price moves to 2591.00 (+1.1R) -> Triggers Breakeven
    broker.set_price("XAUUSD", 2591.00, 2591.25)
    pm.process_positions()

    assert 777 in pm._be_applied, "Breakeven was not applied at +1.1R"

    pm.shutdown()
    state.close()
    print("PASS: PositionManager Breakeven")


if __name__ == "__main__":
    import tempfile
    import shutil
    td = tempfile.mkdtemp()
    try:
        from pathlib import Path
        test_market_regime_detection()
        test_dynamic_spread_guard()
        test_order_idempotency_and_state_machine()
        test_reconciliation_engine(Path(td))
        test_position_manager_breakeven(Path(td))
        print("=" * 60)
        print("ALL EXECUTION & ORDER MANAGEMENT TESTS PASSED!")
        print("=" * 60)
    finally:
        shutil.rmtree(td, ignore_errors=True)

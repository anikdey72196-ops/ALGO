"""
test_execution_metrics.py — Unit & Integration Tests for Execution Quality Metrics.
"""

import os
import time
import sqlite3
import unittest
from datetime import datetime, timezone
from execution_metrics import ExecutionMetricsCollector, ExecutionOrderRecord, ExecutionFillRecord
from metrics_aggregator import MetricsAggregator


def test_order_latency_and_slippage_calculation(temp_db):
    collector = ExecutionMetricsCollector(db_path=temp_db, batch_size=1, flush_interval_sec=0.1)

    # Stage 0: Signal
    order = collector.start_order(
        symbol="XAUUSD",
        strategy_name="SMC Swing",
        direction="BUY",
        requested_price=2580.00,
        atr_14=5.00,
        pip_size=0.10,
        session_killzone="LONDON_OPEN",
        ml_p_tp=0.72,
        ai_conviction=88.5,
        conflict_score=9.2,
        spread_at_signal=0.25,
    )
    assert order.order_id.startswith("ORD-")
    assert order.status == "PENDING"

    # Stage 1: Submission
    time.sleep(0.01)
    collector.mark_submission(
        order_id=order.order_id,
        requested_lot=1.5,
        stop_loss=2570.00,
        take_profit=2600.00,
        spread_at_submit=0.28,
        risk_pct=0.5,
        risk_amount=250.0,
        account_equity=50000.0,
    )

    # Stage 2: Broker Ack
    time.sleep(0.01)
    collector.mark_broker_ack(order_id=order.order_id, filling_mode="FOK")

    # Stage 3: Fill with 0.15 points adverse slippage = 1.5 pips
    time.sleep(0.01)
    filled = collector.mark_filled(
        order_id=order.order_id,
        trade_id=101,
        filled_price=2580.15,
        filled_lot=1.5,
        spread_at_fill=0.30,
        retries_used=0,
    )

    assert filled is not None
    assert filled.status == "FILLED"
    assert filled.slippage_points == 0.15
    assert filled.slippage_pips == 1.5
    assert filled.slippage_pct_atr == 3.0  # (0.15 / 5.0) * 100
    assert filled.latency.total_latency_ms >= 20.0

    # Flush collector to DB
    collector.shutdown()

    # Verify DB persistence
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM execution_orders WHERE order_id = ?", (order.order_id,)).fetchone()
    assert row is not None
    assert row["symbol"] == "XAUUSD"
    assert row["strategy_name"] == "SMC Swing"
    assert row["slippage_pips"] == 1.5
    assert row["status"] == "FILLED"
    assert row["ml_p_tp"] == 0.72
    assert row["ai_conviction"] == 88.5
    conn.close()


def test_rejection_metrics_logging(temp_db):
    collector = ExecutionMetricsCollector(db_path=temp_db, batch_size=1, flush_interval_sec=0.1)

    order = collector.start_order(
        symbol="EURUSD",
        strategy_name="5M Scalp",
        direction="SELL",
        requested_price=1.08500,
        atr_14=0.0020,
        pip_size=0.0001,
    )
    collector.mark_submission(order.order_id, requested_lot=2.0, stop_loss=1.0870, take_profit=1.0810, spread_at_submit=0.00012)
    collector.mark_rejected(order.order_id, rejection_code=10014, rejection_reason="Invalid Volume", retries_used=3)

    collector.shutdown()

    aggregator = MetricsAggregator(db_path=temp_db)
    summary = aggregator.get_summary(window_hours=24)
    assert summary["total_orders"] == 1
    assert summary["filled_orders"] == 0
    assert summary["rejected_orders"] == 1
    assert summary["fill_rate_pct"] == 0.0

    breakdowns = aggregator.get_breakdowns(window_hours=24)
    assert len(breakdowns["rejections"]) == 1
    assert breakdowns["rejections"][0]["reason"] == "Invalid Volume"


def test_aggregation_kpis_and_distributions(temp_db):
    collector = ExecutionMetricsCollector(db_path=temp_db, batch_size=10, flush_interval_sec=0.1)

    # Insert 5 synthetic filled orders
    for i in range(5):
        o = collector.start_order("BTCUSD", "Order Flow Engine", "BUY", 65000.0, 500.0, 1.0)
        collector.mark_submission(o.order_id, 0.5, 64000.0, 67000.0, 1.5)
        collector.mark_broker_ack(o.order_id)
        collector.mark_filled(o.order_id, trade_id=200+i, filled_price=65000.0 + (i * 0.5), filled_lot=0.5, spread_at_fill=1.5)

    collector.shutdown()

    agg = MetricsAggregator(db_path=temp_db)
    summary = agg.get_summary(window_hours=24)
    assert summary["total_orders"] == 5
    assert summary["filled_orders"] == 5
    assert summary["fill_rate_pct"] == 100.0
    assert summary["avg_slippage_pips"] == 1.0  # Mean of [0, 0.5, 1.0, 1.5, 2.0]

    dist = agg.get_slippage_distribution(window_hours=24)
    assert dist["total_measured"] == 5

    hist = agg.get_latency_histogram(window_hours=24, bins=5)
    assert len(hist["buckets"]) == 5


def test_exit_fill_tracking(temp_db):
    collector = ExecutionMetricsCollector(db_path=temp_db, batch_size=1, flush_interval_sec=0.1)

    collector.record_exit_fill(
        trade_id=888,
        exit_type="CLOSED_TP",
        intended_price=2600.0,
        actual_price=2600.5,
        volume=1.0,
        pip_size=0.1,
        atr_14=5.0,
    )
    collector.shutdown()

    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM execution_fills WHERE trade_id = 888").fetchone()
    assert row is not None
    assert row["fill_type"] == "CLOSED_TP"
    assert row["slippage_pips"] == 5.0
    conn.close()


if __name__ == "__main__":
    import tempfile
    import shutil
    td = tempfile.mkdtemp()
    try:
        db = os.path.join(td, "test.db")
        print("Running test_order_latency_and_slippage_calculation...")
        test_order_latency_and_slippage_calculation(db)
        print("PASS")

        db2 = os.path.join(td, "test2.db")
        print("Running test_rejection_metrics_logging...")
        test_rejection_metrics_logging(db2)
        print("PASS")

        db3 = os.path.join(td, "test3.db")
        print("Running test_aggregation_kpis_and_distributions...")
        test_aggregation_kpis_and_distributions(db3)
        print("PASS")

        db4 = os.path.join(td, "test4.db")
        print("Running test_exit_fill_tracking...")
        test_exit_fill_tracking(db4)
        print("PASS")

        print("=" * 50)
        print("ALL EXECUTION QUALITY METRICS TESTS PASSED!")
        print("=" * 50)
    finally:
        shutil.rmtree(td, ignore_errors=True)

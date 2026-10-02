"""
test_reconciliation.py — Unit Tests for ReconciliationEngine and Orphan Adoption / Conflict Handling.
"""

import time
import os
import sqlite3
import pandas as pd
from datetime import datetime, timezone
import pytest

from core.config import Direction
from execution.reconciliation import ReconciliationEngine
from core.state import StateManager, TradeRecord
from execution import MockBrokerAdapter


def test_reconciliation_adopt_brand_new_orphan(tmp_path):
    """Test that a position appearing on broker not in local state is cleanly adopted."""
    db_file = str(tmp_path / "test_reconcile1.db")
    state = StateManager(db_path=db_file)
    broker = MockBrokerAdapter()
    broker.connect()

    # Place an order on broker directly so it has a ticket
    ticket = 58703462470
    broker._positions[ticket] = {
        "ticket": ticket,
        "symbol": "EURUSD",
        "type": 0,  # BUY
        "price_open": 1.08500,
        "sl": 1.08000,
        "tp": 1.09000,
        "volume": 0.1,
        "profit": 15.50,
        "magic": 125456,
        "time": int(time.time()),
    }

    reconciler = ReconciliationEngine(broker=broker, state=state, sync_interval_sec=100.0)
    report = reconciler.reconcile_now()

    assert report["orphans_detected"] == 1
    open_positions = state.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].id == ticket
    assert open_positions[0].symbol == "EURUSD"
    assert open_positions[0].status == "OPEN"

    reconciler.shutdown()
    state.close()


def test_reconciliation_restore_prematurely_closed_orphan(tmp_path):
    """
    Test the exact bug scenario:
    A trade with ticket #58703462470 already exists in trade_log as 'CLOSED'.
    Reconciliation detects it open in MT5.
    Instead of crashing with UNIQUE constraint failed: trade_log.id,
    it must restore the position to 'OPEN' state.
    """
    db_file = str(tmp_path / "test_reconcile2.db")
    state = StateManager(db_path=db_file)
    broker = MockBrokerAdapter()
    broker.connect()

    ticket = 58703462470

    # 1. Trade was originally logged and then marked CLOSED
    initial_trade = TradeRecord(
        id=ticket,
        timestamp=datetime.now(timezone.utc),
        symbol="EURUSD",
        direction=Direction.BUY,
        entry_price=1.08500,
        stop_loss=1.08000,
        take_profit=1.09000,
        lot_size=0.1,
        realized_pnl=0.0,
        status="CLOSED",
    )
    state.record_trade(initial_trade)

    # Verify it is closed locally
    assert len(state.get_open_positions()) == 0

    # 2. Broker still has this position open
    broker._positions[ticket] = {
        "ticket": ticket,
        "symbol": "EURUSD",
        "type": 0,
        "price_open": 1.08500,
        "sl": 1.08100,
        "tp": 1.09100,
        "volume": 0.1,
        "profit": 25.00,
        "magic": 125456,
        "time": int(time.time()),
    }

    reconciler = ReconciliationEngine(broker=broker, state=state, sync_interval_sec=100.0)
    report = reconciler.reconcile_now()

    # Orphan detected and restored
    assert report["orphans_detected"] == 1
    open_positions = state.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].id == ticket
    assert open_positions[0].status == "OPEN"
    assert open_positions[0].stop_loss == 1.08100

    # Running reconcile again should find 0 orphans since it's now locally open
    report2 = reconciler.reconcile_now()
    assert report2["orphans_detected"] == 0

    reconciler.shutdown()
    state.close()


def test_state_record_trade_upsert_on_conflict(tmp_path):
    """Test that record_trade with an existing ID does an UPSERT without throwing an error."""
    db_file = str(tmp_path / "test_upsert.db")
    state = StateManager(db_path=db_file)

    ticket = 12345678
    trade1 = TradeRecord(
        id=ticket,
        timestamp=datetime.now(timezone.utc),
        symbol="GBPUSD",
        direction=Direction.SELL,
        entry_price=1.28000,
        stop_loss=1.28500,
        take_profit=1.27000,
        lot_size=0.05,
        realized_pnl=0.0,
        status="CLOSED",
    )
    state.record_trade(trade1)

    # Re-inserting with updated status OPEN
    trade2 = TradeRecord(
        id=ticket,
        timestamp=datetime.now(timezone.utc),
        symbol="GBPUSD",
        direction=Direction.SELL,
        entry_price=1.28000,
        stop_loss=1.28200,
        take_profit=1.27000,
        lot_size=0.05,
        realized_pnl=10.0,
        status="OPEN",
    )
    # Must NOT raise sqlite3.IntegrityError
    state.record_trade(trade2)

    open_pos = state.get_open_positions()
    assert len(open_pos) == 1
    assert open_pos[0].id == ticket
    assert open_pos[0].status == "OPEN"
    assert open_pos[0].stop_loss == 1.28200

    state.close()


def test_reconciliation_broker_disconnect_preserves_local_trades(tmp_path):
    """Test that if broker temporarily disconnects and returns empty, local trades are NOT falsely closed."""
    db_file = str(tmp_path / "test_disconnect.db")
    state = StateManager(db_path=db_file)
    broker = MockBrokerAdapter()
    broker.connect()

    ticket = 999001
    trade = TradeRecord(
        id=ticket,
        timestamp=datetime.now(timezone.utc),
        symbol="EURUSD",
        direction=Direction.BUY,
        entry_price=1.08500,
        stop_loss=1.08000,
        take_profit=1.09000,
        lot_size=0.1,
        realized_pnl=0.0,
        status="OPEN",
    )
    state.record_trade(trade)
    assert len(state.get_open_positions()) == 1

    # Simulate broker disconnect
    broker.disconnect()
    broker._positions.clear()

    reconciler = ReconciliationEngine(broker=broker, state=state, sync_interval_sec=100.0)
    report = reconciler.reconcile_now()

    # Trades should NOT be marked closed
    assert report["closed_synced"] == 0
    assert len(state.get_open_positions()) == 1

    reconciler.shutdown()
    state.close()

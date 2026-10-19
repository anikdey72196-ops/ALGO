"""
test_open_trades_sync.py — Tests for batched broker deal lookup and StateManager batch_update_trade_pnls.
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest

from core.config import Direction
from core.state import StateManager, TradeRecord
from execution.execution import MockBrokerAdapter, BrokerAdapter, MT5Adapter
from execution.reconciliation import ReconciliationEngine


def test_broker_adapter_default_batch_delegation():
    """Verify base BrokerAdapter.get_deals_pnl_for_orders delegates to get_deal_pnl_for_order."""
    class CustomBroker(BrokerAdapter):
        def connect(self): return True
        def disconnect(self): pass
        def send_bracket_order(self, order): pass
        def get_current_price(self, symbol): return None
        def get_account_equity(self): return 10000.0
        def get_open_positions(self, symbol=None): return []
        def get_deal_pnl_for_order(self, order_id):
            if order_id == 10:
                return (50.0, "CLOSED_TP")
            elif order_id == 20:
                return (-30.0, "CLOSED_SL")
            return None

    broker = CustomBroker()
    res = broker.get_deals_pnl_for_orders([10, 20, 30])
    assert res == {
        10: (50.0, "CLOSED_TP"),
        20: (-30.0, "CLOSED_SL"),
    }


def test_mock_broker_batch_deals():
    """Verify MockBrokerAdapter records and batch queries closed deals."""
    broker = MockBrokerAdapter()
    broker._deal_history[101] = (150.0, "CLOSED_TP")
    broker._deal_history[102] = (-40.0, "CLOSED_SL")

    res = broker.get_deals_pnl_for_orders([101, 102, 103])
    assert res == {
        101: (150.0, "CLOSED_TP"),
        102: (-40.0, "CLOSED_SL"),
    }
    assert broker.get_deal_pnl_for_order(101) == (150.0, "CLOSED_TP")
    assert broker.get_deal_pnl_for_order(999) is None


def test_state_manager_batch_update_trade_pnls(tmp_path):
    """Verify StateManager.batch_update_trade_pnls updates trade_log and daily_state in single batch."""
    db_file = str(tmp_path / "test_batch_state.db")
    state = StateManager(db_path=db_file)

    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)

    # 1. Create trades
    trade1 = TradeRecord(
        id=1,
        timestamp=now,
        symbol="EURUSD",
        direction=Direction.BUY,
        entry_price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0900,
        lot_size=0.1,
        realized_pnl=0.0,
        status="OPEN",
    )
    trade2 = TradeRecord(
        id=2,
        timestamp=yesterday,
        symbol="XAUUSD",
        direction=Direction.BUY,
        entry_price=2600.0,
        stop_loss=2590.0,
        take_profit=2620.0,
        lot_size=0.05,
        realized_pnl=0.0,
        status="OPEN",
    )
    state.record_trade(trade1)
    state.record_trade(trade2)

    assert len(state.get_open_positions()) == 2

    # 2. Batch update
    updates = [
        (1, 100.0, "CLOSED_TP"),
        (2, -50.0, "CLOSED_SL"),
        (999, 0.0, "CLOSED"),  # Non-existent trade should be skipped cleanly
    ]
    state.batch_update_trade_pnls(updates)

    # 3. Verify open positions are now 0
    assert len(state.get_open_positions()) == 0

    # 4. Check trade_log
    all_trades = {t.id: t for t in state.get_all_trades()}
    assert all_trades[1].status == "CLOSED_TP"
    assert all_trades[1].realized_pnl == 100.0
    assert all_trades[2].status == "CLOSED_SL"
    assert all_trades[2].realized_pnl == -50.0

    # 5. Check daily_state
    daily_pnl = state.get_daily_pnl(today=now.date())
    # Trade 1 (+100) closed today, trade 2 (-50) closed today -> total today = +50
    assert daily_pnl == 50.0

    state.close()


def test_reconciliation_batch_deals_sync(tmp_path):
    """Verify ReconciliationEngine uses batch deal query and updates state."""
    db_file = str(tmp_path / "test_reconcile_batch.db")
    state = StateManager(db_path=db_file)
    broker = MockBrokerAdapter()
    broker.connect()

    now = datetime.now(timezone.utc)
    for tid in (501, 502):
        trade = TradeRecord(
            id=tid,
            timestamp=now,
            symbol="EURUSD",
            direction=Direction.BUY,
            entry_price=1.0850,
            stop_loss=1.0800,
            take_profit=1.0900,
            lot_size=0.1,
            realized_pnl=0.0,
            status="OPEN",
        )
        state.record_trade(trade)

    # Both positions are in local open state, but NOT in broker positions (i.e. closed)
    broker._deal_history[501] = (75.0, "CLOSED_TP")
    broker._deal_history[502] = (-25.0, "CLOSED_SL")

    reconciler = ReconciliationEngine(broker=broker, state=state, sync_interval_sec=100.0)
    report = reconciler.reconcile_now()

    assert report["closed_synced"] == 2
    assert len(state.get_open_positions()) == 0

    all_trades = {t.id: t for t in state.get_all_trades()}
    assert all_trades[501].status == "CLOSED_TP"
    assert all_trades[501].realized_pnl == 75.0
    assert all_trades[502].status == "CLOSED_SL"
    assert all_trades[502].realized_pnl == -25.0

    reconciler.shutdown()
    state.close()

"""
tests/test_night_limit_security_guards.py
Security checks & validation for 3-layer Night Limit & Broker Isolation Guards.
Ensures neither database desync nor mock runners can ever bypass position limits.
"""

import os
from datetime import datetime, timezone
import pytest

from core.config import TradingConfig, Direction, RiskConfig
from core.state import StateManager, TradeRecord
from core.risk_engine import RiskEngine
from strategies.strategy import TradeSignal
from execution.execution import MockBrokerAdapter, BracketOrder, MT5Adapter


def test_lock1_risk_engine_blocks_when_broker_at_night_limit(tmp_path):
    """Lock 1: If broker has 2 positions at night, RiskEngine must block even if DB has 0."""
    db_file = str(tmp_path / "test_sec_risk.db")
    state = StateManager(db_path=db_file)
    config = TradingConfig(use_mock_broker=True)
    config.risk.night_limit_enabled = True
    config.risk.night_start_hour = 23
    config.risk.night_end_hour = 8
    config.risk.night_max_open_positions = 2

    risk_engine = RiskEngine(config, state)

    # Database has 0 open positions
    assert len(state.get_open_positions()) == 0

    # Current time: 00:30 UTC (which falls in night window for UTC/IST)
    night_time = datetime(2026, 10, 23, 0, 30, tzinfo=timezone.utc)

    # 2 positions already active on the broker
    broker_positions = [
        {"ticket": 1001, "symbol": "EURUSD"},
        {"ticket": 1002, "symbol": "GBPUSD"},
    ]

    from core.config import MarketBias
    from strategies.strategy import LTFConfirmation

    sig = TradeSignal(
        symbol="EURUSD",
        direction=Direction.BUY,
        entry_price=1.1200,
        stop_loss=1.1180,
        take_profit=1.1260,
        sl_distance=0.0020,
        tp_distance=0.0060,
        htf_bias=MarketBias.BULLISH,
        ltf_confirmation=LTFConfirmation.OB_MITIGATION,
        rr_ratio=3.0,
        timestamp=night_time,
        strategy_name="ICT",
        quality_score=90.0,
    )

    auth = risk_engine.authorize_trade(
        sig,
        current_equity=10000.0,
        current_time=night_time,
        broker_open_positions=broker_positions,
    )

    assert not auth.authorized
    assert "Night Limit" in (auth.rejection_reason or "")
    assert "2/2 open" in (auth.rejection_reason or "")


def test_lock2_mock_broker_never_touches_db_open_positions(tmp_path):
    """Lock 2: TradingBot._sync_open_positions must no-op when use_mock_broker=True."""
    from main import TradingBot

    db_file = str(tmp_path / "test_sec_sync.db")
    state = StateManager(db_path=db_file)

    # Record a live position in DB
    rec = TradeRecord(
        id=999999,
        timestamp=datetime.now(timezone.utc),
        symbol="EURUSD",
        direction=Direction.BUY,
        entry_price=1.1200,
        stop_loss=1.1180,
        take_profit=1.1260,
        lot_size=0.1,
        realized_pnl=0.0,
        status="OPEN",
        strategy_name="ICT",
    )
    state.record_trade(rec)
    assert len(state.get_open_positions()) == 1

    config = TradingConfig(use_mock_broker=True, db_path=db_file)
    bot = TradingBot(config=config, load_saved_settings=False)
    bot.state = state

    # Calling _sync_open_positions with mock broker must NOT mark trade CLOSED
    bot._sync_open_positions()

    open_trades = state.get_open_positions()
    assert len(open_trades) == 1
    assert open_trades[0].id == 999999
    assert open_trades[0].status == "OPEN"


def test_lock3_broker_pre_dispatch_safety_veto():
    """Lock 3: BrokerAdapter.send_bracket_order drops orders directly if limit exceeded."""
    config = TradingConfig(use_mock_broker=True)
    config.risk.night_limit_enabled = True
    config.risk.night_start_hour = 0
    config.risk.night_end_hour = 24  # Force 24h night limit for deterministic test
    config.risk.night_max_open_positions = 2

    broker = MockBrokerAdapter()
    broker.config = config
    broker.set_price("EURUSD", 1.1200, 1.1202)

    # Fill 2 mock orders
    o1 = BracketOrder(symbol="EURUSD", direction=Direction.BUY, lot_size=0.01, entry_price=1.1202, stop_loss=1.1180, take_profit=1.1260)
    o2 = BracketOrder(symbol="EURUSD", direction=Direction.BUY, lot_size=0.01, entry_price=1.1202, stop_loss=1.1180, take_profit=1.1260)
    res1 = broker.send_bracket_order(o1)
    res2 = broker.send_bracket_order(o2)
    assert res1.success
    assert res2.success
    assert len(broker.get_open_positions()) == 2

    # Attempt 3rd order during limit window -> Must be rejected at broker dispatch level!
    o3 = BracketOrder(symbol="EURUSD", direction=Direction.BUY, lot_size=0.01, entry_price=1.1202, stop_loss=1.1180, take_profit=1.1260)
    res3 = broker.send_bracket_order(o3)
    assert not res3.success
    assert res3.error_code == 10027
    assert "pre-dispatch safety veto" in res3.error_message
    assert len(broker.get_open_positions()) == 2


def test_lock4_test_environment_mysql_isolation(tmp_path):
    """Lock 4: Any test execution or mock run is strictly barred from connecting to live MySQL."""
    from core.database import get_db_connection
    import sqlite3

    os.environ["PYTEST_CURRENT_TEST"] = "test_lock4"
    conn = get_db_connection(str(tmp_path / "test_iso.db"))
    assert isinstance(conn, sqlite3.Connection)
    conn.close()

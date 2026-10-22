"""
tests/test_mock_broker_simulate_close.py
Unit tests for MockBrokerAdapter.simulate_close method in execution/execution.py.
"""

import pytest
from core.config import Direction
from execution.execution import MockBrokerAdapter, BracketOrder


def test_simulate_close_nonexistent_order():
    """Verify simulate_close on an unknown order_id returns 0.0 and does not alter equity or history."""
    broker = MockBrokerAdapter(initial_equity=10_000.0)
    initial_equity = broker._equity

    pnl = broker.simulate_close(order_id=9999, close_price=1.1000, point_value=10.0)

    assert pnl == 0.0
    assert broker._equity == initial_equity
    assert 9999 not in broker._deal_history
    assert broker.get_deal_pnl_for_order(9999) is None


def test_simulate_close_buy_profit():
    """Verify simulate_close on EURUSD BUY in profit."""
    broker = MockBrokerAdapter(initial_equity=10_000.0)
    order_id = 1
    broker._positions[order_id] = {
        "order_id": order_id,
        "symbol": "EURUSD",
        "direction": Direction.BUY,
        "lot_size": 0.1,
        "entry_price": 1.08000,
        "stop_loss": 1.07500,
        "take_profit": 1.09000,
        "comment": "test_buy_profit",
    }

    # close_price is 1.08500 (gain of +0.00500 on 0.1 lots, contract_size=100,000)
    # PnL = 0.00500 * 0.1 * 100,000 = 50.0
    close_price = 1.08500
    pnl = broker.simulate_close(order_id, close_price=close_price, point_value=10.0)

    assert pytest.approx(pnl, rel=1e-5) == 50.0
    assert pytest.approx(broker._equity, rel=1e-5) == 10_050.0
    assert order_id not in broker._positions
    assert broker.get_deal_pnl_for_order(order_id) == (pytest.approx(50.0, rel=1e-5), "CLOSED_TP")


def test_simulate_close_buy_loss():
    """Verify simulate_close on EURUSD BUY in loss."""
    broker = MockBrokerAdapter(initial_equity=10_000.0)
    order_id = 2
    broker._positions[order_id] = {
        "order_id": order_id,
        "symbol": "EURUSD",
        "direction": Direction.BUY,
        "lot_size": 0.2,
        "entry_price": 1.08500,
        "stop_loss": 1.08000,
        "take_profit": 1.09500,
        "comment": "test_buy_loss",
    }

    # close_price is 1.08000 (loss of -0.00500 on 0.2 lots, contract_size=100,000)
    # PnL = -0.00500 * 0.2 * 100,000 = -100.0
    close_price = 1.08000
    pnl = broker.simulate_close(order_id, close_price=close_price, point_value=10.0)

    assert pytest.approx(pnl, rel=1e-5) == -100.0
    assert pytest.approx(broker._equity, rel=1e-5) == 9_900.0
    assert order_id not in broker._positions
    assert broker.get_deal_pnl_for_order(order_id) == (pytest.approx(-100.0, rel=1e-5), "CLOSED_SL")


def test_simulate_close_sell_profit():
    """Verify simulate_close on GBPUSD SELL in profit."""
    broker = MockBrokerAdapter(initial_equity=10_000.0)
    order_id = 3
    broker._positions[order_id] = {
        "order_id": order_id,
        "symbol": "GBPUSD",
        "direction": Direction.SELL,
        "lot_size": 0.5,
        "entry_price": 1.28000,
        "stop_loss": 1.28500,
        "take_profit": 1.27000,
        "comment": "test_sell_profit",
    }

    # close_price is 1.27500 (gain of +0.00500 on 0.5 lots, contract_size=100,000)
    # diff = -(1.27500 - 1.28000) = +0.00500
    # PnL = 0.00500 * 0.5 * 100,000 = 250.0
    close_price = 1.27500
    pnl = broker.simulate_close(order_id, close_price=close_price, point_value=10.0)

    assert pytest.approx(pnl, rel=1e-5) == 250.0
    assert pytest.approx(broker._equity, rel=1e-5) == 10_250.0
    assert order_id not in broker._positions
    assert broker.get_deal_pnl_for_order(order_id) == (pytest.approx(250.0, rel=1e-5), "CLOSED_TP")


def test_simulate_close_sell_loss():
    """Verify simulate_close on GBPUSD SELL in loss."""
    broker = MockBrokerAdapter(initial_equity=10_000.0)
    order_id = 4
    broker._positions[order_id] = {
        "order_id": order_id,
        "symbol": "GBPUSD",
        "direction": Direction.SELL,
        "lot_size": 0.1,
        "entry_price": 1.28000,
        "stop_loss": 1.28500,
        "take_profit": 1.27000,
        "comment": "test_sell_loss",
    }

    # close_price is 1.28500 (loss of -0.00500 on 0.1 lots, contract_size=100,000)
    # diff = -(1.28500 - 1.28000) = -0.00500
    # PnL = -0.00500 * 0.1 * 100,000 = -50.0
    close_price = 1.28500
    pnl = broker.simulate_close(order_id, close_price=close_price, point_value=10.0)

    assert pytest.approx(pnl, rel=1e-5) == -50.0
    assert pytest.approx(broker._equity, rel=1e-5) == 9_950.0
    assert order_id not in broker._positions
    assert broker.get_deal_pnl_for_order(order_id) == (pytest.approx(-50.0, rel=1e-5), "CLOSED_SL")


def test_simulate_close_breakeven():
    """Verify simulate_close at exact entry price yields 0.0 PnL and CLOSED_SL status."""
    broker = MockBrokerAdapter(initial_equity=10_000.0)
    order_id = 5
    broker._positions[order_id] = {
        "order_id": order_id,
        "symbol": "EURUSD",
        "direction": Direction.BUY,
        "lot_size": 1.0,
        "entry_price": 1.08500,
        "stop_loss": 1.08000,
        "take_profit": 1.09000,
        "comment": "test_breakeven",
    }

    pnl = broker.simulate_close(order_id, close_price=1.08500, point_value=10.0)

    assert pnl == 0.0
    assert broker._equity == 10_000.0
    assert order_id not in broker._positions
    # status is CLOSED_TP if pnl > 0 else CLOSED_SL
    assert broker.get_deal_pnl_for_order(order_id) == (0.0, "CLOSED_SL")


def test_simulate_close_metals():
    """Verify simulate_close uses contract_size=100.0 for XAU and GOLD."""
    broker = MockBrokerAdapter(initial_equity=10_000.0)

    # 1. XAUUSD BUY: entry 2500, close 2520, lots 0.5 -> diff=20 * 0.5 * 100 = 1000.0
    order_id_1 = 10
    broker._positions[order_id_1] = {
        "order_id": order_id_1,
        "symbol": "XAUUSD",
        "direction": Direction.BUY,
        "lot_size": 0.5,
        "entry_price": 2500.00,
    }
    pnl_1 = broker.simulate_close(order_id_1, close_price=2520.00, point_value=1.0)
    assert pytest.approx(pnl_1, rel=1e-5) == 1000.0

    # 2. GOLD SELL: entry 2500, close 2490, lots 0.2 -> diff=10 * 0.2 * 100 = 200.0
    order_id_2 = 11
    broker._positions[order_id_2] = {
        "order_id": order_id_2,
        "symbol": "GOLD",
        "direction": Direction.SELL,
        "lot_size": 0.2,
        "entry_price": 2500.00,
    }
    pnl_2 = broker.simulate_close(order_id_2, close_price=2490.00, point_value=1.0)
    assert pytest.approx(pnl_2, rel=1e-5) == 200.0


def test_simulate_close_crypto():
    """Verify simulate_close uses contract_size=1.0 for BTC and ETH."""
    broker = MockBrokerAdapter(initial_equity=50_000.0)

    # 1. BTCUSD BUY: entry 65000, close 66000, lots 0.5 -> diff=1000 * 0.5 * 1.0 = 500.0
    order_id_btc = 20
    broker._positions[order_id_btc] = {
        "order_id": order_id_btc,
        "symbol": "BTCUSD",
        "direction": Direction.BUY,
        "lot_size": 0.5,
        "entry_price": 65000.0,
    }
    pnl_btc = broker.simulate_close(order_id_btc, close_price=66000.0, point_value=1.0)
    assert pytest.approx(pnl_btc, rel=1e-5) == 500.0

    # 2. ETHUSD SELL: entry 3200, close 3300, lots 2.0 -> diff=-100 * 2.0 * 1.0 = -200.0
    order_id_eth = 21
    broker._positions[order_id_eth] = {
        "order_id": order_id_eth,
        "symbol": "ETHUSD",
        "direction": Direction.SELL,
        "lot_size": 2.0,
        "entry_price": 3200.0,
    }
    pnl_eth = broker.simulate_close(order_id_eth, close_price=3300.0, point_value=1.0)
    assert pytest.approx(pnl_eth, rel=1e-5) == -200.0


def test_simulate_close_other_symbols_fallback():
    """Verify contract_size fallback based on point_value parameter for unknown symbols."""
    broker = MockBrokerAdapter(initial_equity=10_000.0)

    # point_value == 10.0 -> contract_size = 100_000.0
    order_id_1 = 30
    broker._positions[order_id_1] = {
        "order_id": order_id_1,
        "symbol": "USDJPY",
        "direction": Direction.BUY,
        "lot_size": 0.1,
        "entry_price": 150.00,
    }
    pnl_1 = broker.simulate_close(order_id_1, close_price=150.50, point_value=10.0)
    # diff = 0.50 * 0.1 * 100_000 = 5000.0
    assert pytest.approx(pnl_1, rel=1e-5) == 5000.0

    # point_value != 10.0 -> contract_size = 100.0
    order_id_2 = 31
    broker._positions[order_id_2] = {
        "order_id": order_id_2,
        "symbol": "US30",
        "direction": Direction.BUY,
        "lot_size": 1.0,
        "entry_price": 40000.0,
    }
    pnl_2 = broker.simulate_close(order_id_2, close_price=40050.0, point_value=1.0)
    # diff = 50.0 * 1.0 * 100.0 = 5000.0
    assert pytest.approx(pnl_2, rel=1e-5) == 5000.0


def test_simulate_close_with_send_bracket_order():
    """Verify end-to-end integration: order creation via send_bracket_order followed by simulate_close."""
    broker = MockBrokerAdapter(initial_equity=20_000.0, slippage_points=0.0)
    broker.set_price("EURUSD", bid=1.08500, ask=1.08510)

    order = BracketOrder(
        symbol="EURUSD",
        direction=Direction.BUY,
        lot_size=0.5,
        entry_price=1.08500,
        stop_loss=1.08000,
        take_profit=1.09500,
        magic=1001,
        comment="e2e_simulate_close",
    )
    result = broker.send_bracket_order(order)
    assert result.success is True
    order_id = result.order_id
    assert order_id in broker._positions

    # Close the order
    close_price = 1.09000
    # diff = 1.09000 - 1.08500 = 0.00500 * 0.5 * 100,000 = 250.0
    pnl = broker.simulate_close(order_id, close_price=close_price, point_value=10.0)

    assert pytest.approx(pnl, rel=1e-5) == 250.0
    assert pytest.approx(broker._equity, rel=1e-5) == 20_250.0
    assert order_id not in broker._positions

    # Verify deal history access methods
    deal = broker.get_deal_pnl_for_order(order_id)
    assert deal is not None
    assert pytest.approx(deal[0], rel=1e-5) == 250.0
    assert deal[1] == "CLOSED_TP"

    batch_deals = broker.get_deals_pnl_for_orders([order_id, 9999])
    assert order_id in batch_deals
    assert 9999 not in batch_deals

    # Repeated simulate_close on already closed order returns 0.0
    pnl_again = broker.simulate_close(order_id, close_price=close_price, point_value=10.0)
    assert pnl_again == 0.0
    assert pytest.approx(broker._equity, rel=1e-5) == 20_250.0


def test_simulate_close_cumulative_equity():
    """Verify multiple positions sequentially update cumulative equity accurately."""
    broker = MockBrokerAdapter(initial_equity=10_000.0)

    broker._positions[1] = {"order_id": 1, "symbol": "EURUSD", "direction": Direction.BUY, "lot_size": 0.1, "entry_price": 1.0800}
    broker._positions[2] = {"order_id": 2, "symbol": "EURUSD", "direction": Direction.SELL, "lot_size": 0.1, "entry_price": 1.0850}
    broker._positions[3] = {"order_id": 3, "symbol": "XAUUSD", "direction": Direction.BUY, "lot_size": 0.2, "entry_price": 2500.0}

    # Trade 1: +50.0
    broker.simulate_close(1, close_price=1.0850, point_value=10.0)
    assert pytest.approx(broker._equity, rel=1e-5) == 10_050.0

    # Trade 2: -30.0 (entry 1.0850, close 1.0880 -> diff = -0.0030 * 0.1 * 100,000 = -30.0)
    broker.simulate_close(2, close_price=1.0880, point_value=10.0)
    assert pytest.approx(broker._equity, rel=1e-5) == 10_020.0

    # Trade 3: +100.0 (entry 2500.0, close 2505.0 -> diff = 5.0 * 0.2 * 100 = 100.0)
    broker.simulate_close(3, close_price=2505.0, point_value=1.0)
    assert pytest.approx(broker._equity, rel=1e-5) == 10_120.0

    assert len(broker._positions) == 0
    assert len(broker._deal_history) == 3

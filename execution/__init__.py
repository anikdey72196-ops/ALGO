"""
execution package — Broker adapters (MetaTrader 5 & Mock), order lifecycle managers, execution metrics, and position managers.
"""

from .execution import (
    BrokerAdapter,
    MockBrokerAdapter,
    MT5Adapter,
    BracketOrder,
    PriceQuote,
    OrderResult,
)
from .order_manager import OrderManager, generate_idempotency_key, OrderState
from .position_manager import PositionManager
from .spread_guard import DynamicSpreadGuard
from .reconciliation import ReconciliationEngine
from .execution_metrics import ExecutionMetricsCollector, ExecutionOrderRecord
from .metrics_aggregator import MetricsAggregator

__all__ = [
    "BrokerAdapter",
    "MockBrokerAdapter",
    "MT5Adapter",
    "BracketOrder",
    "PriceQuote",
    "OrderResult",
    "OrderManager",
    "generate_idempotency_key",
    "OrderState",
    "PositionManager",
    "DynamicSpreadGuard",
    "ReconciliationEngine",
    "ExecutionMetricsCollector",
    "ExecutionOrderRecord",
    "MetricsAggregator",
]

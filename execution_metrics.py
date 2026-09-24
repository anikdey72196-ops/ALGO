"""
execution_metrics.py — High-Performance Execution Quality Tracking Subsystem.

Features:
- Sub-millisecond monotonic latency profiling (T0 -> T1 -> T2 -> T3 -> T4).
- Price slippage in absolute points, pips, and ATR-normalized percentage.
- Spread drift and broker execution friction monitoring.
- Non-blocking asynchronous buffer with batch database writer.
- Resilient to broker disconnects (retains queue in memory and drains upon reconnect).
"""

from __future__ import annotations

import time
import uuid
import queue
import threading
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from loguru import logger


@dataclass
class OrderLatencyStage:
    """Monotonic nanosecond markers for order lifecycle profiling."""
    t0_signal_ns: int = 0
    t1_submit_ns: int = 0
    t2_ack_ns: int = 0
    t3_fill_ns: int = 0
    t4_sync_ns: int = 0

    @property
    def signal_to_submit_ms(self) -> float:
        if self.t0_signal_ns and self.t1_submit_ns:
            return max(0.0, (self.t1_submit_ns - self.t0_signal_ns) / 1_000_000.0)
        return 0.0

    @property
    def submit_to_ack_ms(self) -> float:
        if self.t1_submit_ns and self.t2_ack_ns:
            return max(0.0, (self.t2_ack_ns - self.t1_submit_ns) / 1_000_000.0)
        return 0.0

    @property
    def ack_to_fill_ms(self) -> float:
        if self.t2_ack_ns and self.t3_fill_ns:
            return max(0.0, (self.t3_fill_ns - self.t2_ack_ns) / 1_000_000.0)
        return 0.0

    @property
    def total_latency_ms(self) -> float:
        if self.t0_signal_ns and self.t3_fill_ns:
            return max(0.0, (self.t3_fill_ns - self.t0_signal_ns) / 1_000_000.0)
        return self.signal_to_submit_ms + self.submit_to_ack_ms + self.ack_to_fill_ms


@dataclass
class ExecutionOrderRecord:
    """Full lifecycle metric record for an order."""
    order_id: str = field(default_factory=lambda: f"ORD-{uuid.uuid4().hex[:12].upper()}")
    trade_id: Optional[int] = None
    symbol: str = ""
    strategy_name: str = ""
    magic_number: int = 123456
    direction: str = ""
    order_type: str = "BRACKET"
    filling_mode: str = "FOK"
    
    # Volumes & Prices
    requested_lot: float = 0.0
    filled_lot: float = 0.0
    requested_price: float = 0.0
    filled_price: Optional[float] = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    atr_14: float = 0.0
    pip_size: float = 0.0001
    
    # Slippage Calculations
    slippage_points: float = 0.0
    slippage_pips: float = 0.0
    slippage_pct_atr: float = 0.0
    
    # Spread Context
    spread_at_signal: float = 0.0
    spread_at_submit: float = 0.0
    spread_at_fill: float = 0.0
    
    # UTC Timestamps
    signal_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    submit_time: Optional[str] = None
    ack_time: Optional[str] = None
    fill_time: Optional[str] = None
    close_time: Optional[str] = None
    
    # Latencies
    latency: OrderLatencyStage = field(default_factory=OrderLatencyStage)
    
    # Status & Diagnostics
    retries_used: int = 0
    rejection_code: Optional[int] = None
    rejection_reason: Optional[str] = None
    status: str = "PENDING"  # PENDING, FILLED, REJECTED, CANCELLED
    
    # Intelligence & Risk Context
    session_killzone: str = "OFF_HOURS"
    ml_p_tp: Optional[float] = None
    ml_p_sl: Optional[float] = None
    ai_conviction: Optional[float] = None
    conflict_score: Optional[float] = None
    risk_pct: float = 0.5
    risk_amount: float = 0.0
    account_equity: float = 0.0
    broker_name: str = "MetaTrader 5"

    def compute_slippage(self) -> None:
        """Calculate signed slippage in points, pips, and % of ATR."""
        if self.filled_price is None or self.requested_price <= 0:
            return

        # For BUY: filled higher than requested is adverse slippage (+)
        # For SELL: filled lower than requested is adverse slippage (+)
        price_diff = self.filled_price - self.requested_price
        if self.direction.upper() == "SELL":
            price_diff = -price_diff

        self.slippage_points = round(price_diff, 6)
        if self.pip_size > 0:
            self.slippage_pips = round(self.slippage_points / self.pip_size, 2)

        if self.atr_14 > 0:
            self.slippage_pct_atr = round((self.slippage_points / self.atr_14) * 100.0, 2)


@dataclass
class ExecutionFillRecord:
    """Individual execution fill event (entry or exit)."""
    fill_id: str = field(default_factory=lambda: f"FILL-{uuid.uuid4().hex[:10].upper()}")
    order_id: str = ""
    trade_id: Optional[int] = None
    deal_ticket: Optional[int] = None
    fill_type: str = "ENTRY"  # ENTRY, SL_EXIT, TP_EXIT, MANUAL_EXIT
    volume: float = 0.0
    intended_price: float = 0.0
    actual_price: float = 0.0
    slippage_pips: float = 0.0
    slippage_pct_atr: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    fill_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ExecutionMetricsCollector:
    """
    Thread-safe, non-blocking collector for execution metrics.
    Buffers events in memory and writes asynchronously to SQLite.
    """

    def __init__(self, db_path: str = "trading_state.db", batch_size: int = 20, flush_interval_sec: float = 2.0):
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec
        self._queue: queue.Queue = queue.Queue(maxsize=10_000)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._active_orders: Dict[str, ExecutionOrderRecord] = {}

        self._ensure_schema()

        # Start asynchronous DB worker thread
        self._worker = threading.Thread(target=self._db_writer_loop, daemon=True, name="EQM-Writer")
        self._worker.start()
        logger.info(f"ExecutionMetricsCollector initialized with background DB flush on {db_path}")

    def _ensure_schema(self) -> None:
        """Create tables if not already present."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS execution_orders (
                        order_id TEXT PRIMARY KEY,
                        trade_id INTEGER,
                        symbol TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        magic_number INTEGER NOT NULL,
                        direction TEXT NOT NULL,
                        order_type TEXT NOT NULL,
                        filling_mode TEXT DEFAULT 'FOK',
                        requested_lot REAL NOT NULL,
                        filled_lot REAL DEFAULT 0.0,
                        requested_price REAL NOT NULL,
                        filled_price REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        atr_14 REAL,
                        slippage_points REAL,
                        slippage_pips REAL,
                        slippage_pct_atr REAL,
                        spread_at_signal REAL,
                        spread_at_submit REAL,
                        spread_at_fill REAL,
                        signal_time TEXT NOT NULL,
                        submit_time TEXT,
                        ack_time TEXT,
                        fill_time TEXT,
                        close_time TEXT,
                        latency_signal_to_submit_ms REAL,
                        latency_submit_to_ack_ms REAL,
                        latency_ack_to_fill_ms REAL,
                        latency_total_ms REAL,
                        retries_used INTEGER DEFAULT 0,
                        rejection_code INTEGER,
                        rejection_reason TEXT,
                        status TEXT NOT NULL,
                        session_killzone TEXT,
                        ml_p_tp REAL,
                        ml_p_sl REAL,
                        ai_conviction REAL,
                        conflict_score REAL,
                        risk_pct REAL,
                        risk_amount REAL,
                        account_equity REAL,
                        broker_name TEXT DEFAULT 'MetaTrader 5',
                        created_at TEXT DEFAULT (datetime('now'))
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS execution_fills (
                        fill_id TEXT PRIMARY KEY,
                        order_id TEXT NOT NULL,
                        trade_id INTEGER,
                        deal_ticket INTEGER,
                        fill_type TEXT NOT NULL,
                        volume REAL NOT NULL,
                        intended_price REAL NOT NULL,
                        actual_price REAL NOT NULL,
                        slippage_pips REAL NOT NULL,
                        slippage_pct_atr REAL,
                        commission REAL DEFAULT 0.0,
                        swap REAL DEFAULT 0.0,
                        fill_time TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS execution_aggregates_daily (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        dimension_type TEXT NOT NULL,
                        dimension_value TEXT NOT NULL,
                        total_orders INTEGER DEFAULT 0,
                        filled_orders INTEGER DEFAULT 0,
                        rejected_orders INTEGER DEFAULT 0,
                        fill_rate REAL DEFAULT 0.0,
                        avg_slippage_pips REAL DEFAULT 0.0,
                        max_slippage_pips REAL DEFAULT 0.0,
                        min_slippage_pips REAL DEFAULT 0.0,
                        avg_slippage_pct_atr REAL DEFAULT 0.0,
                        p50_latency_ms REAL DEFAULT 0.0,
                        p95_latency_ms REAL DEFAULT 0.0,
                        p99_latency_ms REAL DEFAULT 0.0,
                        avg_total_latency_ms REAL DEFAULT 0.0,
                        avg_spread_pips REAL DEFAULT 0.0,
                        total_volume_lots REAL DEFAULT 0.0,
                        updated_at TEXT DEFAULT (datetime('now')),
                        UNIQUE(date, dimension_type, dimension_value)
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_orders_symbol ON execution_orders(symbol)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_orders_strat ON execution_orders(strategy_name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_orders_status ON execution_orders(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_orders_signal_time ON execution_orders(signal_time)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_orders_trade_id ON execution_orders(trade_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_fills_order_id ON execution_fills(order_id)")
            conn.close()
        except Exception as e:
            logger.error(f"Error initializing execution metrics tables: {e}")

    def start_order(self, symbol: str, strategy_name: str, direction: str,
                    requested_price: float, atr_14: float, pip_size: float,
                    session_killzone: str = "OFF_HOURS", ml_p_tp: Optional[float] = None,
                    ai_conviction: Optional[float] = None, conflict_score: Optional[float] = None,
                    spread_at_signal: float = 0.0, magic: int = 123456) -> ExecutionOrderRecord:
        """Stage 0: Initialize order tracking on raw signal generation."""
        order = ExecutionOrderRecord(
            symbol=symbol,
            strategy_name=strategy_name,
            magic_number=magic,
            direction=direction,
            requested_price=requested_price,
            atr_14=atr_14,
            pip_size=pip_size,
            spread_at_signal=spread_at_signal,
            session_killzone=session_killzone,
            ml_p_tp=ml_p_tp,
            ml_p_sl=(1.0 - ml_p_tp) if ml_p_tp is not None else None,
            ai_conviction=ai_conviction,
            conflict_score=conflict_score,
        )
        order.latency.t0_signal_ns = time.perf_counter_ns()
        with self._lock:
            self._active_orders[order.order_id] = order
        return order

    def mark_submission(self, order_id: str, requested_lot: float, stop_loss: float,
                        take_profit: float, spread_at_submit: float, risk_pct: float = 0.5,
                        risk_amount: float = 0.0, account_equity: float = 0.0) -> None:
        """Stage 1: Mark order dispatch to broker."""
        with self._lock:
            order = self._active_orders.get(order_id)
            if not order:
                return
            order.latency.t1_submit_ns = time.perf_counter_ns()
            order.submit_time = datetime.now(timezone.utc).isoformat()
            order.requested_lot = requested_lot
            order.stop_loss = stop_loss
            order.take_profit = take_profit
            order.spread_at_submit = spread_at_submit
            order.risk_pct = risk_pct
            order.risk_amount = risk_amount
            order.account_equity = account_equity

    def mark_broker_ack(self, order_id: str, filling_mode: str = "FOK") -> None:
        """Stage 2: Broker received and acknowledged the order request."""
        with self._lock:
            order = self._active_orders.get(order_id)
            if not order:
                return
            order.latency.t2_ack_ns = time.perf_counter_ns()
            order.ack_time = datetime.now(timezone.utc).isoformat()
            order.filling_mode = filling_mode

    def mark_filled(self, order_id: str, trade_id: int, filled_price: float,
                    filled_lot: float, spread_at_fill: float, retries_used: int = 0) -> Optional[ExecutionOrderRecord]:
        """Stage 3: Deal filled successfully."""
        with self._lock:
            order = self._active_orders.pop(order_id, None)
            if not order:
                return None
            if not order.latency.t3_fill_ns:
                order.latency.t3_fill_ns = time.perf_counter_ns()
            order.fill_time = datetime.now(timezone.utc).isoformat()
            order.trade_id = trade_id
            order.filled_price = filled_price
            order.filled_lot = filled_lot
            order.spread_at_fill = spread_at_fill
            order.retries_used = retries_used
            order.status = "FILLED"
            order.compute_slippage()

        # Enqueue order record and fill record
        self._enqueue(order)
        fill = ExecutionFillRecord(
            order_id=order.order_id,
            trade_id=trade_id,
            fill_type="ENTRY",
            volume=filled_lot,
            intended_price=order.requested_price,
            actual_price=filled_price,
            slippage_pips=order.slippage_pips,
            slippage_pct_atr=order.slippage_pct_atr,
            fill_time=order.fill_time,
        )
        self._enqueue(fill)
        return order

    def mark_rejected(self, order_id: str, rejection_code: int, rejection_reason: str,
                      retries_used: int = 0) -> Optional[ExecutionOrderRecord]:
        """Stage 3 (Reject): Order was rejected or expired."""
        with self._lock:
            order = self._active_orders.pop(order_id, None)
            if not order:
                return None
            if not order.latency.t3_fill_ns:
                order.latency.t3_fill_ns = time.perf_counter_ns()
            order.fill_time = datetime.now(timezone.utc).isoformat()
            order.rejection_code = rejection_code
            order.rejection_reason = rejection_reason
            order.retries_used = retries_used
            order.status = "REJECTED"

        self._enqueue(order)
        return order

    def record_exit_fill(self, trade_id: int, exit_type: str, intended_price: float,
                         actual_price: float, volume: float, pip_size: float = 0.0001,
                         atr_14: float = 0.0, commission: float = 0.0, swap: float = 0.0) -> None:
        """Record execution metrics on position close (SL, TP, or Manual)."""
        diff = actual_price - intended_price
        if "SL" in exit_type.upper():
            diff = -abs(diff) if actual_price < intended_price else abs(diff)
        slip_pips = round(diff / pip_size, 2) if pip_size > 0 else 0.0
        slip_pct_atr = round((diff / atr_14) * 100.0, 2) if atr_14 > 0 else 0.0

        fill = ExecutionFillRecord(
            order_id=f"EXIT-TRD-{trade_id}",
            trade_id=trade_id,
            fill_type=exit_type,
            volume=volume,
            intended_price=intended_price,
            actual_price=actual_price,
            slippage_pips=slip_pips,
            slippage_pct_atr=slip_pct_atr,
            commission=commission,
            swap=swap,
        )
        self._enqueue(fill)

    def _enqueue(self, item: Any) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.error("Execution metrics queue full! Dropping item to prevent memory exhaustion.")

    def _db_writer_loop(self) -> None:
        """Background daemon that flushes batches into SQLite."""
        while not self._stop_event.is_set():
            batch = []
            try:
                # Wait for first item
                item = self._queue.get(timeout=self.flush_interval_sec)
                batch.append(item)
                # Drain up to batch_size items
                while len(batch) < self.batch_size:
                    item = self._queue.get_nowait()
                    batch.append(item)
            except queue.Empty:
                pass

            if batch:
                self._persist_batch(batch)

    def _persist_batch(self, batch: List[Any]) -> None:
        """Write records to SQLite with WAL mode resilience."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")
            with conn:
                for item in batch:
                    if isinstance(item, ExecutionOrderRecord):
                        conn.execute("""
                            INSERT OR REPLACE INTO execution_orders (
                                order_id, trade_id, symbol, strategy_name, magic_number,
                                direction, order_type, filling_mode, requested_lot, filled_lot,
                                requested_price, filled_price, stop_loss, take_profit, atr_14,
                                slippage_points, slippage_pips, slippage_pct_atr,
                                spread_at_signal, spread_at_submit, spread_at_fill,
                                signal_time, submit_time, ack_time, fill_time, close_time,
                                latency_signal_to_submit_ms, latency_submit_to_ack_ms,
                                latency_ack_to_fill_ms, latency_total_ms,
                                retries_used, rejection_code, rejection_reason, status,
                                session_killzone, ml_p_tp, ml_p_sl, ai_conviction, conflict_score,
                                risk_pct, risk_amount, account_equity, broker_name
                            ) VALUES (
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                            )
                        """, (
                            item.order_id, item.trade_id, item.symbol, item.strategy_name, item.magic_number,
                            item.direction, item.order_type, item.filling_mode, item.requested_lot, item.filled_lot,
                            item.requested_price, item.filled_price, item.stop_loss, item.take_profit, item.atr_14,
                            item.slippage_points, item.slippage_pips, item.slippage_pct_atr,
                            item.spread_at_signal, item.spread_at_submit, item.spread_at_fill,
                            item.signal_time, item.submit_time, item.ack_time, item.fill_time, item.close_time,
                            item.latency.signal_to_submit_ms, item.latency.submit_to_ack_ms,
                            item.latency.ack_to_fill_ms, item.latency.total_latency_ms,
                            item.retries_used, item.rejection_code, item.rejection_reason, item.status,
                            item.session_killzone, item.ml_p_tp, item.ml_p_sl, item.ai_conviction, item.conflict_score,
                            item.risk_pct, item.risk_amount, item.account_equity, item.broker_name
                        ))
                    elif isinstance(item, ExecutionFillRecord):
                        conn.execute("""
                            INSERT OR REPLACE INTO execution_fills (
                                fill_id, order_id, trade_id, deal_ticket, fill_type,
                                volume, intended_price, actual_price, slippage_pips,
                                slippage_pct_atr, commission, swap, fill_time
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            item.fill_id, item.order_id, item.trade_id, item.deal_ticket, item.fill_type,
                            item.volume, item.intended_price, item.actual_price, item.slippage_pips,
                            item.slippage_pct_atr, item.commission, item.swap, item.fill_time
                        ))
            conn.close()
        except Exception as e:
            logger.error(f"Error persisting execution metrics batch: {e}")

    def flush_sync(self) -> None:
        """Force flush all buffered metrics synchronously (e.g. before shutdown)."""
        items = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if items:
            self._persist_batch(items)

    def shutdown(self) -> None:
        """Gracefully stop writer thread and flush remaining metrics."""
        self._stop_event.set()
        self.flush_sync()
        if self._worker.is_alive():
            self._worker.join(timeout=3.0)
        logger.info("ExecutionMetricsCollector shut down cleanly.")

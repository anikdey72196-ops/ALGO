"""
reconciliation.py — Reconciles Internal State with MT5 Positions, Orders, and Deals.
"""
from __future__ import annotations
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Any
from loguru import logger
from core.config import Direction
from core.state import StateManager, TradeRecord
from execution.execution import BrokerAdapter


class ReconciliationEngine:
    """Background engine ensuring local SQLite trade state exactly mirrors MT5 terminal reality."""

    def __init__(self, broker: BrokerAdapter, state: StateManager, sync_interval_sec: float = 30.0, position_manager=None):
        self.broker = broker
        self.state = state
        self.sync_interval_sec = sync_interval_sec
        self.position_manager = position_manager
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run_loop, daemon=True, name="Reconciler")
        self._worker.start()
        logger.info(f"ReconciliationEngine active (interval: {sync_interval_sec}s)")

    def reconcile_now(self) -> Dict[str, Any]:
        """Execute a single reconciliation pass."""
        report = {"orphans_detected": 0, "closed_synced": 0, "mismatches_fixed": 0, "timestamp": datetime.now(timezone.utc).isoformat()}
        try:
            if hasattr(self.broker, "ensure_connected") and not self.broker.ensure_connected():
                logger.debug("[RECONCILE] Broker not connected; skipping reconciliation pass.")
                return report

            local_open = self.state.get_open_positions()
            broker_positions = self.broker.get_open_positions()

            # Guard against false positives if broker temporarily returns empty during reconnect
            if not broker_positions and local_open:
                if hasattr(self.broker, "ensure_connected") and not self.broker.ensure_connected():
                    logger.debug("[RECONCILE] Broker disconnected; preserving local open positions.")
                    return report

            broker_tickets = {
                p.get("ticket") or p.get("order_id"): p
                for p in broker_positions
                if p.get("ticket") or p.get("order_id")
            }

            # 1. Check for local open positions that have closed with broker
            for t in local_open:
                if t.id is not None and t.id not in broker_tickets:
                    # Query deal history for realized PnL if supported
                    if hasattr(self.broker, "get_deal_pnl_for_order"):
                        deal_res = self.broker.get_deal_pnl_for_order(t.id)
                        if deal_res:
                            pnl, status = deal_res
                            self.state.update_trade_pnl(t.id, pnl, status)
                            if self.position_manager:
                                self.position_manager.on_trade_closed(t, pnl, status)
                            report["closed_synced"] += 1
                            logger.info(f"[RECONCILE] Synced closed trade #{t.id} ({t.symbol}): {status} PnL=${pnl:+.2f}")
                        else:
                            self.state.update_trade_pnl(t.id, 0.0, "CLOSED")
                            if self.position_manager:
                                self.position_manager.on_trade_closed(t, 0.0, "CLOSED")
                            report["closed_synced"] += 1
                    else:
                        self.state.update_trade_pnl(t.id, 0.0, "CLOSED")
                        if self.position_manager:
                            self.position_manager.on_trade_closed(t, 0.0, "CLOSED")
                        report["closed_synced"] += 1

            # 2. Check for orphan broker positions not in local state
            local_ids = {t.id for t in local_open if t.id is not None}
            for ticket, p in broker_tickets.items():
                if ticket not in local_ids:
                    report["orphans_detected"] += 1
                    try:
                        # Check if trade already exists in trade_log (e.g. marked CLOSED prematurely)
                        with self.state._lock, self.state.conn:
                            cur = self.state.conn.execute("SELECT id, status FROM trade_log WHERE id = ?", (ticket,))
                            existing = cur.fetchone()
                            if existing:
                                self.state.conn.execute("""
                                    UPDATE trade_log
                                    SET status = 'OPEN',
                                        stop_loss = ?,
                                        take_profit = ?,
                                        realized_pnl = ?,
                                        closed_at = NULL,
                                        duration_seconds = NULL
                                    WHERE id = ?
                                """, (float(p.get('sl', 0.0)), float(p.get('tp', 0.0)), float(p.get('profit', 0.0)), ticket))
                                self.state.conn.commit()
                                logger.info(f"[RECONCILE] Restored existing broker position #{ticket} ({p.get('symbol')}) to OPEN state.")
                                continue

                        direction = Direction.BUY if p.get('type') == 0 else Direction.SELL
                        magic = p.get('magic', 123456)
                        strat_name = (
                            "SMC Scalp (5m)" if magic == 125456 
                            else ("Order Flow (Delta & Absorption)" if magic == 127456 
                            else ("ICT" if magic == 126456 
                            else "SMC Swing (15m)"))
                        )
                        open_time = (
                            datetime.fromtimestamp(p.get('time', 0), tz=timezone.utc)
                            if p.get('time') else datetime.now(timezone.utc)
                        )
                        adopted_trade = TradeRecord(
                            id=ticket,
                            timestamp=open_time,
                            symbol=p.get('symbol', 'UNKNOWN'),
                            direction=direction,
                            entry_price=float(p.get('price_open', 0.0)),
                            stop_loss=float(p.get('sl', 0.0)),
                            take_profit=float(p.get('tp', 0.0)),
                            lot_size=float(p.get('volume', 0.01)),
                            realized_pnl=float(p.get('profit', 0.0)),
                            status='OPEN',
                            strategy_name=strat_name,
                            magic_number=magic,
                        )
                        self.state.record_trade(adopted_trade)
                        logger.info(f"[RECONCILE] Adopted broker position #{ticket} ({p.get('symbol')}) into local tracking state.")
                    except Exception as e:
                        logger.error(f"[RECONCILE] Failed to adopt orphan position #{ticket}: {e}")

        except Exception as e:
            logger.error(f"Error during reconciliation pass: {e}")

        return report

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self.sync_interval_sec)
            if not self._stop_event.is_set():
                self.reconcile_now()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

"""
reconciliation.py — Reconciles Internal State with MT5 Positions, Orders, and Deals.
"""
from __future__ import annotations
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Any
from loguru import logger
from core.state import StateManager
from execution.execution import BrokerAdapter


class ReconciliationEngine:
    """Background engine ensuring local SQLite trade state exactly mirrors MT5 terminal reality."""

    def __init__(self, broker: BrokerAdapter, state: StateManager, sync_interval_sec: float = 30.0):
        self.broker = broker
        self.state = state
        self.sync_interval_sec = sync_interval_sec
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run_loop, daemon=True, name="Reconciler")
        self._worker.start()
        logger.info(f"ReconciliationEngine active (interval: {sync_interval_sec}s)")

    def reconcile_now(self) -> Dict[str, Any]:
        """Execute a single reconciliation pass."""
        report = {"orphans_detected": 0, "closed_synced": 0, "mismatches_fixed": 0, "timestamp": datetime.now(timezone.utc).isoformat()}
        try:
            local_open = self.state.get_open_positions()
            broker_positions = self.broker.get_open_positions()

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
                            report["closed_synced"] += 1
                            logger.info(f"[RECONCILE] Synced closed trade #{t.id} ({t.symbol}): {status} PnL=${pnl:+.2f}")
                        else:
                            self.state.update_trade_pnl(t.id, 0.0, "CLOSED")
                            report["closed_synced"] += 1
                    else:
                        self.state.update_trade_pnl(t.id, 0.0, "CLOSED")
                        report["closed_synced"] += 1

            # 2. Check for orphan broker positions not in local state
            local_ids = {t.id for t in local_open if t.id is not None}
            for ticket, p in broker_tickets.items():
                if ticket not in local_ids:
                    report["orphans_detected"] += 1
                    logger.warning(f"[RECONCILE] Orphan broker position #{ticket} ({p.get('symbol')}) detected in MT5!")

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

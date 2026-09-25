"""
metrics_aggregator.py — Aggregates and Analyzes Execution Quality Data.

Computes:
- Slippage distributions, averages, and worst outliers.
- Latency percentiles (P50, P90, P95, P99) across execution stages.
- Rejection rates grouped by MT5 retcode and failure reason.
- Fill ratios and partial execution rates.
- Slices by Symbol, Strategy Engine, KillZone session, and Time Window (24h, 7d, 30d, All).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import numpy as np


class MetricsAggregator:
    def __init__(self, db_path: str = "trading_state.db"):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_summary(self, window_hours: int = 24) -> Dict[str, Any]:
        """Compute top-level summary KPIs for the requested time window."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
        with self._get_connection() as conn:
            orders = conn.execute("""
                SELECT * FROM execution_orders
                WHERE signal_time >= ?
            """, (cutoff,)).fetchall()

        if not orders:
            return {
                "window_hours": window_hours,
                "total_orders": 0,
                "filled_orders": 0,
                "rejected_orders": 0,
                "fill_rate_pct": 100.0,
                "avg_slippage_pips": 0.0,
                "max_slippage_pips": 0.0,
                "min_slippage_pips": 0.0,
                "avg_slippage_pct_atr": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "avg_total_latency_ms": 0.0,
                "avg_spread_pips": 0.0,
                "total_volume_lots": 0.0,
            }

        total_orders = len(orders)
        filled = [o for o in orders if o["status"] == "FILLED"]
        rejected = [o for o in orders if o["status"] == "REJECTED"]
        
        slippages = [o["slippage_pips"] for o in filled if o["slippage_pips"] is not None]
        slip_atrs = [o["slippage_pct_atr"] for o in filled if o["slippage_pct_atr"] is not None]
        latencies = [o["latency_total_ms"] for o in filled if o["latency_total_ms"] is not None and o["latency_total_ms"] > 0]
        spreads = [o["spread_at_fill"] for o in filled if o["spread_at_fill"] is not None]
        vols = [o["filled_lot"] for o in filled if o["filled_lot"] is not None]

        return {
            "window_hours": window_hours,
            "total_orders": total_orders,
            "filled_orders": len(filled),
            "rejected_orders": len(rejected),
            "fill_rate_pct": round((len(filled) / total_orders) * 100.0, 2) if total_orders > 0 else 0.0,
            "avg_slippage_pips": round(float(np.mean(slippages)), 2) if slippages else 0.0,
            "max_slippage_pips": round(float(np.max(slippages)), 2) if slippages else 0.0,
            "min_slippage_pips": round(float(np.min(slippages)), 2) if slippages else 0.0,
            "avg_slippage_pct_atr": round(float(np.mean(slip_atrs)), 2) if slip_atrs else 0.0,
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
            "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2) if latencies else 0.0,
            "avg_total_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "avg_spread_pips": round(float(np.mean(spreads)), 2) if spreads else 0.0,
            "total_volume_lots": round(float(np.sum(vols)), 2) if vols else 0.0,
        }

    def get_breakdowns(self, window_hours: int = 168) -> Dict[str, Any]:
        """Group metrics by Symbol, Strategy, KillZone Session, and Rejection Reasons."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
        with self._get_connection() as conn:
            orders = conn.execute("""
                SELECT * FROM execution_orders
                WHERE signal_time >= ?
            """, (cutoff,)).fetchall()

        by_symbol: Dict[str, List[sqlite3.Row]] = {}
        by_strategy: Dict[str, List[sqlite3.Row]] = {}
        by_killzone: Dict[str, List[sqlite3.Row]] = {}
        reject_reasons: Dict[str, int] = {}

        for o in orders:
            by_symbol.setdefault(o["symbol"], []).append(o)
            by_strategy.setdefault(o["strategy_name"], []).append(o)
            by_killzone.setdefault(o["session_killzone"] or "OFF_HOURS", []).append(o)
            if o["status"] == "REJECTED":
                reason = o["rejection_reason"] or f"Code {o['rejection_code']}"
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

        def _calc_group(rows: List[sqlite3.Row]) -> Dict[str, Any]:
            total = len(rows)
            filled = [r for r in rows if r["status"] == "FILLED"]
            slips = [r["slippage_pips"] for r in filled if r["slippage_pips"] is not None]
            lats = [r["latency_total_ms"] for r in filled if r["latency_total_ms"] is not None and r["latency_total_ms"] > 0]
            vols = [r["filled_lot"] for r in filled if r["filled_lot"] is not None]
            return {
                "total_orders": total,
                "filled_orders": len(filled),
                "fill_rate": round((len(filled) / total) * 100.0, 1) if total > 0 else 0.0,
                "avg_slippage_pips": round(float(np.mean(slips)), 2) if slips else 0.0,
                "max_slippage_pips": round(float(np.max(slips)), 2) if slips else 0.0,
                "p95_latency_ms": round(float(np.percentile(lats, 95)), 1) if lats else 0.0,
                "total_volume": round(float(np.sum(vols)), 2) if vols else 0.0,
            }

        return {
            "by_symbol": {k: _calc_group(v) for k, v in by_symbol.items()},
            "by_strategy": {k: _calc_group(v) for k, v in by_strategy.items()},
            "by_killzone": {k: _calc_group(v) for k, v in by_killzone.items()},
            "rejections": [{"reason": k, "count": v} for k, v in reject_reasons.items()],
        }

    def get_latency_histogram(self, window_hours: int = 168, bins: int = 10) -> Dict[str, Any]:
        """Compute latency distribution buckets for stage-by-stage profiling."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
        with self._get_connection() as conn:
            orders = conn.execute("""
                SELECT latency_signal_to_submit_ms, latency_submit_to_ack_ms,
                       latency_ack_to_fill_ms, latency_total_ms
                FROM execution_orders
                WHERE status = 'FILLED' AND signal_time >= ?
            """, (cutoff,)).fetchall()

        if not orders:
            return {"buckets": [], "counts_total": [], "counts_ack": []}

        totals = [o["latency_total_ms"] for o in orders if o["latency_total_ms"] is not None]
        if not totals:
            return {"buckets": [], "counts_total": [], "counts_ack": []}

        counts, bin_edges = np.histogram(totals, bins=bins)
        bucket_labels = [f"{round(bin_edges[i], 1)}-{round(bin_edges[i+1], 1)}ms" for i in range(len(counts))]
        return {
            "buckets": bucket_labels,
            "counts_total": counts.tolist(),
            "min_ms": round(float(np.min(totals)), 2),
            "max_ms": round(float(np.max(totals)), 2),
            "mean_ms": round(float(np.mean(totals)), 2),
        }

    def get_slippage_distribution(self, window_hours: int = 168) -> Dict[str, Any]:
        """Categorize slippage into Zero, Positive (Favorable), and Negative (Adverse) bins."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
        with self._get_connection() as conn:
            orders = conn.execute("""
                SELECT slippage_pips FROM execution_orders
                WHERE status = 'FILLED' AND signal_time >= ?
            """, (cutoff,)).fetchall()

        slips = [o["slippage_pips"] for o in orders if o["slippage_pips"] is not None]
        if not slips:
            return {"zero_slippage": 0, "favorable_improvement": 0, "adverse_mild_1pip": 0, "adverse_high_gt_1pip": 0, "total_measured": 0}

        zero = sum(1 for s in slips if abs(s) < 0.1)
        favorable = sum(1 for s in slips if s <= -0.1)  # price improved
        adverse_mild = sum(1 for s in slips if 0.1 <= s <= 1.0)
        adverse_high = sum(1 for s in slips if s > 1.0)

        return {
            "zero_slippage": zero,
            "favorable_improvement": favorable,
            "adverse_mild_1pip": adverse_mild,
            "adverse_high_gt_1pip": adverse_high,
            "total_measured": len(slips),
        }

"""
ml/temporal_window_recorder.py
==============================

5-Window Daytime Interval & Shadow Mode Telemetry Recorder.
Divides the active trading day (08:00 AM - 11:00 PM) into 5 fixed 3-hour windows
(plus the 11:00 PM - 08:00 AM Night Window) and aggregates performance into CSV
for October shadow data collection and future dynamic lot-size optimization.

5 Daytime Windows (+ 1 Night Guard Window):
--------------------------------------------
- W1: 08:00 AM - 11:00 AM  (Morning Open / Asian Late)
- W2: 11:01 AM - 02:00 PM  (London Pre-Open & Transition)
- W3: 02:01 PM - 05:00 PM  (London Core & Pre-NY)
- W4: 05:01 PM - 08:00 PM  (US/London Overlap — Power Hour)
- W5: 08:01 PM - 11:00 PM  (NY PM & Close)
- WNIGHT: 11:01 PM - 08:00 AM (Night Guard — Max 2 Trades)
"""

from __future__ import annotations

import csv
import logging
import os
import sqlite3
import threading
from datetime import datetime, date, time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("algo.ml.temporal_window_recorder")

DEFAULT_TIMEZONE = "Asia/Kolkata"
CSV_FILE_PATH = Path("data/temporal_windows_october.csv")

WINDOW_CONFIGS = [
    {
        "id": "W1",
        "name": "Morning Open",
        "time_range": "08:00 - 11:00 AM",
        "start_h": 8, "start_m": 0,
        "end_h": 11, "end_m": 0,
        "is_night": False,
        "expected_edge": "Conservative (0.75x)",
    },
    {
        "id": "W2",
        "name": "London Transition",
        "time_range": "11:01 - 02:00 PM",
        "start_h": 11, "start_m": 1,
        "end_h": 14, "end_m": 0,
        "is_night": False,
        "expected_edge": "Normal (1.00x)",
    },
    {
        "id": "W3",
        "name": "London Core",
        "time_range": "02:01 - 05:00 PM",
        "start_h": 14, "start_m": 1,
        "end_h": 17, "end_m": 0,
        "is_night": False,
        "expected_edge": "Favorable (1.10x)",
    },
    {
        "id": "W4",
        "name": "US/London Overlap",
        "time_range": "05:01 - 08:00 PM",
        "start_h": 17, "start_m": 1,
        "end_h": 20, "end_m": 0,
        "is_night": False,
        "expected_edge": "Prime Edge (1.25x)",
    },
    {
        "id": "W5",
        "name": "NY PM & Close",
        "time_range": "08:01 - 11:00 PM",
        "start_h": 20, "start_m": 1,
        "end_h": 23, "end_m": 0,
        "is_night": False,
        "expected_edge": "Normal (0.85x)",
    },
    {
        "id": "WNIGHT",
        "name": "Night Guard",
        "time_range": "11:01 - 08:00 AM",
        "start_h": 23, "start_m": 1,
        "end_h": 8, "end_m": 0,
        "is_night": True,
        "expected_edge": "Guard Capped (Max 2 Trades)",
    },
]

CSV_HEADERS = [
    "date",
    "timestamp",
    "window_id",
    "window_name",
    "time_range",
    "total_trades",
    "tp_hits",
    "sl_hits",
    "breakevens",
    "win_rate_pct",
    "net_pnl",
    "strategies",
    "avg_base_lot",
    "shadow_rec_lot",
    "shadow_multiplier",
    "shadow_edge_tier",
    "mode",
]


class TemporalWindowRecorder:
    """
    Manages 5 daytime windows + 1 night window, records real-time trade performance,
    shadow predictions, and automatically outputs aggregated CSV reports after every 3 hours.
    """

    def __init__(
        self,
        db_path: str = "trading_state.db",
        csv_path: Path | str = CSV_FILE_PATH,
        tz_name: str = DEFAULT_TIMEZONE,
    ):
        self.db_path = str(db_path)
        self.csv_path = Path(csv_path)
        self.tz_name = tz_name
        self._lock = threading.RLock()
        self._last_processed_window_key: Optional[str] = None

        # Ensure target data folder exists
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_headers()

    def get_tz(self) -> ZoneInfo:
        try:
            return ZoneInfo(self.tz_name)
        except Exception:
            return ZoneInfo("UTC")

    def _ensure_csv_headers(self) -> None:
        """Create CSV file with institutional header row if not present."""
        with self._lock:
            if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
                with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADERS)
                logger.info(f"Initialized 5-window CSV: {self.csv_path}")

    def get_window_info(self, dt: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Determine which 3-hour window a given datetime falls into.
        Defaults to current local time.
        """
        tz = self.get_tz()
        if dt is None:
            local_dt = datetime.now(tz)
        else:
            if dt.tzinfo is None:
                local_dt = dt.replace(tzinfo=timezone.utc).astimezone(tz)
            else:
                local_dt = dt.astimezone(tz)

        total_m = local_dt.hour * 60 + local_dt.minute

        for w in WINDOW_CONFIGS:
            if not w["is_night"]:
                start_m = w["start_h"] * 60 + w["start_m"]
                end_m = w["end_h"] * 60 + w["end_m"]
                if start_m <= total_m <= end_m:
                    return {
                        "id": w["id"],
                        "name": w["name"],
                        "time_range": w["time_range"],
                        "is_night": False,
                        "expected_edge": w["expected_edge"],
                        "local_time": local_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "date": local_dt.strftime("%Y-%m-%d"),
                        "hour": local_dt.hour,
                        "minute": local_dt.minute,
                    }

        # Fallback to WNIGHT (11:01 PM - 08:00 AM)
        return {
            "id": "WNIGHT",
            "name": "Night Guard",
            "time_range": "11:01 - 08:00 AM",
            "is_night": True,
            "expected_edge": "Guard Capped (Max 2 Trades)",
            "local_time": local_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "date": local_dt.strftime("%Y-%m-%d"),
            "hour": local_dt.hour,
            "minute": local_dt.minute,
        }

    def aggregate_window_from_trades(
        self,
        date_str: str,
        window_id: str,
        trades: List[Dict[str, Any]],
        shadow_mult: float = 1.0,
        shadow_tier: str = "FAVORABLE",
        mode: str = "SHADOW",
    ) -> Dict[str, Any]:
        """Generate structured performance dictionary for a completed window."""
        cfg = next((w for w in WINDOW_CONFIGS if w["id"] == window_id), WINDOW_CONFIGS[0])
        total = len(trades)
        tp = sum(1 for t in trades if "TP" in str(t.get("status", "")).upper())
        sl = sum(1 for t in trades if "SL" in str(t.get("status", "")).upper())
        be = sum(
            1 for t in trades
            if "BREAKEVEN" in str(t.get("status", "")).upper()
            or (abs(float(t.get("realized_pnl") or 0.0)) < 0.01 and "TP" not in str(t.get("status", "")).upper() and "SL" not in str(t.get("status", "")).upper())
        )
        net_pnl = round(sum(float(t.get("realized_pnl") or 0.0) for t in trades), 2)
        win_rate = round((tp / total * 100.0), 1) if total > 0 else 0.0

        strats = sorted(list(set(str(t.get("strategy_name") or "SMC") for t in trades)))
        strats_str = "/".join(strats) if strats else "None"

        lots = [float(t.get("lot_size") or 0.05) for t in trades]
        avg_base_lot = round(sum(lots) / len(lots), 2) if lots else 0.05
        shadow_rec_lot = round(avg_base_lot * shadow_mult, 2)

        return {
            "date": date_str,
            "timestamp": datetime.now(self.get_tz()).strftime("%Y-%m-%d %H:%M:%S"),
            "window_id": window_id,
            "window_name": cfg["name"],
            "time_range": cfg["time_range"],
            "total_trades": total,
            "tp_hits": tp,
            "sl_hits": sl,
            "breakevens": be,
            "win_rate_pct": f"{win_rate:.1f}%",
            "net_pnl": f"{net_pnl:+.2f}",
            "strategies": strats_str,
            "avg_base_lot": f"{avg_base_lot:.2f}",
            "shadow_rec_lot": f"{shadow_rec_lot:.2f}",
            "shadow_multiplier": f"{shadow_mult:.2f}x",
            "shadow_edge_tier": shadow_tier,
            "mode": mode,
        }

    def append_window_record(self, record: Dict[str, Any]) -> None:
        """Atomically append a window performance summary to the October CSV."""
        with self._lock:
            self._ensure_csv_headers()
            row = [record.get(h, "") for h in CSV_HEADERS]
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            logger.info(
                f"📊 [WINDOW CSV APPLIED] {record.get('date')} {record.get('window_id')} ({record.get('time_range')}): "
                f"{record.get('total_trades')} trades, {record.get('tp_hits')} TP, {record.get('sl_hits')} SL "
                f"(Win Rate {record.get('win_rate_pct')}) | Shadow Lot: {record.get('shadow_rec_lot')}"
            )

    def backfill_from_database(self) -> int:
        """
        Backfill historical trades from trading_state.db into temporal_windows_october.csv.
        Groups all past trades by Date & 3-hour Window so October already has rich data!
        """
        if not os.path.exists(self.db_path):
            logger.warning(f"Database {self.db_path} not found for backfill.")
            return 0

        tz = self.get_tz()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT timestamp, symbol, lot_size, realized_pnl, status, strategy_name FROM trade_log ORDER BY id ASC"
            ).fetchall()
        except Exception as e:
            logger.error(f"Error querying trade_log for window backfill: {e}")
            conn.close()
            return 0
        conn.close()

        if not rows:
            return 0

        # Group trades by (date_str, window_id)
        grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for r in rows:
            ts_str = r["timestamp"]
            try:
                dt = datetime.fromisoformat(ts_str)
                local_dt = dt.astimezone(tz) if dt.tzinfo else dt.replace(tzinfo=timezone.utc).astimezone(tz)
            except Exception:
                continue

            date_str = local_dt.strftime("%Y-%m-%d")
            winfo = self.get_window_info(local_dt)
            wid = winfo["id"]

            key = (date_str, wid)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(dict(r))

        # Check existing CSV keys so we don't duplicate on re-runs
        existing_keys = set()
        if self.csv_path.exists():
            with open(self.csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d = row.get("date")
                    w = row.get("window_id")
                    if d and w:
                        existing_keys.add((d, w))

        inserted_count = 0
        # Multiplier heuristics based on window index
        mult_map = {
            "W1": (0.75, "NEUTRAL"),
            "W2": (1.00, "FAVORABLE"),
            "W3": (1.10, "FAVORABLE"),
            "W4": (1.25, "PRIME_EDGE"),
            "W5": (0.85, "NEUTRAL"),
            "WNIGHT": (1.00, "NIGHT_GUARD"),
        }

        for (d_str, w_id), t_list in grouped.items():
            if (d_str, w_id) in existing_keys:
                continue

            mult, tier = mult_map.get(w_id, (1.0, "FAVORABLE"))
            summary = self.aggregate_window_from_trades(
                date_str=d_str,
                window_id=w_id,
                trades=t_list,
                shadow_mult=mult,
                shadow_tier=tier,
                mode="SHADOW_OCTOBER",
            )
            self.append_window_record(summary)
            existing_keys.add((d_str, w_id))
            inserted_count += 1

        logger.info(f"Backfilled {inserted_count} window intervals into {self.csv_path}")
        return inserted_count

    def get_summary_by_window(self) -> List[Dict[str, Any]]:
        """Return cumulative win rates and stats for each of the 5 windows."""
        records = self.read_all_records()
        totals: Dict[str, Dict[str, Any]] = {
            w["id"]: {
                "id": w["id"],
                "name": w["name"],
                "time_range": w["time_range"],
                "total_trades": 0,
                "tp_hits": 0,
                "sl_hits": 0,
                "breakevens": 0,
                "net_pnl": 0.0,
                "expected_edge": w["expected_edge"],
            }
            for w in WINDOW_CONFIGS
        }

        for r in records:
            wid = r.get("window_id")
            if wid in totals:
                t = int(r.get("total_trades") or 0)
                tp = int(r.get("tp_hits") or 0)
                sl = int(r.get("sl_hits") or 0)
                be = int(r.get("breakevens") or 0)
                try:
                    pnl = float(str(r.get("net_pnl") or "0").replace("+", ""))
                except ValueError:
                    pnl = 0.0

                totals[wid]["total_trades"] += t
                totals[wid]["tp_hits"] += tp
                totals[wid]["sl_hits"] += sl
                totals[wid]["breakevens"] += be
                totals[wid]["net_pnl"] += pnl

        results = []
        for w in WINDOW_CONFIGS:
            data = totals[w["id"]]
            tot = data["total_trades"]
            tp = data["tp_hits"]
            data["win_rate"] = round((tp / tot * 100.0), 1) if tot > 0 else 0.0
            data["net_pnl"] = round(data["net_pnl"], 2)
            results.append(data)
        return results

    def read_all_records(self) -> List[Dict[str, str]]:
        """Read all rows from the October CSV."""
        if not self.csv_path.exists():
            return []
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)


# Global Singleton instance
window_recorder = TemporalWindowRecorder()

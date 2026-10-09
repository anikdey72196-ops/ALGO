"""
core/cloud_sync.py — Enterprise-grade bi-directional ML and Trade Data Synchronization Engine.

Enables distributed training and synchronization between cloud (e.g. AWS 24/7 Demo)
and local nodes (e.g. XM Real execution).
"""

from __future__ import annotations

import os
import io
import json
import zlib
import gzip
import sqlite3
import hashlib
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import joblib
from loguru import logger


# Default artifacts directory
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "ml" / "artifacts"
DB_PATH = Path(__file__).resolve().parent.parent / "trading_state.db"


class CloudSyncEngine:
    """
    Local database and model artifacts sync manager.
    Handles export/import of ML events, trade logs, and atomic model synchronization.
    """

    def __init__(self, db_path: Path | str = DB_PATH, artifacts_dir: Path | str = ARTIFACTS_DIR):
        self.db_path = Path(db_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def get_db_connection(self) -> sqlite3.Connection:
        """Create a thread-safe connection to trading_state.db with WAL mode."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    # ─────────────────────────────────────────────────────────────
    # Model Artifacts Manifest & Verification
    # ─────────────────────────────────────────────────────────────

    def get_models_manifest(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute SHA256 hash, byte size, and modified timestamp for all .joblib and .json artifacts.
        """
        manifest: Dict[str, Dict[str, Any]] = {}
        if not self.artifacts_dir.exists():
            return manifest

        for file_path in self.artifacts_dir.glob("*.*"):
            if file_path.suffix not in (".joblib", ".json"):
                continue

            stat = file_path.stat()
            h = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)

            manifest[file_path.name] = {
                "filename": file_path.name,
                "size_bytes": stat.st_size,
                "sha256": h.hexdigest(),
                "mtime": stat.st_mtime,
                "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        return manifest

    def save_model_artifact(self, filename: str, content: bytes, expected_sha256: str | None = None) -> Tuple[bool, str]:
        """
        Atomically write a model artifact after verifying integrity and valid joblib deserialization.
        """
        filename = Path(filename).name  # Prevent directory traversal
        if not (filename.endswith(".joblib") or filename.endswith(".json")):
            return False, f"Invalid artifact filename: {filename}"

        # 1. SHA256 Checksum validation
        calc_hash = hashlib.sha256(content).hexdigest()
        if expected_sha256 and calc_hash.lower() != expected_sha256.lower():
            return False, f"Checksum mismatch for {filename}: expected {expected_sha256}, got {calc_hash}"

        # 2. Integrity load verification for joblib models (prevent corrupted models from breaking bot)
        if filename.endswith(".joblib"):
            try:
                bio = io.BytesIO(content)
                _ = joblib.load(bio)
            except Exception as e:
                return False, f"Joblib integrity verification failed for {filename}: {e}"

        target_file = self.artifacts_dir / filename
        temp_file = self.artifacts_dir / f"{filename}.sync_{os.getpid()}.tmp"

        with self._lock:
            try:
                with open(temp_file, "wb") as f:
                    f.write(content)
                # Atomic replace on OS level
                temp_file.replace(target_file)
                logger.info(f"Model artifact '{filename}' updated successfully ({len(content):,} bytes, SHA: {calc_hash[:8]}).")
                return True, f"Updated {filename}"
            except Exception as e:
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)
                return False, f"Failed to write model file: {e}"

    # ─────────────────────────────────────────────────────────────
    # Database Events Export & Deduplicated Import
    # ─────────────────────────────────────────────────────────────

    def export_events(self, since_ts: str | None = None, limit: int = 10000) -> Dict[str, Any]:
        """
        Export ml_events, trade_log, and partial_tp_events with optional timestamp filter.
        """
        conn = self.get_db_connection()
        try:
            # 1. ml_events
            ml_query = "SELECT * FROM ml_events"
            params: list[Any] = []
            if since_ts:
                ml_query += " WHERE ts > ?"
                params.append(since_ts)
            ml_query += " ORDER BY ts ASC LIMIT ?"
            params.append(limit)

            ml_rows = conn.execute(ml_query, params).fetchall()
            ml_events = [dict(r) for r in ml_rows]

            # 2. trade_log
            trade_query = "SELECT * FROM trade_log"
            trade_params: list[Any] = []
            if since_ts:
                trade_query += " WHERE timestamp > ?"
                trade_params.append(since_ts)
            trade_query += " ORDER BY timestamp ASC LIMIT ?"
            trade_params.append(limit)

            trade_rows = conn.execute(trade_query, trade_params).fetchall()
            trade_logs = [dict(r) for r in trade_rows]

            # 3. partial_tp_events (if table exists)
            partial_tp_events: list[dict[str, Any]] = []
            try:
                ptp_query = "SELECT * FROM partial_tp_events"
                ptp_params: list[Any] = []
                if since_ts:
                    ptp_query += " WHERE timestamp > ?"
                    ptp_params.append(since_ts)
                ptp_query += " ORDER BY timestamp ASC LIMIT ?"
                ptp_params.append(limit)
                ptp_rows = conn.execute(ptp_query, ptp_params).fetchall()
                partial_tp_events = [dict(r) for r in ptp_rows]
            except sqlite3.OperationalError:
                pass

            payload = {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "since_ts": since_ts,
                "counts": {
                    "ml_events": len(ml_events),
                    "trade_log": len(trade_logs),
                    "partial_tp_events": len(partial_tp_events),
                },
                "ml_events": ml_events,
                "trade_log": trade_logs,
                "partial_tp_events": partial_tp_events,
            }
            return payload
        finally:
            conn.close()

    def import_events(self, payload: Dict[str, Any]) -> Dict[str, int]:
        """
        Merge remote records into local tables atomically without duplicating records.
        """
        ml_events = payload.get("ml_events", [])
        trade_logs = payload.get("trade_log", [])
        partial_tp_events = payload.get("partial_tp_events", [])

        stats = {
            "ml_events_inserted": 0,
            "trade_log_inserted": 0,
            "partial_tp_inserted": 0,
        }

        conn = self.get_db_connection()
        try:
            with conn:
                # 1. Insert ml_events with INSERT OR IGNORE (keyed on unique event_id)
                if ml_events:
                    cols = [
                        "event_id", "ts", "symbol", "timeframe", "kind", "direction",
                        "entry", "stop", "target", "features", "label", "outcome",
                        "r_multiple", "label_ts", "p_genuine", "model_version",
                        "allowed", "strategy"
                    ]
                    placeholders = ", ".join(["?"] * len(cols))
                    col_str = ", ".join(cols)
                    insert_sql = f"INSERT OR IGNORE INTO ml_events ({col_str}) VALUES ({placeholders})"

                    batch = []
                    for ev in ml_events:
                        batch.append([ev.get(c) for c in cols])

                    cur = conn.executemany(insert_sql, batch)
                    stats["ml_events_inserted"] = cur.rowcount if cur.rowcount > 0 else 0

                # 2. Insert trade_log with deduplication check
                if trade_logs:
                    trade_cols = [
                        "timestamp", "symbol", "direction", "entry_price", "stop_loss",
                        "take_profit", "lot_size", "realized_pnl", "status",
                        "strategy_name", "magic_number"
                    ]
                    insert_trade_sql = f"""
                        INSERT INTO trade_log ({", ".join(trade_cols)})
                        SELECT {", ".join(["?"] * len(trade_cols))}
                        WHERE NOT EXISTS (
                            SELECT 1 FROM trade_log
                            WHERE timestamp = ? AND symbol = ? AND entry_price = ? AND magic_number = ?
                        )
                    """
                    t_count = 0
                    for t in trade_logs:
                        vals = [t.get(c) for c in trade_cols]
                        check_vals = [t.get("timestamp"), t.get("symbol"), t.get("entry_price"), t.get("magic_number")]
                        cur = conn.execute(insert_trade_sql, vals + check_vals)
                        if cur.rowcount > 0:
                            t_count += 1
                    stats["trade_log_inserted"] = t_count

                # 3. Insert partial_tp_events with deduplication
                if partial_tp_events:
                    ptp_cols = [
                        "trade_id", "symbol", "strategy_name", "direction", "timestamp",
                        "current_price", "r_multiple", "p_reversal", "p_full_tp",
                        "predicted_max_r", "action", "close_pct", "closed_lot",
                        "remaining_lot", "nearest_resistance", "structure_confluence_count",
                        "features_json", "realized_max_r", "outcome_label", "is_shadow"
                    ]
                    insert_ptp_sql = f"""
                        INSERT INTO partial_tp_events ({", ".join(ptp_cols)})
                        SELECT {", ".join(["?"] * len(ptp_cols))}
                        WHERE NOT EXISTS (
                            SELECT 1 FROM partial_tp_events
                            WHERE trade_id = ? AND timestamp = ? AND action = ?
                        )
                    """
                    ptp_count = 0
                    for p in partial_tp_events:
                        vals = [p.get(c) for c in ptp_cols]
                        check_vals = [p.get("trade_id"), p.get("timestamp"), p.get("action")]
                        cur = conn.execute(insert_ptp_sql, vals + check_vals)
                        if cur.rowcount > 0:
                            ptp_count += 1
                    stats["partial_tp_inserted"] = ptp_count

            logger.info(f"Sync import completed: {stats}")
            return stats
        finally:
            conn.close()

    def get_sync_status(self) -> Dict[str, Any]:
        """Return total event counts and model versions on this local instance."""
        conn = self.get_db_connection()
        try:
            ml_count = conn.execute("SELECT count(*) FROM ml_events").fetchone()[0]
            trade_count = conn.execute("SELECT count(*) FROM trade_log").fetchone()[0]
            ptp_count = 0
            try:
                ptp_count = conn.execute("SELECT count(*) FROM partial_tp_events").fetchone()[0]
            except sqlite3.OperationalError:
                pass

            models = self.get_models_manifest()
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "db_path": str(self.db_path),
                "counts": {
                    "ml_events": ml_count,
                    "trade_log": trade_count,
                    "partial_tp_events": ptp_count,
                },
                "models_count": len(models),
                "models": list(models.keys()),
            }
        finally:
            conn.close()


class CloudSyncClient:
    """
    High-level client for executing sync operations with a remote cloud instance (e.g. AWS).
    """

    def __init__(self, remote_url: str, auth_token: str, local_engine: CloudSyncEngine | None = None, timeout: float = 30.0):
        self.remote_url = remote_url.rstrip("/")
        self.auth_token = auth_token
        self.engine = local_engine or CloudSyncEngine()
        self.timeout = timeout
        self.session = requests.Session()
        if self.auth_token:
            self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})

    def check_health(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify connectivity with remote server and compare event counts."""
        url = f"{self.remote_url}/api/sync/status"
        try:
            r = self.session.get(url, timeout=self.timeout)
            if r.status_code == 200:
                return True, r.json()
            return False, {"error": f"HTTP {r.status_code}: {r.text}"}
        except Exception as e:
            return False, {"error": str(e)}

    def push_events_to_cloud(self, since_ts: str | None = None) -> Dict[str, Any]:
        """Export local events and send to remote cloud server."""
        payload = self.engine.export_events(since_ts=since_ts)
        url = f"{self.remote_url}/api/sync/events/import"
        try:
            r = self.session.post(url, json=payload, timeout=self.timeout)
            if r.status_code == 200:
                res = r.json()
                logger.info(f"Pushed {payload['counts']} to cloud. Cloud import result: {res}")
                return {"status": "success", "pushed": payload["counts"], "remote_result": res}
            return {"status": "error", "error": f"HTTP {r.status_code}: {r.text}"}
        except Exception as e:
            logger.error(f"Failed to push events to cloud: {e}")
            return {"status": "error", "error": str(e)}

    def pull_events_from_cloud(self, since_ts: str | None = None) -> Dict[str, Any]:
        """Fetch remote events from cloud and merge into local database."""
        url = f"{self.remote_url}/api/sync/events/export"
        params = {"since_ts": since_ts} if since_ts else {}
        try:
            r = self.session.post(url, params=params, timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                import_res = self.engine.import_events(data)
                logger.info(f"Pulled {data['counts']} from cloud. Local import result: {import_res}")
                return {"status": "success", "pulled": data["counts"], "imported": import_res}
            return {"status": "error", "error": f"HTTP {r.status_code}: {r.text}"}
        except Exception as e:
            logger.error(f"Failed to pull events from cloud: {e}")
            return {"status": "error", "error": str(e)}

    def sync_models_from_cloud(self) -> Dict[str, Any]:
        """
        Compare local vs cloud model manifests.
        Download and verify any newer or missing models from cloud.
        """
        manifest_url = f"{self.remote_url}/api/sync/models/manifest"
        try:
            r = self.session.get(manifest_url, timeout=self.timeout)
            if r.status_code != 200:
                return {"status": "error", "error": f"Failed to fetch remote manifest: HTTP {r.status_code}"}

            remote_manifest = r.json().get("models", {})
            local_manifest = self.engine.get_models_manifest()

            downloaded = []
            skipped = []
            errors = []

            for filename, r_meta in remote_manifest.items():
                l_meta = local_manifest.get(filename)
                # If local does not exist or remote hash is different and remote mtime is newer
                need_download = False
                if not l_meta:
                    need_download = True
                elif l_meta["sha256"] != r_meta["sha256"]:
                    if r_meta["mtime"] > l_meta["mtime"]:
                        need_download = True

                if need_download:
                    dl_url = f"{self.remote_url}/api/sync/models/download/{filename}"
                    dl_res = self.session.get(dl_url, timeout=self.timeout)
                    if dl_res.status_code == 200:
                        ok, msg = self.engine.save_model_artifact(filename, dl_res.content, r_meta["sha256"])
                        if ok:
                            downloaded.append(filename)
                        else:
                            errors.append(f"{filename}: {msg}")
                    else:
                        errors.append(f"{filename}: HTTP {dl_res.status_code}")
                else:
                    skipped.append(filename)

            return {
                "status": "success",
                "downloaded": downloaded,
                "up_to_date": skipped,
                "errors": errors,
            }
        except Exception as e:
            logger.error(f"Failed to sync models from cloud: {e}")
            return {"status": "error", "error": str(e)}

    def trigger_remote_retrain(self) -> Dict[str, Any]:
        """Request remote cloud instance to retrain ML models using combined dataset."""
        url = f"{self.remote_url}/api/sync/trigger_retrain"
        try:
            r = self.session.post(url, timeout=self.timeout)
            return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}: {r.text}"}
        except Exception as e:
            return {"error": str(e)}

    def full_two_way_sync(self) -> Dict[str, Any]:
        """
        Execute full two-way production synchronization:
        1. Push local trades to cloud
        2. Pull cloud trades to local
        3. Synchronize newest model weights
        """
        logger.info(f"Starting Full Two-Way Sync with {self.remote_url}...")
        push_res = self.push_events_to_cloud()
        pull_res = self.pull_events_from_cloud()
        model_res = self.sync_models_from_cloud()

        summary = {
            "status": "success" if push_res.get("status") == "success" and pull_res.get("status") == "success" else "partial",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "push": push_res,
            "pull": pull_res,
            "models": model_res,
        }
        logger.info(f"Full Two-Way Sync finished: {summary['status']}")
        return summary

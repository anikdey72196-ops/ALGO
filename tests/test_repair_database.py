"""
test_repair_database.py — Tests for scripts/repair_database.py CSV fallback reconciliation.
"""

import os
import csv
import sqlite3
import tempfile
from pathlib import Path
import pytest

from scripts.repair_database import sync_from_csv_fallbacks

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_sync_from_csv_fallbacks_batch(monkeypatch):
    """Verify that sync_from_csv_fallbacks restores missing CSV rows in batch with executemany."""
    orig_cwd = os.getcwd()
    tmpdir = tempfile.TemporaryDirectory()
    try:
        tmp_path = Path(tmpdir.name)
        os.chdir(tmp_path)

        # 1. Create a mock trades_history.csv
        csv_file = tmp_path / "trades_history.csv"
        fieldnames = [
            "id", "timestamp", "closed_at", "symbol", "direction",
            "entry_price", "stop_loss", "take_profit", "lot_size",
            "realized_pnl", "status", "duration_seconds", "strategy_name", "magic_number"
        ]
        rows = [
            {
                "id": "1",
                "timestamp": "2026-10-18T10:00:00Z",
                "closed_at": "2026-10-18T10:10:00Z",
                "symbol": "XAUUSD",
                "direction": "BUY",
                "entry_price": "2650.5",
                "stop_loss": "2640.0",
                "take_profit": "2670.0",
                "lot_size": "0.05",
                "realized_pnl": "50.0",
                "status": "CLOSED",
                "duration_seconds": "600",
                "strategy_name": "SMC",
                "magic_number": "123456",
            },
            {
                "id": "2",
                "timestamp": "2026-10-18T11:00:00Z",
                "closed_at": "2026-10-18T11:15:00Z",
                "symbol": "EURUSD",
                "direction": "SELL",
                "entry_price": "1.0850",
                "stop_loss": "1.0870",
                "take_profit": "1.0810",
                "lot_size": "0.10",
                "realized_pnl": "-20.0",
                "status": "CLOSED",
                "duration_seconds": "900",
                "strategy_name": "ICT",
                "magic_number": "126456",
            },
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # 2. Create target database with trade_log schema
        db_path = tmp_path / "trading_state.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE trade_log (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                closed_at TEXT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                lot_size REAL,
                realized_pnl REAL,
                status TEXT,
                duration_seconds REAL,
                strategy_name TEXT,
                magic_number INTEGER
            );
        """)
        conn.commit()
        conn.close()

        # 3. First reconciliation: should batch restore both rows
        sync_from_csv_fallbacks(db_path)

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        saved = cur.execute("SELECT id, symbol, realized_pnl, strategy_name FROM trade_log ORDER BY id;").fetchall()
        conn.close()

        assert len(saved) == 2
        assert saved[0] == (1, "XAUUSD", 50.0, "SMC")
        assert saved[1] == (2, "EURUSD", -20.0, "ICT")

        # 4. Second reconciliation with 1 new row (id 3) and 2 existing (id 1, 2)
        rows.append({
            "id": "3",
            "timestamp": "2026-10-18T12:00:00Z",
            "closed_at": "2026-10-18T12:20:00Z",
            "symbol": "GBPUSD",
            "direction": "BUY",
            "entry_price": "1.3020",
            "stop_loss": "1.3000",
            "take_profit": "1.3060",
            "lot_size": "0.15",
            "realized_pnl": "60.0",
            "status": "CLOSED",
            "duration_seconds": "1200",
            "strategy_name": "ORDER_FLOW",
            "magic_number": "127456",
        })
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        sync_from_csv_fallbacks(db_path)

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        total_rows = cur.execute("SELECT count(*) FROM trade_log;").fetchone()[0]
        row_3 = cur.execute("SELECT id, symbol, realized_pnl, strategy_name FROM trade_log WHERE id = 3;").fetchone()
        conn.close()

        assert total_rows == 3
        assert row_3 == (3, "GBPUSD", 60.0, "ORDER_FLOW")
    finally:
        os.chdir(orig_cwd)
        tmpdir.cleanup()

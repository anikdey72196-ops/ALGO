"""
test_repair_database.py — Tests for scripts/repair_database.py CSV fallback reconciliation.
"""

import os
import csv
import sqlite3
import tempfile
from pathlib import Path
import pytest

from scripts.repair_database import sync_from_csv_fallbacks, is_safe_table_name, get_table_counts, salvage_table_data

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_is_safe_table_name():
    """Verify table identifier validator strictly allows safe names and blocks injection payloads."""
    safe_names = ["trade_log", "daily_state", "bot_sessions", "Table1", "t", "a_b_c"]
    for name in safe_names:
        assert is_safe_table_name(name) is True, f"Expected {name} to be safe"

    unsafe_names = [
        "trade_log; DROP TABLE test; --",
        "trade_log' OR '1'='1",
        'trade_log" OR "1"="1',
        "table name with space",
        "table-with-hyphen",
        "table.name",
        "table/*comment*/",
        "",
        "a" * 65,  # Exceeds max length
        None,
        123,
    ]
    for name in unsafe_names:
        assert is_safe_table_name(name) is False, f"Expected {name} to be rejected"


def test_get_table_counts_sql_injection_defense():
    """Verify get_table_counts safely handles tables with injection payloads in sqlite_master."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_injection.db"
        conn = sqlite3.connect(str(db_path))

        # Create a legitimate table
        conn.execute("CREATE TABLE valid_table (id INT);")
        conn.execute("INSERT INTO valid_table VALUES (1), (2), (3);")

        # Create a table with a malicious name containing SQL injection syntax
        malicious_name = 'valid_table; CREATE TABLE injected_table (x INT); --'
        conn.execute(f'CREATE TABLE "{malicious_name}" (id INT);')
        conn.commit()
        conn.close()

        # Run get_table_counts
        counts = get_table_counts(db_path)

        assert counts.get("valid_table") == 3
        assert counts.get(malicious_name) is None

        # Verify that the injected statement was NOT executed
        conn = sqlite3.connect(str(db_path))
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        conn.close()

        assert "injected_table" not in tables


def test_salvage_table_data_security():
    """Verify salvage_table_data blocks tables with unsafe identifiers and sanitizes columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "src.db"
        dst_path = Path(tmpdir) / "dst.db"

        src_conn = sqlite3.connect(str(src_path))
        dst_conn = sqlite3.connect(str(dst_path))

        # 1. Unsafe table name should be rejected immediately
        salvaged, errors = salvage_table_data(src_conn, dst_conn, "test; DROP TABLE something; --")
        assert salvaged == 0
        assert errors == 1

        # 2. Legitimate table should be salvaged properly
        src_conn.execute("CREATE TABLE safe_table (id INT, val TEXT);")
        dst_conn.execute("CREATE TABLE safe_table (id INT, val TEXT);")
        src_conn.execute("INSERT INTO safe_table VALUES (1, 'apple'), (2, 'banana');")
        src_conn.commit()

        salvaged, errors = salvage_table_data(src_conn, dst_conn, "safe_table")
        assert salvaged == 2
        assert errors == 0

        res = dst_conn.execute("SELECT id, val FROM safe_table ORDER BY id;").fetchall()
        assert res == [(1, 'apple'), (2, 'banana')]

        src_conn.close()
        dst_conn.close()


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

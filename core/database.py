"""
core/database.py — Unified Database Abstraction Layer for ALGO Trading.
Supports both MySQL and SQLite seamlessly with automatic query translation,
thread-safe connections, and dictionary/index row access (sqlite3.Row parity).
"""

from __future__ import annotations
import os
import re
import threading
import sqlite3
from typing import Any, Iterable, List, Optional, Tuple, Dict
from loguru import logger

try:
    import pymysql
    import pymysql.cursors
    HAS_PYMYSQL = True
except ImportError:
    HAS_PYMYSQL = False


class DBRow(dict):
    """Row wrapper that supports both key access (row['col']) and index access (row[0])."""
    def __init__(self, data: dict, columns: list[str], values: tuple):
        super().__init__(data)
        self._columns = columns
        self._values = values

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._values[item]
        return super().__getitem__(item)

    def keys(self):
        return self._columns

    def values(self):
        return self._values


def _translate_sqlite_to_mysql(sql: str) -> str:
    """
    Translates common SQLite SQL constructs into MySQL-compatible syntax.
    Handles:
    - PRAGMA statements -> ignored (no-op)
    - Parameter placeholders: '?' -> '%s'
    - 'INSERT OR REPLACE INTO' -> 'REPLACE INTO'
    - 'INSERT OR IGNORE INTO' -> 'INSERT IGNORE INTO'
    - SQLite ON CONFLICT(col) DO UPDATE SET ... excluded.field -> ON DUPLICATE KEY UPDATE field=VALUES(field)
    - datetime('now') -> NOW()
    """
    cleaned = sql.strip()
    if cleaned.upper().startswith("PRAGMA "):
        return "SELECT 1"

    # 1. Parameter placeholder translation (? -> %s) outside of single/double quotes
    parts = []
    in_single = False
    in_double = False
    i = 0
    while i < len(cleaned):
        ch = cleaned[i]
        if ch == "'" and not in_double:
            in_single = not in_single
            parts.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            parts.append(ch)
        elif ch == '?' and not in_single and not in_double:
            parts.append('%s')
        else:
            parts.append(ch)
        i += 1
    sql = "".join(parts)

    # 2. Syntax replacements
    sql = re.sub(r'\bINSERT\s+OR\s+REPLACE\s+INTO\b', 'REPLACE INTO', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bINSERT\s+OR\s+IGNORE\s+INTO\b', 'INSERT IGNORE INTO', sql, flags=re.IGNORECASE)
    sql = re.sub(r"datetime\('now'\)", "NOW()", sql, flags=re.IGNORECASE)

    # 3. SQLite ON CONFLICT translation for ml_events
    if "ON CONFLICT" in sql.upper():
        conflict_match = re.search(
            r'ON\s+CONFLICT\s*\([^)]*\)\s*DO\s+UPDATE\s+SET\s+(.*)',
            sql,
            flags=re.IGNORECASE | re.DOTALL
        )
        if conflict_match:
            update_clause = conflict_match.group(1).strip()
            # Replace excluded.column with VALUES(column)
            update_clause_mysql = re.sub(
                r'excluded\.(\w+)',
                r'VALUES(\1)',
                update_clause,
                flags=re.IGNORECASE
            )
            prefix = sql[:conflict_match.start()].strip()
            sql = f"{prefix} ON DUPLICATE KEY UPDATE {update_clause_mysql}"

    return sql


class MySQLCursorWrapper:
    """Cursor wrapper for PyMySQL providing sqlite3-like behavior."""
    def __init__(self, raw_cursor):
        self._cur = raw_cursor

    def execute(self, sql: str, params: Optional[Iterable[Any]] = None):
        mysql_sql = _translate_sqlite_to_mysql(sql)
        if mysql_sql == "SELECT 1" and sql.strip().upper().startswith("PRAGMA "):
            return self
        if params is not None:
            if isinstance(params, (list, tuple)):
                self._cur.execute(mysql_sql, tuple(params))
            else:
                self._cur.execute(mysql_sql, (params,))
        else:
            self._cur.execute(mysql_sql)
        return self

    def executemany(self, sql: str, seq_of_params: Iterable[Iterable[Any]]):
        mysql_sql = _translate_sqlite_to_mysql(sql)
        self._cur.executemany(mysql_sql, seq_of_params)
        return self

    def fetchone(self) -> Optional[DBRow]:
        row = self._cur.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in self._cur.description] if self._cur.description else []
        vals = tuple(row[c] for c in cols)
        return DBRow(row, cols, vals)

    def fetchall(self) -> List[DBRow]:
        rows = self._cur.fetchall()
        if not rows:
            return []
        cols = [desc[0] for desc in self._cur.description] if self._cur.description else []
        result = []
        for r in rows:
            vals = tuple(r[c] for c in cols)
            result.append(DBRow(r, cols, vals))
        return result

    @property
    def lastrowid(self) -> Optional[int]:
        return self._cur.lastrowid

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    @property
    def description(self):
        return self._cur.description

    def close(self):
        try:
            self._cur.close()
        except Exception:
            pass


class MySQLConnectionWrapper:
    """Thread-aware PyMySQL connection wrapper with context manager and sqlite parity."""
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.row_factory = None  # Parity attribute
        self._raw_conn = None
        self._connect()

    def _connect(self):
        if not HAS_PYMYSQL:
            raise ImportError("pymysql is required for MySQL database backend. Run `pip install pymysql`.")
        self._raw_conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
        )

    def _ensure_connected(self):
        try:
            if self._raw_conn:
                self._raw_conn.ping(reconnect=True)
            else:
                self._connect()
        except Exception:
            self._connect()

    def cursor(self) -> MySQLCursorWrapper:
        self._ensure_connected()
        raw_cur = self._raw_conn.cursor()
        return MySQLCursorWrapper(raw_cur)

    def execute(self, sql: str, params: Optional[Iterable[Any]] = None) -> MySQLCursorWrapper:
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def commit(self):
        try:
            if self._raw_conn:
                self._raw_conn.commit()
        except Exception:
            pass

    def rollback(self):
        try:
            if self._raw_conn:
                self._raw_conn.rollback()
        except Exception:
            pass

    def close(self):
        try:
            if self._raw_conn:
                self._raw_conn.close()
        except Exception:
            pass

    def __enter__(self):
        self._ensure_connected()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()


def get_db_connection(db_path: str = "trading_state.db"):
    """
    Factory that returns either a MySQL connection wrapper or an SQLite connection,
    based on the DB_BACKEND environment variable.
    """
    backend = os.getenv("DB_BACKEND", "sqlite").strip().lower()

    if backend == "mysql":
        host = os.getenv("DB_HOST", "localhost")
        port = int(os.getenv("DB_PORT", 3306))
        user = os.getenv("DB_USER", "root")
        password = os.getenv("DB_PASSWORD", "")
        database = os.getenv("DB_NAME", "ALGO")

        try:
            logger.debug(f"Connecting to MySQL ({user}@{host}:{port}/{database})...")
            conn = MySQLConnectionWrapper(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
            )
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to MySQL ({e}). Falling back to SQLite '{db_path}'.")

    # Fallback / Default: SQLite
    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 30000")
    except Exception:
        pass
    return conn

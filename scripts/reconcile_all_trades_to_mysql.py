"""
scripts/reconcile_all_trades_to_mysql.py
=========================================
1. Cleans up junk/test records (IDs 1, 2, 3 and 9000-9999, CLOSED_TEST) from MySQL ALGO database.
2. Fetches all real closed positions from MT5 (Account 1302182232 - XMGlobal-MT5 6) with exact net PnL (profit + swap + commission).
3. Inserts/updates real MT5 trades into MySQL `trade_log`.
4. Migrates any missing cloud/historical trades from `trading_state.db` into MySQL `trade_log`.
5. Refreshes the `trades_journal` view so all Net P&L and Trade IDs are authentic.
"""

import os
import sys
import sqlite3
import pymysql
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "ALGO")
MT5_ACCOUNT = int(os.getenv("MT5_LOGIN", "1302182232"))
MT5_SERVER = os.getenv("MT5_SERVER", "XMGlobal-MT5 6")

def get_session_name(dt: datetime) -> str:
    hour = dt.hour
    if 7 <= hour < 12:
        return "LONDON_OPEN"
    elif 12 <= hour < 17:
        return "NY_AM"
    elif 17 <= hour < 22:
        return "NY_PM"
    else:
        return "ASIA_TOKYO"

def get_strategy_name_from_magic(magic: int) -> str:
    if magic == 125456:
        return "SMC Scalp (5m)"
    elif magic == 127456:
        return "Order Flow (Delta & Absorption)"
    elif magic == 126456:
        return "ICT Institutional"
    elif magic == 128456:
        return "Trend Reversal Strategy"
    else:
        return "SMC Swing (15m)"

def run_reconciliation():
    print("=" * 60)
    print("🚀 RECONCILING ALL TRADES TO MYSQL (SINGLE SOURCE OF TRUTH)")
    print("=" * 60)

    # 1. Connect to MySQL
    my_conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor
    )
    my_cur = my_conn.cursor()

    # 2. Delete test/junk records
    print("\n[Step 1/5] Cleaning up test fixture rows (IDs 1, 2, 3, 9000-9999, CLOSED_TEST)...")
    my_cur.execute("""
        DELETE FROM trade_log 
        WHERE id IN (1, 2, 3) 
           OR (id >= 9000 AND id <= 9999) 
           OR status = 'CLOSED_TEST'
    """)
    deleted_count = my_cur.rowcount
    print(f"  ✓ Deleted {deleted_count} junk/test rows.")

    # 3. Pull real closed deals from MT5 XM account
    print(f"\n[Step 2/5] Fetching live closed deals from MT5 ({MT5_SERVER} - Acc #{MT5_ACCOUNT})...")
    mt5_trades = []
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            deals = mt5.history_deals_get(datetime(2020, 1, 1), datetime.now())
            if deals:
                positions = {}
                for d in deals:
                    pos_id = d.position_id
                    if not pos_id:
                        continue
                    if pos_id not in positions:
                        positions[pos_id] = {
                            "position_id": pos_id,
                            "symbol": d.symbol,
                            "magic": d.magic,
                            "open_deal": None,
                            "close_deal": None,
                            "profit": 0.0,
                            "swap": 0.0,
                            "commission": 0.0,
                        }
                    p = positions[pos_id]
                    p["profit"] += d.profit
                    p["swap"] += d.swap
                    p["commission"] += d.commission
                    if d.entry == 0:  # In
                        p["open_deal"] = d
                    elif d.entry == 1:  # Out
                        p["close_deal"] = d

                # Filter only fully closed positions
                for pos_id, p in positions.items():
                    if p["close_deal"] and p["open_deal"]:
                        open_d = p["open_deal"]
                        close_d = p["close_deal"]
                        open_dt = datetime.fromtimestamp(open_d.time, tz=timezone.utc)
                        close_dt = datetime.fromtimestamp(close_d.time, tz=timezone.utc)
                        net_pnl = round(p["profit"] + p["swap"] + p["commission"], 2)
                        direction = "BUY" if open_d.type == 0 else "SELL"
                        duration = max(0.0, (close_dt - open_dt).total_seconds())
                        status = "CLOSED_TP" if net_pnl > 0 else "CLOSED_SL"
                        strat = get_strategy_name_from_magic(open_d.magic)
                        session = get_session_name(open_dt)

                        mt5_trades.append({
                            "id": pos_id,
                            "timestamp": open_dt.isoformat(),
                            "symbol": open_d.symbol,
                            "direction": direction,
                            "entry_price": float(open_d.price),
                            "stop_loss": 0.0,
                            "take_profit": 0.0,
                            "lot_size": float(open_d.volume),
                            "realized_pnl": net_pnl,
                            "status": status,
                            "strategy_name": strat,
                            "magic_number": open_d.magic,
                            "closed_at": close_dt.isoformat(),
                            "duration_seconds": duration,
                            "account_id": MT5_ACCOUNT,
                            "broker_name": MT5_SERVER,
                            "session": session,
                        })
                print(f"  ✓ Processed {len(mt5_trades)} real closed positions from MT5.")
            mt5.shutdown()
        else:
            print("  ⚠️ Could not initialize MT5 desktop terminal directly.")
    except Exception as e:
        print(f"  ⚠️ Error reading MT5 history: {e}")

    # 4. Insert real MT5 trades into MySQL
    if mt5_trades:
        print(f"\n[Step 3/5] Inserting/Updating {len(mt5_trades)} real MT5 trades into MySQL...")
        insert_sql = """
            INSERT INTO trade_log (
                id, timestamp, symbol, direction, entry_price, stop_loss, take_profit,
                lot_size, realized_pnl, status, strategy_name, magic_number,
                closed_at, duration_seconds, account_id, broker_name, session
            ) VALUES (
                %(id)s, %(timestamp)s, %(symbol)s, %(direction)s, %(entry_price)s, %(stop_loss)s, %(take_profit)s,
                %(lot_size)s, %(realized_pnl)s, %(status)s, %(strategy_name)s, %(magic_number)s,
                %(closed_at)s, %(duration_seconds)s, %(account_id)s, %(broker_name)s, %(session)s
            )
            ON DUPLICATE KEY UPDATE
                realized_pnl = VALUES(realized_pnl),
                status = VALUES(status),
                closed_at = VALUES(closed_at),
                duration_seconds = VALUES(duration_seconds),
                broker_name = VALUES(broker_name),
                account_id = VALUES(account_id);
        """
        my_cur.executemany(insert_sql, mt5_trades)
        print(f"  ✓ Upserted {len(mt5_trades)} real XM trades into MySQL.")

    # 5. Migrate any missing cloud / historical trades from trading_state.db
    sqlite_path = Path("trading_state.db")
    if sqlite_path.exists():
        print("\n[Step 4/5] Syncing historical/cloud trades from trading_state.db...")
        sq_conn = sqlite3.connect(str(sqlite_path))
        sq_conn.row_factory = sqlite3.Row
        sq_cur = sq_conn.cursor()

        sq_cur.execute("""
            SELECT id, timestamp, symbol, direction, entry_price, stop_loss, take_profit,
                   lot_size, realized_pnl, status, strategy_name, magic_number,
                   closed_at, duration_seconds, account_id, broker_name, session
            FROM trade_log
            WHERE status != 'CLOSED_TEST' AND (id IS NULL OR (id NOT IN (1, 2, 3) AND (id < 9000 OR id > 9999)))
        """)
        sq_rows = sq_cur.fetchall()
        print(f"  ✓ Found {len(sq_rows)} historical rows in SQLite.")

        migrated_from_sqlite = 0
        insert_sq_sql = """
            INSERT IGNORE INTO trade_log (
                id, timestamp, symbol, direction, entry_price, stop_loss, take_profit,
                lot_size, realized_pnl, status, strategy_name, magic_number,
                closed_at, duration_seconds, account_id, broker_name, session
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        batch = []
        for r in sq_rows:
            ts_str = str(r["timestamp"] or "")
            broker = r["broker_name"] or ("XMGlobal-MT5 6" if ts_str >= "2026-10-01" else "MetaQuotes-Demo")
            session = r["session"]
            if not session and ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    session = get_session_name(dt)
                except Exception:
                    session = "LONDON_OPEN"
            session = session or "LONDON_OPEN"
            batch.append((
                r["id"], r["timestamp"], r["symbol"], r["direction"], r["entry_price"],
                r["stop_loss"], r["take_profit"], r["lot_size"], r["realized_pnl"],
                r["status"], r["strategy_name"], r["magic_number"], r["closed_at"],
                r["duration_seconds"], r["account_id"], broker, session
            ))
            if len(batch) >= 200:
                my_cur.executemany(insert_sq_sql, batch)
                migrated_from_sqlite += len(batch)
                batch = []
        if batch:
            my_cur.executemany(insert_sq_sql, batch)
            migrated_from_sqlite += len(batch)

        print(f"  ✓ Migrated {migrated_from_sqlite} rows from SQLite into MySQL.")
        sq_conn.close()

    # 5. Recreate the clean VIEW
    print("\n[Step 5/6] Recreating clean VIEW `trades_journal`...")
    view_sql = """
    CREATE OR REPLACE VIEW trades_journal AS
    SELECT 
        id AS `Trade ID`,
        SUBSTRING(timestamp, 12, 8) AS `TIME`,
        SUBSTRING(timestamp, 1, 10) AS `DATE`,
        broker_name AS `BROKER NAME`,
        symbol AS `INSTRUMENTS`,
        ROUND(realized_pnl, 2) AS `NET P&L`,
        strategy_name AS `STRATEGY`,
        session AS `SESSION`
    FROM trade_log
    ORDER BY id DESC;
    """
    my_cur.execute(view_sql)
    print("  ✓ Created optimized VIEW `trades_journal`.")

    # 6. Sync daily_state (trade_count & realized_pnl)
    print("\n[Step 6/6] Syncing daily_state table in MySQL...")
    my_cur.execute("""
        INSERT INTO daily_state (date, realized_pnl, trade_count, circuit_breaker_active)
        SELECT 
            SUBSTR(timestamp, 1, 10) as dt,
            ROUND(COALESCE(SUM(CASE WHEN status NOT IN ('OPEN', 'CLOSED_TEST') THEN realized_pnl ELSE 0.0 END), 0.0), 2) as pnl,
            COUNT(*) as cnt,
            0
        FROM trade_log
        WHERE status != 'CLOSED_TEST'
        GROUP BY dt
        ON DUPLICATE KEY UPDATE
            trade_count = VALUES(trade_count),
            realized_pnl = VALUES(realized_pnl);
    """)
    print(f"  ✓ Synced {my_cur.rowcount} daily_state summary records.")

    # 7. Summary
    my_cur.execute("SELECT COUNT(*) AS total, SUM(realized_pnl) AS total_pnl FROM trade_log")
    summary = my_cur.fetchone()
    print("\n" + "=" * 60)
    print("🎉 RECONCILIATION COMPLETE!")
    print(f"Total Trades in MySQL: {summary['total']}")
    print(f"Total Realized Net PnL: ${summary['total_pnl']:,.2f}")
    print("=" * 60)

    my_cur.execute("SELECT * FROM trades_journal LIMIT 5")
    rows = my_cur.fetchall()
    print("\nSample Rows from `trades_journal`:")
    for r in rows:
        print(r)

    my_conn.close()

if __name__ == "__main__":
    run_reconciliation()

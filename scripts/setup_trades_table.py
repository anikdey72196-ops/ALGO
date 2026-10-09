"""
scripts/setup_trades_table.py — Sets up the exact user-requested trade columns & views in MySQL.
Fields:
- Trade ID
- TIME
- DATE
- BROKER NAME
- INSTRUMENTS
- NET P&L
- STRATEGY
- SESSION
"""

import os
import sys
import pymysql
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
BROKER = os.getenv("MT5_SERVER", "XMGlobal-MT5 6") or "XMGlobal-MT5 6"
HISTORICAL_BROKER = "MetaQuotes-Demo"

def setup():
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor
    )
    cur = conn.cursor()

    print(f"Connecting to MySQL ({DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME})...")

    # 1. Add broker_name and session columns to trade_log if not already present
    cur.execute("SHOW COLUMNS FROM trade_log LIKE 'broker_name';")
    if not cur.fetchone():
        print("Adding `broker_name` column to trade_log...")
        cur.execute(f"ALTER TABLE trade_log ADD COLUMN broker_name VARCHAR(64) DEFAULT '{BROKER}';")

    cur.execute("SHOW COLUMNS FROM trade_log LIKE 'session';")
    if not cur.fetchone():
        print("Adding `session` column to trade_log...")
        cur.execute("ALTER TABLE trade_log ADD COLUMN session VARCHAR(64) DEFAULT 'LONDON_OPEN';")

    # 2. Backfill existing records with broker_name and intelligent session name
    cur.execute(f"""
        UPDATE trade_log SET 
            broker_name = '{HISTORICAL_BROKER}'
        WHERE broker_name IS NULL OR broker_name = '' OR broker_name = '{BROKER}';
    """)

    cur.execute("""
        UPDATE trade_log SET 
            session = CASE 
                WHEN CAST(SUBSTRING(timestamp, 12, 2) AS UNSIGNED) BETWEEN 7 AND 11 THEN 'LONDON_OPEN'
                WHEN CAST(SUBSTRING(timestamp, 12, 2) AS UNSIGNED) BETWEEN 12 AND 16 THEN 'NY_AM'
                WHEN CAST(SUBSTRING(timestamp, 12, 2) AS UNSIGNED) BETWEEN 17 AND 21 THEN 'NY_PM'
                ELSE 'ASIA_TOKYO'
            END
        WHERE session IS NULL OR session = '' OR session = 'LONDON_OPEN';
    """)

    # 3. Create dedicated VIEW with the exact column names requested by the user
    view_sql = """
    CREATE OR REPLACE VIEW trades_journal AS
    SELECT 
        id AS `Trade ID`,
        SUBSTRING(timestamp, 12, 8) AS `TIME`,
        SUBSTRING(timestamp, 1, 10) AS `DATE`,
        broker_name AS `BROKER NAME`,
        symbol AS `INSTRUMENTS`,
        realized_pnl AS `NET P&L`,
        strategy_name AS `STRATEGY`,
        session AS `SESSION`
    FROM trade_log
    ORDER BY id DESC;
    """
    cur.execute(view_sql)
    print("✅ Created VIEW `trades_journal` with exact columns requested.")

    # 4. Preview top 5 rows
    cur.execute("SELECT * FROM trades_journal LIMIT 5;")
    rows = cur.fetchall()

    print("\n" + "=" * 95)
    print(f"{'Trade ID':<15} | {'TIME':<10} | {'DATE':<12} | {'BROKER NAME':<16} | {'INSTRUMENTS':<12} | {'NET P&L':<10} | {'STRATEGY':<22} | {'SESSION':<12}")
    print("=" * 95)
    for r in rows:
        pnl = f"${r['NET P&L']:+.2f}" if r['NET P&L'] is not None else "$0.00"
        print(f"{r['Trade ID']:<15} | {str(r['TIME']):<10} | {str(r['DATE']):<12} | {str(r['BROKER NAME']):<16} | {str(r['INSTRUMENTS']):<12} | {pnl:<10} | {str(r['STRATEGY'])[:22]:<22} | {str(r['SESSION']):<12}")
    print("=" * 95)

    conn.close()

if __name__ == "__main__":
    setup()

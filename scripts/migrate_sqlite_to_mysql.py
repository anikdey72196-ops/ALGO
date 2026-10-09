"""
scripts/migrate_sqlite_to_mysql.py — Automated Migration from SQLite to MySQL.
Transfers all tables and data from trading_state.db into the configured MySQL database.
"""

import os
import sys
import sqlite3
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
SQLITE_PATH = "trading_state.db"

TABLE_SCHEMAS = {
    "daily_state": """
        CREATE TABLE IF NOT EXISTS daily_state (
            date VARCHAR(32) PRIMARY KEY,
            realized_pnl DOUBLE DEFAULT 0.0,
            trade_count INT DEFAULT 0,
            circuit_breaker_active INT DEFAULT 0
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "trade_log": """
        CREATE TABLE IF NOT EXISTS trade_log (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            timestamp VARCHAR(64),
            symbol VARCHAR(32),
            direction VARCHAR(16),
            entry_price DOUBLE,
            stop_loss DOUBLE,
            take_profit DOUBLE,
            lot_size DOUBLE,
            realized_pnl DOUBLE DEFAULT 0.0,
            status VARCHAR(32) DEFAULT 'OPEN',
            strategy_name VARCHAR(64) DEFAULT 'SMC',
            magic_number BIGINT DEFAULT 123456,
            closed_at VARCHAR(64) NULL,
            duration_seconds DOUBLE DEFAULT 0.0,
            account_id BIGINT NULL,
            broker_name VARCHAR(64) DEFAULT 'XMGlobal-MT5 6',
            session VARCHAR(64) DEFAULT 'LONDON_OPEN',
            INDEX idx_trade_log_status (status),
            INDEX idx_trade_log_symbol (symbol),
            INDEX idx_trade_log_timestamp (timestamp)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "bot_sessions": """
        CREATE TABLE IF NOT EXISTS bot_sessions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            activation_time VARCHAR(64) NOT NULL,
            deactivation_time VARCHAR(64) NULL,
            duration_seconds DOUBLE NULL,
            symbols TEXT NULL,
            lot_size VARCHAR(64) NULL,
            trigger_source VARCHAR(64) DEFAULT 'Web Dashboard',
            deactivation_reason TEXT NULL,
            status VARCHAR(32) DEFAULT 'ACTIVE',
            INDEX idx_bot_sessions_status (status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "ml_events": """
        CREATE TABLE IF NOT EXISTS ml_events (
            event_id VARCHAR(128) PRIMARY KEY,
            ts VARCHAR(64) NOT NULL,
            symbol VARCHAR(32) NOT NULL,
            timeframe VARCHAR(16) NOT NULL,
            kind VARCHAR(32) NOT NULL,
            direction VARCHAR(16) NOT NULL,
            entry DOUBLE NOT NULL,
            stop DOUBLE NOT NULL,
            target DOUBLE NOT NULL,
            features LONGTEXT NOT NULL,
            label INT NULL,
            outcome VARCHAR(32) NULL,
            r_multiple DOUBLE NULL,
            label_ts VARCHAR(64) NULL,
            p_genuine DOUBLE NULL,
            model_version VARCHAR(64) NULL,
            allowed INT NULL,
            strategy VARCHAR(64) NOT NULL DEFAULT 'SMC',
            INDEX idx_ml_events_sym (symbol),
            INDEX idx_ml_events_ts (ts)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "execution_orders": """
        CREATE TABLE IF NOT EXISTS execution_orders (
            order_id VARCHAR(128) PRIMARY KEY,
            trade_id INT NULL,
            symbol VARCHAR(32) NOT NULL,
            strategy_name VARCHAR(64) NOT NULL,
            magic_number BIGINT NOT NULL,
            direction VARCHAR(16) NOT NULL,
            order_type VARCHAR(32) NOT NULL,
            filling_mode VARCHAR(32) DEFAULT 'FOK',
            requested_lot DOUBLE NOT NULL,
            filled_lot DOUBLE DEFAULT 0.0,
            requested_price DOUBLE NOT NULL,
            filled_price DOUBLE NULL,
            stop_loss DOUBLE NULL,
            take_profit DOUBLE NULL,
            atr_14 DOUBLE NULL,
            slippage_points DOUBLE NULL,
            slippage_pips DOUBLE NULL,
            slippage_pct_atr DOUBLE NULL,
            spread_at_signal DOUBLE NULL,
            spread_at_submit DOUBLE NULL,
            spread_at_fill DOUBLE NULL,
            signal_time VARCHAR(64) NOT NULL,
            submit_time VARCHAR(64) NULL,
            ack_time VARCHAR(64) NULL,
            fill_time VARCHAR(64) NULL,
            close_time VARCHAR(64) NULL,
            latency_signal_to_submit_ms DOUBLE NULL,
            latency_submit_to_ack_ms DOUBLE NULL,
            latency_ack_to_fill_ms DOUBLE NULL,
            latency_total_ms DOUBLE NULL,
            retries_used INT DEFAULT 0,
            rejection_code INT NULL,
            rejection_reason TEXT NULL,
            status VARCHAR(32) NOT NULL,
            session_killzone VARCHAR(64) NULL,
            ml_p_tp DOUBLE NULL,
            ml_p_sl DOUBLE NULL,
            ai_conviction DOUBLE NULL,
            conflict_score DOUBLE NULL,
            risk_pct DOUBLE NULL,
            risk_amount DOUBLE NULL,
            account_equity DOUBLE NULL,
            broker_name VARCHAR(64) DEFAULT 'MetaTrader 5',
            created_at VARCHAR(64) NULL,
            INDEX idx_exec_orders_symbol (symbol),
            INDEX idx_exec_orders_strat (strategy_name),
            INDEX idx_exec_orders_status (status),
            INDEX idx_exec_orders_signal_time (signal_time)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "execution_fills": """
        CREATE TABLE IF NOT EXISTS execution_fills (
            fill_id VARCHAR(128) PRIMARY KEY,
            order_id VARCHAR(128) NOT NULL,
            trade_id INT NULL,
            deal_ticket BIGINT NULL,
            fill_type VARCHAR(32) NOT NULL,
            volume DOUBLE NOT NULL,
            intended_price DOUBLE NOT NULL,
            actual_price DOUBLE NOT NULL,
            slippage_pips DOUBLE NOT NULL,
            slippage_pct_atr DOUBLE NULL,
            commission DOUBLE DEFAULT 0.0,
            swap DOUBLE DEFAULT 0.0,
            fill_time VARCHAR(64) NOT NULL,
            INDEX idx_exec_fills_order_id (order_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "execution_aggregates_daily": """
        CREATE TABLE IF NOT EXISTS execution_aggregates_daily (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date VARCHAR(32) NOT NULL,
            dimension_type VARCHAR(32) NOT NULL,
            dimension_value VARCHAR(64) NOT NULL,
            total_orders INT DEFAULT 0,
            filled_orders INT DEFAULT 0,
            rejected_orders INT DEFAULT 0,
            fill_rate DOUBLE DEFAULT 0.0,
            avg_slippage_pips DOUBLE DEFAULT 0.0,
            max_slippage_pips DOUBLE DEFAULT 0.0,
            min_slippage_pips DOUBLE DEFAULT 0.0,
            avg_slippage_pct_atr DOUBLE DEFAULT 0.0,
            p50_latency_ms DOUBLE DEFAULT 0.0,
            p95_latency_ms DOUBLE DEFAULT 0.0,
            p99_latency_ms DOUBLE DEFAULT 0.0,
            avg_total_latency_ms DOUBLE DEFAULT 0.0,
            avg_spread_pips DOUBLE DEFAULT 0.0,
            total_volume_lots DOUBLE DEFAULT 0.0,
            updated_at VARCHAR(64) NULL,
            UNIQUE KEY uq_date_dim (date, dimension_type, dimension_value)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "partial_tp_events": """
        CREATE TABLE IF NOT EXISTS partial_tp_events (
            id INT AUTO_INCREMENT PRIMARY KEY,
            trade_id INT NOT NULL,
            symbol VARCHAR(32) NOT NULL,
            strategy_name VARCHAR(64) NOT NULL,
            direction VARCHAR(16) NOT NULL,
            timestamp VARCHAR(64) NOT NULL,
            current_price DOUBLE NOT NULL,
            r_multiple DOUBLE NOT NULL,
            p_reversal DOUBLE NOT NULL,
            p_full_tp DOUBLE NOT NULL,
            predicted_max_r DOUBLE NULL,
            action VARCHAR(32) NOT NULL,
            close_pct DOUBLE NOT NULL,
            closed_lot DOUBLE DEFAULT 0.0,
            remaining_lot DOUBLE NOT NULL,
            nearest_resistance VARCHAR(128) NULL,
            structure_confluence_count INT DEFAULT 0,
            features_json LONGTEXT NULL,
            realized_max_r DOUBLE NULL,
            outcome_label VARCHAR(64) NULL,
            is_shadow INT DEFAULT 0,
            INDEX idx_ptp_trade_id (trade_id),
            INDEX idx_ptp_timestamp (timestamp)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "daily_bias_log": """
        CREATE TABLE IF NOT EXISTS daily_bias_log (
            date VARCHAR(32) NOT NULL,
            symbol VARCHAR(32) NOT NULL,
            bias VARCHAR(32) NOT NULL,
            trend_clarity DOUBLE DEFAULT 0.0,
            ema_value DOUBLE DEFAULT 0.0,
            current_price DOUBLE DEFAULT 0.0,
            dealing_range_low DOUBLE NULL,
            dealing_range_high DOUBLE NULL,
            fib_50 DOUBLE NULL,
            is_discount INT DEFAULT 0,
            is_premium INT DEFAULT 0,
            zone_status VARCHAR(64) DEFAULT 'EQUILIBRIUM',
            first_zone_type VARCHAR(64) DEFAULT 'NONE',
            all_zones_failed INT DEFAULT 0,
            reversal_risk VARCHAR(32) DEFAULT 'LOW',
            choch_detected INT DEFAULT 0,
            summary TEXT NULL,
            updated_at VARCHAR(64) NULL,
            PRIMARY KEY (date, symbol)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
}

def migrate():
    print("=" * 60)
    print("  🚀 STARTING SQLITE -> MYSQL MIGRATION FOR ALGO TRADING")
    print("=" * 60)
    print(f"Connecting to MySQL at {DB_USER}@{DB_HOST}:{DB_PORT} / {DB_NAME}...")

    my_conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        autocommit=True,
    )
    my_cur = my_conn.cursor()

    if not os.path.exists(SQLITE_PATH):
        print(f"❌ SQLite file not found: {SQLITE_PATH}")
        return

    sq_conn = sqlite3.connect(SQLITE_PATH)
    sq_conn.row_factory = sqlite3.Row
    sq_cur = sq_conn.cursor()

    total_migrated = 0

    for table_name, create_sql in TABLE_SCHEMAS.items():
        print(f"\n📦 Processing table: `{table_name}`...")
        my_cur.execute(create_sql)

        # Fetch columns from SQLite table
        try:
            sq_cur.execute(f"PRAGMA table_info({table_name});")
            col_info = sq_cur.fetchall()
            if not col_info:
                print(f"  ⚠️ Table `{table_name}` does not exist in SQLite. Skipping data copy.")
                continue
            cols = [c[1] for c in col_info]
        except Exception as e:
            print(f"  ⚠️ Could not read info for `{table_name}`: {e}")
            continue

        # Fetch rows from SQLite
        sq_cur.execute(f"SELECT * FROM {table_name}")
        rows = sq_cur.fetchall()
        row_count = len(rows)

        if row_count == 0:
            print(f"  ℹ️ `{table_name}` has 0 rows in SQLite. Table structure created.")
            continue

        # Prepare batch insert with ON DUPLICATE KEY UPDATE / INSERT IGNORE
        col_list = ", ".join(f"`{c}`" for c in cols)
        placeholders = ", ".join(["%s"] * len(cols))
        insert_sql = f"INSERT IGNORE INTO `{table_name}` ({col_list}) VALUES ({placeholders})"

        batch_data = []
        for r in rows:
            batch_data.append(tuple(r[c] for c in cols))

        # Insert in chunks of 500
        chunk_size = 500
        inserted_for_table = 0
        for i in range(0, len(batch_data), chunk_size):
            chunk = batch_data[i:i + chunk_size]
            my_cur.executemany(insert_sql, chunk)
            inserted_for_table += len(chunk)

        print(f"  ✅ Migrated {inserted_for_table} rows into `{table_name}`.")
        total_migrated += inserted_for_table

    sq_conn.close()
    my_conn.close()

    print("\n" + "=" * 60)
    print(f"  🎉 MIGRATION COMPLETE! Successfully transferred {total_migrated:,} records.")
    print("=" * 60)

if __name__ == "__main__":
    migrate()

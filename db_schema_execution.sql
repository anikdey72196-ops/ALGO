-- ============================================================================
-- EXECUTION QUALITY METRICS SCHEMA
-- Compatible with SQLite (trading_state.db) and PostgreSQL / TimescaleDB
-- ============================================================================

-- 1. Main Execution Orders Table (One row per order lifecycle)
CREATE TABLE IF NOT EXISTS execution_orders (
    order_id TEXT PRIMARY KEY,
    trade_id INTEGER,                              -- Foreign link to trade_log.id
    symbol TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    magic_number INTEGER NOT NULL,
    direction TEXT NOT NULL,
    order_type TEXT NOT NULL,                       -- MARKET, BRACKET_BUY, BRACKET_SELL
    filling_mode TEXT DEFAULT 'FOK',                -- FOK, IOC, RETURN
    
    -- Volume & Prices
    requested_lot REAL NOT NULL,
    filled_lot REAL DEFAULT 0.0,
    requested_price REAL NOT NULL,
    filled_price REAL,
    stop_loss REAL,
    take_profit REAL,
    atr_14 REAL,
    
    -- Slippage Measurements
    slippage_points REAL,                           -- (filled_price - requested_price) * sign
    slippage_pips REAL,
    slippage_pct_atr REAL,                          -- (slippage / ATR) * 100
    
    -- Spread Context
    spread_at_signal REAL,
    spread_at_submit REAL,
    spread_at_fill REAL,
    
    -- Timestamps (ISO-8601 UTC)
    signal_time TEXT NOT NULL,
    submit_time TEXT,
    ack_time TEXT,
    fill_time TEXT,
    close_time TEXT,
    
    -- Latencies (milliseconds with microsecond precision)
    latency_signal_to_submit_ms REAL,               -- T1 - T0
    latency_submit_to_ack_ms REAL,                  -- T2 - T1
    latency_ack_to_fill_ms REAL,                    -- T3 - T2
    latency_total_ms REAL,                          -- T3 - T0
    
    -- Rejections & Retries
    retries_used INTEGER DEFAULT 0,
    rejection_code INTEGER,
    rejection_reason TEXT,
    status TEXT NOT NULL,                           -- FILLED, REJECTED, PARTIAL, CANCELLED
    
    -- Strategy & Intelligence Metadata
    session_killzone TEXT,                          -- LONDON_OPEN, NY_AM, ASIA, OFF_HOURS
    ml_p_tp REAL,                                   -- ML Genuine Setup Probability
    ml_p_sl REAL,                                   -- ML Stop Loss Probability
    ai_conviction REAL,                             -- AI Sentinel Confidence (0-100)
    conflict_score REAL,                            -- ConflictResolver Score
    risk_pct REAL,                                  -- Equity Risk %
    risk_amount REAL,                               -- Account Risk $
    account_equity REAL,
    broker_name TEXT DEFAULT 'MetaTrader 5',
    created_at TEXT DEFAULT (datetime('now'))
);

-- 2. Execution Fills Table (Supports partial fills & exit executions)
CREATE TABLE IF NOT EXISTS execution_fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    trade_id INTEGER,
    deal_ticket INTEGER,
    fill_type TEXT NOT NULL,                        -- ENTRY, PARTIAL_EXIT, SL_EXIT, TP_EXIT, MANUAL_EXIT
    volume REAL NOT NULL,
    intended_price REAL NOT NULL,
    actual_price REAL NOT NULL,
    slippage_pips REAL NOT NULL,
    slippage_pct_atr REAL,
    commission REAL DEFAULT 0.0,
    swap REAL DEFAULT 0.0,
    fill_time TEXT NOT NULL,
    FOREIGN KEY(order_id) REFERENCES execution_orders(order_id)
);

-- 3. Execution Daily & Rolling Aggregates Table
CREATE TABLE IF NOT EXISTS execution_aggregates_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    dimension_type TEXT NOT NULL,                   -- OVERALL, SYMBOL, STRATEGY, KILLZONE, BROKER
    dimension_value TEXT NOT NULL,                  -- e.g. 'XAUUSD', 'SMC Swing', 'LONDON_OPEN'
    
    total_orders INTEGER DEFAULT 0,
    filled_orders INTEGER DEFAULT 0,
    rejected_orders INTEGER DEFAULT 0,
    fill_rate REAL DEFAULT 0.0,
    
    avg_slippage_pips REAL DEFAULT 0.0,
    max_slippage_pips REAL DEFAULT 0.0,
    min_slippage_pips REAL DEFAULT 0.0,
    avg_slippage_pct_atr REAL DEFAULT 0.0,
    
    p50_latency_ms REAL DEFAULT 0.0,
    p95_latency_ms REAL DEFAULT 0.0,
    p99_latency_ms REAL DEFAULT 0.0,
    avg_total_latency_ms REAL DEFAULT 0.0,
    
    avg_spread_pips REAL DEFAULT 0.0,
    total_volume_lots REAL DEFAULT 0.0,
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(date, dimension_type, dimension_value)
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_exec_orders_symbol ON execution_orders(symbol);
CREATE INDEX IF NOT EXISTS idx_exec_orders_strat ON execution_orders(strategy_name);
CREATE INDEX IF NOT EXISTS idx_exec_orders_status ON execution_orders(status);
CREATE INDEX IF NOT EXISTS idx_exec_orders_signal_time ON execution_orders(signal_time);
CREATE INDEX IF NOT EXISTS idx_exec_orders_trade_id ON execution_orders(trade_id);
CREATE INDEX IF NOT EXISTS idx_exec_fills_order_id ON execution_fills(order_id);
CREATE INDEX IF NOT EXISTS idx_exec_agg_lookup ON execution_aggregates_daily(date, dimension_type, dimension_value);

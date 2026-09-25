from __future__ import annotations
import sqlite3
import csv
import threading
from datetime import datetime, date, timezone
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from core.config import Direction


@dataclass
class TradeRecord:
    """Record of an executed trade."""
    id: int | None
    timestamp: datetime
    symbol: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    realized_pnl: float  # 0.0 while open
    status: str  # 'OPEN', 'CLOSED_TP', 'CLOSED_SL', 'CLOSED_MANUAL'
    strategy_name: str = "SMC"
    magic_number: int = 123456
    closed_at: datetime | None = None
    duration_seconds: float = 0.0


@dataclass
class BotSessionRecord:
    """Record of a bot activation / deactivation session."""
    id: int | None
    activation_time: datetime
    deactivation_time: datetime | None
    duration_seconds: float | None
    symbols: str
    lot_size: str
    trigger_source: str
    deactivation_reason: str | None
    status: str  # 'ACTIVE', 'COMPLETED', 'INTERRUPTED'


def normalize_strategy_display_name(name: str | None) -> str:
    """Map internal IDs, ticker suffixes, and legacy string variants to standard UI display names."""
    if not name:
        return "SMC Swing"
    s = str(name).strip()
    u = s.upper()
    if "FLOW" in u or "DELTA" in u or "ORDER_FLOW" in u or "ABSORPTION" in u:
        return "Order Flow Engine"
    elif "SCALP" in u or "5M" in u:
        return "5M Scalp"
    elif "ICT" in u:
        return "ICT Institutional"
    elif "SWING" in u or "SMC" in u:
        return "SMC Swing"
    return s


class StateManager:
    """Persistent state backed by SQLite and automatic CSV file export."""
    
    def __init__(
        self,
        db_path: str = 'trading_state.db',
        csv_path: str = 'trades_history.csv',
        sessions_csv_path: str = 'sessions_history.csv',
    ):
        """Initialize DB connection, create tables, and sync CSV files."""
        self.db_path = db_path
        self.csv_path = csv_path
        self.sessions_csv_path = sessions_csv_path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self._init_schema()
        self._ensure_daily_row(datetime.now(timezone.utc).date())
        self.cleanup_interrupted_sessions()
        self._sync_trades_csv()
        self._sync_sessions_csv()
    
    def _init_schema(self) -> None:
        """Create tables: daily_state, trade_log, bot_sessions with migrations."""
        with self._lock, self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_state (
                    date TEXT PRIMARY KEY,
                    realized_pnl REAL DEFAULT 0.0,
                    trade_count INTEGER DEFAULT 0,
                    circuit_breaker_active INTEGER DEFAULT 0
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    lot_size REAL,
                    realized_pnl REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'OPEN',
                    strategy_name TEXT DEFAULT 'SMC',
                    magic_number INTEGER DEFAULT 123456
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS bot_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    activation_time TEXT NOT NULL,
                    deactivation_time TEXT,
                    duration_seconds REAL,
                    symbols TEXT,
                    lot_size TEXT,
                    trigger_source TEXT DEFAULT 'Web Dashboard',
                    deactivation_reason TEXT,
                    status TEXT DEFAULT 'ACTIVE'
                )
            ''')

            # Ensure migrations for existing databases
            try:
                self.conn.execute("ALTER TABLE trade_log ADD COLUMN strategy_name TEXT DEFAULT 'SMC'")
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute("ALTER TABLE trade_log ADD COLUMN magic_number INTEGER DEFAULT 123456")
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute("ALTER TABLE trade_log ADD COLUMN closed_at TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                self.conn.execute("ALTER TABLE trade_log ADD COLUMN duration_seconds REAL DEFAULT 0.0")
            except sqlite3.OperationalError:
                pass

            # Execution Quality Metrics Tables
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS execution_orders (
                    order_id TEXT PRIMARY KEY,
                    trade_id INTEGER,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    magic_number INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    filling_mode TEXT DEFAULT 'FOK',
                    requested_lot REAL NOT NULL,
                    filled_lot REAL DEFAULT 0.0,
                    requested_price REAL NOT NULL,
                    filled_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    atr_14 REAL,
                    slippage_points REAL,
                    slippage_pips REAL,
                    slippage_pct_atr REAL,
                    spread_at_signal REAL,
                    spread_at_submit REAL,
                    spread_at_fill REAL,
                    signal_time TEXT NOT NULL,
                    submit_time TEXT,
                    ack_time TEXT,
                    fill_time TEXT,
                    close_time TEXT,
                    latency_signal_to_submit_ms REAL,
                    latency_submit_to_ack_ms REAL,
                    latency_ack_to_fill_ms REAL,
                    latency_total_ms REAL,
                    retries_used INTEGER DEFAULT 0,
                    rejection_code INTEGER,
                    rejection_reason TEXT,
                    status TEXT NOT NULL,
                    session_killzone TEXT,
                    ml_p_tp REAL,
                    ml_p_sl REAL,
                    ai_conviction REAL,
                    conflict_score REAL,
                    risk_pct REAL,
                    risk_amount REAL,
                    account_equity REAL,
                    broker_name TEXT DEFAULT 'MetaTrader 5',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS execution_fills (
                    fill_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    trade_id INTEGER,
                    deal_ticket INTEGER,
                    fill_type TEXT NOT NULL,
                    volume REAL NOT NULL,
                    intended_price REAL NOT NULL,
                    actual_price REAL NOT NULL,
                    slippage_pips REAL NOT NULL,
                    slippage_pct_atr REAL,
                    commission REAL DEFAULT 0.0,
                    swap REAL DEFAULT 0.0,
                    fill_time TEXT NOT NULL
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS execution_aggregates_daily (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    dimension_type TEXT NOT NULL,
                    dimension_value TEXT NOT NULL,
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
                )
            ''')

            # Performance Indexes
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_log_status ON trade_log (status)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_log_symbol ON trade_log (symbol)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_log_timestamp ON trade_log (timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_bot_sessions_status ON bot_sessions (status)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_orders_symbol ON execution_orders(symbol)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_orders_strat ON execution_orders(strategy_name)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_orders_status ON execution_orders(status)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_orders_signal_time ON execution_orders(signal_time)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_orders_trade_id ON execution_orders(trade_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_exec_fills_order_id ON execution_fills(order_id)")

            # Auto-close any orphaned mock test trade IDs left over from automated tests
            self.conn.execute("UPDATE trade_log SET status = 'CLOSED_TEST' WHERE status = 'OPEN' AND id >= 9000 AND id <= 9999")

        logger.info(f"Database schema initialized at {self.db_path}")
    
    def _ensure_daily_row(self, today: date) -> None:
        """Insert a row for today if not already present."""
        date_str = today.isoformat()
        with self._lock, self.conn:
            self.conn.execute('''
                INSERT OR IGNORE INTO daily_state (date, realized_pnl, trade_count, circuit_breaker_active)
                VALUES (?, 0.0, 0, 0)
            ''', (date_str,))
        logger.debug(f"Ensured daily row exists for {date_str}")
    
    def record_trade(self, trade: TradeRecord) -> int:
        """Insert a trade record. Returns the row id."""
        date_str = trade.timestamp.date().isoformat()
        self._ensure_daily_row(trade.timestamp.date())
        
        with self._lock, self.conn:
            if trade.id is not None:
                cursor = self.conn.execute('''
                    INSERT INTO trade_log (
                        id, timestamp, symbol, direction, entry_price, 
                        stop_loss, take_profit, lot_size, realized_pnl, status,
                        strategy_name, magic_number
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.id,
                    trade.timestamp.isoformat(),
                    trade.symbol,
                    trade.direction.name if hasattr(trade.direction, 'name') else str(trade.direction),
                    trade.entry_price,
                    trade.stop_loss,
                    trade.take_profit,
                    trade.lot_size,
                    trade.realized_pnl,
                    trade.status,
                    trade.strategy_name,
                    trade.magic_number,
                ))
                trade_id = trade.id
            else:
                cursor = self.conn.execute('''
                    INSERT INTO trade_log (
                        timestamp, symbol, direction, entry_price, 
                        stop_loss, take_profit, lot_size, realized_pnl, status,
                        strategy_name, magic_number
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.timestamp.isoformat(),
                    trade.symbol,
                    trade.direction.name if hasattr(trade.direction, 'name') else str(trade.direction),
                    trade.entry_price,
                    trade.stop_loss,
                    trade.take_profit,
                    trade.lot_size,
                    trade.realized_pnl,
                    trade.status,
                    trade.strategy_name,
                    trade.magic_number,
                ))
                trade_id = cursor.lastrowid
                trade.id = trade_id
            
            # Update daily state
            self.conn.execute('''
                UPDATE daily_state 
                SET trade_count = trade_count + 1,
                    realized_pnl = realized_pnl + ?
                WHERE date = ?
            ''', (trade.realized_pnl, date_str))
            
        logger.info(f"Recorded trade {trade_id} ({trade.strategy_name}) for {trade.symbol} at {trade.entry_price}")
        self._sync_trades_csv()
        return trade_id

    
    def update_trade_pnl(self, trade_id: int, pnl: float, status: str) -> None:
        """Update a trade's realized PnL and status, recording close timestamp and duration."""
        with self._lock, self.conn:
            # Get existing pnl to calculate diff for daily_state update
            cursor = self.conn.execute('SELECT timestamp, realized_pnl FROM trade_log WHERE id = ?', (trade_id,))
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Trade {trade_id} not found for PnL update.")
                return
            
            old_pnl = float(row['realized_pnl'] or 0.0)
            trade_timestamp = datetime.fromisoformat(row['timestamp'])
            trade_date_str = trade_timestamp.date().isoformat()
            
            now_dt = datetime.now(timezone.utc)
            close_date = now_dt.date()
            close_date_str = close_date.isoformat()
            self._ensure_daily_row(close_date)

            if trade_timestamp.tzinfo is None:
                trade_dt = trade_timestamp.replace(tzinfo=timezone.utc)
            else:
                trade_dt = trade_timestamp
            duration_sec = max(0.0, (now_dt - trade_dt).total_seconds())

            self.conn.execute('''
                UPDATE trade_log
                SET realized_pnl = ?, status = ?, closed_at = ?, duration_seconds = ?
                WHERE id = ?
            ''', (pnl, status, now_dt.isoformat(), duration_sec, trade_id))
            
            # Credit realized PnL to the actual close date (today)
            if trade_date_str == close_date_str:
                pnl_diff = pnl - old_pnl
                if pnl_diff != 0:
                    self.conn.execute('''
                        UPDATE daily_state
                        SET realized_pnl = realized_pnl + ?
                        WHERE date = ?
                    ''', (pnl_diff, close_date_str))
            else:
                # Multi-day trade: remove old open-date pnl if any, and record realized pnl on close date
                if old_pnl != 0:
                    self.conn.execute('''
                        UPDATE daily_state
                        SET realized_pnl = realized_pnl - ?
                        WHERE date = ?
                    ''', (old_pnl, trade_date_str))
                self.conn.execute('''
                    UPDATE daily_state
                    SET realized_pnl = realized_pnl + ?
                    WHERE date = ?
                ''', (pnl, close_date_str))
                
        logger.info(f"Updated trade {trade_id}: PnL={pnl}, status={status}, duration={format_duration(duration_sec)}")
        self._sync_trades_csv()

    def _sync_trades_csv(self) -> None:
        """Export current trade_log table into trades_history.csv."""
        try:
            trades = self.get_all_trades(limit=5000)
            fieldnames = [
                "id", "timestamp", "closed_at", "symbol", "direction", "entry_price",
                "stop_loss", "take_profit", "lot_size", "realized_pnl",
                "status", "duration_seconds", "strategy_name", "magic_number"
            ]
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for t in trades:
                    writer.writerow({
                        "id": t.id,
                        "timestamp": t.timestamp.isoformat(),
                        "closed_at": t.closed_at.isoformat() if t.closed_at else "",
                        "symbol": t.symbol,
                        "direction": t.direction.value if hasattr(t.direction, 'value') else str(t.direction),
                        "entry_price": t.entry_price,
                        "stop_loss": t.stop_loss,
                        "take_profit": t.take_profit,
                        "lot_size": t.lot_size,
                        "realized_pnl": t.realized_pnl,
                        "status": t.status,
                        "duration_seconds": t.duration_seconds,
                        "strategy_name": t.strategy_name,
                        "magic_number": t.magic_number,
                    })
        except Exception as e:
            logger.error(f"Failed to sync trades CSV: {e}")

    def _sync_sessions_csv(self) -> None:
        """Export current bot_sessions table into sessions_history.csv."""
        try:
            sessions = self.get_activation_history(limit=5000)
            fieldnames = [
                "id", "activation_time", "deactivation_time", "formatted_duration",
                "duration_seconds", "symbols", "lot_size", "trigger_source",
                "deactivation_reason", "status"
            ]
            with open(self.sessions_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for s in sessions:
                    writer.writerow({
                        "id": s.get("id"),
                        "activation_time": s.get("activation_time"),
                        "deactivation_time": s.get("deactivation_time"),
                        "formatted_duration": s.get("formatted_duration"),
                        "duration_seconds": s.get("duration_seconds"),
                        "symbols": s.get("symbols"),
                        "lot_size": s.get("lot_size"),
                        "trigger_source": s.get("trigger_source"),
                        "deactivation_reason": s.get("deactivation_reason"),
                        "status": s.get("status"),
                    })
        except Exception as e:
            logger.error(f"Failed to sync sessions CSV: {e}")
    
    def get_daily_pnl(self, today: date | None = None) -> float:
        """Sum of realized PnL for today (UTC)."""
        if today is None:
            today = datetime.now(timezone.utc).date()
        date_str = today.isoformat()
        
        with self._lock:
            cursor = self.conn.execute('SELECT realized_pnl FROM daily_state WHERE date = ?', (date_str,))
            row = cursor.fetchone()
        return float(row['realized_pnl']) if row else 0.0
    
    def get_trade_count(self, today: date | None = None) -> int:
        """Number of trades executed today (UTC)."""
        if today is None:
            today = datetime.now(timezone.utc).date()
        date_str = today.isoformat()
        
        with self._lock:
            cursor = self.conn.execute('SELECT trade_count FROM daily_state WHERE date = ?', (date_str,))
            row = cursor.fetchone()
        return int(row['trade_count']) if row else 0
    
    def is_circuit_breaker_active(self, today: date | None = None) -> bool:
        """Check if the circuit breaker flag is set for today."""
        if today is None:
            today = datetime.now(timezone.utc).date()
        date_str = today.isoformat()
        
        with self._lock:
            cursor = self.conn.execute('SELECT circuit_breaker_active FROM daily_state WHERE date = ?', (date_str,))
            row = cursor.fetchone()
        return bool(row['circuit_breaker_active']) if row else False
    
    def activate_circuit_breaker(self, today: date | None = None) -> None:
        """Set the circuit breaker flag for today."""
        if today is None:
            today = datetime.now(timezone.utc).date()
        date_str = today.isoformat()
        self._ensure_daily_row(today)
        
        with self._lock, self.conn:
            self.conn.execute('''
                UPDATE daily_state
                SET circuit_breaker_active = 1
                WHERE date = ?
            ''', (date_str,))
        logger.warning(f"Circuit breaker activated for {date_str}")
    
    def reset_daily_state(self) -> None:
        """Called at 00:00 UTC to start fresh. Insert new daily row."""
        today = datetime.now(timezone.utc).date()
        self._ensure_daily_row(today)
        logger.info(f"Daily state reset for {today.isoformat()}")
    
    def get_open_positions(self) -> list[TradeRecord]:
        """Return all trades with status='OPEN'."""
        with self._lock:
            cursor = self.conn.execute('''
                SELECT id, timestamp, symbol, direction, entry_price, 
                       stop_loss, take_profit, lot_size, realized_pnl, status,
                       strategy_name, magic_number, closed_at, duration_seconds
                FROM trade_log
                WHERE status = 'OPEN'
            ''')
            rows = cursor.fetchall()
        
        open_trades = []
        for row in rows:
            dir_str = row['direction']
            try:
                direction = Direction[dir_str]
            except KeyError:
                direction = getattr(Direction, dir_str.upper(), dir_str)
                
            trade = TradeRecord(
                id=row['id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                symbol=row['symbol'],
                direction=direction,
                entry_price=row['entry_price'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                lot_size=row['lot_size'],
                realized_pnl=row['realized_pnl'],
                status=row['status'],
                strategy_name=row['strategy_name'] if 'strategy_name' in row.keys() and row['strategy_name'] else "SMC",
                magic_number=row['magic_number'] if 'magic_number' in row.keys() and row['magic_number'] else 123456,
                closed_at=datetime.fromisoformat(row['closed_at']) if 'closed_at' in row.keys() and row['closed_at'] else None,
                duration_seconds=float(row['duration_seconds'] or 0.0) if 'duration_seconds' in row.keys() and row['duration_seconds'] else 0.0,
            )
            open_trades.append(trade)
            
        return open_trades
    
    def get_all_trades(self, limit: int = 100) -> list[TradeRecord]:
        """Return all historical trades ordered by id DESC."""
        with self._lock:
            cursor = self.conn.execute('''
                SELECT id, timestamp, symbol, direction, entry_price, 
                       stop_loss, take_profit, lot_size, realized_pnl, status,
                       strategy_name, magic_number, closed_at, duration_seconds
                FROM trade_log
                ORDER BY id DESC
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
        
        trades = []
        for row in rows:
            dir_str = row['direction']
            try:
                direction = Direction[dir_str]
            except KeyError:
                direction = getattr(Direction, dir_str.upper(), dir_str)
                
            trade = TradeRecord(
                id=row['id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                symbol=row['symbol'],
                direction=direction,
                entry_price=row['entry_price'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                lot_size=row['lot_size'],
                realized_pnl=row['realized_pnl'],
                status=row['status'],
                strategy_name=row['strategy_name'] if 'strategy_name' in row.keys() and row['strategy_name'] else "SMC",
                magic_number=row['magic_number'] if 'magic_number' in row.keys() and row['magic_number'] else 123456,
                closed_at=datetime.fromisoformat(row['closed_at']) if 'closed_at' in row.keys() and row['closed_at'] else None,
                duration_seconds=float(row['duration_seconds'] or 0.0) if 'duration_seconds' in row.keys() and row['duration_seconds'] else 0.0,
            )
            trades.append(trade)
            
        return trades

    def get_stats_by_strategy(self) -> dict[str, dict]:
        """
        Calculate win rate, trade counts, and PnL grouped by strategy.
        Returns: { strategy_name: { 'total': int, 'wins': int, 'losses': int, 'win_rate': float, 'pnl': float } }
        """
        with self._lock:
            cursor = self.conn.execute('''
                SELECT strategy_name, status, realized_pnl
                FROM trade_log
                WHERE status != 'OPEN'
            ''')
            rows = cursor.fetchall()
        
        stats: dict[str, dict] = {}
        for row in rows:
            strat = row['strategy_name'] or "SMC"
            if strat not in stats:
                stats[strat] = {
                    "strategy_name": strat,
                    "total": 0,
                    "total_trades": 0,
                    "wins": 0,
                    "winning_trades": 0,
                    "losses": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "pnl": 0.0,
                    "total_pnl": 0.0,
                }
            
            stats[strat]["total"] += 1
            stats[strat]["total_trades"] += 1
            pnl = float(row['realized_pnl'] or 0.0)
            stats[strat]["pnl"] += pnl
            stats[strat]["total_pnl"] += pnl
            
            if row['status'] == 'CLOSED_TP' or pnl > 0:
                stats[strat]["wins"] += 1
                stats[strat]["winning_trades"] += 1
            elif row['status'] == 'CLOSED_SL' or pnl < 0:
                stats[strat]["losses"] += 1
                stats[strat]["losing_trades"] += 1

        for strat, data in stats.items():
            if data["total"] > 0:
                data["win_rate"] = round((data["wins"] / data["total"]) * 100.0, 1)
                data["pnl"] = round(data["pnl"], 2)
                data["total_pnl"] = round(data["total_pnl"], 2)
                
        return stats

    def get_stats_by_pair(self) -> dict[str, dict]:
        """
        Calculate win rate, trade counts, and PnL grouped by pair / symbol.
        Returns: { symbol: { 'total': int, 'wins': int, 'losses': int, 'win_rate': float, 'pnl': float } }
        """
        with self._lock:
            cursor = self.conn.execute('''
                SELECT symbol, status, realized_pnl
                FROM trade_log
                WHERE status != 'OPEN'
            ''')
            rows = cursor.fetchall()
        
        stats: dict[str, dict] = {}
        for row in rows:
            sym = row['symbol']
            if sym not in stats:
                stats[sym] = {
                    "symbol": sym,
                    "total": 0,
                    "total_trades": 0,
                    "wins": 0,
                    "winning_trades": 0,
                    "losses": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "pnl": 0.0,
                    "total_pnl": 0.0,
                }
            
            stats[sym]["total"] += 1
            stats[sym]["total_trades"] += 1
            pnl = float(row['realized_pnl'] or 0.0)
            stats[sym]["pnl"] += pnl
            stats[sym]["total_pnl"] += pnl
            
            if row['status'] == 'CLOSED_TP' or pnl > 0:
                stats[sym]["wins"] += 1
                stats[sym]["winning_trades"] += 1
            elif row['status'] == 'CLOSED_SL' or pnl < 0:
                stats[sym]["losses"] += 1
                stats[sym]["losing_trades"] += 1

        for sym, data in stats.items():
            if data["total"] > 0:
                data["win_rate"] = round((data["wins"] / data["total"]) * 100.0, 1)
                data["pnl"] = round(data["pnl"], 2)
                data["total_pnl"] = round(data["total_pnl"], 2)
                
        return stats

    def get_performance_metrics(self) -> dict:
        """
        Comprehensive analytics for the dashboard:
        - total_trades, total_open, total_closed
        - total_tp, total_sl, tp_amount, sl_amount
        - win_rate (pct), win_loss_diff (can be negative)
        - avg_planned_rr, realized_rr
        - avg_holding_time, total_holding_time
        - profit_factor, best_trade, worst_trade
        - strategy_breakdown, pair_breakdown
        """
        with self._lock:
            cursor = self.conn.execute('''
                SELECT id, timestamp, symbol, strategy_name, entry_price, stop_loss, take_profit,
                       realized_pnl, status, duration_seconds, closed_at
                FROM trade_log
            ''')
            rows = cursor.fetchall()

        total_trades = len(rows)
        closed_rows = [r for r in rows if r['status'] != 'OPEN']
        open_rows = [r for r in rows if r['status'] == 'OPEN']

        total_closed = len(closed_rows)
        total_open = len(open_rows)

        tp_rows = [r for r in closed_rows if r['status'] == 'CLOSED_TP' or (r['realized_pnl'] and r['realized_pnl'] > 0)]
        sl_rows = [r for r in closed_rows if r['status'] == 'CLOSED_SL' or (r['realized_pnl'] and r['realized_pnl'] < 0)]

        total_tp = len(tp_rows)
        total_sl = len(sl_rows)

        total_tp_pnl = sum(float(r['realized_pnl'] or 0.0) for r in tp_rows)
        total_sl_pnl = sum(float(r['realized_pnl'] or 0.0) for r in sl_rows)
        total_pnl = sum(float(r['realized_pnl'] or 0.0) for r in closed_rows)

        win_rate = round((total_tp / total_closed * 100.0), 1) if total_closed > 0 else 0.0
        win_loss_diff = total_tp - total_sl  # Can be positive or negative

        # Single-pass calculation of by_strategy and by_pair from closed_rows
        by_strat: dict[str, dict] = {}
        by_pair: dict[str, dict] = {}
        for r in closed_rows:
            raw_strat = r['strategy_name'] if ('strategy_name' in r.keys() and r['strategy_name']) else "SMC"
            strat = normalize_strategy_display_name(raw_strat)
            sym = r['symbol'] if ('symbol' in r.keys() and r['symbol']) else "UNKNOWN"
            pnl = float(r['realized_pnl'] or 0.0)
            is_win = (r['status'] == 'CLOSED_TP' or pnl > 0)
            is_loss = (r['status'] == 'CLOSED_SL' or pnl < 0)

            if strat not in by_strat:
                by_strat[strat] = {"strategy_name": strat, "total": 0, "total_trades": 0, "wins": 0, "winning_trades": 0, "losses": 0, "losing_trades": 0, "win_rate": 0.0, "pnl": 0.0, "total_pnl": 0.0}
            by_strat[strat]["total"] += 1
            by_strat[strat]["total_trades"] += 1
            by_strat[strat]["pnl"] += pnl
            by_strat[strat]["total_pnl"] += pnl
            if is_win:
                by_strat[strat]["wins"] += 1
                by_strat[strat]["winning_trades"] += 1
            elif is_loss:
                by_strat[strat]["losses"] += 1
                by_strat[strat]["losing_trades"] += 1

            if sym not in by_pair:
                by_pair[sym] = {"symbol": sym, "total": 0, "total_trades": 0, "wins": 0, "winning_trades": 0, "losses": 0, "losing_trades": 0, "win_rate": 0.0, "pnl": 0.0, "total_pnl": 0.0}
            by_pair[sym]["total"] += 1
            by_pair[sym]["total_trades"] += 1
            by_pair[sym]["pnl"] += pnl
            by_pair[sym]["total_pnl"] += pnl
            if is_win:
                by_pair[sym]["wins"] += 1
                by_pair[sym]["winning_trades"] += 1
            elif is_loss:
                by_pair[sym]["losses"] += 1
                by_pair[sym]["losing_trades"] += 1

        for d in by_strat.values():
            if d["total"] > 0:
                d["win_rate"] = round((d["wins"] / d["total"]) * 100.0, 1)
                d["pnl"] = round(d["pnl"], 2)
                d["total_pnl"] = round(d["total_pnl"], 2)

        for d in by_pair.values():
            if d["total"] > 0:
                d["win_rate"] = round((d["wins"] / d["total"]) * 100.0, 1)
                d["pnl"] = round(d["pnl"], 2)
                d["total_pnl"] = round(d["total_pnl"], 2)

        # Risk-to-reward calculation
        rr_list = []
        for r in rows:
            entry = float(r['entry_price'] or 0.0)
            sl = float(r['stop_loss'] or 0.0)
            tp = float(r['take_profit'] or 0.0)
            sl_dist = abs(entry - sl)
            tp_dist = abs(tp - entry)
            if sl_dist > 0.000001:
                rr_list.append(tp_dist / sl_dist)

        avg_planned_rr = round(sum(rr_list) / len(rr_list), 2) if rr_list else 2.50

        avg_win_pnl = total_tp_pnl / total_tp if total_tp > 0 else 0.0
        avg_loss_pnl = abs(total_sl_pnl / total_sl) if total_sl > 0 else 0.0
        realized_rr = round(avg_win_pnl / avg_loss_pnl, 2) if avg_loss_pnl > 0.0001 else avg_planned_rr

        # Holding duration
        durations = []
        now_dt = datetime.now(timezone.utc)
        for r in closed_rows:
            dur = float(r['duration_seconds'] or 0.0)
            if dur <= 0.0 and r['timestamp']:
                try:
                    t_dt = datetime.fromisoformat(r['timestamp'])
                    if t_dt.tzinfo is None:
                        t_dt = t_dt.replace(tzinfo=timezone.utc)
                    if r['closed_at']:
                        c_dt = datetime.fromisoformat(r['closed_at'])
                        if c_dt.tzinfo is None:
                            c_dt = c_dt.replace(tzinfo=timezone.utc)
                        dur = max(0.0, (c_dt - t_dt).total_seconds())
                    else:
                        dur = max(0.0, (now_dt - t_dt).total_seconds())
                except Exception:
                    dur = 0.0
            if dur > 0.0:
                durations.append(dur)

        total_holding_sec = sum(durations)
        avg_holding_sec = total_holding_sec / len(durations) if durations else 0.0

        profit_factor = round(abs(total_tp_pnl) / max(0.01, abs(total_sl_pnl)), 2) if abs(total_sl_pnl) > 0 else (round(total_tp_pnl, 2) if total_tp_pnl > 0 else 0.0)

        all_pnls = [float(r['realized_pnl'] or 0.0) for r in closed_rows]
        best_trade = round(max(all_pnls), 2) if all_pnls else 0.0
        worst_trade = round(min(all_pnls), 2) if all_pnls else 0.0

        return {
            "total_trades": total_trades,
            "total_closed_trades": total_closed,
            "total_open_trades": total_open,
            "total_tp": total_tp,
            "total_sl": total_sl,
            "winning_trades": total_tp,
            "losing_trades": total_sl,
            "total_tp_pnl": round(total_tp_pnl, 2),
            "total_sl_pnl": round(total_sl_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "win_rate": win_rate,
            "accuracy": win_rate,
            "win_loss_diff": win_loss_diff,
            "win_loss_ratio": round(total_tp / max(1, total_sl), 2),
            "avg_planned_rr": avg_planned_rr,
            "formatted_avg_rr": f"1:{avg_planned_rr:.2f}",
            "realized_rr": realized_rr,
            "formatted_realized_rr": f"1:{realized_rr:.2f}",
            "avg_holding_seconds": round(avg_holding_sec, 1),
            "formatted_avg_holding_time": format_duration(avg_holding_sec),
            "total_holding_seconds": round(total_holding_sec, 1),
            "formatted_total_holding_time": format_duration(total_holding_sec),
            "profit_factor": profit_factor,
            "best_trade_pnl": best_trade,
            "worst_trade_pnl": worst_trade,
            "by_strategy": by_strat,
            "by_pair": by_pair,
        }


    def cleanup_interrupted_sessions(self, reason: str = "Interrupted / Server Restarted") -> None:
        """Mark any lingering 'ACTIVE' sessions as 'INTERRUPTED'."""
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        with self._lock, self.conn:
            cursor = self.conn.execute("SELECT id, activation_time FROM bot_sessions WHERE status = 'ACTIVE'")
            rows = cursor.fetchall()
            for r in rows:
                try:
                    act_time = datetime.fromisoformat(r['activation_time'])
                    if act_time.tzinfo is None:
                        act_time = act_time.replace(tzinfo=timezone.utc)
                    duration_sec = max(0.0, (now - act_time).total_seconds())
                except Exception:
                    duration_sec = 0.0
                self.conn.execute('''
                    UPDATE bot_sessions
                    SET deactivation_time = ?,
                        duration_seconds = ?,
                        deactivation_reason = ?,
                        status = 'INTERRUPTED'
                    WHERE id = ?
                ''', (now_iso, duration_sec, reason, r['id']))
                logger.info(f"Cleaned up stale session #{r['id']} -> INTERRUPTED ({reason})")

    def record_activation(
        self,
        symbols: list[str] | str,
        lot_size: str | float | None = None,
        trigger_source: str = "Web Dashboard",
    ) -> int:
        """
        Record a new bot activation event.
        Closes any currently active dangling session before creating a new one.
        """
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        
        # Clean up any currently ACTIVE sessions
        self.cleanup_interrupted_sessions(reason="Superseded by new activation")
        
        if isinstance(symbols, list):
            symbols_str = ", ".join(s for s in symbols if s and s != "NONE")
        else:
            symbols_str = str(symbols)
            
        lot_size_str = str(lot_size) if lot_size is not None else "Dynamic"
        
        with self._lock, self.conn:
            cursor = self.conn.execute('''
                INSERT INTO bot_sessions (
                    activation_time, symbols, lot_size, trigger_source, status
                ) VALUES (?, ?, ?, ?, 'ACTIVE')
            ''', (now_iso, symbols_str, lot_size_str, trigger_source))
            session_id = cursor.lastrowid
            
        logger.info(f"[ACTIVATED] Recorded bot activation session #{session_id} for {symbols_str} (Trigger: {trigger_source})")
        self._sync_sessions_csv()
        return session_id

    def record_deactivation(
        self,
        reason: str = "Manual User Stop",
        session_id: int | None = None,
    ) -> int | None:
        """
        Record a bot deactivation event.
        Computes elapsed active duration in seconds and completes the session.
        """
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        
        with self._lock, self.conn:
            if session_id is not None:
                cursor = self.conn.execute(
                    "SELECT id, activation_time FROM bot_sessions WHERE id = ? AND status = 'ACTIVE'",
                    (session_id,)
                )
            else:
                cursor = self.conn.execute(
                    "SELECT id, activation_time FROM bot_sessions WHERE status = 'ACTIVE' ORDER BY id DESC LIMIT 1"
                )
            row = cursor.fetchone()
            if not row:
                logger.warning("No active session found to deactivate.")
                return None
                
            active_id = row['id']
            try:
                act_time = datetime.fromisoformat(row['activation_time'])
                if act_time.tzinfo is None:
                    act_time = act_time.replace(tzinfo=timezone.utc)
                duration_sec = max(0.0, (now - act_time).total_seconds())
            except Exception:
                duration_sec = 0.0
                
            self.conn.execute('''
                UPDATE bot_sessions
                SET deactivation_time = ?,
                    duration_seconds = ?,
                    deactivation_reason = ?,
                    status = 'COMPLETED'
                WHERE id = ?
            ''', (now_iso, duration_sec, reason, active_id))
            
        logger.info(f"[DEACTIVATED] Recorded bot deactivation for session #{active_id}. Duration: {format_duration(duration_sec)} ({reason})")
        self._sync_sessions_csv()
        return active_id

    def get_active_session(self) -> dict | None:
        """Return the currently active session if one exists, with live running duration."""
        with self._lock:
            cursor = self.conn.execute(
                "SELECT * FROM bot_sessions WHERE status = 'ACTIVE' ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
        if not row:
            return None
            
        now = datetime.now(timezone.utc)
        try:
            act_time = datetime.fromisoformat(row['activation_time'])
            if act_time.tzinfo is None:
                act_time = act_time.replace(tzinfo=timezone.utc)
            duration_sec = max(0.0, (now - act_time).total_seconds())
        except Exception:
            duration_sec = 0.0
            
        return {
            "id": row["id"],
            "activation_time": row["activation_time"],
            "activation_time_formatted": act_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "deactivation_time": None,
            "deactivation_time_formatted": "Active Now",
            "duration_seconds": round(duration_sec, 1),
            "formatted_duration": format_duration(duration_sec),
            "symbols": row["symbols"] or "All",
            "lot_size": row["lot_size"] or "Dynamic",
            "trigger_source": row["trigger_source"] or "Web Dashboard",
            "deactivation_reason": None,
            "status": "ACTIVE",
        }

    def get_activation_history(self, limit: int = 100) -> list[dict]:
        """Return historical bot activation / deactivation sessions ordered by id DESC."""
        with self._lock:
            cursor = self.conn.execute(
                "SELECT * FROM bot_sessions ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = cursor.fetchall()
        now = datetime.now(timezone.utc)
        records = []
        for row in rows:
            try:
                act_dt = datetime.fromisoformat(row["activation_time"])
                if act_dt.tzinfo is None:
                    act_dt = act_dt.replace(tzinfo=timezone.utc)
                act_str = act_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                act_dt = now
                act_str = row["activation_time"]

            deact_raw = row["deactivation_time"]
            if deact_raw:
                try:
                    deact_dt = datetime.fromisoformat(deact_raw)
                    if deact_dt.tzinfo is None:
                        deact_dt = deact_dt.replace(tzinfo=timezone.utc)
                    deact_str = deact_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                except Exception:
                    deact_str = deact_raw
            else:
                deact_str = "Active Now"

            # Compute live duration for currently ACTIVE session
            if row["status"] == "ACTIVE":
                dur_sec = max(0.0, (now - act_dt).total_seconds())
            else:
                dur_sec = row["duration_seconds"] or 0.0

            records.append({
                "id": row["id"],
                "activation_time": row["activation_time"],
                "activation_time_formatted": act_str,
                "deactivation_time": deact_raw,
                "deactivation_time_formatted": deact_str,
                "duration_seconds": round(dur_sec, 1),
                "formatted_duration": format_duration(dur_sec),
                "symbols": row["symbols"] or "All",
                "lot_size": row["lot_size"] or "Dynamic",
                "trigger_source": row["trigger_source"] or "Web Dashboard",
                "deactivation_reason": row["deactivation_reason"] or ("Active Now" if row["status"] == "ACTIVE" else "---"),
                "status": row["status"],
            })
        return records

    def get_activation_stats(self) -> dict:
        """Return aggregate operational uptime statistics."""
        history = self.get_activation_history(limit=500)
        total_sessions = len(history)
        active_session = self.get_active_session()
        
        total_uptime_seconds = sum(item["duration_seconds"] for item in history)
        completed_sessions = [s for s in history if s["status"] == "COMPLETED"]
        completed_durations = [s["duration_seconds"] for s in completed_sessions]
        avg_duration_sec = (sum(completed_durations) / len(completed_durations)) if completed_durations else 0.0
        
        last_activation = history[0]["activation_time_formatted"] if history else None
        last_deactivation = None
        for s in history:
            if s["status"] != "ACTIVE" and s["deactivation_time"]:
                last_deactivation = s["deactivation_time_formatted"]
                break
                
        return {
            "total_sessions": total_sessions,
            "is_active": active_session is not None,
            "current_session": active_session,
            "total_uptime_seconds": round(total_uptime_seconds, 1),
            "formatted_total_uptime": format_duration(total_uptime_seconds),
            "average_duration_seconds": round(avg_duration_sec, 1),
            "formatted_avg_duration": format_duration(avg_duration_sec),
            "last_activation_time": last_activation,
            "last_deactivation_time": last_deactivation,
        }

    def clear_activation_history(self) -> None:
        """Clear all completed/interrupted session history records (keeps active session if one exists)."""
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM bot_sessions WHERE status != 'ACTIVE'")
        logger.info("Cleared non-active bot session history.")
        self._sync_sessions_csv()

    def clear_trade_history(self) -> None:
        """Clear historical trade records (keeps OPEN positions)."""
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM trade_log WHERE status != 'OPEN'")
            self.conn.execute("UPDATE daily_state SET trade_count = 0, realized_pnl = 0.0")
        logger.info("Cleared closed trade records and reset daily counters.")
        self._sync_trades_csv()

    def close(self) -> None:
        """Close DB connection."""
        with self._lock:
            self.conn.close()
        logger.info("Database connection closed")


def format_duration(seconds: float | None) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds is None:
        return "---"
    sec = int(round(seconds))
    if sec < 60:
        return f"{sec}s"
    minutes = sec // 60
    rem_sec = sec % 60
    if minutes < 60:
        return f"{minutes}m {rem_sec:02d}s"
    hours = minutes // 60
    rem_min = minutes % 60
    return f"{hours}h {rem_min:02d}m {rem_sec:02d}s"

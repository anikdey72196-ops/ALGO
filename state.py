from __future__ import annotations
import sqlite3
from datetime import datetime, date, timezone
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from config import Direction


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


class StateManager:
    """Persistent state backed by SQLite."""
    
    def __init__(self, db_path: str = 'trading_state.db'):
        """Initialize DB connection and create tables if not exist."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        self._ensure_daily_row(datetime.now(timezone.utc).date())
    
    def _init_schema(self) -> None:
        """Create tables: daily_state, trade_log."""
        with self.conn:
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
                    status TEXT DEFAULT 'OPEN'
                )
            ''')
        logger.info(f"Database schema initialized at {self.db_path}")
    
    def _ensure_daily_row(self, today: date) -> None:
        """Insert a row for today if not already present."""
        date_str = today.isoformat()
        with self.conn:
            self.conn.execute('''
                INSERT OR IGNORE INTO daily_state (date, realized_pnl, trade_count, circuit_breaker_active)
                VALUES (?, 0.0, 0, 0)
            ''', (date_str,))
        logger.debug(f"Ensured daily row exists for {date_str}")
    
    def record_trade(self, trade: TradeRecord) -> int:
        """Insert a trade record. Returns the row id."""
        date_str = trade.timestamp.date().isoformat()
        self._ensure_daily_row(trade.timestamp.date())
        
        with self.conn:
            if trade.id is not None:
                cursor = self.conn.execute('''
                    INSERT INTO trade_log (
                        id, timestamp, symbol, direction, entry_price, 
                        stop_loss, take_profit, lot_size, realized_pnl, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    trade.status
                ))
                trade_id = trade.id
            else:
                cursor = self.conn.execute('''
                    INSERT INTO trade_log (
                        timestamp, symbol, direction, entry_price, 
                        stop_loss, take_profit, lot_size, realized_pnl, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.timestamp.isoformat(),
                    trade.symbol,
                    trade.direction.name if hasattr(trade.direction, 'name') else str(trade.direction),
                    trade.entry_price,
                    trade.stop_loss,
                    trade.take_profit,
                    trade.lot_size,
                    trade.realized_pnl,
                    trade.status
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
            
        logger.info(f"Recorded trade {trade_id} for {trade.symbol} at {trade.entry_price}")
        return trade_id
    
    def update_trade_pnl(self, trade_id: int, pnl: float, status: str) -> None:
        """Update a trade's realized PnL and status."""
        with self.conn:
            # Get existing pnl to calculate diff for daily_state update
            cursor = self.conn.execute('SELECT timestamp, realized_pnl FROM trade_log WHERE id = ?', (trade_id,))
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Trade {trade_id} not found for PnL update.")
                return
            
            old_pnl = row['realized_pnl']
            trade_timestamp = datetime.fromisoformat(row['timestamp'])
            date_str = trade_timestamp.date().isoformat()
            
            self.conn.execute('''
                UPDATE trade_log
                SET realized_pnl = ?, status = ?
                WHERE id = ?
            ''', (pnl, status, trade_id))
            
            # Update daily_state
            pnl_diff = pnl - old_pnl
            if pnl_diff != 0:
                self.conn.execute('''
                    UPDATE daily_state
                    SET realized_pnl = realized_pnl + ?
                    WHERE date = ?
                ''', (pnl_diff, date_str))
                
        logger.info(f"Updated trade {trade_id}: PnL={pnl}, status={status}")
    
    def get_daily_pnl(self, today: date | None = None) -> float:
        """Sum of realized PnL for today (UTC)."""
        if today is None:
            today = datetime.now(timezone.utc).date()
        date_str = today.isoformat()
        
        cursor = self.conn.execute('SELECT realized_pnl FROM daily_state WHERE date = ?', (date_str,))
        row = cursor.fetchone()
        return float(row['realized_pnl']) if row else 0.0
    
    def get_trade_count(self, today: date | None = None) -> int:
        """Number of trades executed today (UTC)."""
        if today is None:
            today = datetime.now(timezone.utc).date()
        date_str = today.isoformat()
        
        cursor = self.conn.execute('SELECT trade_count FROM daily_state WHERE date = ?', (date_str,))
        row = cursor.fetchone()
        return int(row['trade_count']) if row else 0
    
    def is_circuit_breaker_active(self, today: date | None = None) -> bool:
        """Check if the circuit breaker flag is set for today."""
        if today is None:
            today = datetime.now(timezone.utc).date()
        date_str = today.isoformat()
        
        cursor = self.conn.execute('SELECT circuit_breaker_active FROM daily_state WHERE date = ?', (date_str,))
        row = cursor.fetchone()
        return bool(row['circuit_breaker_active']) if row else False
    
    def activate_circuit_breaker(self, today: date | None = None) -> None:
        """Set the circuit breaker flag for today."""
        if today is None:
            today = datetime.now(timezone.utc).date()
        date_str = today.isoformat()
        self._ensure_daily_row(today)
        
        with self.conn:
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
        cursor = self.conn.execute('''
            SELECT id, timestamp, symbol, direction, entry_price, 
                   stop_loss, take_profit, lot_size, realized_pnl, status
            FROM trade_log
            WHERE status = 'OPEN'
        ''')
        
        open_trades = []
        for row in cursor.fetchall():
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
                status=row['status']
            )
            open_trades.append(trade)
            
        return open_trades
    
    def get_all_trades(self, limit: int = 100) -> list[TradeRecord]:
        """Return all historical trades ordered by id DESC."""
        cursor = self.conn.execute('''
            SELECT id, timestamp, symbol, direction, entry_price, 
                   stop_loss, take_profit, lot_size, realized_pnl, status
            FROM trade_log
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        
        trades = []
        for row in cursor.fetchall():
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
                status=row['status']
            )
            trades.append(trade)
            
        return trades

    def close(self) -> None:

        """Close DB connection."""
        self.conn.close()
        logger.info("Database connection closed")

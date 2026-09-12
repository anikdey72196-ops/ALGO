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
        self.cleanup_interrupted_sessions()
    
    def _init_schema(self) -> None:
        """Create tables: daily_state, trade_log, bot_sessions."""
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

    def cleanup_interrupted_sessions(self, reason: str = "Interrupted / Server Restarted") -> None:
        """Mark any lingering 'ACTIVE' sessions as 'INTERRUPTED'."""
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        with self.conn:
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
        
        with self.conn:
            cursor = self.conn.execute('''
                INSERT INTO bot_sessions (
                    activation_time, symbols, lot_size, trigger_source, status
                ) VALUES (?, ?, ?, ?, 'ACTIVE')
            ''', (now_iso, symbols_str, lot_size_str, trigger_source))
            session_id = cursor.lastrowid
            
        logger.info(f"[ACTIVATED] Recorded bot activation session #{session_id} for {symbols_str} (Trigger: {trigger_source})")
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
        
        with self.conn:
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
        return active_id

    def get_active_session(self) -> dict | None:
        """Return the currently active session if one exists, with live running duration."""
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
        cursor = self.conn.execute(
            "SELECT * FROM bot_sessions ORDER BY id DESC LIMIT ?", (limit,)
        )
        now = datetime.now(timezone.utc)
        records = []
        for row in cursor.fetchall():
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
        with self.conn:
            self.conn.execute("DELETE FROM bot_sessions WHERE status != 'ACTIVE'")
        logger.info("Cleared non-active bot session history.")

    def close(self) -> None:
        """Close DB connection."""
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

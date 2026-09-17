from __future__ import annotations
import os
import time as _time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from loguru import logger
from config import Direction, InstrumentConfig

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


@dataclass(frozen=True)
class PriceQuote:
    """Current bid/ask quote for an instrument."""
    symbol: str
    bid: float
    ask: float
    spread: float  # ask - bid in price terms
    timestamp: datetime


@dataclass(frozen=True)
class BracketOrder:
    """Atomic bracket order: entry + stop loss + take profit."""
    symbol: str
    direction: Direction
    lot_size: float
    entry_price: float  # For market orders, this is the expected fill price
    stop_loss: float
    take_profit: float
    magic: int = 123456
    comment: str = ''



@dataclass
class OrderResult:
    """Result of an order submission."""
    success: bool
    order_id: int | None = None
    fill_price: float | None = None
    error_code: int | None = None
    error_message: str | None = None
    retries_used: int = 0


class BrokerAdapter(ABC):
    """Abstract base class for broker execution."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the broker. Returns True on success."""
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the broker."""
    
    @abstractmethod
    def send_bracket_order(self, order: BracketOrder) -> OrderResult:
        """
        Submit an atomic bracket order (entry + SL + TP in one payload).
        MUST NOT send unbracketed market orders.
        """
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> PriceQuote | None:
        """Get the current bid/ask quote."""
    
    @abstractmethod
    def get_account_equity(self) -> float:
        """Get current account equity."""
    
    @abstractmethod
    def get_open_positions(self, symbol: str | None = None) -> list[dict]:
        """Get open positions, optionally filtered by symbol."""


class MT5Adapter(BrokerAdapter):
    r"""
    MetaTrader 5 execution adapter.
    
    Deployment Notes:
      1. Install MetaTrader 5 terminal on a Windows VPS or desktop.
      2. pip install MetaTrader5
      3. Log in to your MT5 account in the terminal (or supply MT5_LOGIN, MT5_PASSWORD, MT5_SERVER).
      4. Set mt5_path to the terminal executable path (defaults to standard C:\Program Files\MetaTrader 5\terminal64.exe).
      5. Call connect() before trading.
    """
    
    def __init__(self, mt5_path: str | None = None, login: int | None = None,
                 password: str | None = None, server: str | None = None,
                 max_retries: int = 3, retry_delay: float = 1.0):
        self.mt5_path = mt5_path or os.environ.get("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
        self.login_id = login
        self.password = password
        self.server = server
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._symbol_cache: dict[str, str] = {}
    
    def resolve_symbol(self, symbol: str) -> str:
        """Find the broker's exact symbol name, matching prefixes/suffixes (e.g. XAUUSDm, XAUUSD.a, GOLD)."""
        if mt5 is None:
            return symbol
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        # First check direct name
        info = mt5.symbol_info(symbol)
        if info is not None:
            self._symbol_cache[symbol] = symbol
            return symbol

        # Fetch all available broker symbols
        all_symbols = mt5.symbols_get()
        if not all_symbols:
            return symbol

        target = symbol.upper()
        # Exact match case-insensitive
        for s in all_symbols:
            if s.name.upper() == target:
                self._symbol_cache[symbol] = s.name
                return s.name

        # Substring / broker suffix match (e.g. XAUUSDm, XAUUSD.pro, GOLD)
        for s in all_symbols:
            s_up = s.name.upper()
            if target in s_up or (target == "XAUUSD" and "GOLD" in s_up):
                logger.info(f"Resolved symbol '{symbol}' -> broker symbol '{s.name}'")
                self._symbol_cache[symbol] = s.name
                return s.name

        return symbol

    def connect(self) -> bool:
        if mt5 is None:
            logger.error("MetaTrader5 package not installed. Cannot connect.")
            return False

        initialized = False
        # 1. Attempt connecting to an already running MT5 terminal
        try:
            initialized = mt5.initialize()
            if initialized:
                logger.info("Attached to running MetaTrader 5 terminal.")
        except Exception as e:
            logger.debug(f"Direct mt5.initialize() attempt: {e}")

        # 2. If not initialized, try with configured or standard terminal paths
        if not initialized:
            candidate_paths = []
            if self.mt5_path and os.path.exists(self.mt5_path):
                candidate_paths.append(self.mt5_path)
            candidate_paths.extend([
                r"C:\Program Files\MetaTrader 5\terminal64.exe",
                r"C:\Program Files (x86)\MetaTrader 5\terminal.exe",
            ])
            for path in candidate_paths:
                if os.path.exists(path):
                    logger.info(f"Attempting MT5 initialization with terminal path: {path}")
                    if mt5.initialize(path=path):
                        initialized = True
                        break

        if not initialized:
            err = mt5.last_error()
            logger.error(
                f"MT5 initialization failed (code: {err}). "
                "Please make sure the MetaTrader 5 desktop app is open and logged into your account."
            )
            return False

        # 3. Optional explicit login if credentials were provided
        if self.login_id and self.password and self.server:
            if not mt5.login(self.login_id, password=self.password, server=self.server):
                err = mt5.last_error()
                logger.error(f"MT5 login failed for account #{self.login_id}: {err}")
                return False

        acc = mt5.account_info()
        if acc is not None:
            logger.info(
                f"✅ Connected to MetaTrader 5 | Account #{acc.login} ({acc.server}) | "
                f"Balance: ${acc.balance:,.2f} | Equity: ${acc.equity:,.2f} | Leverage: 1:{acc.leverage}"
            )
        else:
            logger.warning("Connected to MT5, but no account is active. Please log in inside the MT5 terminal.")
        return True

    def disconnect(self) -> None:
        if mt5:
            mt5.shutdown()
            logger.info("Disconnected from MetaTrader 5.")

    def send_bracket_order(self, order: BracketOrder) -> OrderResult:
        if mt5 is None:
            return OrderResult(success=False, error_message="MT5 not installed")

        broker_symbol = self.resolve_symbol(order.symbol)
        mt5.symbol_select(broker_symbol, True)

        order_type = mt5.ORDER_TYPE_BUY if order.direction == Direction.BUY else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_symbol,
            "volume": order.lot_size,
            "type": order_type,
            "price": order.entry_price,
            "sl": order.stop_loss,
            "tp": order.take_profit,
            "deviation": 20,
            "magic": order.magic,
            "comment": order.comment,

            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        last_retcode = None
        for attempt in range(self.max_retries):
            result = mt5.order_send(request)
            if result is None:
                err = mt5.last_error()
                logger.error(f"MT5 order_send failed on attempt {attempt+1}. Error: {err}")
                last_retcode = err[0] if isinstance(err, tuple) else None
            else:
                last_retcode = result.retcode
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Order executed successfully! Order ID: {result.order} for {broker_symbol}")
                    return OrderResult(
                        success=True,
                        order_id=result.order,
                        fill_price=result.price,
                        retries_used=attempt
                    )
                else:
                    logger.warning(f"Order failed on attempt {attempt+1} with retcode: {result.retcode}")
            
            _time.sleep(self.retry_delay)
            
        return OrderResult(
            success=False, 
            retries_used=self.max_retries, 
            error_code=last_retcode, 
            error_message="Order exhausted max retries"
        )

    def get_current_price(self, symbol: str) -> PriceQuote | None:
        if mt5 is None:
            return None
        broker_symbol = self.resolve_symbol(symbol)
        mt5.symbol_select(broker_symbol, True)
        tick = mt5.symbol_info_tick(broker_symbol)
        if tick is None:
            return None
        return PriceQuote(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            spread=tick.ask - tick.bid,
            timestamp=datetime.fromtimestamp(tick.time, tz=timezone.utc)
        )
    
    def get_account_equity(self) -> float:
        if mt5 is None:
            return 0.0
        acc_info = mt5.account_info()
        if acc_info is None:
            return 0.0
        return acc_info.equity

    def get_account_info(self) -> dict | None:
        """Return dict of account details (login, server, balance, equity, leverage)."""
        if mt5 is None:
            return None
        acc = mt5.account_info()
        if acc is None:
            return None
        return {
            "login": acc.login,
            "server": acc.server,
            "balance": acc.balance,
            "equity": acc.equity,
            "profit": acc.profit,
            "margin": acc.margin,
            "margin_free": acc.margin_free,
            "currency": acc.currency,
            "company": acc.company,
            "leverage": acc.leverage,
        }
    
    def get_open_positions(self, symbol: str | None = None) -> list[dict]:
        if mt5 is None:
            return []
        kwargs = {}
        if symbol:
            kwargs['symbol'] = self.resolve_symbol(symbol)
        positions = mt5.positions_get(**kwargs)
        if positions is None:
            return []
        return [p._asdict() for p in positions]

    def get_deal_pnl_for_order(self, order_id: int) -> tuple[float, str] | None:
        """
        Check MT5 deal history to see if an order/position has closed.
        Returns (realized_pnl, status) where status is 'CLOSED_TP', 'CLOSED_SL', or 'CLOSED_MANUAL'.
        """
        if mt5 is None:
            return None
        now = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(now - timedelta(days=7), now)
        if not deals:
            return None
        
        pos_deals = [d for d in deals if d.position_id == order_id]
        if not pos_deals:
            return None
            
        closing_deals = [d for d in pos_deals if d.entry == 1]  # DEAL_ENTRY_OUT
        if closing_deals:
            cd = closing_deals[-1]
            total_profit = sum(d.profit + d.commission + d.swap for d in pos_deals)
            comment = (cd.comment or "").lower()
            status = "CLOSED_TP" if "tp" in comment else ("CLOSED_SL" if "sl" in comment else "CLOSED_MANUAL")
            return total_profit, status
        return None


class MockBrokerAdapter(BrokerAdapter):
    """Mock broker for local testing. Simulates fills and tracks positions."""
    
    def __init__(self, initial_equity: float = 10_000.0, slippage_points: float = 0.5,
                 pip_size: float = 0.0001):
        self._equity = initial_equity
        self.slippage_points = slippage_points
        self.pip_size = pip_size
        self._positions: dict[int, dict] = {}
        self._prices: dict[str, PriceQuote] = {}
        self._next_order_id = 1
        
        self.set_price("XAUUSD", 2580.00, 2580.25)
        self.set_price("EURUSD", 1.08500, 1.08512)
        self.set_price("GBPUSD", 1.28500, 1.28515)
        self.set_price("BTCUSD", 65200.00, 65200.50)
        self.set_price("ETHUSD", 3210.00, 3210.30)
    
    def connect(self) -> bool:
        logger.info("MockBrokerAdapter connected.")
        return True
        
    def disconnect(self) -> None:
        logger.info("MockBrokerAdapter disconnected.")
        
    def set_price(self, symbol: str, bid: float, ask: float) -> None:
        """Set a mock price for testing."""
        self._prices[symbol] = PriceQuote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            spread=ask - bid,
            timestamp=datetime.now(timezone.utc)
        )
        
    def send_bracket_order(self, order: BracketOrder) -> OrderResult:
        if order.symbol not in self._prices:
            return OrderResult(success=False, error_message=f"No mock price set for symbol: {order.symbol}")
            
        # Apply simulated slippage
        slippage = random.uniform(-self.slippage_points, self.slippage_points) * self.pip_size
        actual_fill = order.entry_price + slippage
        
        order_id = self._next_order_id
        self._next_order_id += 1
        
        self._positions[order_id] = {
            "order_id": order_id,
            "symbol": order.symbol,
            "direction": order.direction,
            "lot_size": order.lot_size,
            "entry_price": actual_fill,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            "comment": order.comment
        }
        
        logger.info(f"Mock execution for {order.symbol}: Order ID {order_id} filled @ {actual_fill:.5f}")
        return OrderResult(success=True, order_id=order_id, fill_price=actual_fill)
        
    def simulate_close(self, order_id: int, close_price: float, point_value: float) -> float:
        """Simulate closing a position. Returns realized PnL."""
        if order_id not in self._positions:
            return 0.0
            
        pos = self._positions.pop(order_id)
        diff = close_price - pos['entry_price']
        if pos['direction'] == Direction.SELL:
            diff = -diff
            
        pnl = diff * pos['lot_size'] * point_value
        self._equity += pnl
        logger.info(f"Mock position closed: Order ID {order_id} @ {close_price:.5f}, Realized PnL: {pnl:.2f}")
        return pnl
        
    def get_current_price(self, symbol: str) -> PriceQuote | None:
        return self._prices.get(symbol)
        
    def get_account_equity(self) -> float:
        return self._equity
        
    def get_open_positions(self, symbol: str | None = None) -> list[dict]:
        return [p for p in self._positions.values() if (symbol is None or p['symbol'] == symbol)]

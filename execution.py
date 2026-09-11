from __future__ import annotations
import time as _time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    """
    MetaTrader 5 execution adapter.
    
    Deployment Notes:
      1. Install MetaTrader 5 terminal on a Windows VPS.
      2. pip install MetaTrader5
      3. Log in to your MT5 account in the terminal.
      4. Set mt5_path to the terminal executable path.
      5. Call connect() before trading.
    """
    
    def __init__(self, mt5_path: str | None = None, login: int | None = None,
                 password: str | None = None, server: str | None = None,
                 max_retries: int = 3, retry_delay: float = 1.0):
        self.mt5_path = mt5_path
        self.login_id = login
        self.password = password
        self.server = server
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def connect(self) -> bool:
        if mt5 is None:
            logger.error("MetaTrader5 package not installed. Cannot connect.")
            return False
            
        kwargs = {}
        if self.mt5_path:
            kwargs['path'] = self.mt5_path
            
        if not mt5.initialize(**kwargs):
            logger.error(f"MT5 initialization failed, error code: {mt5.last_error()}")
            return False
            
        if self.login_id and self.password and self.server:
            if not mt5.login(self.login_id, self.password, self.server):
                logger.error(f"MT5 login failed, error code: {mt5.last_error()}")
                return False
                
        logger.info("Successfully connected to MetaTrader 5.")
        return True

    def disconnect(self) -> None:
        if mt5:
            mt5.shutdown()
            logger.info("Disconnected from MetaTrader 5.")

    def send_bracket_order(self, order: BracketOrder) -> OrderResult:
        if mt5 is None:
            return OrderResult(success=False, error_message="MT5 not installed")
            
        order_type = mt5.ORDER_TYPE_BUY if order.direction == Direction.BUY else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": order.lot_size,
            "type": order_type,
            "price": order.entry_price,
            "sl": order.stop_loss,
            "tp": order.take_profit,
            "deviation": 20,
            "magic": 123456,
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
                    logger.info(f"Order executed successfully! Order ID: {result.order}")
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
        tick = mt5.symbol_info_tick(symbol)
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
    
    def get_open_positions(self, symbol: str | None = None) -> list[dict]:
        if mt5 is None:
            return []
        kwargs = {}
        if symbol:
            kwargs['symbol'] = symbol
        positions = mt5.positions_get(**kwargs)
        if positions is None:
            return []
        return [p._asdict() for p in positions]


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

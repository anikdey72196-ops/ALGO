from __future__ import annotations
import math
from dataclasses import dataclass
from loguru import logger
from config import AccountConfig, RiskConfig, InstrumentConfig, TradingConfig, get_instrument
from state import StateManager
from strategy import TradeSignal


@dataclass(frozen=True)
class AuthorizationResult:
    """Result of trade authorization."""
    authorized: bool
    lot_size: float = 0.0
    rejection_reason: str | None = None
    risk_amount: float = 0.0  # Dollar amount at risk
    account_equity: float = 0.0


class RiskEngine:
    """Position sizing, circuit breakers, and trade authorization."""
    
    def __init__(self, config: TradingConfig, state: StateManager):
        self.config = config
        self.state = state
    
    def calculate_lot_size(
        self,
        equity: float,
        risk_pct: float,
        sl_distance_price: float,
        instrument: InstrumentConfig,
    ) -> float:
        """
        Calculate position size using the formula:
        
        Lot Size = (Equity × Risk%) / (SL Distance in Points × Point Value)
        
        Where:
          SL Distance in Points = sl_distance_price / (10 ** -instrument.digits)
          (Converting price distance to points based on digits)
        
        Then clamp to [min_lot, max_lot] and round to lot_step.
        """
        sl_distance_in_points = sl_distance_price / (10 ** -instrument.digits)
        
        if sl_distance_in_points <= 0 or instrument.point_value <= 0:
            return 0.0
            
        lot_size = (equity * risk_pct) / (sl_distance_in_points * instrument.point_value)
        
        # Clamp to min and max lot
        lot_size = max(instrument.min_lot, min(instrument.max_lot, lot_size))
        
        # Round to lot step
        lot_size = math.floor(lot_size / instrument.lot_step) * instrument.lot_step
        
        # Round to 2 decimal places to avoid floating point issues
        return round(lot_size, 2)
    
    def check_circuit_breakers(self, equity: float) -> tuple[bool, str | None]:
        """
        Check all circuit breaker conditions:
        1. Is circuit breaker already activated for today? -> blocked
        2. Has daily trade count reached max? -> blocked
        3. Has daily realized PnL exceeded max drawdown? -> activate breaker, blocked
        
        Returns (is_blocked: bool, reason: str | None).
        """
        if self.state.is_circuit_breaker_active():
            return True, 'Circuit breaker active: daily loss limit reached'
            
        # --- Daily trade limit temporarily disabled for extensive retesting ---
        # count = self.state.get_trade_count()
        # max_trades = self.config.risk.max_daily_trades
        # if count >= max_trades:
        #     return True, f'Daily trade limit reached ({count}/{max_trades})'

            
        pnl = self.state.get_daily_pnl()
        limit = equity * self.config.account.max_daily_drawdown_pct
        if pnl < 0 and abs(pnl) >= limit:
            self.state.activate_circuit_breaker()
            msg = f'Daily drawdown limit breached: ${abs(pnl):.2f} >= ${limit:.2f}'
            logger.error(f"Circuit breaker activated! {msg}")
            return True, msg
            
        return False, None
    
    def authorize_trade(
        self,
        signal: TradeSignal,
        current_equity: float,
    ) -> AuthorizationResult:
        """
        Full authorization pipeline:
        1. Check circuit breakers.
        2. Look up instrument config.
        3. Calculate lot size.
        4. Validate lot size > 0 and within bounds.
        5. Return AuthorizationResult.
        """
        is_blocked, reason = self.check_circuit_breakers(current_equity)
        if is_blocked:
            logger.warning(f"Trade rejected: {reason}")
            return AuthorizationResult(
                authorized=False, 
                rejection_reason=reason, 
                account_equity=current_equity
            )
            
        try:
            instrument = get_instrument(self.config, signal.symbol)
        except ValueError as e:
            logger.warning(f"Trade rejected: {e}")
            return AuthorizationResult(
                authorized=False, 
                rejection_reason=str(e), 
                account_equity=current_equity
            )
            
        sl_distance = signal.sl_distance if signal.sl_distance > 0 else abs(signal.entry_price - signal.stop_loss)
        if sl_distance <= 0:
            msg = 'Invalid SL distance'
            logger.warning(f"Trade rejected: {msg}")
            return AuthorizationResult(
                authorized=False, 
                rejection_reason=msg, 
                account_equity=current_equity
            )
            
        if self.config.fixed_lot_size is not None and self.config.fixed_lot_size > 0:
            lot_size = max(instrument.min_lot, min(instrument.max_lot, self.config.fixed_lot_size))
            lot_size = math.floor(lot_size / instrument.lot_step) * instrument.lot_step
            lot_size = round(lot_size, 2)
            logger.info(f"Using manual fixed lot size override: {lot_size}")
        else:
            lot_size = self.calculate_lot_size(
                equity=current_equity, 
                risk_pct=self.config.account.risk_pct, 
                sl_distance_price=sl_distance, 
                instrument=instrument
            )
        
        if lot_size < instrument.min_lot:
            msg = 'Calculated lot size below minimum'
            logger.warning(f"Trade rejected: {msg}")
            return AuthorizationResult(
                authorized=False, 
                rejection_reason=msg, 
                account_equity=current_equity
            )
            
        sl_distance_in_points = sl_distance / (10 ** -instrument.digits)
        risk_amount = lot_size * sl_distance_in_points * instrument.point_value
        
        logger.info(f"Trade authorized: {signal.symbol} {signal.direction} {lot_size} lots. Risk: ${risk_amount:.2f}")
        return AuthorizationResult(
            authorized=True,
            lot_size=lot_size,
            risk_amount=risk_amount,
            account_equity=current_equity
        )

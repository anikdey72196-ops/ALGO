"""
position_manager.py — Asynchronous Trailing Stops, Breakeven, and Profit Scaling.
"""
from __future__ import annotations
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, List
from loguru import logger
from core.config import Direction
from core.state import StateManager
from execution.execution import BrokerAdapter

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


from core.market_regime import MarketRegimeDetector

class PositionManager:
    """Manages active bracket orders: +1R breakeven (specifically for sideways markets), trails, and session closes."""

    def __init__(
        self,
        broker: BrokerAdapter,
        state: StateManager,
        poll_interval_sec: float = 5.0,
        history_provider=None,
        breakeven_sideways_only: bool = True,
    ):
        self.broker = broker
        self.state = state
        self.poll_interval_sec = poll_interval_sec
        self.history_provider = history_provider
        self.breakeven_sideways_only = breakeven_sideways_only
        self._stop_event = threading.Event()
        self._be_applied: set[int] = set()
        self._partial_tp_applied: set[int] = set()
        self._reversal_shielded: set[int] = set()
        self._worker = threading.Thread(target=self._run_loop, daemon=True, name="PositionMgr")
        self._worker.start()

    def protect_against_reversal(self, trade, reversal_analysis) -> bool:
        """Shields an active position when an opposing CHoCH is confirmed in the market."""
        if trade.id in self._reversal_shielded:
            return False

        quote = self.broker.get_current_price(trade.symbol)
        if not quote:
            return False

        current_price = quote.bid if trade.direction == Direction.BUY else quote.ask
        entry = trade.entry_price
        pip_sz = 0.1 if ("XAU" in trade.symbol or "BTC" in trade.symbol) else 0.0001

        # Move SL to secure profit or lock in breakeven
        if trade.direction == Direction.BUY:
            new_sl = max(trade.stop_loss, entry + pip_sz)
            if new_sl >= current_price:
                new_sl = current_price - 2 * pip_sz
        else:
            new_sl = min(trade.stop_loss, entry - pip_sz)
            if new_sl <= current_price:
                new_sl = current_price + 2 * pip_sz

        success = self._modify_mt5_sl_tp(trade.id, trade.symbol, new_sl, trade.take_profit)
        if success:
            self._reversal_shielded.add(trade.id)
            logger.info(
                f"🛡️ [REVERSAL SHIELD ACTIVATED] Position #{trade.id} ({trade.symbol} {trade.direction.value}) "
                f"SL secured to {new_sl:.5f} due to {reversal_analysis.choch_type.value} CHoCH (Prob: {reversal_analysis.reversal_probability:.0f}%)"
            )
        return success

    def process_positions(self) -> None:
        """Evaluate open positions against dynamic management rules."""
        try:
            open_trades = self.state.get_open_positions()
            if not open_trades:
                return

            now_utc = datetime.now(timezone.utc)

            for trade in open_trades:
                if trade.id is None:
                    continue

                quote = self.broker.get_current_price(trade.symbol)
                if not quote:
                    continue

                current_price = quote.bid if trade.direction == Direction.BUY else quote.ask
                entry = trade.entry_price
                orig_sl = trade.stop_loss
                risk_dist = abs(entry - orig_sl) if orig_sl > 0 else 0.0

                if risk_dist <= 0:
                    continue

                # ── 1. Breakeven at +1R (Enforced ONLY during Sideways / Ranging Markets) ──
                if trade.id not in self._be_applied:
                    r_gain = (current_price - entry) / risk_dist if trade.direction == Direction.BUY else (entry - current_price) / risk_dist
                    if r_gain >= 1.0:
                        # Check regime if sideways_only is enabled
                        is_sideways = not self.breakeven_sideways_only
                        if self.breakeven_sideways_only and self.history_provider is not None:
                            try:
                                df = self.history_provider(trade.symbol, "5m", 100)
                                if df is not None and len(df) >= 30:
                                    regime = MarketRegimeDetector.analyze(df)
                                    is_sideways = bool(regime.is_sideways)
                            except Exception as e:
                                logger.debug(f"Could not evaluate regime for BE on {trade.symbol}: {e}")

                        if is_sideways:
                            pip_sz = 0.1 if ("XAU" in trade.symbol or "BTC" in trade.symbol) else 0.0001
                            new_sl = entry + (pip_sz if trade.direction == Direction.BUY else -pip_sz)
                            self._modify_mt5_sl_tp(trade.id, trade.symbol, new_sl, trade.take_profit)
                            self._be_applied.add(trade.id)
                            logger.info(f"🛡️ [SIDEWAYS BE] Position #{trade.id} ({trade.symbol}) locked in Breakeven @ {new_sl:.5f} (+{r_gain:.2f}R reached in ranging market)")
                        else:
                            logger.debug(f"Trending market active on {trade.symbol} (+{r_gain:.2f}R). Skipping BE lock to allow trend continuation.")

                # ── 2. Session Close Guard (e.g. 21:50 UTC) ──
                if "SCALP" in trade.strategy_name.upper() or "ICT" in trade.strategy_name.upper():
                    if now_utc.hour == 21 and now_utc.minute >= 50:
                        logger.info(f"⏰ [SESSION CLOSE] Closing intraday trade #{trade.id} ({trade.symbol}) before daily rollover.")
                        self._close_mt5_position(trade.id, trade.symbol, trade.lot_size, trade.direction)

        except Exception as e:
            logger.error(f"Error in PositionManager evaluation: {e}")

    def _modify_mt5_sl_tp(self, ticket: int, symbol: str, new_sl: float, new_tp: float) -> bool:
        if mt5 is None:
            return True
        try:
            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": symbol,
                "sl": round(new_sl, 5),
                "tp": round(new_tp, 5),
            }
            res = mt5.order_send(req)
            return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
        except Exception as e:
            logger.error(f"Failed to modify SL/TP for position #{ticket}: {e}")
            return False

    def _close_mt5_position(self, ticket: int, symbol: str, lot: float, direction: Direction) -> bool:
        if mt5 is None:
            return True
        try:
            order_type = mt5.ORDER_TYPE_SELL if direction == Direction.BUY else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(symbol)
            price = (tick.bid if direction == Direction.BUY else tick.ask) if tick else 0.0
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": price,
                "deviation": 20,
            }
            res = mt5.order_send(req)
            return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
        except Exception as e:
            logger.error(f"Failed to close position #{ticket}: {e}")
            return False

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self.poll_interval_sec)
            if not self._stop_event.is_set():
                self.process_positions()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

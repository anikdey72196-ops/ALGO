"""
position_manager.py — Asynchronous Trailing Stops, Breakeven, and Profit Scaling.
"""
from __future__ import annotations
import json
import math
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from loguru import logger
from core.config import Direction
from core.state import StateManager
from execution.execution import BrokerAdapter

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from core.market_regime import MarketRegimeDetector

try:
    from ml.smart_partial_tp import SmartPartialTPService, PartialTPVerdict
except ImportError:
    SmartPartialTPService = None
    PartialTPVerdict = None


class PositionManager:
    """Manages active bracket orders: +1R breakeven (specifically for sideways markets), smart ML partial TP, and session closes."""

    def __init__(
        self,
        broker: BrokerAdapter,
        state: StateManager,
        poll_interval_sec: float = 5.0,
        history_provider=None,
        breakeven_sideways_only: bool = True,
        config=None,
        smart_partial_tp_service=None,
    ):
        self.broker = broker
        self.state = state
        self.poll_interval_sec = poll_interval_sec
        self.history_provider = history_provider
        self.breakeven_sideways_only = breakeven_sideways_only
        self.config = config
        self.smart_partial_tp_service = smart_partial_tp_service or (SmartPartialTPService.get_instance() if SmartPartialTPService else None)
        self._stop_event = threading.Event()
        self._be_applied: set[int] = set()
        self._partial_tp_applied: set[int] = set()
        self._trade_features: dict[int, dict] = {}
        self._remaining_lots: dict[int, float] = {}
        self._last_partial_time: dict[int, float] = {}
        self._partial_stages_taken: dict[int, int] = {}
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
                r_gain = (current_price - entry) / risk_dist if trade.direction == Direction.BUY else (entry - current_price) / risk_dist
                if trade.id not in self._be_applied:
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

                # ── 2. Smart ML Partial Take-Profit (Structural Reversal & Runner Protection) ──
                if trade.id not in self._remaining_lots:
                    self._remaining_lots[trade.id] = float(trade.lot_size)
                remaining_lot = self._remaining_lots[trade.id]

                # Resolve position rules
                rules = None
                if self.config and hasattr(self.config, "position_management"):
                    rules = self.config.position_management.get_rules(trade.strategy_name, trade.symbol)

                smart_enabled = getattr(rules, "smart_partial_tp_enabled", True) if rules else True
                min_r = getattr(rules, "smart_partial_tp_min_r", 0.6) if rules else 0.6
                cooldown_sec = getattr(rules, "smart_partial_tp_cooldown_bars", 3) * 60.0 if rules else 180.0
                shadow_mode = getattr(rules, "smart_partial_tp_shadow_mode", False) if rules else False

                now_ts = time.time()
                last_time = self._last_partial_time.get(trade.id, 0.0)
                stages_taken = self._partial_stages_taken.get(trade.id, 0)

                if (
                    smart_enabled
                    and self.smart_partial_tp_service is not None
                    and r_gain >= min_r
                    and (now_ts - last_time >= cooldown_sec)
                    and stages_taken < 3
                    and remaining_lot > 0.01
                ):
                    bars_held = 10
                    if getattr(trade, "timestamp", None):
                        try:
                            t_entry = datetime.fromisoformat(str(trade.timestamp).replace("Z", "+00:00"))
                            bars_held = max(1, int((now_utc - t_entry).total_seconds() / 300))
                        except Exception:
                            pass

                    df_hist = None
                    if self.history_provider:
                        try:
                            df_hist = self.history_provider(trade.symbol, "5m", 100)
                        except Exception as e:
                            logger.debug(f"History provider error on {trade.symbol}: {e}")

                    verdict = self.smart_partial_tp_service.evaluate(
                        symbol=trade.symbol,
                        strategy_name=trade.strategy_name or "SMC",
                        direction=trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction),
                        entry_price=entry,
                        sl_price=orig_sl,
                        tp_price=trade.take_profit,
                        current_price=current_price,
                        current_lot=remaining_lot,
                        bars_since_entry=bars_held,
                        df=df_hist,
                        shadow_mode=shadow_mode,
                        reversal_threshold=getattr(rules, "smart_partial_tp_reversal_threshold", 0.60) if rules else 0.60,
                        runner_threshold=getattr(rules, "smart_partial_tp_runner_threshold", 0.65) if rules else 0.65,
                    )
                    if verdict and hasattr(verdict, "features") and verdict.features:
                        self._trade_features[trade.id] = verdict.features

                    if verdict.action == "PARTIAL_CLOSE" and verdict.close_pct > 0.0:
                        raw_close = remaining_lot * verdict.close_pct
                        close_lot = round(math.floor(raw_close / 0.01) * 0.01, 2)
                        # Keep at least 0.01 runner lot
                        if remaining_lot - close_lot < 0.01:
                            close_lot = round(remaining_lot - 0.01, 2)

                        if close_lot >= 0.01:
                            closed_ok = True
                            if not verdict.is_shadow:
                                closed_ok = self._close_mt5_position(trade.id, trade.symbol, close_lot, trade.direction)

                            if closed_ok:
                                new_rem = round(remaining_lot - close_lot, 2)
                                self._remaining_lots[trade.id] = new_rem
                                self._last_partial_time[trade.id] = now_ts
                                self._partial_stages_taken[trade.id] = stages_taken + 1
                                self._partial_tp_applied.add(trade.id)

                                # Secure partial profit with SL bump
                                pip_sz = 0.1 if ("XAU" in trade.symbol or "BTC" in trade.symbol) else 0.0001
                                new_sl = entry + (pip_sz if trade.direction == Direction.BUY else -pip_sz)
                                self._modify_mt5_sl_tp(trade.id, trade.symbol, new_sl, trade.take_profit)

                                # Persist event to DB
                                try:
                                    self.state.record_partial_tp_event(
                                        trade_id=trade.id,
                                        symbol=trade.symbol,
                                        strategy_name=trade.strategy_name or "SMC",
                                        direction=trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction),
                                        current_price=current_price,
                                        r_multiple=round(r_gain, 2),
                                        p_reversal=round(verdict.p_reversal, 3),
                                        p_full_tp=round(verdict.p_full_tp, 3),
                                        predicted_max_r=round(verdict.predicted_max_r, 2),
                                        action=verdict.action,
                                        close_pct=round(verdict.close_pct, 2),
                                        closed_lot=close_lot,
                                        remaining_lot=new_rem,
                                        nearest_resistance=verdict.nearest_resistance,
                                        structure_confluence_count=int(verdict.features.get("structure_confluence_count", 0)),
                                        features_json=json.dumps(verdict.features),
                                        is_shadow=verdict.is_shadow,
                                    )
                                except Exception as db_err:
                                    logger.debug(f"Failed to record partial TP event in DB: {db_err}")

                                logger.info(
                                    f"💰 [SMART PARTIAL TP EXECUTED] Trade #{trade.id} ({trade.symbol}): "
                                    f"Closed {close_lot} lots ({verdict.close_pct:.0%}) @ {current_price:.5f} (+{r_gain:.2f}R). "
                                    f"Remaining: {new_rem} lots. Reason: {verdict.reason}"
                                )
                    else:
                        # Action is HOLD
                        self._last_partial_time[trade.id] = now_ts
                        try:
                            self.state.record_partial_tp_event(
                                trade_id=trade.id,
                                symbol=trade.symbol,
                                strategy_name=trade.strategy_name or "SMC",
                                direction=trade.direction.value if hasattr(trade.direction, "value") else str(trade.direction),
                                current_price=current_price,
                                r_multiple=round(r_gain, 2),
                                p_reversal=round(verdict.p_reversal, 3),
                                p_full_tp=round(verdict.p_full_tp, 3),
                                predicted_max_r=round(verdict.predicted_max_r, 2),
                                action=verdict.action,
                                close_pct=0.0,
                                closed_lot=0.0,
                                remaining_lot=remaining_lot,
                                nearest_resistance=verdict.nearest_resistance,
                                structure_confluence_count=int(verdict.features.get("structure_confluence_count", 0)),
                                features_json=json.dumps(verdict.features),
                                is_shadow=verdict.is_shadow,
                            )
                        except Exception as db_err:
                            logger.debug(f"Failed to record HOLD event in DB: {db_err}")

                        logger.info(
                            f"🏃 [SMART RUNNER HOLD] Trade #{trade.id} ({trade.symbol}): "
                            f"Holding runner @ {current_price:.5f} (+{r_gain:.2f}R). Reason: {verdict.reason}"
                        )

                # ── 3. Session Close Guard (e.g. 21:50 UTC) ──
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

    def on_trade_closed(self, trade, pnl: float, status: str) -> None:
        """Feed completed trade back into SmartPartialTPService for continuous self-learning."""
        if not self.smart_partial_tp_service:
            return
        try:
            features = self._trade_features.get(trade.id)
            last_ev = None
            if not features and hasattr(self.state, "get_latest_partial_tp_event") and trade.id is not None:
                last_ev = self.state.get_latest_partial_tp_event(trade.id)
                if last_ev and last_ev.get("features_json"):
                    try:
                        features = json.loads(last_ev["features_json"])
                    except Exception:
                        features = None

            entry = float(getattr(trade, "entry_price", 0.0) or 0.0)
            sl = float(getattr(trade, "stop_loss", 0.0) or 0.0)
            sl_dist = abs(entry - sl)

            if sl_dist > 0:
                exit_price = trade.take_profit if status == "CLOSED_TP" else trade.stop_loss
                if exit_price:
                    price_diff = (exit_price - entry) if trade.direction == Direction.BUY else (entry - exit_price)
                    final_r = price_diff / sl_dist
                else:
                    final_r = 1.0 if status == "CLOSED_TP" else -1.0
            else:
                final_r = 1.0 if status == "CLOSED_TP" else -1.0

            hit_tp = (status == "CLOSED_TP") or (final_r >= 1.5)

            if features:
                self.smart_partial_tp_service.record_trade_outcome(
                    features=features,
                    hit_tp=hit_tp,
                    final_r=round(final_r, 2),
                )
                logger.info(
                    f"🧠 [MODEL 3 SELF-LEARNING] SmartPartialTP learned Trade #{trade.id} outcome: "
                    f"Status={status}, final_r={final_r:+.2f}R, hit_tp={hit_tp}"
                )
                if last_ev and "id" in last_ev and hasattr(self.state, "update_partial_tp_outcome"):
                    self.state.update_partial_tp_outcome(
                        last_ev["id"],
                        realized_max_r=round(final_r, 2),
                        outcome_label="FULL_TP" if hit_tp else "REVERSED",
                    )

            if trade.id is not None:
                self._trade_features.pop(trade.id, None)
                self._remaining_lots.pop(trade.id, None)
                self._partial_stages_taken.pop(trade.id, None)
                self._last_partial_time.pop(trade.id, None)
                self._partial_tp_applied.discard(trade.id)
                self._be_applied.discard(trade.id)
                self._reversal_shielded.discard(trade.id)
        except Exception as e:
            logger.debug(f"Failed to record trade outcome in PositionManager: {e}")

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)


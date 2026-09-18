

from __future__ import annotations

import os
import sys
import signal
import asyncio
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

# pyrefly: ignore [missing-import]
from loguru import logger
# pyrefly: ignore [missing-import]
from apscheduler.schedulers.asyncio import AsyncIOScheduler
# pyrefly: ignore [missing-import]
from apscheduler.triggers.interval import IntervalTrigger

from config import TradingConfig, DEFAULT_CONFIG, get_instrument, Direction, normalize_strategy_key
from state import StateManager, TradeRecord
from news_filter import NewsFilter
from strategy import StrategyEngine, TradeSignal
from conflict_resolver import ConflictResolver
from risk_engine import RiskEngine
from ai_analyst import AIAnalyst, AIDecision
from ml.trap_detector import TrapDetectorConfig, TrapDetectorService, EventKind
from execution import (

    BrokerAdapter, MockBrokerAdapter, MT5Adapter,
    BracketOrder, PriceQuote,
)


# ─────────────────────────────────────────────
#  Logging Configuration
# ─────────────────────────────────────────────

def setup_logging(log_path: str) -> None:
    """Configure loguru with structured JSON logging and file rotation."""
    # Remove default stderr handler
    logger.remove()

    # Console handler — human-readable
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS UTC}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )

    # File handler — JSON structured, rotated daily, kept 30 days
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path,
        format="{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {name}:{function}:{line} | {message}",
        rotation="00:00",  # Rotate at midnight
        retention="30 days",
        compression="gz",
        level="DEBUG",
        serialize=True,  # JSON format
    )


# ─────────────────────────────────────────────
#  Broker Factory
# ─────────────────────────────────────────────

def create_broker(config: TradingConfig) -> BrokerAdapter:
    """Create the appropriate broker adapter based on configuration."""
    if config.use_mock_broker:
        logger.info("Using MockBrokerAdapter for local testing.")
        return MockBrokerAdapter(initial_equity=config.account.equity)
    else:
        logger.info("Using MT5Adapter for live real-market trading.")
        return MT5Adapter(
            mt5_path=config.mt5_path or os.environ.get("MT5_PATH"),
            login=config.mt5_login or (int(os.environ.get("MT5_LOGIN", "0")) if os.environ.get("MT5_LOGIN", "0").isdigit() and int(os.environ.get("MT5_LOGIN", "0")) > 0 else None),
            password=config.mt5_password or os.environ.get("MT5_PASSWORD"),
            server=config.mt5_server or os.environ.get("MT5_SERVER"),
        )


# ─────────────────────────────────────────────
#  Trading Loop
# ─────────────────────────────────────────────

class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config: TradingConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self._last_utc_day: int | None = None
        self.is_active: bool = False
        self.recent_logs: list[str] = []
        self._tick_lock = threading.Lock()

        # Initialize components
        self.state = StateManager(db_path=self.config.db_path)
        self.news_filter = NewsFilter(
            calendar_path=self.config.news_calendar_path,
            blackout_minutes=self.config.news_blackout_minutes,
        )
        self.strategy = StrategyEngine(self.config.timeframes)
        self.conflict_resolver = ConflictResolver(self.config.risk)
        self.risk_engine = RiskEngine(self.config, self.state)
        self.ai_analyst = AIAnalyst(self.config)
        self.broker = create_broker(self.config)

        # ML Trap Detector — veto gate for FVG / Sweep setups
        trap_cfg = TrapDetectorConfig(
            db_path=self.config.db_path,
            model_path="ml/artifacts/trap_detector.joblib",
            p_genuine_threshold=0.55,
            shadow_until_samples=200,
        )
        self.trap_svc = TrapDetectorService(trap_cfg)


        # Price data cache (in production, fetch from broker or data provider)
        self._htf_cache: dict[str, object] = {}
        self._ltf_cache: dict[str, object] = {}

        # Load persisted settings if present
        self.load_settings()

    def load_settings(self) -> None:
        """Load persisted user settings from bot_settings.json."""
        settings_path = Path("bot_settings.json")
        if settings_path.exists():
            try:
                import json
                with open(settings_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                
                # Pair 1 & Pair 2 independent configurations
                if "pair1" in saved and isinstance(saved["pair1"], dict):
                    p1 = saved["pair1"]
                    self.config.pair1.symbol = p1.get("symbol", self.config.pair1.symbol)
                    self.config.pair1.fixed_lot_size = p1.get("fixed_lot_size")
                    self.config.pair1.fixed_sl_pips = p1.get("fixed_sl_pips")
                    self.config.pair1.enabled = p1.get("enabled", True)
                
                if "pair2" in saved and isinstance(saved["pair2"], dict):
                    p2 = saved["pair2"]
                    self.config.pair2.symbol = p2.get("symbol", self.config.pair2.symbol)
                    self.config.pair2.fixed_lot_size = p2.get("fixed_lot_size")
                    self.config.pair2.fixed_sl_pips = p2.get("fixed_sl_pips")
                    self.config.pair2.enabled = p2.get("enabled", True)

                # Enabled strategies
                if "enabled_strategies" in saved and isinstance(saved["enabled_strategies"], list):
                    self.config.enabled_strategies = saved["enabled_strategies"]

                if "selected_symbols" in saved and isinstance(saved["selected_symbols"], list):
                    self.config.selected_symbols = saved["selected_symbols"]
                else:
                    self.config.selected_symbols = [self.config.pair1.symbol, self.config.pair2.symbol]

                if "strategy_type" in saved and isinstance(saved["strategy_type"], str):
                    self.config.strategy_type = saved["strategy_type"]
                if "fixed_lot_size" in saved:
                    self.config.fixed_lot_size = saved["fixed_lot_size"]
                if "fixed_sl_pips" in saved:
                    self.config.fixed_sl_pips = saved["fixed_sl_pips"]
                if "ai_confirmation_enabled" in saved:
                    self.config.ai_confirmation_enabled = bool(saved["ai_confirmation_enabled"])

                logger.info(
                    f"Loaded persisted settings from {settings_path}: "
                    f"Strategies={self.config.enabled_strategies} | "
                    f"Pair1={self.config.pair1.symbol} (lot={self.config.pair1.fixed_lot_size}, sl={self.config.pair1.fixed_sl_pips}) | "
                    f"Pair2={self.config.pair2.symbol} (lot={self.config.pair2.fixed_lot_size}, sl={self.config.pair2.fixed_sl_pips})"
                )
            except Exception as e:
                logger.warning(f"Failed to load bot_settings.json: {e}")

    def save_settings(self) -> None:
        """Persist user dashboard settings to bot_settings.json."""
        try:
            import json
            data = {
                "pair1": {
                    "symbol": self.config.pair1.symbol,
                    "fixed_lot_size": self.config.pair1.fixed_lot_size,
                    "fixed_sl_pips": self.config.pair1.fixed_sl_pips,
                    "enabled": self.config.pair1.enabled,
                },
                "pair2": {
                    "symbol": self.config.pair2.symbol,
                    "fixed_lot_size": self.config.pair2.fixed_lot_size,
                    "fixed_sl_pips": self.config.pair2.fixed_sl_pips,
                    "enabled": self.config.pair2.enabled,
                },
                "enabled_strategies": self.config.enabled_strategies,
                "selected_symbols": [self.config.pair1.symbol, self.config.pair2.symbol],
                "strategy_type": self.config.strategy_type,
                "fixed_lot_size": self.config.fixed_lot_size,
                "fixed_sl_pips": self.config.fixed_sl_pips,
                "ai_confirmation_enabled": self.config.ai_confirmation_enabled,
            }
            with open("bot_settings.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save bot settings to bot_settings.json: {e}")


    def log(self, message: str, level: str = "INFO") -> None:
        """Helper to append to recent logs and send to logger."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.recent_logs.append(entry)
        if len(self.recent_logs) > 200:
            self.recent_logs.pop(0)
        
        if level == "INFO":
            logger.info(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)
        else:
            logger.debug(message)

    def startup(self) -> bool:
        """Connect to broker and perform startup checks."""
        setup_logging(self.config.log_path)
        logger.info("=" * 60)
        logger.info("TRADING BOT STARTING UP")
        logger.info(f"  Instruments: {[i.symbol for i in self.config.instruments]}")
        logger.info(f"  Active Selected: {self.config.selected_symbols}")
        logger.info(f"  Risk per trade: {self.config.account.risk_pct*100:.1f}%")
        logger.info(f"  Fixed Lot Size: {self.config.fixed_lot_size}")
        logger.info(f"  Max daily drawdown: {self.config.account.max_daily_drawdown_pct*100:.1f}%")
        logger.info(f"  Min R:R ratio: {self.config.risk.min_rr_ratio}")
        logger.info(f"  Max trades/day: {self.config.risk.max_daily_trades}")
        logger.info(f"  HTF: {self.config.timeframes.htf} | LTF: {self.config.timeframes.ltf}")
        logger.info(f"  News blackout: ±{self.config.news_blackout_minutes} min")
        logger.info(f"  Mock broker: {self.config.use_mock_broker}")
        logger.info("=" * 60)

        connected = self.broker.connect()
        if not connected:
            logger.error("Failed to connect to broker. Aborting.")
            return False
        return True

    def shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("Trading bot shutting down...")
        if self.is_active:
            self.state.record_deactivation("Server Shutdown")
            self.is_active = False
        self.trap_svc.close()
        self.broker.disconnect()
        self.state.close()
        logger.info("Shutdown complete.")

    def _check_day_rollover(self) -> None:
        """Reset daily state if we've crossed into a new UTC day."""
        today = datetime.now(timezone.utc).date()
        today_ordinal = today.toordinal()
        if self._last_utc_day is None:
            self._last_utc_day = today_ordinal
        elif today_ordinal > self._last_utc_day:
            logger.info(f"New UTC day detected: {today.isoformat()}. Resetting daily state.")
            self.state.reset_daily_state()
            self._last_utc_day = today_ordinal

    def _sync_open_positions(self) -> None:
        """
        Check database open trades against live broker positions / deal history.
        If a trade reached TP or SL and is no longer open in MT5, update state.
        """
        try:
            open_trades = self.state.get_open_positions()
            if not open_trades:
                return

            live_positions = self.broker.get_open_positions()
            live_order_ids = {
                p.get("ticket") or p.get("order_id")
                for p in live_positions
                if p.get("ticket") or p.get("order_id")
            }

            for trade in open_trades:
                # If still open with broker, skip
                if trade.id in live_order_ids:
                    continue

                # Query broker for closed deal result
                if hasattr(self.broker, "get_deal_pnl_for_order") and trade.id is not None:
                    deal_res = self.broker.get_deal_pnl_for_order(trade.id)
                    if deal_res:
                        pnl, status = deal_res
                        self.state.update_trade_pnl(trade.id, pnl, status)
                        self.log(
                            f"🔔 Position #{trade.id} ({trade.symbol}) CLOSED in broker: {status} | "
                            f"Realized PnL: ${pnl:+.2f}"
                        )
        except Exception as e:
            logger.error(f"Error syncing open positions: {e}")

    def tick(self) -> None:
        """
        Main trading loop — called once per LTF bar close.
        Only runs analysis and execution if is_active == True.
        Evaluates Pair 1 and Pair 2 independently across all enabled strategies simultaneously.
        """
        if not self.is_active:
            logger.debug("TradingBot is DEACTIVATED (Idle). Skipping tick.")
            return

        if not self._tick_lock.acquire(blocking=False):
            self.log("Tick already in progress. Skipping duplicate run.", level="DEBUG")
            return

        try:
            self._execute_tick()
        finally:
            self._tick_lock.release()

    def _execute_tick(self) -> None:
        now_utc = datetime.now(timezone.utc)

        # Clear per-tick candle cache so fresh streaming candles are fetched every tick
        self._htf_cache.clear()
        self._ltf_cache.clear()

        # Step 1: Day rollover & position sync
        self._check_day_rollover()
        self._sync_open_positions()

        # Step 1b: Max concurrent open positions guard (e.g. max 6 open trades)
        open_trades = self.state.get_open_positions()
        max_open = getattr(self.config.risk, 'max_open_positions', 6)
        if len(open_trades) >= max_open:
            self.log(
                f"⏸️ MAX OPEN TRADES ACTIVE ({len(open_trades)}/{max_open} positions open). "
                f"Skipping new trade execution until an existing position closes.",
                level="INFO"
            )
            return

        # Build independent pair targets
        active_pairs = []
        if getattr(self.config, 'pair1', None) and self.config.pair1.enabled and self.config.pair1.symbol:
            active_pairs.append({
                "pair_num": 1,
                "symbol": self.config.pair1.symbol,
                "fixed_lot_size": self.config.pair1.fixed_lot_size,
                "fixed_sl_pips": self.config.pair1.fixed_sl_pips,
            })
        if getattr(self.config, 'pair2', None) and self.config.pair2.enabled and self.config.pair2.symbol:
            active_pairs.append({
                "pair_num": 2,
                "symbol": self.config.pair2.symbol,
                "fixed_lot_size": self.config.pair2.fixed_lot_size,
                "fixed_sl_pips": self.config.pair2.fixed_sl_pips,
            })
        if not active_pairs:
            for i, sym in enumerate(self.config.selected_symbols[:2], start=1):
                active_pairs.append({
                    "pair_num": i,
                    "symbol": sym,
                    "fixed_lot_size": self.config.fixed_lot_size,
                    "fixed_sl_pips": self.config.fixed_sl_pips,
                })

        strats_str = ", ".join(self.config.enabled_strategies)
        pairs_str = ", ".join(f"Pair {p['pair_num']}: {p['symbol']}" for p in active_pairs)
        self.log(f"─── TICK @ {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} | Pairs: [{pairs_str}] | Active Strategies: [{strats_str}] ───")

        for pair_info in active_pairs:
            symbol = pair_info["symbol"]
            pair_num = pair_info["pair_num"]
            pair_lot = pair_info["fixed_lot_size"]
            pair_sl = pair_info["fixed_sl_pips"]

            try:
                instrument = get_instrument(self.config, symbol)
            except ValueError:
                logger.warning(f"  Symbol '{symbol}' not in configured instruments. Skipping.")
                continue
            # ── Check open positions for this symbol (max 3 for XAUUSD, max 3 for EURUSD) ──
            open_symbol_trades = [t for t in open_trades if t.symbol == symbol]
            max_per_symbol = getattr(self.config.risk, 'max_open_per_symbol', 3)
            if len(open_symbol_trades) >= max_per_symbol:
                self.log(
                    f"  ⏸️ Max open positions reached for {symbol} ({len(open_symbol_trades)}/{max_per_symbol} open). "
                    f"Skipping new entry for Pair {pair_num}."
                )
                continue

            htf_tf = "1H"
            ltf_tf = "5m" if "SMC_SCALP_5M" in self.config.enabled_strategies or "ICT" in self.config.enabled_strategies else self.config.timeframes.ltf

            self.log(f"Analyzing Pair {pair_num} ({symbol}) | Sizing: lot={pair_lot or 'Dynamic'}, SL={pair_sl or 'Dynamic'} pips | HTF: {htf_tf} | LTF: {ltf_tf}...")

            # ── Step 2a: Fetch current price ──
            quote = self.broker.get_current_price(symbol)
            if quote is None:
                logger.warning(f"  No price available for {symbol}. Skipping.")
                continue

            current_spread = quote.spread
            logger.debug(f"  Price: bid={quote.bid:.5f} ask={quote.ask:.5f} spread={current_spread:.5f}")

            # ── Step 2a: News blackout check ──
            news_result = self.news_filter.check_news_blackout(symbol, now_utc)
            if news_result.blocked:
                logger.info(f"  ⛔ NEWS BLACKOUT: {news_result.reason}")
                continue

            # ── Step 2a: Spread check ──
            point_size = 10 ** -instrument.digits
            spread_result = NewsFilter.check_spread(
                current_spread=current_spread,
                avg_spread=instrument.avg_spread_points * point_size,
                max_spread_multiple=self.config.risk.max_spread_multiple,
            )
            if spread_result.blocked:
                self.log(f"  ⛔ SPREAD EXCESSIVE: {spread_result.reason}", level="WARNING")
                continue

            # ── Step 2b: Fetch OHLCV data ──
            htf_data = self._get_ohlcv(symbol, htf_tf)
            ltf_data = self._get_ohlcv(symbol, ltf_tf)
            if htf_data is None or ltf_data is None:
                logger.warning(f"  No OHLCV data for {symbol} ({htf_tf}/{ltf_tf}). Skipping.")
                continue

            # ── Step 2b: Exclude strategies that already have an active open position for this symbol ──
            # (Every strategy can execute at most 1 trade at a time per symbol)
            active_strats_on_symbol = {
                normalize_strategy_key(t.strategy_name, t.magic_number)
                for t in open_symbol_trades
            }
            eval_strats = [
                s for s in self.config.enabled_strategies
                if normalize_strategy_key(s) not in active_strats_on_symbol
            ]
            if not eval_strats:
                self.log(
                    f"  ⏸️ All enabled strategies ({list(active_strats_on_symbol)}) already have active trades for {symbol}. "
                    f"Skipping Pair {pair_num}."
                )
                continue

            # ── Step 2b: Generate signals across enabled strategies ──
            htf_analysis = self.strategy.htf_analyzer.analyze(htf_data)
            signals = self.strategy.evaluate_all(
                symbol=symbol,
                htf_data=htf_data,
                ltf_data=ltf_data,
                instrument=instrument,
                current_spread=current_spread,
                fixed_sl_pips=pair_sl,
                enabled_strategies=eval_strats,
                htf_analysis=htf_analysis,
            )

            if not signals:
                logger.info(f"  No signals generated for Pair {pair_num} ({symbol}).")
                continue

            logger.info(f"  {len(signals)} raw signal(s) generated for {symbol} across strategies.")

            # ── Step 2c: Conflict resolution ──
            filter_result = self.conflict_resolver.resolve(signals, current_spread)

            if filter_result.rejection_reasons:
                for reason in filter_result.rejection_reasons:
                    self.log(f"  🚫 {reason}")

            if filter_result.accepted_signal is None:
                self.log(f"  No signal survived quality gates for Pair {pair_num} ({symbol}).")
                continue

            best_signal = filter_result.accepted_signal
            self.log(
                f"  ✅ [{best_signal.strategy_name}] SIGNAL DETECTED: {best_signal.direction.value} {symbol} "
                f"| Entry={best_signal.entry_price:.5f} "
                f"| SL={best_signal.stop_loss:.5f} "
                f"| TP={best_signal.take_profit:.5f} "
                f"| R:R={best_signal.rr_ratio:.2f} "
                f"| Score={best_signal.quality_score:.1f} ({best_signal.ltf_confirmation.value})"
            )

            # ── Step 2c-1.5: ML Trap Detector Gate (FVG / Sweep veto) ──
            trap_kind = self._signal_to_trap_kind(best_signal)
            if trap_kind is not None and ltf_data is not None:
                try:
                    trap_dir = "long" if best_signal.direction == Direction.BUY else "short"
                    ml_df = self._prepare_df_for_trap_detector(ltf_data)
                    trap_ev = self.trap_svc.observe_event(
                        symbol=symbol,
                        timeframe=ltf_tf,
                        kind=trap_kind,
                        direction=trap_dir,
                        entry=best_signal.entry_price,
                        stop=best_signal.stop_loss,
                        target=best_signal.take_profit,
                        df=ml_df,
                        bar_index=len(ml_df) - 1,
                    )
                    if not trap_ev.allowed:
                        self.log(
                            f"  🪤 TRAP GATE VETO | p_genuine={trap_ev.p_genuine:.3f} "
                            f"| {trap_kind.value} | model={trap_ev.model_version}",
                            level="WARNING",
                        )
                        continue
                    self.log(
                        f"  🔬 Trap Gate: p_genuine={trap_ev.p_genuine:.3f} "
                        f"| {trap_kind.value} | mode={'shadow' if trap_ev.model_version == 'cold' else 'gated'}"
                    )
                except Exception as e:
                    self.log(f"  ⚠️ Trap Gate error (failing open): {e}", level="WARNING")

            # ── Step 2c-2: AI Second-Opinion Confirmation Gate ──
            ai_verdict = self.ai_analyst.evaluate_setup(best_signal, htf_analysis, current_spread)

            if not ai_verdict.confirmed:
                self.log(f"  🤖 AI GATE REJECTED ({ai_verdict.confidence:.1f}%): {ai_verdict.reason}", level="WARNING")
                continue

            self.log(f"  🤖 AI CONFIRMED ({ai_verdict.confidence:.1f}%): {ai_verdict.reason}")

            # ── Step 2c-3: Risk authorization (Pair-specific sizing override) ──
            equity = self.broker.get_account_equity()
            auth = self.risk_engine.authorize_trade(best_signal, equity, fixed_lot_size=pair_lot)

            if not auth.authorized:
                self.log(f"  🛑 RISK REJECTED: {auth.rejection_reason}", level="WARNING")
                continue

            self.log(
                f"  💰 AUTHORIZED: {auth.lot_size} lots "
                f"| Risk: ${auth.risk_amount:.2f} "
                f"| Equity: ${auth.account_equity:.2f}"
            )

            # ── Step 2d: Execute bracket order with strategy Magic Number ──
            bracket = BracketOrder(
                symbol=symbol,
                direction=best_signal.direction,
                lot_size=auth.lot_size,
                entry_price=quote.ask if best_signal.direction == Direction.BUY else quote.bid,
                stop_loss=best_signal.stop_loss,
                take_profit=best_signal.take_profit,
                magic=best_signal.magic_number,
                comment=f"{best_signal.strategy_name[:6]}_{best_signal.direction.value}_{best_signal.ltf_confirmation.value[:8]}",
            )

            order_result = self.broker.send_bracket_order(bracket)

            if order_result.success:
                self.log(
                    f"  ✅ ORDER FILLED: ID={order_result.order_id} "
                    f"@ {order_result.fill_price:.5f} ({bracket.symbol} {bracket.lot_size} lots | {best_signal.strategy_name})"
                )
                # Record in state with strategy attribution
                trade_record = TradeRecord(
                    id=order_result.order_id,
                    timestamp=now_utc,
                    symbol=symbol,
                    direction=best_signal.direction,
                    entry_price=order_result.fill_price or bracket.entry_price,
                    stop_loss=bracket.stop_loss,
                    take_profit=bracket.take_profit,
                    lot_size=bracket.lot_size,
                    realized_pnl=0.0,
                    status="OPEN",
                    strategy_name=best_signal.strategy_name,
                    magic_number=best_signal.magic_number,
                )
                self.state.record_trade(trade_record)
            else:
                self.log(
                    f"  ❌ ORDER FAILED: {order_result.error_message} "
                    f"(code={order_result.error_code}, retries={order_result.retries_used})",
                    level="ERROR"
                )

        self.log(
            f"─── TICK COMPLETE | "
            f"Daily PnL: ${self.state.get_daily_pnl():.2f} | "
            f"Trades today: {self.state.get_trade_count()} | "
            f"Circuit breaker: {'ACTIVE' if self.state.is_circuit_breaker_active() else 'OFF'} ───"
        )


    def _get_ohlcv(self, symbol: str, timeframe: str, count: int = 300):
        """
        Fetch OHLCV data for a symbol/timeframe.
        In mock mode: loads synthetic demo data from mock_data.py.
        In live real-market mode: fetches streaming candlesticks from MetaTrader 5.
        """
        # Check cache only for default 300-bar fetch
        cache_key = f"{symbol}_{timeframe}"
        if count == 300 and cache_key in self._htf_cache:
            return self._htf_cache[cache_key]

        # In mock mode, generate data on first call
        if self.config.use_mock_broker:
            try:
                from mock_data import get_demo_datasets
                datasets = get_demo_datasets()
                if symbol in datasets and timeframe in datasets[symbol]:
                    df = datasets[symbol][timeframe]
                    if count == 300:
                        self._htf_cache[cache_key] = df
                    return df
            except ImportError:
                logger.warning("mock_data module not available.")
            return None

        # In live real-market mode (MetaTrader 5):
        try:
            import MetaTrader5 as mt5
            import pandas as pd

            tf_map = {
                '1m': mt5.TIMEFRAME_M1,
                '5m': mt5.TIMEFRAME_M5,
                '15m': mt5.TIMEFRAME_M15,
                '1H': mt5.TIMEFRAME_H1,
                '4H': mt5.TIMEFRAME_H4,
                '1D': mt5.TIMEFRAME_D1,
            }
            if timeframe not in tf_map:
                logger.warning(f"Unsupported timeframe '{timeframe}' for MT5.")
                return None

            if hasattr(self.broker, 'ensure_connected'):
                self.broker.ensure_connected()

            broker_symbol = symbol
            if hasattr(self.broker, 'resolve_symbol'):
                broker_symbol = self.broker.resolve_symbol(symbol)

            mt5.symbol_select(broker_symbol, True)
            rates = mt5.copy_rates_from_pos(broker_symbol, tf_map[timeframe], 0, count)
            if rates is None or len(rates) == 0:
                err = mt5.last_error()
                logger.warning(f"MT5 returned no rates for {broker_symbol} ({timeframe}). Error: {err}")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df['volume'] = df['tick_volume']
            if count == 300:
                self._htf_cache[cache_key] = df
            return df
        except Exception as e:
            logger.error(f"Error fetching live MT5 candles for {symbol} ({timeframe}): {e}")
            return None

    def _prepare_df_for_trap_detector(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has a DatetimeIndex for the ML trap detector without mutating original."""
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        if 'time' in df.columns:
            df_copy = df.copy(deep=False)
            return df_copy.set_index(pd.to_datetime(df_copy['time'], utc=True))
        return df

    # ── Trap Detector helpers ──

    _TRAP_KIND_MAP: dict[str, EventKind] = {
        "FVG_MITIGATION":    EventKind.FVG_BULL,   # direction resolved at call site
        "OB_PLUS_FVG":       EventKind.FVG_BULL,
        "ICT_KILLZONE_FVG":  EventKind.FVG_BULL,
        "LIQUIDITY_SWEEP":   EventKind.SWEEP_BSL,  # direction resolved at call site
    }

    def _signal_to_trap_kind(self, sig: TradeSignal) -> EventKind | None:
        """Map a TradeSignal's LTF confirmation to a TrapDetector EventKind.

        Returns None for confirmation types that don't correspond to an FVG or
        sweep pattern (OB-only, scalps, etc.) — those pass through unfiltered.
        """
        base = self._TRAP_KIND_MAP.get(sig.ltf_confirmation.value)
        if base is None:
            return None
        # Resolve directional variant
        if base in (EventKind.FVG_BULL, EventKind.FVG_BEAR):
            return EventKind.FVG_BULL if sig.direction == Direction.BUY else EventKind.FVG_BEAR
        # Sweeps: BSL taken → potential short; SSL taken → potential long
        if sig.direction == Direction.BUY:
            return EventKind.SWEEP_SSL
        return EventKind.SWEEP_BSL

    def set_data(self, symbol: str, timeframe: str, df) -> None:
        """Manually inject OHLCV data (useful for backtesting)."""
        cache_key = f"{symbol}_{timeframe}"
        self._htf_cache[cache_key] = df


# ─────────────────────────────────────────────
#  Scheduler & Entry Point
# ─────────────────────────────────────────────

def parse_ltf_to_seconds(ltf: str) -> int:
    """Convert a timeframe string like '15m' or '1H' to seconds."""
    ltf = ltf.strip()
    if ltf.endswith('m'):
        return int(ltf[:-1]) * 60
    elif ltf.endswith('H'):
        return int(ltf[:-1]) * 3600
    elif ltf.endswith('D'):
        return int(ltf[:-1]) * 86400
    return 900  # Default 15 minutes


async def run_scheduled(config: TradingConfig | None = None) -> None:
    """Run the trading bot on a schedule."""
    bot = TradingBot(config)
    if not bot.startup():
        return

    interval_seconds = 60  # Fast 1-minute scan loop
    logger.info(f"Scheduling tick every {interval_seconds}s (1-minute continuous scan)")

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        bot.tick,
        trigger=IntervalTrigger(seconds=interval_seconds),
        id="trading_tick",
        name="Trading Tick",
        max_instances=1,
        misfire_grace_time=30,
    )

    # Run one tick immediately on startup
    bot.tick()

    # Schedule the ML Trap Detector labeler as a background task
    labeler_task = asyncio.create_task(
        bot.trap_svc.run_labeler(
            history_provider=lambda sym, tf, n: bot._prepare_df_for_trap_detector(
                bot._get_ohlcv(sym, tf, count=max(n, 300))
            )
        )
    )

    scheduler.start()
    logger.info("Scheduler started. Press Ctrl+C to stop.")

    # Keep running until interrupted
    stop_event = asyncio.Event()

    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}. Stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        await stop_event.wait()
    finally:
        labeler_task.cancel()
        scheduler.shutdown(wait=False)
        bot.shutdown()


def main() -> None:
    """Entry point."""
    asyncio.run(run_scheduled())


if __name__ == "__main__":
    main()

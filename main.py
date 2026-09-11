"""
main.py — Async orchestrator for the automated trading bot.

This is the entry point that wires all modules together and runs the trading
loop on a schedule (every LTF bar close).

Deployment to Cloud VPS:
  1. Provision a Windows VPS (e.g., AWS EC2 Windows, Contabo, or Hetzner).
     Linux VPS works for mock/backtest mode; MT5 requires Windows.
  2. Install Python 3.11+:
       winget install Python.Python.3.11
  3. Clone or upload the project files to the VPS.
  4. Install dependencies:
       pip install -r requirements.txt
       pip install MetaTrader5   # Windows only, for live trading
  5. Configure broker credentials:
       - Edit DEFAULT_CONFIG in config.py, or
       - Set environment variables MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH.
  6. Adjust config.py:
       - Set use_mock_broker = False for live trading.
       - Set account.equity to your starting balance.
       - Adjust instruments, risk parameters, and timeframes.
  7. Run the bot:
       python main.py
  8. For persistent operation, use a process manager:
       pip install pywin32  # for Windows service, or
       # Use Task Scheduler, or nssm to install as a Windows service.
       # On Linux (mock mode): nohup python main.py &
"""

from __future__ import annotations

import os
import sys
import signal
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path

from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import TradingConfig, DEFAULT_CONFIG, get_instrument, Direction
from state import StateManager, TradeRecord
from news_filter import NewsFilter
from strategy import StrategyEngine, TradeSignal
from conflict_resolver import ConflictResolver
from risk_engine import RiskEngine
from ai_analyst import AIAnalyst, AIDecision
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
        logger.info("Using MT5Adapter for live trading.")
        return MT5Adapter(
            mt5_path=os.environ.get("MT5_PATH"),
            login=int(os.environ.get("MT5_LOGIN", "0")) or None,
            password=os.environ.get("MT5_PASSWORD"),
            server=os.environ.get("MT5_SERVER"),
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


        # Price data cache (in production, fetch from broker or data provider)
        self._htf_cache: dict[str, object] = {}
        self._ltf_cache: dict[str, object] = {}

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

    def tick(self) -> None:
        """
        Main trading loop — called once per LTF bar close.
        Only runs analysis and execution if is_active == True.
        """
        if not self.is_active:
            logger.debug("TradingBot is DEACTIVATED (Idle). Skipping tick.")
            return

        now_utc = datetime.now(timezone.utc)
        self.log(f"─── TICK @ {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} (Active Charts: {self.config.selected_symbols[:2]}) ───")

        # Step 1: Day rollover
        self._check_day_rollover()

        active_symbols = set(self.config.selected_symbols[:2])
        for instrument in self.config.instruments:
            symbol = instrument.symbol
            if symbol not in active_symbols:
                continue

            self.log(f"Analyzing {symbol}...")

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
            htf_data = self._get_ohlcv(symbol, self.config.timeframes.htf)
            ltf_data = self._get_ohlcv(symbol, self.config.timeframes.ltf)
            if htf_data is None or ltf_data is None:
                logger.warning(f"  No OHLCV data for {symbol}. Skipping.")
                continue

            # ── Step 2b: Generate signals ──
            signals = self.strategy.generate_signals(
                symbol=symbol,
                htf_data=htf_data,
                ltf_data=ltf_data,
                instrument=instrument,
                current_spread=current_spread,
                fixed_sl_pips=self.config.fixed_sl_pips,
            )


            if not signals:
                logger.info(f"  No signals generated for {symbol}.")
                continue

            logger.info(f"  {len(signals)} raw signal(s) generated for {symbol}.")

            # ── Step 2c: Conflict resolution ──
            filter_result = self.conflict_resolver.resolve(signals, current_spread)

            if filter_result.rejection_reasons:
                for reason in filter_result.rejection_reasons:
                    self.log(f"  🚫 {reason}")

            if filter_result.accepted_signal is None:
                self.log(f"  No signal survived quality gates for {symbol}.")
                continue

            best_signal = filter_result.accepted_signal
            self.log(
                f"  ✅ SMC SIGNAL DETECTED: {best_signal.direction.value} {symbol} "
                f"| Entry={best_signal.entry_price:.5f} "
                f"| SL={best_signal.stop_loss:.5f} "
                f"| TP={best_signal.take_profit:.5f} "
                f"| R:R={best_signal.rr_ratio:.2f} "
                f"| Score={best_signal.quality_score:.1f} ({best_signal.ltf_confirmation.value})"
            )

            # ── Step 2c-2: AI Second-Opinion Confirmation Gate ──
            htf_analysis = self.strategy.htf_analyzer.analyze(htf_data)
            ai_verdict = self.ai_analyst.evaluate_setup(best_signal, htf_analysis, current_spread)

            if not ai_verdict.confirmed:
                self.log(f"  🤖 AI GATE REJECTED ({ai_verdict.confidence:.1f}%): {ai_verdict.reason}", level="WARNING")
                continue

            self.log(f"  🤖 AI CONFIRMED ({ai_verdict.confidence:.1f}%): {ai_verdict.reason}")

            # ── Step 2c-3: Risk authorization ──
            equity = self.broker.get_account_equity()
            auth = self.risk_engine.authorize_trade(best_signal, equity)

            if not auth.authorized:
                self.log(f"  🛑 RISK REJECTED: {auth.rejection_reason}", level="WARNING")
                continue


            self.log(
                f"  💰 AUTHORIZED: {auth.lot_size} lots "
                f"| Risk: ${auth.risk_amount:.2f} "
                f"| Equity: ${auth.account_equity:.2f}"
            )

            # ── Step 2d: Execute bracket order ──
            bracket = BracketOrder(
                symbol=symbol,
                direction=best_signal.direction,
                lot_size=auth.lot_size,
                entry_price=quote.ask if best_signal.direction == Direction.BUY else quote.bid,
                stop_loss=best_signal.stop_loss,
                take_profit=best_signal.take_profit,
                comment=f"BOT|{best_signal.ltf_confirmation.value}|RR{best_signal.rr_ratio:.1f}",
            )

            order_result = self.broker.send_bracket_order(bracket)

            if order_result.success:
                self.log(
                    f"  ✅ ORDER FILLED: ID={order_result.order_id} "
                    f"@ {order_result.fill_price:.5f} ({bracket.symbol} {bracket.lot_size} lots)"
                )
                # Record in state
                trade_record = TradeRecord(
                    id=None,
                    timestamp=now_utc,
                    symbol=symbol,
                    direction=best_signal.direction,
                    entry_price=order_result.fill_price or bracket.entry_price,
                    stop_loss=bracket.stop_loss,
                    take_profit=bracket.take_profit,
                    lot_size=bracket.lot_size,
                    realized_pnl=0.0,
                    status="OPEN",
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

    def _get_ohlcv(self, symbol: str, timeframe: str):
        """
        Fetch OHLCV data for a symbol/timeframe.

        In production, this would call self.broker or a data API to get
        the latest bars. For the mock adapter, we use cached demo data.
        """
        # Check cache
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self._htf_cache:
            return self._htf_cache[cache_key]

        # In mock mode, generate data on first call
        if self.config.use_mock_broker:
            try:
                from mock_data import get_demo_datasets
                datasets = get_demo_datasets()
                if symbol in datasets and timeframe in datasets[symbol]:
                    df = datasets[symbol][timeframe]
                    self._htf_cache[cache_key] = df
                    return df
            except ImportError:
                logger.warning("mock_data module not available.")
            return None

        # In live mode, you would fetch from MT5:
        # import MetaTrader5 as mt5
        # timeframe_map = {'1m': mt5.TIMEFRAME_M1, '5m': mt5.TIMEFRAME_M5, ...}
        # rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, 300)
        # df = pd.DataFrame(rates)
        # df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        return None

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

    interval_seconds = parse_ltf_to_seconds(bot.config.timeframes.ltf)
    logger.info(f"Scheduling tick every {interval_seconds}s ({bot.config.timeframes.ltf})")

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
        scheduler.shutdown(wait=False)
        bot.shutdown()


def main() -> None:
    """Entry point."""
    asyncio.run(run_scheduled())


if __name__ == "__main__":
    main()

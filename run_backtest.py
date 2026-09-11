"""
run_backtest.py — Local backtest runner for the automated trading bot.

Wires all modules together with MockBrokerAdapter and synthetic data,
runs through the trading pipeline, and prints a detailed summary.

Usage:
    python run_backtest.py

This script requires NO broker connection and NO external data feeds.
It uses deterministic mock data (seeded RNG) for reproducibility.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta

from loguru import logger

from config import TradingConfig, DEFAULT_CONFIG, get_instrument, Direction
from state import StateManager, TradeRecord
from news_filter import NewsFilter
from strategy import StrategyEngine
from conflict_resolver import ConflictResolver
from risk_engine import RiskEngine
from execution import MockBrokerAdapter, BracketOrder
from mock_data import get_demo_datasets, generate_mock_price_quote


# ─────────────────────────────────────────────
#  Logging Setup (console only for backtest)
# ─────────────────────────────────────────────

def setup_backtest_logging() -> None:
    """Minimal console logging for backtest."""
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )


# ─────────────────────────────────────────────
#  Backtest Engine
# ─────────────────────────────────────────────

def run_backtest() -> None:
    """Run a full end-to-end backtest with mock data."""
    setup_backtest_logging()

    logger.info("=" * 70)
    logger.info("  AUTOMATED TRADING BOT — LOCAL BACKTEST")
    logger.info("=" * 70)

    # ── Configuration ──
    config = TradingConfig(
        db_path=":memory:",  # In-memory SQLite for backtest
        use_mock_broker=True,
        news_blackout_minutes=30,
    )

    # ── Initialize components ──
    state = StateManager(db_path=config.db_path)
    news_filter = NewsFilter(
        calendar_path=config.news_calendar_path,
        blackout_minutes=config.news_blackout_minutes,
    )
    strategy = StrategyEngine(config.timeframes)
    conflict_resolver = ConflictResolver(config.risk)
    risk_engine = RiskEngine(config, state)
    broker = MockBrokerAdapter(initial_equity=config.account.equity)
    broker.connect()

    # ── Load demo datasets ──
    datasets = get_demo_datasets()
    logger.info(f"Loaded demo datasets: {list(datasets.keys())}")

    # ── Statistics tracking ──
    stats = {
        "total_signals": 0,
        "signals_rejected_htf": 0,
        "signals_rejected_rr": 0,
        "signals_rejected_spread": 0,
        "signals_rejected_news": 0,
        "signals_rejected_spread_filter": 0,
        "signals_rejected_risk": 0,
        "trades_executed": 0,
        "trades_failed": 0,
        "decisions": [],  # Log of all decisions
    }

    # ── Simulate ticks ──
    # We'll iterate over the LTF data in windows, simulating 50 bar-close events.
    num_ticks = 50
    window_size = 100  # Minimum bars needed for analysis

    for instrument in config.instruments:
        symbol = instrument.symbol

        if symbol not in datasets:
            logger.warning(f"No demo data for {symbol}. Skipping.")
            continue

        htf_data = datasets[symbol].get(config.timeframes.htf)
        ltf_data = datasets[symbol].get(config.timeframes.ltf)

        if htf_data is None or ltf_data is None:
            logger.warning(f"Missing HTF or LTF data for {symbol}. Skipping.")
            continue

        logger.info(f"\n{'─' * 50}")
        logger.info(f"  Backtesting {symbol} ({num_ticks} ticks)")
        logger.info(f"  HTF data: {len(htf_data)} bars | LTF data: {len(ltf_data)} bars")
        logger.info(f"{'─' * 50}")

        # Use the last bar's time to simulate timestamps
        # Slide a window through LTF data
        max_start = len(ltf_data) - window_size
        if max_start < 0:
            max_start = 0
        step = max(1, max_start // num_ticks)

        for tick_idx in range(num_ticks):
            start_idx = min(tick_idx * step, max_start)
            end_idx = start_idx + window_size
            if end_idx > len(ltf_data):
                break

            ltf_window = ltf_data.iloc[start_idx:end_idx].copy()
            # Use last bar's time as "now"
            if 'time' in ltf_window.columns:
                sim_time = ltf_window['time'].iloc[-1]
                if not isinstance(sim_time, datetime):
                    sim_time = datetime.now(timezone.utc)
                elif sim_time.tzinfo is None:
                    sim_time = sim_time.replace(tzinfo=timezone.utc)
            else:
                sim_time = datetime.now(timezone.utc)

            # Set mock price from last bar
            last_close = ltf_window['close'].iloc[-1]
            spread_price = instrument.avg_spread_points * (instrument.pip_size / 10)
            broker.set_price(
                symbol,
                bid=last_close - spread_price / 2,
                ask=last_close + spread_price / 2,
            )

            quote = broker.get_current_price(symbol)
            current_spread = quote.spread if quote else 0.0

            # ── News check ──
            news_result = news_filter.check_news_blackout(symbol, sim_time)
            if news_result.blocked:
                stats["signals_rejected_news"] += 1
                stats["decisions"].append(
                    f"TICK {tick_idx:03d} | {symbol} | ⛔ NEWS: {news_result.reason}"
                )
                continue

            # ── Spread check ──
            spread_result = NewsFilter.check_spread(
                current_spread=current_spread,
                avg_spread=instrument.avg_spread_points * (instrument.pip_size / 10),
                max_spread_multiple=config.risk.max_spread_multiple,
            )
            if spread_result.blocked:
                stats["signals_rejected_spread_filter"] += 1
                stats["decisions"].append(
                    f"TICK {tick_idx:03d} | {symbol} | ⛔ SPREAD: {spread_result.reason}"
                )
                continue

            # ── Signal generation ──
            signals = strategy.generate_signals(
                symbol=symbol,
                htf_data=htf_data,
                ltf_data=ltf_window,
                instrument=instrument,
                current_spread=current_spread,
            )

            stats["total_signals"] += len(signals)

            if not signals:
                stats["decisions"].append(
                    f"TICK {tick_idx:03d} | {symbol} | — No signals"
                )
                continue

            # ── Conflict resolution ──
            filter_result = conflict_resolver.resolve(signals, current_spread)

            # Track rejection reasons
            for reason in filter_result.rejection_reasons:
                if "HTF_CONFLICT" in reason:
                    stats["signals_rejected_htf"] += 1
                elif "LOW_RR" in reason:
                    stats["signals_rejected_rr"] += 1
                elif "SPREAD_FRICTION" in reason:
                    stats["signals_rejected_spread"] += 1

            if filter_result.accepted_signal is None:
                stats["decisions"].append(
                    f"TICK {tick_idx:03d} | {symbol} | 🚫 All {len(signals)} signals filtered out"
                )
                continue

            best = filter_result.accepted_signal

            # ── Risk authorization ──
            equity = broker.get_account_equity()
            auth = risk_engine.authorize_trade(best, equity)

            if not auth.authorized:
                stats["signals_rejected_risk"] += 1
                stats["decisions"].append(
                    f"TICK {tick_idx:03d} | {symbol} | 🛑 RISK: {auth.rejection_reason}"
                )
                continue

            # ── Execute bracket order ──
            bracket = BracketOrder(
                symbol=symbol,
                direction=best.direction,
                lot_size=auth.lot_size,
                entry_price=quote.ask if best.direction == Direction.BUY else quote.bid,
                stop_loss=best.stop_loss,
                take_profit=best.take_profit,
                comment=f"BT|{best.ltf_confirmation.value}|RR{best.rr_ratio:.1f}",
            )

            order_result = broker.send_bracket_order(bracket)

            if order_result.success:
                stats["trades_executed"] += 1
                # Record trade
                trade = TradeRecord(
                    id=None,
                    timestamp=sim_time,
                    symbol=symbol,
                    direction=best.direction,
                    entry_price=order_result.fill_price or bracket.entry_price,
                    stop_loss=bracket.stop_loss,
                    take_profit=bracket.take_profit,
                    lot_size=bracket.lot_size,
                    realized_pnl=0.0,
                    status="OPEN",
                )
                trade_id = state.record_trade(trade)

                # Simulate immediate close for backtest PnL (random TP/SL hit)
                import random
                random.seed(tick_idx)
                hit_tp = random.random() < 0.55  # Slight edge toward TP
                close_price = bracket.take_profit if hit_tp else bracket.stop_loss
                pnl = broker.simulate_close(
                    order_result.order_id, close_price, instrument.point_value
                )
                status = "CLOSED_TP" if hit_tp else "CLOSED_SL"
                state.update_trade_pnl(trade_id, pnl, status)

                stats["decisions"].append(
                    f"TICK {tick_idx:03d} | {symbol} | ✅ {best.direction.value} "
                    f"@ {order_result.fill_price:.5f} | "
                    f"SL={bracket.stop_loss:.5f} TP={bracket.take_profit:.5f} | "
                    f"Lots={auth.lot_size} | R:R={best.rr_ratio:.2f} | "
                    f"Score={best.quality_score:.1f} | "
                    f"{best.ltf_confirmation.value} | "
                    f"Result: {status} PnL=${pnl:.2f}"
                )
            else:
                stats["trades_failed"] += 1
                stats["decisions"].append(
                    f"TICK {tick_idx:03d} | {symbol} | ❌ ORDER FAILED: {order_result.error_message}"
                )

    # ── Print Summary ──
    logger.info("\n" + "=" * 70)
    logger.info("  BACKTEST SUMMARY")
    logger.info("=" * 70)

    # Decision log
    logger.info("\n── Decision Log ──")
    for d in stats["decisions"]:
        logger.info(f"  {d}")

    # Statistics
    logger.info(f"\n── Statistics ──")
    logger.info(f"  Total raw signals generated:     {stats['total_signals']}")
    logger.info(f"  Rejected — HTF conflict:         {stats['signals_rejected_htf']}")
    logger.info(f"  Rejected — Low R:R (<{config.risk.min_rr_ratio}):      {stats['signals_rejected_rr']}")
    logger.info(f"  Rejected — Spread friction:      {stats['signals_rejected_spread']}")
    logger.info(f"  Rejected — News blackout:        {stats['signals_rejected_news']}")
    logger.info(f"  Rejected — Spread filter:        {stats['signals_rejected_spread_filter']}")
    logger.info(f"  Rejected — Risk/circuit breaker: {stats['signals_rejected_risk']}")
    logger.info(f"  Trades executed:                 {stats['trades_executed']}")
    logger.info(f"  Orders failed:                   {stats['trades_failed']}")

    final_equity = broker.get_account_equity()
    starting_equity = config.account.equity
    net_pnl = final_equity - starting_equity
    logger.info(f"\n── Financial Summary ──")
    logger.info(f"  Starting equity:  ${starting_equity:,.2f}")
    logger.info(f"  Final equity:     ${final_equity:,.2f}")
    logger.info(f"  Net PnL:          ${net_pnl:,.2f} ({net_pnl/starting_equity*100:+.2f}%)")
    logger.info(f"  Daily PnL (DB):   ${state.get_daily_pnl():,.2f}")
    logger.info(f"  Trades recorded:  {state.get_trade_count()}")
    logger.info(f"  Circuit breaker:  {'ACTIVE' if state.is_circuit_breaker_active() else 'OFF'}")

    logger.info("\n" + "=" * 70)
    logger.info("  BACKTEST COMPLETE")
    logger.info("=" * 70)

    # Cleanup
    broker.disconnect()
    state.close()


if __name__ == "__main__":
    run_backtest()

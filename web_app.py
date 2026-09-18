"""
web_app.py — FastAPI Web Control Dashboard for the Automated Trading Bot.

Features:
- Activation / Deactivation toggle: Starts/stops the execution loop on demand.
- Custom fixed lot size input: Allows manual override of lot sizing.
- Dual Chart Selection: Allows user to pick 2 active markets (e.g. XAUUSD & EURUSD).
- Market Lock Enforcement: Once activated, symbol selectors cannot be modified until deactivated.
- Real-time State Polling: Exposes live equity, daily PnL, circuit breaker status, and recent activity logs.
"""

from __future__ import annotations

import os
import asyncio
from datetime import datetime, timezone
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import TradingConfig, DEFAULT_CONFIG, InstrumentConfig, get_instrument
from main import TradingBot, parse_ltf_to_seconds
from state import format_duration


# ─────────────────────────────────────────────
#  Pydantic Schemas for Web API
# ─────────────────────────────────────────────

class PairConfigItem(BaseModel):
    symbol: str
    fixed_lot_size: Optional[float] = None
    fixed_sl_pips: Optional[float] = None
    enabled: bool = True


class BotConfigUpdate(BaseModel):
    pair1: Optional[PairConfigItem] = None
    pair2: Optional[PairConfigItem] = None
    enabled_strategies: Optional[List[str]] = None
    selected_symbols: Optional[List[str]] = None
    strategy_type: Optional[str] = None
    fixed_lot_size: Optional[float] = None
    fixed_sl_pips: Optional[float] = None
    ai_confirmation_enabled: Optional[bool] = None


class BotStateResponse(BaseModel):
    is_active: bool
    selected_symbols: List[str]
    pair1: dict
    pair2: dict
    enabled_strategies: List[str]
    strategy_type: str
    fixed_lot_size: Optional[float]
    fixed_sl_pips: Optional[float]
    ai_confirmation_enabled: bool
    ai_confidence_threshold: float
    equity: float
    daily_pnl: float
    trades_today: int
    circuit_breaker_active: bool
    recent_logs: List[str]
    available_symbols: List[str]
    mock_mode: bool
    broker_info: Optional[dict] = None
    accuracy: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    total_closed_trades: int = 0
    active_session: Optional[dict] = None
    available_strategies: List[str] = ["SMC", "SMC_SCALP_5M", "ICT"]
    stats_by_strategy: dict = {}
    stats_by_pair: dict = {}
    performance_metrics: dict = {}





# ─────────────────────────────────────────────
#  Global Bot Instance & Background Scheduler
# ─────────────────────────────────────────────

bot_instance: TradingBot = TradingBot(DEFAULT_CONFIG)
bot_instance.startup()
scheduler_instance: AsyncIOScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize background tick scheduler."""
    global bot_instance, scheduler_instance
    logger.info("Initializing Web Control Server...")

    config = bot_instance.config

    # Start background scheduler — fast 60-second (1-minute) interval for high responsiveness
    scheduler_instance = AsyncIOScheduler(timezone="UTC")
    interval = 60
    
    # Tick job runs every interval, but bot.tick() checks `if not self.is_active: return`
    scheduler_instance.add_job(
        bot_instance.tick,
        trigger=IntervalTrigger(seconds=interval),
        id="web_trading_tick",
        name="Web Trading Tick",
        max_instances=1,
        misfire_grace_time=30,
    )
    scheduler_instance.start()
    logger.info(f"Background tick scheduler active (fast 60s scan interval). Bot starts DEACTIVATED.")

    yield

    # Shutdown
    if scheduler_instance:
        scheduler_instance.shutdown(wait=False)
    if bot_instance:
        if bot_instance.is_active:
            bot_instance.state.record_deactivation("Server Shutdown")
            bot_instance.is_active = False
        bot_instance.shutdown()
    logger.info("Web Control Server shut down cleanly.")



app = FastAPI(title="Algorithmic Trading Bot Dashboard", lifespan=lifespan)

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ─────────────────────────────────────────────
#  Web API Routes
# ─────────────────────────────────────────────

@app.get("/favicon.ico")
async def get_favicon():
    """Silence browser favicon 404 request."""
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serve the single-page control dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")



@app.get("/api/state", response_model=BotStateResponse)
async def get_bot_state():
    """Return live system state, metrics, and logs."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    equity = bot_instance.broker.get_account_equity()
    daily_pnl = bot_instance.state.get_daily_pnl()
    trade_count = bot_instance.state.get_trade_count()
    cb_active = bot_instance.state.is_circuit_breaker_active()
    active_session = bot_instance.state.get_active_session()

    available = [i.symbol for i in bot_instance.config.instruments]
    broker_info = None
    if hasattr(bot_instance.broker, "get_account_info"):
        broker_info = bot_instance.broker.get_account_info()

    # Compute performance metrics in a single database pass
    metrics = bot_instance.state.get_performance_metrics()
    accuracy = metrics.get("accuracy", 0.0)
    winning_trades = metrics.get("winning_trades", 0)
    losing_trades = metrics.get("losing_trades", 0)
    total_closed = metrics.get("total_closed_trades", 0)
    stats_by_strategy = metrics.get("by_strategy", {})
    stats_by_pair = metrics.get("by_pair", {})

    pair1_data = {
        "symbol": bot_instance.config.pair1.symbol,
        "fixed_lot_size": bot_instance.config.pair1.fixed_lot_size,
        "fixed_sl_pips": bot_instance.config.pair1.fixed_sl_pips,
        "enabled": bot_instance.config.pair1.enabled,
    }
    pair2_data = {
        "symbol": bot_instance.config.pair2.symbol,
        "fixed_lot_size": bot_instance.config.pair2.fixed_lot_size,
        "fixed_sl_pips": bot_instance.config.pair2.fixed_sl_pips,
        "enabled": bot_instance.config.pair2.enabled,
    }

    return BotStateResponse(
        is_active=bot_instance.is_active,
        selected_symbols=bot_instance.config.selected_symbols[:2],
        pair1=pair1_data,
        pair2=pair2_data,
        enabled_strategies=bot_instance.config.enabled_strategies,
        strategy_type=bot_instance.config.strategy_type,
        fixed_lot_size=bot_instance.config.fixed_lot_size,
        fixed_sl_pips=bot_instance.config.fixed_sl_pips,
        ai_confirmation_enabled=bot_instance.config.ai_confirmation_enabled,
        ai_confidence_threshold=bot_instance.config.ai_confidence_threshold,
        equity=round(equity, 2),
        daily_pnl=round(daily_pnl, 2),
        trades_today=trade_count,
        circuit_breaker_active=cb_active,
        recent_logs=list(reversed(bot_instance.recent_logs[-50:])),
        available_symbols=available,
        mock_mode=bot_instance.config.use_mock_broker,
        broker_info=broker_info,
        accuracy=accuracy,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        total_closed_trades=total_closed,
        active_session=active_session,
        available_strategies=["SMC", "SMC_SCALP_5M", "ICT"],
        stats_by_strategy=stats_by_strategy,
        stats_by_pair=stats_by_pair,
        performance_metrics=metrics,
    )


@app.post("/api/activate")
async def activate_bot():
    """Engage the trading bot. Once active, chart selections are locked."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    if bot_instance.is_active:
        return {"status": "already_active", "message": "Bot is already active."}

    bot_instance.is_active = True
    session_id = bot_instance.state.record_activation(
        symbols=bot_instance.config.selected_symbols[:2],
        lot_size=f"P1:{bot_instance.config.pair1.fixed_lot_size or 'Dyn'} | P2:{bot_instance.config.pair2.fixed_lot_size or 'Dyn'}",
        trigger_source="Web Dashboard",
    )
    bot_instance.log(
        f"🟢 BOT ACTIVATED by user (Session #{session_id}). Monitoring: "
        f"Pair 1 ({bot_instance.config.pair1.symbol}, lot={bot_instance.config.pair1.fixed_lot_size or 'Dyn'}, SL={bot_instance.config.pair1.fixed_sl_pips or 'Dyn'}) | "
        f"Pair 2 ({bot_instance.config.pair2.symbol}, lot={bot_instance.config.pair2.fixed_lot_size or 'Dyn'}, SL={bot_instance.config.pair2.fixed_sl_pips or 'Dyn'}) | "
        f"Active Strategies: {bot_instance.config.enabled_strategies}"
    )

    # Immediately trigger a tick cycle
    asyncio.create_task(asyncio.to_thread(bot_instance.tick))
    return {
        "status": "success",
        "session_id": session_id,
        "message": f"Bot activated successfully (Session #{session_id}). Market selection locked.",
    }


@app.post("/api/deactivate")
async def deactivate_bot():
    """Deactivate trading bot. Stops execution loop and unlocks market selection."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    if not bot_instance.is_active:
        return {"status": "already_inactive", "message": "Bot is already deactivated."}

    bot_instance.is_active = False
    closed_id = bot_instance.state.record_deactivation("Manual User Stop")
    bot_instance.log(f"🔴 BOT DEACTIVATED by user (Session #{closed_id or '---'}). No trade execution will occur.")
    return {
        "status": "success",
        "session_id": closed_id,
        "message": "Bot deactivated. Trade execution halted.",
    }


@app.post("/api/configure")
async def update_configuration(payload: BotConfigUpdate):
    """
    Update selected charts (Pair 1 & Pair 2), strategies, and lot sizes.
    STRICT RULE: Cannot change symbols or strategies if bot is currently ACTIVE.
    """
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    # Handle Pair 1 & Pair 2 update
    if payload.pair1 is not None:
        p1_sym = payload.pair1.symbol.strip().upper()
        if bot_instance.is_active and p1_sym != bot_instance.config.pair1.symbol:
            raise HTTPException(
                status_code=400,
                detail="Market type changes are LOCKED during activation! Deactivate the bot first to switch Pair 1."
            )
        bot_instance.config.pair1.symbol = p1_sym
        bot_instance.config.pair1.fixed_lot_size = payload.pair1.fixed_lot_size
        bot_instance.config.pair1.fixed_sl_pips = payload.pair1.fixed_sl_pips
        bot_instance.config.pair1.enabled = payload.pair1.enabled

    if payload.pair2 is not None:
        p2_sym = payload.pair2.symbol.strip().upper()
        if bot_instance.is_active and p2_sym != bot_instance.config.pair2.symbol:
            raise HTTPException(
                status_code=400,
                detail="Market type changes are LOCKED during activation! Deactivate the bot first to switch Pair 2."
            )
        bot_instance.config.pair2.symbol = p2_sym
        bot_instance.config.pair2.fixed_lot_size = payload.pair2.fixed_lot_size
        bot_instance.config.pair2.fixed_sl_pips = payload.pair2.fixed_sl_pips
        bot_instance.config.pair2.enabled = payload.pair2.enabled

    # Handle enabled strategies update
    if payload.enabled_strategies is not None:
        valid_strats = [s for s in payload.enabled_strategies if s in ("SMC", "SMC_SCALP_5M", "ICT")]
        if not valid_strats:
            raise HTTPException(status_code=400, detail="At least 1 strategy must be enabled.")
        if bot_instance.is_active and set(valid_strats) != set(bot_instance.config.enabled_strategies):
            raise HTTPException(
                status_code=400,
                detail="Strategy configuration is LOCKED during activation! Deactivate the bot first to toggle strategies."
            )
        bot_instance.config.enabled_strategies = valid_strats

    # Backward compatibility with selected_symbols
    if payload.selected_symbols is not None and not payload.pair1 and not payload.pair2:
        valid_symbols = [s.strip().upper() for s in payload.selected_symbols if s and s != "NONE"]
        if valid_symbols:
            curr_symbols = [bot_instance.config.pair1.symbol, bot_instance.config.pair2.symbol]
            new_p1 = valid_symbols[0] if len(valid_symbols) >= 1 else bot_instance.config.pair1.symbol
            new_p2 = valid_symbols[1] if len(valid_symbols) >= 2 else ""
            if bot_instance.is_active and (new_p1 != bot_instance.config.pair1.symbol or (len(valid_symbols) >= 2 and new_p2 != bot_instance.config.pair2.symbol)):
                raise HTTPException(
                    status_code=400,
                    detail="Market type changes are LOCKED during activation! Deactivate the bot first to switch symbols."
                )
            bot_instance.config.pair1.symbol = new_p1
            bot_instance.config.pair1.enabled = True
            if len(valid_symbols) >= 2:
                bot_instance.config.pair2.symbol = valid_symbols[1]
                bot_instance.config.pair2.enabled = True
            else:
                bot_instance.config.pair2.enabled = False
            bot_instance.config.selected_symbols = valid_symbols
        else:
            bot_instance.config.selected_symbols = []
    else:
        active_syms = []
        if bot_instance.config.pair1.enabled and bot_instance.config.pair1.symbol:
            active_syms.append(bot_instance.config.pair1.symbol)
        if bot_instance.config.pair2.enabled and bot_instance.config.pair2.symbol:
            active_syms.append(bot_instance.config.pair2.symbol)
        bot_instance.config.selected_symbols = active_syms

    if payload.strategy_type is not None:
        new_strat = payload.strategy_type.strip().upper()
        if bot_instance.is_active and new_strat != bot_instance.config.strategy_type:
            raise HTTPException(
                status_code=400,
                detail="Strategy type changes are LOCKED during activation! Deactivate the bot first to change strategy."
            )
        bot_instance.config.strategy_type = new_strat

    if payload.fixed_lot_size is not None:
        bot_instance.config.fixed_lot_size = payload.fixed_lot_size
    if payload.fixed_sl_pips is not None:
        bot_instance.config.fixed_sl_pips = payload.fixed_sl_pips
    if payload.ai_confirmation_enabled is not None:
        bot_instance.config.ai_confirmation_enabled = payload.ai_confirmation_enabled

    # Persist updated configuration to bot_settings.json
    bot_instance.save_settings()

    bot_instance.log(
        f"⚙️ Configuration updated: "
        f"Pair 1={bot_instance.config.pair1.symbol} (lot={bot_instance.config.pair1.fixed_lot_size or 'Dyn'}, SL={bot_instance.config.pair1.fixed_sl_pips or 'Dyn'}) | "
        f"Pair 2={bot_instance.config.pair2.symbol} (lot={bot_instance.config.pair2.fixed_lot_size or 'Dyn'}, SL={bot_instance.config.pair2.fixed_sl_pips or 'Dyn'}) | "
        f"Strategies={bot_instance.config.enabled_strategies} | "
        f"AI Confirmation={'ON' if bot_instance.config.ai_confirmation_enabled else 'OFF'}"
    )

    return {
        "status": "success",
        "pair1": bot_instance.config.pair1.model_dump(),
        "pair2": bot_instance.config.pair2.model_dump(),
        "enabled_strategies": bot_instance.config.enabled_strategies,
        "selected_symbols": bot_instance.config.selected_symbols,
        "strategy_type": bot_instance.config.strategy_type,
        "fixed_lot_size": bot_instance.config.fixed_lot_size,
        "fixed_sl_pips": bot_instance.config.fixed_sl_pips,
        "ai_confirmation_enabled": bot_instance.config.ai_confirmation_enabled,
    }





@app.post("/api/trigger_tick")
async def manual_trigger_tick():
    """Manually invoke a tick cycle for testing while active."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    if not bot_instance.is_active:
        raise HTTPException(status_code=400, detail="Bot is deactivated. Activate it first to run a tick.")
    
    asyncio.create_task(asyncio.to_thread(bot_instance.tick))
    return {"status": "success", "message": "Manual tick cycle triggered."}


@app.get("/api/trades")
async def get_trade_history():
    """Return historical executed trades, entry/exit prices, status, holding times, and performance metrics."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    
    records = bot_instance.state.get_all_trades(limit=100)
    trades_data = []
    now_dt = datetime.now(timezone.utc)

    for r in records:
        dur_sec = r.duration_seconds
        if (dur_sec <= 0.0 or dur_sec is None) and r.timestamp:
            try:
                t_dt = r.timestamp if r.timestamp.tzinfo else r.timestamp.replace(tzinfo=timezone.utc)
                if r.status != 'OPEN' and r.closed_at:
                    c_dt = r.closed_at if r.closed_at.tzinfo else r.closed_at.replace(tzinfo=timezone.utc)
                    dur_sec = max(0.0, (c_dt - t_dt).total_seconds())
                else:
                    dur_sec = max(0.0, (now_dt - t_dt).total_seconds())
            except Exception:
                dur_sec = 0.0

        # Calculate Planned R:R
        sl_dist = abs(r.entry_price - r.stop_loss)
        tp_dist = abs(r.take_profit - r.entry_price)
        planned_rr = round(tp_dist / sl_dist, 2) if sl_dist > 0.00001 else 2.0

        trades_data.append({
            "id": r.id,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "closed_at": r.closed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if r.closed_at else None,
            "symbol": r.symbol,
            "direction": r.direction.value if hasattr(r.direction, 'value') else str(r.direction),
            "strategy_name": r.strategy_name,
            "entry_price": round(r.entry_price, 5),
            "stop_loss": round(r.stop_loss, 5),
            "take_profit": round(r.take_profit, 5),
            "planned_rr": planned_rr,
            "lot_size": round(r.lot_size, 2),
            "realized_pnl": round(r.realized_pnl, 2),
            "status": r.status,
            "duration_seconds": round(dur_sec, 1),
            "holding_time_formatted": bot_instance.state.__class__.__module__ and format_duration(dur_sec),
        })

    metrics = bot_instance.state.get_performance_metrics()

    return {
        "trades": trades_data,
        "metrics": metrics,
        "total_trades": metrics.get("total_trades", len(trades_data)),
        "total_closed_trades": metrics.get("total_closed_trades", 0),
        "total_open_trades": metrics.get("total_open_trades", 0),
        "total_tp": metrics.get("total_tp", 0),
        "total_sl": metrics.get("total_sl", 0),
        "total_pnl": metrics.get("total_pnl", 0.0),
        "accuracy": metrics.get("accuracy", 0.0),
        "win_rate": metrics.get("win_rate", 0.0),
        "win_loss_diff": metrics.get("win_loss_diff", 0),
        "winning_trades": metrics.get("winning_trades", 0),
        "losing_trades": metrics.get("losing_trades", 0),
        "formatted_avg_rr": metrics.get("formatted_avg_rr", "1:2.50"),
        "formatted_realized_rr": metrics.get("formatted_realized_rr", "1:2.50"),
        "formatted_avg_holding_time": metrics.get("formatted_avg_holding_time", "---"),
        "formatted_total_holding_time": metrics.get("formatted_total_holding_time", "---"),
        "profit_factor": metrics.get("profit_factor", 0.0),
    }


@app.get("/api/performance_metrics")
@app.get("/api/analytics")
async def get_analytics_metrics():
    """Return comprehensive performance metrics, SL/TP stats, win rates, RR ratios, and holding times."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    return bot_instance.state.get_performance_metrics()


@app.post("/api/trades/clear")
async def clear_trades():
    """Clear closed trade ledger history."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    bot_instance.state.clear_trade_history()
    bot_instance.log("🧹 Trade ledger cleared.")
    return {"status": "success", "message": "Trade history cleared."}


@app.get("/api/trades/export_csv")
@app.get("/api/trades/download_csv")
async def export_trades_csv():
    """Download trade ledger as a CSV file."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    bot_instance.state._sync_trades_csv()
    csv_file = Path(bot_instance.state.csv_path)
    if not csv_file.exists():
        raise HTTPException(status_code=404, detail="Trades CSV not found")
    return FileResponse(
        path=str(csv_file),
        filename="trades_history.csv",
        media_type="text/csv"
    )


@app.get("/api/sessions/export_csv")
async def export_sessions_csv():
    """Download bot sessions history as a CSV file."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    bot_instance.state._sync_sessions_csv()
    csv_file = Path(bot_instance.state.sessions_csv_path)
    if not csv_file.exists():
        raise HTTPException(status_code=404, detail="Sessions CSV not found")
    return FileResponse(
        path=str(csv_file),
        filename="sessions_history.csv",
        media_type="text/csv"
    )


@app.get("/api/activation_history")
@app.get("/api/activation-history")
async def get_activation_history(limit: int = 100):
    """Return historical bot activation / deactivation records and stats."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    history = bot_instance.state.get_activation_history(limit=limit)
    stats = bot_instance.state.get_activation_stats()
    return {
        "sessions": history,
        "stats": stats,
    }


@app.post("/api/activation_history/clear")
@app.post("/api/activation-history/clear")
async def clear_activation_history():
    """Clear historical completed/interrupted sessions."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    bot_instance.state.clear_activation_history()
    bot_instance.log("🧹 Bot activation/deactivation session history cleared.")
    return {"status": "success", "message": "Activation history cleared."}


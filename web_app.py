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
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import TradingConfig, DEFAULT_CONFIG, InstrumentConfig, get_instrument
from main import TradingBot, parse_ltf_to_seconds


# ─────────────────────────────────────────────
#  Pydantic Schemas for Web API
# ─────────────────────────────────────────────

class BotConfigUpdate(BaseModel):
    selected_symbols: List[str] = Field(
        ...,
        min_length=1,
        max_length=2,
        description="List of exactly 1 or 2 symbols to monitor simultaneously."
    )
    strategy_type: Optional[str] = Field(
        default=None,
        description="Trading strategy algorithm selection (e.g. SMC, EMA_CROSS)."
    )
    fixed_lot_size: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Manual fixed lot size override. None uses risk % dynamic sizing."
    )
    fixed_sl_pips: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Manual fixed Stop Loss in pips. None uses SMC dynamic structure stop loss."
    )
    ai_confirmation_enabled: Optional[bool] = Field(
        default=True,
        description="Enable/disable Gemini AI second-opinion confirmation gate."
    )


class BotStateResponse(BaseModel):
    is_active: bool
    selected_symbols: List[str]
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
    available_strategies: List[str] = ["SMC", "SMC_SCALP_5M"]




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

    # Start background scheduler
    scheduler_instance = AsyncIOScheduler(timezone="UTC")
    interval = parse_ltf_to_seconds(config.timeframes.ltf)
    
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
    logger.info(f"Background tick scheduler active (interval: {interval}s). Bot starts DEACTIVATED.")

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

    # Calculate win accuracy across all historical closed trades
    all_trades = bot_instance.state.get_all_trades(limit=500)
    closed_trades = [t for t in all_trades if t.status != "OPEN"]
    winning_trades = [t for t in closed_trades if t.realized_pnl > 0]
    losing_trades = [t for t in closed_trades if t.realized_pnl < 0]
    accuracy = round((len(winning_trades) / len(closed_trades) * 100), 1) if closed_trades else 0.0

    return BotStateResponse(
        is_active=bot_instance.is_active,
        selected_symbols=bot_instance.config.selected_symbols[:2],
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
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        total_closed_trades=len(closed_trades),
        active_session=active_session,
        available_strategies=["SMC", "SMC_SCALP_5M"],
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
        lot_size=bot_instance.config.fixed_lot_size,
        trigger_source="Web Dashboard",
    )
    bot_instance.log(
        f"🟢 BOT ACTIVATED by user (Session #{session_id}). Monitoring: {bot_instance.config.selected_symbols[:2]} | "
        f"Strategy: {bot_instance.config.strategy_type} | "
        f"Lot Size: {bot_instance.config.fixed_lot_size or 'Dynamic (0.5% risk)'}"
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
    Update selected charts, strategy, and lot size.
    STRICT RULE: Cannot change symbols or strategy if bot is currently ACTIVE.
    """
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    # Clean symbols: filter out 'NONE', empty strings, and duplicates
    valid_symbols = []
    for s in payload.selected_symbols:
        s_clean = s.strip().upper() if s else ""
        if s_clean and s_clean != "NONE" and s_clean not in valid_symbols:
            valid_symbols.append(s_clean)

    if not valid_symbols:
        raise HTTPException(status_code=400, detail="At least 1 chart must be selected (cannot both be NONE).")

    # Lock enforcement
    if bot_instance.is_active:
        current_set = set(bot_instance.config.selected_symbols[:2])
        new_set = set(valid_symbols[:2])
        if current_set != new_set:
            bot_instance.log("⚠️ Configuration rejected: Cannot change market symbols while bot is ACTIVE.", level="WARNING")
            raise HTTPException(
                status_code=400,
                detail="Market type changes are LOCKED during activation! Deactivate the bot first to switch charts."
            )
        if payload.strategy_type is not None and payload.strategy_type != bot_instance.config.strategy_type:
            bot_instance.log("⚠️ Configuration rejected: Cannot change strategy type while bot is ACTIVE.", level="WARNING")
            raise HTTPException(
                status_code=400,
                detail="Strategy type changes are LOCKED during activation! Deactivate the bot first to switch strategies."
            )

    # Validate symbols exist
    available = {i.symbol for i in bot_instance.config.instruments}
    for s in valid_symbols:
        if s not in available:
            raise HTTPException(status_code=400, detail=f"Symbol '{s}' is not supported.")

    # Update config
    bot_instance.config.selected_symbols = valid_symbols[:2]
    if payload.strategy_type is not None:
        bot_instance.config.strategy_type = payload.strategy_type
        if scheduler_instance:
            try:
                active_ltf = "5m" if payload.strategy_type == "SMC_SCALP_5M" else bot_instance.config.timeframes.ltf
                new_interval = parse_ltf_to_seconds(active_ltf)
                job = scheduler_instance.get_job("web_trading_tick")
                if job:
                    job.reschedule(trigger=IntervalTrigger(seconds=new_interval))
                    logger.info(f"Rescheduled tick job for interval: {new_interval}s ({active_ltf})")
            except Exception as e:
                logger.warning(f"Could not reschedule tick interval: {e}")
    bot_instance.config.fixed_lot_size = payload.fixed_lot_size
    bot_instance.config.fixed_sl_pips = payload.fixed_sl_pips
    if payload.ai_confirmation_enabled is not None:
        bot_instance.config.ai_confirmation_enabled = payload.ai_confirmation_enabled

    # Persist updated configuration to bot_settings.json
    bot_instance.save_settings()

    bot_instance.log(
        f"⚙️ Configuration updated: Symbols={bot_instance.config.selected_symbols} | "
        f"Strategy={bot_instance.config.strategy_type} | "
        f"Lot Size={payload.fixed_lot_size if payload.fixed_lot_size else 'Dynamic'} | "
        f"Fixed SL={f'{payload.fixed_sl_pips} pips' if payload.fixed_sl_pips else 'Dynamic SMC'} | "
        f"AI Confirmation={'ON' if bot_instance.config.ai_confirmation_enabled else 'OFF'}"
    )


    return {
        "status": "success",
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
    """Return historical executed trades, entry/exit prices, status, and net PnL."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    
    records = bot_instance.state.get_all_trades(limit=100)
    trades_data = []
    total_realized_pnl = 0.0

    for r in records:
        total_realized_pnl += r.realized_pnl
        trades_data.append({
            "id": r.id,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "symbol": r.symbol,
            "direction": r.direction.value if hasattr(r.direction, 'value') else str(r.direction),
            "entry_price": round(r.entry_price, 5),
            "stop_loss": round(r.stop_loss, 5),
            "take_profit": round(r.take_profit, 5),
            "lot_size": round(r.lot_size, 2),
            "realized_pnl": round(r.realized_pnl, 2),
            "status": r.status,
        })

    closed = [t for t in trades_data if t["status"] != "OPEN"]
    wins = [t for t in closed if t["realized_pnl"] > 0]
    losses = [t for t in closed if t["realized_pnl"] < 0]
    accuracy = round((len(wins) / len(closed) * 100), 1) if closed else 0.0

    return {
        "trades": trades_data,
        "total_trades": len(trades_data),
        "total_pnl": round(total_realized_pnl, 2),
        "accuracy": accuracy,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "total_closed_trades": len(closed),
    }


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


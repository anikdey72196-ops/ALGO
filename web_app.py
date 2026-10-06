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
import time
import socket
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Any
from pathlib import Path
from contextlib import asynccontextmanager

# Lightweight in-memory cache for /api/state to prevent redundant SQLite & MT5 query storms
_state_cache: dict[str, Any] = {"response": None, "ts": 0.0}
_STATE_CACHE_TTL = 1.5  # seconds

def invalidate_state_cache():
    global _state_cache
    _state_cache["ts"] = 0.0
    _state_cache["response"] = None

# pyrefly: ignore [missing-import]
import io
import csv
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from loguru import logger
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from core.config import TradingConfig, DEFAULT_CONFIG, InstrumentConfig, get_instrument
from core.state import format_duration
from main import TradingBot, parse_ltf_to_seconds

try:
    from ml.smart_partial_tp import SmartPartialTPService
except ImportError:
    SmartPartialTPService = None


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
    pair3: Optional[PairConfigItem] = None
    enabled_strategies: Optional[List[str]] = None
    selected_symbols: Optional[List[str]] = None
    strategy_type: Optional[str] = None
    fixed_lot_size: Optional[float] = None
    fixed_sl_pips: Optional[float] = None
    ai_confirmation_enabled: Optional[bool] = None
    pair_configs: Optional[dict] = None
    max_open_positions: Optional[int] = None
    night_limit_enabled: Optional[bool] = None
    night_max_open_positions: Optional[int] = None
    night_start_hour: Optional[int] = None
    night_end_hour: Optional[int] = None
    night_timezone_mode: Optional[str] = None
    ml_gating_enabled: Optional[bool] = None
    ml_max_sl_probability: Optional[float] = None
    ml_shadow_mode: Optional[bool] = None
    temporal_ml_enabled: Optional[bool] = None
    reversal_strategy_enabled: Optional[bool] = None
    directional_loss_cooldown_enabled: Optional[bool] = None


class BotStateResponse(BaseModel):
    is_active: bool
    selected_symbols: List[str]
    pair1: dict
    pair2: dict
    pair3: dict
    pairs_config: dict = {}
    enabled_strategies: List[str]
    strategy_type: str
    fixed_lot_size: Optional[float]
    fixed_sl_pips: Optional[float]
    max_open_positions: int = 15
    effective_max_open_positions: int = 15
    is_night_window: bool = False
    night_limit_enabled: bool = True
    night_max_open_positions: int = 2
    night_start_hour: int = 23
    night_end_hour: int = 8
    night_timezone_mode: str = "Asia/Kolkata"
    directional_loss_cooldown_enabled: bool = False
    ai_confirmation_enabled: bool
    ai_confidence_threshold: float
    ml_gating_enabled: bool = True
    ml_max_sl_probability: float = 0.50
    ml_shadow_mode: bool = True
    temporal_ml_enabled: bool = True
    reversal_strategy_enabled: bool = False
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
    available_strategies: List[str] = ["SMC", "SMC_SCALP_5M", "ICT", "ORDER_FLOW", "TREND_REVERSAL"]
    stats_by_strategy: dict = {}
    stats_by_pair: dict = {}
    performance_metrics: dict = {}
    ml_trap_detector: dict = {}
    trap_stats: dict = {}
    temporal_ml_model: dict = {}
    open_positions_by_strategy: dict = {}
    execution_summary: dict = {}
    trend_reversal: dict = {}
    ml_logs: List[str] = []
    current_window: dict = {}
    temporal_ml_shadow_mode: bool = True
    daily_bias: dict = {}


class CompactBotState(BaseModel):
    """Ultra-lightweight state DTO optimized for high-frequency mobile polling and WebSockets."""
    is_active: bool
    equity: float
    daily_pnl: float
    trades_today: int
    circuit_breaker_active: bool
    accuracy: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    total_closed_trades: int = 0
    performance_metrics: dict = {}
    current_window: dict = {}
    daily_bias: dict = {}
    ai_confirmation_enabled: bool = True
    ai_confidence_threshold: float = 0.75
    enabled_strategies: List[str] = []
    reversal_strategy_enabled: bool = False
    selected_symbols: List[str] = []
    pairs_config: dict = {}
    broker_info: Optional[dict] = None


# ─────────────────────────────────────────────
#  Global Bot Instance & Background Scheduler
# ─────────────────────────────────────────────

bot_instance: TradingBot = TradingBot(DEFAULT_CONFIG)
bot_instance.startup()
scheduler_instance: AsyncIOScheduler | None = None
labeler_task_instance: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize background tick scheduler and ML labeler."""
    global bot_instance, scheduler_instance, labeler_task_instance
    logger.info("Initializing Web Control Server...")

    config = bot_instance.config

    # Start background scheduler — fast 60-second (1-minute) interval for high responsiveness
    scheduler_instance = AsyncIOScheduler(timezone="UTC")
    interval = 60
    
    # Tick job runs every interval, but bot.tick() checks `if not self.is_active: return`
    async def safe_tick_wrapper():
        if bot_instance.is_active:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, bot_instance.tick)

    scheduler_instance.add_job(
        safe_tick_wrapper,
        trigger=IntervalTrigger(seconds=interval),
        id="web_trading_tick",
        name="Web Trading Tick",
        max_instances=1,
        misfire_grace_time=30,
    )
    scheduler_instance.start()
    logger.info(f"Background tick scheduler active (fast 60s scan interval). Bot starts DEACTIVATED.")

    # Schedule ML Trap Detector background labeler task
    labeler_task_instance = asyncio.create_task(
        bot_instance.trap_svc.run_labeler(
            history_provider=lambda sym, tf, n: bot_instance._prepare_df_for_trap_detector(
                bot_instance._get_ohlcv(sym, tf, count=max(n, 300))
            )
        )
    )
    logger.info("ML Trap Detector background labeler task active.")

    yield

    # Shutdown
    if labeler_task_instance:
        labeler_task_instance.cancel()
    if scheduler_instance:
        scheduler_instance.shutdown(wait=False)
    if bot_instance:
        if bot_instance.is_active:
            bot_instance.state.record_deactivation("Server Shutdown")
            bot_instance.is_active = False
        bot_instance.shutdown()
    logger.info("Web Control Server shut down cleanly.")



app = FastAPI(title="Algorithmic Trading Bot Dashboard", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=300, compresslevel=5)

# Templates and static directories
TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_all_lan_ips() -> list[str]:
    """Discover all non-loopback IPv4 addresses (Wi-Fi, Ethernet, Tailscale)."""
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        for addr_info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = addr_info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass

    return sorted(list(ips)) if ips else ["127.0.0.1"]


# ─────────────────────────────────────────────
#  Web API Routes
# ─────────────────────────────────────────────

@app.get("/favicon.ico")
async def get_favicon():
    """Silence browser favicon 404 request."""
    return Response(status_code=204)


@app.get("/manifest.json")
async def get_manifest():
    """Serve PWA Web App Manifest for mobile installation."""
    manifest_path = STATIC_DIR / "manifest.json"
    if manifest_path.exists():
        return FileResponse(manifest_path, media_type="application/manifest+json")
    raise HTTPException(status_code=404, detail="Manifest not found")


@app.get("/sw.js")
async def get_service_worker():
    """Serve PWA Service Worker for mobile caching and home screen install."""
    sw_path = STATIC_DIR / "sw.js"
    if sw_path.exists():
        return FileResponse(sw_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service worker not found")


@app.get("/terminal", response_class=HTMLResponse)
async def serve_terminal(request: Request):
    """Serve the dedicated Mobile Terminal PWA web app."""
    tpl = "terminal_controls.html" if (TEMPLATES_DIR / "terminal_controls.html").exists() else "terminal.html"
    return templates.TemplateResponse(request=request, name=tpl)


@app.get("/terminal/controls", response_class=HTMLResponse)
async def serve_terminal_controls(request: Request):
    """Serve Glacier Terminal Controls tab."""
    tpl = "terminal_controls.html" if (TEMPLATES_DIR / "terminal_controls.html").exists() else "terminal.html"
    return templates.TemplateResponse(request=request, name=tpl)


@app.get("/terminal/strategy", response_class=HTMLResponse)
async def serve_terminal_strategy(request: Request):
    """Serve Glacier Terminal Strategy tab."""
    tpl = "terminal_strategy.html" if (TEMPLATES_DIR / "terminal_strategy.html").exists() else "terminal_controls.html"
    return templates.TemplateResponse(request=request, name=tpl)


@app.get("/terminal/analysis", response_class=HTMLResponse)
async def serve_terminal_analysis(request: Request):
    """Serve Glacier Terminal Analysis tab."""
    tpl = "terminal_analysis.html" if (TEMPLATES_DIR / "terminal_analysis.html").exists() else "terminal_controls.html"
    return templates.TemplateResponse(request=request, name=tpl)



@app.get("/api/network-info")
async def get_network_info():
    """Return local LAN IP addresses and phone terminal URLs for QR code display."""
    ips = get_all_lan_ips()
    primary_ip = ips[0] if ips else "127.0.0.1"
    port = 8000

    terminal_urls = [f"http://{ip}:{port}/terminal" for ip in ips]
    dashboard_urls = [f"http://{ip}:{port}/" for ip in ips]

    return {
        "primary_ip": primary_ip,
        "port": port,
        "available_ips": ips,
        "terminal_url": terminal_urls[0] if terminal_urls else f"http://127.0.0.1:{port}/terminal",
        "terminal_urls": terminal_urls,
        "dashboard_urls": dashboard_urls,
    }


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )



def get_compact_state_payload() -> CompactBotState:
    """Compute lightweight state payload tailored for mobile UI and WebSockets."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    equity = bot_instance.broker.get_account_equity() if bot_instance.broker else 0.0
    daily_summary = bot_instance.state.get_daily_summary()
    metrics = bot_instance.state.get_performance_metrics()

    broker_info = None
    if hasattr(bot_instance.broker, "get_account_info"):
        broker_info = bot_instance.broker.get_account_info()

    pairs_cfg_data = {}
    default_meta = {
        "XAUUSD": {"name": "Gold / USD", "lot": 0.05, "sl": 25.0},
        "EURUSD": {"name": "Euro / USD", "lot": 0.10, "sl": 15.0},
        "GBPUSD": {"name": "Pound / USD", "lot": 0.12, "sl": 20.0},
        "BTCUSD": {"name": "Bitcoin / USD", "lot": 0.01, "sl": 150.0},
        "ETHUSD": {"name": "Ethereum / USD", "lot": 0.05, "sl": 80.0},
    }
    for s_name, meta in default_meta.items():
        lot_val = meta["lot"]
        sl_val = meta["sl"]
        en_val = s_name in bot_instance.config.selected_symbols

        if hasattr(bot_instance.config, "pair_configs") and s_name in bot_instance.config.pair_configs:
            cfg = bot_instance.config.pair_configs[s_name]
            if cfg.fixed_lot_size is not None:
                lot_val = cfg.fixed_lot_size
            if cfg.fixed_sl_pips is not None:
                sl_val = cfg.fixed_sl_pips
            en_val = cfg.enabled and en_val
        elif bot_instance.config.pair1.symbol == s_name:
            if bot_instance.config.pair1.fixed_lot_size is not None:
                lot_val = bot_instance.config.pair1.fixed_lot_size
            if bot_instance.config.pair1.fixed_sl_pips is not None:
                sl_val = bot_instance.config.pair1.fixed_sl_pips
        elif bot_instance.config.pair2.symbol == s_name:
            if bot_instance.config.pair2.fixed_lot_size is not None:
                lot_val = bot_instance.config.pair2.fixed_lot_size
            if bot_instance.config.pair2.fixed_sl_pips is not None:
                sl_val = bot_instance.config.pair2.fixed_sl_pips
        elif bot_instance.config.pair3.symbol == s_name:
            if bot_instance.config.pair3.fixed_lot_size is not None:
                lot_val = bot_instance.config.pair3.fixed_lot_size
            if bot_instance.config.pair3.fixed_sl_pips is not None:
                sl_val = bot_instance.config.pair3.fixed_sl_pips

        pairs_cfg_data[s_name.lower()] = {
            "symbol": s_name,
            "name": meta["name"],
            "lot": lot_val,
            "sl": sl_val,
            "active": en_val,
        }

    return CompactBotState(
        is_active=bot_instance.is_active,
        equity=round(equity, 2),
        daily_pnl=round(daily_summary["realized_pnl"], 2),
        trades_today=daily_summary["trade_count"],
        circuit_breaker_active=daily_summary["circuit_breaker_active"],
        accuracy=metrics.get("accuracy", 0.0),
        winning_trades=metrics.get("winning_trades", 0),
        losing_trades=metrics.get("losing_trades", 0),
        total_closed_trades=metrics.get("total_closed_trades", 0),
        performance_metrics=metrics,
        current_window=getattr(bot_instance, "window_recorder", None).get_window_info() if hasattr(bot_instance, "window_recorder") and bot_instance.window_recorder else {},
        daily_bias=bot_instance.get_daily_bias_summary() if hasattr(bot_instance, "get_daily_bias_summary") else {},
        ai_confirmation_enabled=bot_instance.config.ai_confirmation_enabled,
        ai_confidence_threshold=bot_instance.config.ai_confidence_threshold,
        enabled_strategies=bot_instance.config.enabled_strategies,
        reversal_strategy_enabled=getattr(bot_instance.config, "reversal_strategy_enabled", False),
        selected_symbols=bot_instance.config.selected_symbols,
        pairs_config=pairs_cfg_data,
        broker_info=broker_info,
    )


@app.get("/api/state/compact", response_model=CompactBotState)
async def get_compact_state():
    """Lightweight compact state endpoint for mobile screens."""
    return get_compact_state_payload()


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    """Real-time WebSocket connection streaming compact bot state with low bandwidth and zero polling overhead."""
    await websocket.accept()
    try:
        while True:
            if bot_instance:
                compact_state = get_compact_state_payload()
                state_json = compact_state.model_dump_json() if hasattr(compact_state, "model_dump_json") else compact_state.json()
                await websocket.send_text(state_json)
            await asyncio.sleep(2.5)
    except (WebSocketDisconnect, ConnectionResetError):
        pass
    except Exception as e:
        logger.debug(f"WebSocket client disconnected: {e}")


@app.get("/api/state")
async def get_bot_state(compact: bool = False):
    """Return live system state, metrics, and logs (pass ?compact=true for lightweight payload)."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    if compact:
        return get_compact_state_payload()

    global _state_cache
    now = time.time()
    if _state_cache["response"] is not None and (now - _state_cache["ts"]) < _STATE_CACHE_TTL:
        return _state_cache["response"]

    equity = bot_instance.broker.get_account_equity()
    daily_summary = bot_instance.state.get_daily_summary()
    daily_pnl = daily_summary["realized_pnl"]
    trade_count = daily_summary["trade_count"]
    cb_active = daily_summary["circuit_breaker_active"]
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
    pair3_data = {
        "symbol": bot_instance.config.pair3.symbol,
        "fixed_lot_size": bot_instance.config.pair3.fixed_lot_size,
        "fixed_sl_pips": bot_instance.config.pair3.fixed_sl_pips,
        "enabled": bot_instance.config.pair3.enabled,
    }

    # Build multi-pair lot size and SL configuration dictionary
    pairs_cfg_data = {}
    default_meta = {
        "XAUUSD": {"name": "Gold / USD", "lot": 0.05, "sl": 25.0},
        "EURUSD": {"name": "Euro / USD", "lot": 0.10, "sl": 15.0},
        "GBPUSD": {"name": "Pound / USD", "lot": 0.12, "sl": 20.0},
        "BTCUSD": {"name": "Bitcoin / USD", "lot": 0.01, "sl": 150.0},
        "ETHUSD": {"name": "Ethereum / USD", "lot": 0.05, "sl": 80.0},
    }
    for s_name, meta in default_meta.items():
        lot_val = meta["lot"]
        sl_val = meta["sl"]
        en_val = s_name in bot_instance.config.selected_symbols

        if hasattr(bot_instance.config, "pair_configs") and s_name in bot_instance.config.pair_configs:
            cfg = bot_instance.config.pair_configs[s_name]
            if cfg.fixed_lot_size is not None:
                lot_val = cfg.fixed_lot_size
            if cfg.fixed_sl_pips is not None:
                sl_val = cfg.fixed_sl_pips
            en_val = cfg.enabled and en_val
        elif bot_instance.config.pair1.symbol == s_name:
            if bot_instance.config.pair1.fixed_lot_size is not None:
                lot_val = bot_instance.config.pair1.fixed_lot_size
            if bot_instance.config.pair1.fixed_sl_pips is not None:
                sl_val = bot_instance.config.pair1.fixed_sl_pips
        elif bot_instance.config.pair2.symbol == s_name:
            if bot_instance.config.pair2.fixed_lot_size is not None:
                lot_val = bot_instance.config.pair2.fixed_lot_size
            if bot_instance.config.pair2.fixed_sl_pips is not None:
                sl_val = bot_instance.config.pair2.fixed_sl_pips
        elif bot_instance.config.pair3.symbol == s_name:
            if bot_instance.config.pair3.fixed_lot_size is not None:
                lot_val = bot_instance.config.pair3.fixed_lot_size
            if bot_instance.config.pair3.fixed_sl_pips is not None:
                sl_val = bot_instance.config.pair3.fixed_sl_pips

        pairs_cfg_data[s_name.lower()] = {
            "symbol": s_name,
            "name": meta["name"],
            "lot": lot_val,
            "sl": sl_val,
            "active": en_val,
        }

    # ML Trap Detector status & statistics
    trap_stats = {}
    if hasattr(bot_instance, "trap_svc") and bot_instance.trap_svc:
        try:
            store_stats = bot_instance.trap_svc.store.stats()
            today_stats = bot_instance.trap_svc.store.stats_today() if hasattr(bot_instance.trap_svc.store, "stats_today") else {}
            strat_stats = bot_instance.trap_svc.store.stats_by_strategy() if hasattr(bot_instance.trap_svc.store, "stats_by_strategy") else {}
            is_shadow = (
                bot_instance.trap_svc.cfg.shadow_until_samples > 0
                and bot_instance.trap_svc.model.n_samples < bot_instance.trap_svc.cfg.shadow_until_samples
                and not bot_instance.trap_svc.model._sgd_fitted
                and bot_instance.trap_svc.model.lgbm is None
            )
            strategy_models = {}
            if hasattr(bot_instance.trap_svc, "models"):
                for s_name, m in bot_instance.trap_svc.models.items():
                    s_stat = strat_stats.get(s_name, {})
                    strategy_models[s_name] = {
                        "model_version": m.version,
                        "n_samples": m.n_samples,
                        "total_events": s_stat.get("total", 0),
                        "labeled_events": s_stat.get("labeled", 0),
                        "traps_caught": s_stat.get("traps", 0),
                        "genuine_setups": s_stat.get("genuine", 0),
                    }
            shadow_perf = {}
            if hasattr(bot_instance.trap_svc, "shadow_prediction_stats"):
                try:
                    shadow_perf = bot_instance.trap_svc.shadow_prediction_stats(
                        threshold=getattr(bot_instance.trap_svc.cfg, "max_sl_probability", 0.50)
                    )
                except Exception as ex:
                    shadow_perf = {"error": str(ex)}

            trap_stats = {
                "model_version": bot_instance.trap_svc.model.version,
                "n_samples": bot_instance.trap_svc.model.n_samples,
                "shadow_until": bot_instance.trap_svc.cfg.shadow_until_samples,
                "mode": "shadow" if is_shadow else "gated",
                "threshold": getattr(bot_instance.trap_svc.cfg, "max_sl_probability", 0.50),
                "max_sl_probability": getattr(bot_instance.trap_svc.cfg, "max_sl_probability", 0.50),
                "total_events": store_stats.get("total", 0),
                "labeled_events": store_stats.get("labeled", 0),
                "traps_caught": store_stats.get("traps", 0),
                "genuine_setups": store_stats.get("genuine", 0),
                "traps_today": today_stats.get("traps_caught", 0),
                "vetoed_today": today_stats.get("vetoed", 0),
                "total_today": today_stats.get("total", 0),
                "gating_enabled": getattr(bot_instance.config, "ml_gating_enabled", True),
                "strategies": strategy_models,
                "shadow_performance": shadow_perf,
            }
        except Exception as e:
            trap_stats = {"error": str(e)}

    # Open positions breakdown by strategy
    open_positions_by_strategy = {"SMC": 0, "SMC_SCALP_5M": 0, "ICT": 0, "ORDER_FLOW": 0, "TREND_REVERSAL": 0}
    try:
        open_trades = bot_instance.state.get_open_positions()
        for t in open_trades:
            strat_name = getattr(t, "strategy_name", None) or "SMC"
            open_positions_by_strategy[strat_name] = open_positions_by_strategy.get(strat_name, 0) + 1
    except Exception as e:
        logger.warning(f"Failed to fetch open positions by strategy: {e}")

    # Execution Quality summary (last 24h)
    exec_summary = {}
    if hasattr(bot_instance, "eqm_aggregator") and bot_instance.eqm_aggregator:
        try:
            exec_summary = bot_instance.eqm_aggregator.get_summary(window_hours=24)
        except Exception as e:
            logger.warning(f"Failed to fetch execution quality summary: {e}")

    # Real-time Trend Reversal & CHoCH status
    trend_reversal_data = {}
    if hasattr(bot_instance, "trend_reversal_status"):
        for sym, rev in bot_instance.trend_reversal_status.items():
            trend_reversal_data[sym] = rev.to_dict()

    # Temporal & Session ML Model statistics
    temporal_stats = {}
    if hasattr(bot_instance, "temporal_svc") and bot_instance.temporal_svc:
        try:
            temporal_stats = bot_instance.temporal_svc.get_summary()
        except Exception as e:
            temporal_stats = {"error": str(e)}

    return BotStateResponse(
        is_active=bot_instance.is_active,
        selected_symbols=bot_instance.config.selected_symbols,
        pair1=pair1_data,
        pair2=pair2_data,
        pair3=pair3_data,
        pairs_config=pairs_cfg_data,
        enabled_strategies=bot_instance.config.enabled_strategies,
        strategy_type=bot_instance.config.strategy_type,
        fixed_lot_size=bot_instance.config.fixed_lot_size,
        fixed_sl_pips=bot_instance.config.fixed_sl_pips,
        max_open_positions=getattr(bot_instance.config.risk, 'max_open_positions', 15),
        effective_max_open_positions=bot_instance.config.risk.get_effective_max_open_positions() if hasattr(bot_instance.config.risk, 'get_effective_max_open_positions') else getattr(bot_instance.config.risk, 'max_open_positions', 15),
        is_night_window=bot_instance.config.risk.is_night_window() if hasattr(bot_instance.config.risk, 'is_night_window') else False,
        night_limit_enabled=getattr(bot_instance.config.risk, 'night_limit_enabled', True),
        night_max_open_positions=getattr(bot_instance.config.risk, 'night_max_open_positions', 2),
        night_start_hour=getattr(bot_instance.config.risk, 'night_start_hour', 23),
        night_end_hour=getattr(bot_instance.config.risk, 'night_end_hour', 8),
        night_timezone_mode=getattr(bot_instance.config.risk, 'night_timezone_mode', 'Asia/Kolkata'),
        directional_loss_cooldown_enabled=getattr(bot_instance.config.risk, 'directional_loss_cooldown_enabled', False),
        ai_confirmation_enabled=bot_instance.config.ai_confirmation_enabled,
        ai_confidence_threshold=bot_instance.config.ai_confidence_threshold,
        ml_gating_enabled=getattr(bot_instance.config, "ml_gating_enabled", True),
        ml_max_sl_probability=getattr(bot_instance.config, "ml_max_sl_probability", 0.50),
        ml_shadow_mode=getattr(bot_instance.config, "ml_shadow_mode", False),
        temporal_ml_enabled=getattr(bot_instance.config, "temporal_ml_enabled", True),
        reversal_strategy_enabled=getattr(bot_instance.config, "reversal_strategy_enabled", False),
        equity=round(equity, 2),
        daily_pnl=round(daily_pnl, 2),
        trades_today=trade_count,
        circuit_breaker_active=cb_active,
        recent_logs=list(reversed(bot_instance.recent_logs[-150:])),
        available_symbols=available,
        mock_mode=bot_instance.config.use_mock_broker,
        broker_info=broker_info,
        accuracy=accuracy,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        total_closed_trades=total_closed,
        active_session=active_session,
        available_strategies=["SMC", "SMC_SCALP_5M", "ICT", "ORDER_FLOW", "TREND_REVERSAL"],
        stats_by_strategy=stats_by_strategy,
        stats_by_pair=stats_by_pair,
        performance_metrics=metrics,
        ml_trap_detector=trap_stats,
        trap_stats=trap_stats,
        temporal_ml_model=temporal_stats,
        open_positions_by_strategy=open_positions_by_strategy,
        execution_summary=exec_summary,
        trend_reversal=trend_reversal_data,
        ml_logs=list(reversed(getattr(bot_instance, "ml_logs", [])[-100:])),
        current_window=getattr(bot_instance, "window_recorder", None).get_window_info() if hasattr(bot_instance, "window_recorder") and bot_instance.window_recorder else {},
        temporal_ml_shadow_mode=getattr(bot_instance.config, "temporal_ml_shadow_mode", True),
        daily_bias=bot_instance.get_daily_bias_summary() if hasattr(bot_instance, "get_daily_bias_summary") else {},
    )
    _state_cache["response"] = res
    _state_cache["ts"] = time.time()
    return res


@app.get("/api/logs")
async def get_system_logs(category: Optional[str] = None, limit: int = 200):
    """
    Return recent logs with optional category filtering (ML, TRADE, REVERSAL, WARN, ALL).
    Combines in-memory ring buffers and disk logs if necessary.
    """
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    cat = (category or "ALL").upper()
    
    if cat in ("ML", "AI"):
        # Prioritize dedicated ml_logs buffer
        logs = list(getattr(bot_instance, "ml_logs", []))
        for l in bot_instance.recent_logs:
            if any(k in l for k in ["ML", "TRAP", "Trap", "🪤", "🔬", "🤖", "AI", "TEMPORAL", "DECISION", "vetoed"]):
                if l not in logs:
                    logs.append(l)
    elif cat == "REVERSAL":
        logs = [l for l in bot_instance.recent_logs if any(k in l for k in ["REVERSAL", "CHoCH", "🚨", "🛡️", "SWEEP"])]
    elif cat == "TRADE":
        logs = [l for l in bot_instance.recent_logs if any(k in l for k in ["ORDER", "FILLED", "TRADE", "POSITION", "✅", "AUTHORIZED", "ACTIVATED", "SIGNAL"])]
    elif cat == "WARN":
        logs = [l for l in bot_instance.recent_logs if any(k in l for k in ["⚠️", "WARN", "🚫", "⛔", "ERROR", "FAILED", "SKIPPED", "REJECTED"])]
    else:
        logs = list(bot_instance.recent_logs)

    return {
        "status": "ok",
        "category": cat,
        "count": len(logs),
        "logs": list(reversed(logs[-limit:]))
    }


@app.get("/api/logs/compact")
async def get_compact_logs(limit: int = 80):
    """Return only the last `limit` log lines trimmed to 160 characters for high-speed mobile network transfers."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    raw_logs = list(bot_instance.recent_logs[-limit:])
    trimmed = [
        (l[:160] + "…") if len(l) > 160 else l
        for l in raw_logs
    ]
    return {
        "status": "ok",
        "count": len(trimmed),
        "logs": list(reversed(trimmed))
    }


@app.get("/api/trend-reversal")
async def get_trend_reversal_status():
    """Return real-time CHoCH & Trend Reversal analysis for all monitored pairs."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    return {
        "status": "ok",
        "data": {
            sym: rev.to_dict()
            for sym, rev in getattr(bot_instance, "trend_reversal_status", {}).items()
        }
    }


@app.get("/api/daily-bias")
async def get_daily_bias_api(symbol: Optional[str] = None, date: Optional[str] = None):
    """
    Return today's directional bias summary or specific symbol/date bias.
    Allows quick polling and switching between pairs on the frontend card.
    """
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    if date:
        res = bot_instance.state.get_all_daily_biases(date_str=date)
        return {"date": date, "symbols": res}
    return bot_instance.get_daily_bias_summary(target_symbol=symbol)


@app.get("/api/daily-bias/history")
async def get_daily_bias_history_api(symbol: Optional[str] = None, limit: int = 14):
    """Return historical daily bias log entries for auditing and review."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")
    return bot_instance.state.get_daily_bias_history(symbol=symbol, limit=limit)


@app.post("/api/activate")
async def activate_bot():
    """Engage the trading bot. Once active, chart selections are locked."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    if bot_instance.is_active:
        return {"status": "already_active", "message": "Bot is already active."}

    # Ensure all enabled pairs are included in selected_symbols
    active_syms = list(bot_instance.config.selected_symbols) if bot_instance.config.selected_symbols else []
    for p_cfg in (bot_instance.config.pair1, bot_instance.config.pair2, bot_instance.config.pair3):
        if p_cfg and getattr(p_cfg, "enabled", False) and p_cfg.symbol:
            p_s = p_cfg.symbol.strip().upper()
            if p_s and p_s != "NONE" and p_s not in active_syms:
                active_syms.append(p_s)
    bot_instance.config.selected_symbols = active_syms

    bot_instance.is_active = True
    invalidate_state_cache()
    session_id = bot_instance.state.record_activation(
        symbols=bot_instance.config.selected_symbols,
        lot_size=f"P1:{bot_instance.config.pair1.fixed_lot_size or 'Dyn'} | P2:{bot_instance.config.pair2.fixed_lot_size or 'Dyn'} | P3:{bot_instance.config.pair3.fixed_lot_size or 'Dyn'}",
        trigger_source="Web Dashboard",
    )
    bot_instance.log(
        f"🟢 BOT ACTIVATED by user (Session #{session_id}). Monitoring pairs: [{', '.join(bot_instance.config.selected_symbols)}] | "
        f"Pair 1 ({bot_instance.config.pair1.symbol}, lot={bot_instance.config.pair1.fixed_lot_size or 'Dyn'}, SL={bot_instance.config.pair1.fixed_sl_pips or 'Dyn'}) | "
        f"Pair 2 ({bot_instance.config.pair2.symbol}, lot={bot_instance.config.pair2.fixed_lot_size or 'Dyn'}, SL={bot_instance.config.pair2.fixed_sl_pips or 'Dyn'}) | "
        f"Pair 3 ({bot_instance.config.pair3.symbol}, lot={bot_instance.config.pair3.fixed_lot_size or 'Dyn'}, SL={bot_instance.config.pair3.fixed_sl_pips or 'Dyn'}) | "
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
    invalidate_state_cache()
    closed_id = bot_instance.state.record_deactivation("Manual User Stop")
    bot_instance.log(f"🔴 BOT DEACTIVATED by user (Session #{closed_id or '---'}). No trade execution will occur.")
    return {
        "status": "success",
        "session_id": closed_id,
        "message": "Bot deactivated. Trade execution halted.",
    }


@app.post("/api/update")
async def handle_dashboard_update(payload: dict):
    """
    Unified dashboard update endpoint supporting state toggles,
    strategy toggles, pair toggles, and configuration saves.
    """
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    invalidate_state_cache()
    action = payload.get("action")
    if action == "set_status":
        status = payload.get("status")
        if status in ("RUNNING", "ACTIVATED"):
            if not bot_instance.is_active:
                await activate_bot()
            return {"status": "success", "botStatus": "RUNNING"}
        elif status in ("HALTED", "IDLE", "DEACTIVATED"):
            if bot_instance.is_active:
                await deactivate_bot()
            return {"status": "success", "botStatus": status}

    elif action == "toggle_strategy":
        if bot_instance.is_active:
            raise HTTPException(
                status_code=400,
                detail="Strategy modification is LOCKED while bot is ACTIVE! Switch bot to STANDBY first."
            )
        strat_id = payload.get("strategy")
        active = payload.get("active", True)
        strat_map = {
            "smc": "SMC",
            "swing": "SMC",
            "scalp5m": "SMC_SCALP_5M",
            "scalp-5m": "SMC_SCALP_5M",
            "scalp_5m": "SMC_SCALP_5M",
            "smc_scalp_5m": "SMC_SCALP_5M",
            "ict": "ICT",
            "ict-inst": "ICT",
            "orderflow": "ORDER_FLOW",
            "order-flow": "ORDER_FLOW",
            "order_flow": "ORDER_FLOW",
            "reversal": "TREND_REVERSAL",
            "trend_reversal": "TREND_REVERSAL",
            "trend-reversal": "TREND_REVERSAL",
        }
        lookup_key = str(strat_id).lower().replace(" ", "_")
        strat_name = strat_map.get(lookup_key, strat_map.get(strat_id, str(strat_id).upper().replace("-", "_")))
        current = list(bot_instance.config.enabled_strategies)
        if active and strat_name not in current:
            current.append(strat_name)
        elif not active and strat_name in current:
            current.remove(strat_name)
        bot_instance.config.enabled_strategies = current

        bot_instance.config.reversal_strategy_enabled = ("TREND_REVERSAL" in current)

        if hasattr(bot_instance, "strategy") and hasattr(bot_instance.strategy, "set_enabled_strategies"):
            bot_instance.strategy.set_enabled_strategies(current)

        bot_instance.save_settings()
        invalidate_state_cache()
        return {
            "status": "success",
            "enabled_strategies": current,
            "reversal_strategy_enabled": getattr(bot_instance.config, "reversal_strategy_enabled", False),
            "message": f"Strategy {strat_name} {'enabled' if active else 'disabled'}"
        }

    elif action == "toggle_pair":
        if bot_instance.is_active:
            raise HTTPException(
                status_code=400,
                detail="Pair modification is LOCKED while bot is ACTIVE! Switch bot to STANDBY first."
            )
        pair_id = payload.get("pair")
        active = payload.get("active", True)
        sym_map = {"xauusd": "XAUUSD", "eurusd": "EURUSD", "gbpusd": "GBPUSD", "btcusd": "BTCUSD", "ethusd": "ETHUSD"}
        sym = sym_map.get(pair_id, str(pair_id).upper())
        current = list(bot_instance.config.selected_symbols)
        if active and sym not in current:
            current.append(sym)
        elif not active and sym in current:
            current.remove(sym)
        bot_instance.config.selected_symbols = current
        bot_instance.save_settings()
        return {"status": "success", "selected_symbols": current}

    elif action == "toggle_trailing":
        if bot_instance.is_active:
            raise HTTPException(
                status_code=400,
                detail="Trailing Stop setting is LOCKED while bot is ACTIVE! Switch bot to STANDBY first."
            )
        enabled = payload.get("enabled", True)
        bot_instance.config.trailing_stop_mode = "STRUCTURE" if enabled else "NONE"
        bot_instance.save_settings()
        return {"status": "success", "trailing_stop_mode": bot_instance.config.trailing_stop_mode}

    elif action == "update_pair_settings":
        if bot_instance.is_active:
            raise HTTPException(
                status_code=400,
                detail="Pair risk settings are LOCKED while bot is ACTIVE! Switch bot to STANDBY first."
            )
        pair_id = payload.get("pair")
        lot_val = payload.get("lot")
        sl_val = payload.get("sl")
        sym_map = {"xauusd": "XAUUSD", "eurusd": "EURUSD", "gbpusd": "GBPUSD", "btcusd": "BTCUSD", "ethusd": "ETHUSD"}
        sym = sym_map.get(str(pair_id).lower(), str(pair_id).upper())

        from core.config import PairSettings
        if not hasattr(bot_instance.config, "pair_configs"):
            bot_instance.config.pair_configs = {}
        if sym not in bot_instance.config.pair_configs:
            bot_instance.config.pair_configs[sym] = PairSettings(symbol=sym)

        if lot_val is not None:
            val_f = round(float(lot_val), 2)
            bot_instance.config.pair_configs[sym].fixed_lot_size = val_f
            if sym == bot_instance.config.pair1.symbol:
                bot_instance.config.pair1.fixed_lot_size = val_f
            elif sym == bot_instance.config.pair2.symbol:
                bot_instance.config.pair2.fixed_lot_size = val_f
            elif sym == bot_instance.config.pair3.symbol:
                bot_instance.config.pair3.fixed_lot_size = val_f

        if sl_val is not None:
            sl_f = round(float(sl_val), 1)
            bot_instance.config.pair_configs[sym].fixed_sl_pips = sl_f
            if sym == bot_instance.config.pair1.symbol:
                bot_instance.config.pair1.fixed_sl_pips = sl_f
            elif sym == bot_instance.config.pair2.symbol:
                bot_instance.config.pair2.fixed_sl_pips = sl_f
            elif sym == bot_instance.config.pair3.symbol:
                bot_instance.config.pair3.fixed_sl_pips = sl_f

        bot_instance.save_settings()
        return {
            "status": "success",
            "symbol": sym,
            "lot": bot_instance.config.pair_configs[sym].fixed_lot_size,
            "sl": bot_instance.config.pair_configs[sym].fixed_sl_pips,
        }

    elif action == "save_configuration":
        if bot_instance.is_active:
            raise HTTPException(
                status_code=400,
                detail="Configuration changes are LOCKED during activation! Deactivate the bot first."
            )
        config = payload.get("config", {})
        strats = config.get("strategies", {})
        if strats:
            enabled_strats = []
            strat_mapping = {
                "smc": "SMC", "swing": "SMC",
                "scalp5m": "SMC_SCALP_5M", "scalp-5m": "SMC_SCALP_5M", "scalp_5m": "SMC_SCALP_5M", "smc_scalp_5m": "SMC_SCALP_5M",
                "ict": "ICT", "ict-inst": "ICT",
                "orderflow": "ORDER_FLOW", "order-flow": "ORDER_FLOW", "order_flow": "ORDER_FLOW",
                "reversal": "TREND_REVERSAL", "trend_reversal": "TREND_REVERSAL", "trend-reversal": "TREND_REVERSAL"
            }
            for k, v in strats.items():
                if v:
                    strat_name = strat_mapping.get(str(k).lower().replace(" ", "_"), str(k).upper().replace("-", "_"))
                    if strat_name not in enabled_strats:
                        enabled_strats.append(strat_name)
            bot_instance.config.enabled_strategies = enabled_strats
            bot_instance.config.reversal_strategy_enabled = ("TREND_REVERSAL" in enabled_strats)
            if hasattr(bot_instance, "strategy") and hasattr(bot_instance.strategy, "set_enabled_strategies"):
                bot_instance.strategy.set_enabled_strategies(enabled_strats)

        pairs = config.get("pairs", {})
        if pairs:
            enabled_pairs = []
            from core.config import PairSettings
            if not hasattr(bot_instance.config, "pair_configs"):
                bot_instance.config.pair_configs = {}

            for k, v in pairs.items():
                sym = {"xauusd": "XAUUSD", "eurusd": "EURUSD", "gbpusd": "GBPUSD", "btcusd": "BTCUSD", "ethusd": "ETHUSD"}.get(str(k).lower(), str(k).upper())
                is_active = True
                lot_val = None
                sl_val = None

                if isinstance(v, dict):
                    is_active = bool(v.get("active", True))
                    lot_val = v.get("lot")
                    sl_val = v.get("sl")
                elif isinstance(v, bool):
                    is_active = v

                if is_active:
                    enabled_pairs.append(sym)

                if sym not in bot_instance.config.pair_configs:
                    bot_instance.config.pair_configs[sym] = PairSettings(symbol=sym)

                bot_instance.config.pair_configs[sym].enabled = is_active
                if lot_val is not None:
                    lf = round(float(lot_val), 2)
                    bot_instance.config.pair_configs[sym].fixed_lot_size = lf
                    if sym == bot_instance.config.pair1.symbol: bot_instance.config.pair1.fixed_lot_size = lf
                    elif sym == bot_instance.config.pair2.symbol: bot_instance.config.pair2.fixed_lot_size = lf
                    elif sym == bot_instance.config.pair3.symbol: bot_instance.config.pair3.fixed_lot_size = lf
                if sl_val is not None:
                    sf = round(float(sl_val), 1)
                    bot_instance.config.pair_configs[sym].fixed_sl_pips = sf
                    if sym == bot_instance.config.pair1.symbol: bot_instance.config.pair1.fixed_sl_pips = sf
                    elif sym == bot_instance.config.pair2.symbol: bot_instance.config.pair2.fixed_sl_pips = sf
                    elif sym == bot_instance.config.pair3.symbol: bot_instance.config.pair3.fixed_sl_pips = sf

            bot_instance.config.selected_symbols = enabled_pairs

        if "aiGateEnabled" in config:
            bot_instance.config.ai_confirmation_enabled = bool(config["aiGateEnabled"])

        bot_instance.save_settings()
        return {"status": "success", "message": "Configuration and pair Lot/SL settings saved"}

    return {"status": "success", "message": "Acknowledged"}


@app.post("/api/configure")
async def update_configuration(payload: BotConfigUpdate):
    """
    Update selected charts (Pair 1 & Pair 2), strategies, and lot sizes.
    STRICT RULE: Cannot change symbols or strategies if bot is currently ACTIVE.
    """
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    invalidate_state_cache()

    # Handle Pair 1 & Pair 2 update
    if payload.pair1 is not None:
        p1_sym = payload.pair1.symbol.strip().upper()
        if bot_instance.is_active:
            if p1_sym != bot_instance.config.pair1.symbol:
                raise HTTPException(
                    status_code=400,
                    detail="Market pair changes are LOCKED during activation! Deactivate the bot first to switch Pair 1."
                )
            if payload.pair1.fixed_lot_size != bot_instance.config.pair1.fixed_lot_size:
                raise HTTPException(
                    status_code=400,
                    detail="Pair 1 lot size is LOCKED during activation! Deactivate the bot first to modify lot size."
                )
            if payload.pair1.fixed_sl_pips != bot_instance.config.pair1.fixed_sl_pips:
                raise HTTPException(
                    status_code=400,
                    detail="Pair 1 Stop Loss (SL) is LOCKED during activation! Deactivate the bot first to modify SL."
                )
            if payload.pair1.enabled != bot_instance.config.pair1.enabled:
                raise HTTPException(
                    status_code=400,
                    detail="Pair 1 status is LOCKED during activation! Deactivate the bot first to enable/disable."
                )
        bot_instance.config.pair1.symbol = p1_sym
        bot_instance.config.pair1.fixed_lot_size = payload.pair1.fixed_lot_size
        bot_instance.config.pair1.fixed_sl_pips = payload.pair1.fixed_sl_pips
        bot_instance.config.pair1.enabled = payload.pair1.enabled

    if payload.pair2 is not None:
        p2_sym = payload.pair2.symbol.strip().upper()
        if bot_instance.is_active:
            if p2_sym != bot_instance.config.pair2.symbol:
                raise HTTPException(
                    status_code=400,
                    detail="Market pair changes are LOCKED during activation! Deactivate the bot first to switch Pair 2."
                )
            if payload.pair2.fixed_lot_size != bot_instance.config.pair2.fixed_lot_size:
                raise HTTPException(
                    status_code=400,
                    detail="Pair 2 lot size is LOCKED during activation! Deactivate the bot first to modify lot size."
                )
            if payload.pair2.fixed_sl_pips != bot_instance.config.pair2.fixed_sl_pips:
                raise HTTPException(
                    status_code=400,
                    detail="Pair 2 Stop Loss (SL) is LOCKED during activation! Deactivate the bot first to modify SL."
                )
            if payload.pair2.enabled != bot_instance.config.pair2.enabled:
                raise HTTPException(
                    status_code=400,
                    detail="Pair 2 status is LOCKED during activation! Deactivate the bot first to enable/disable."
                )
        bot_instance.config.pair2.symbol = p2_sym
        bot_instance.config.pair2.fixed_lot_size = payload.pair2.fixed_lot_size
        bot_instance.config.pair2.fixed_sl_pips = payload.pair2.fixed_sl_pips
        bot_instance.config.pair2.enabled = payload.pair2.enabled

    if payload.pair3 is not None:
        p3_sym = payload.pair3.symbol.strip().upper()
        if bot_instance.is_active:
            if p3_sym != bot_instance.config.pair3.symbol:
                raise HTTPException(
                    status_code=400,
                    detail="Market pair changes are LOCKED during activation! Deactivate the bot first to switch Pair 3."
                )
            if payload.pair3.fixed_lot_size != bot_instance.config.pair3.fixed_lot_size:
                raise HTTPException(
                    status_code=400,
                    detail="Pair 3 lot size is LOCKED during activation! Deactivate the bot first to modify lot size."
                )
            if payload.pair3.fixed_sl_pips != bot_instance.config.pair3.fixed_sl_pips:
                raise HTTPException(
                    status_code=400,
                    detail="Pair 3 Stop Loss (SL) is LOCKED during activation! Deactivate the bot first to modify SL."
                )
            if payload.pair3.enabled != bot_instance.config.pair3.enabled:
                raise HTTPException(
                    status_code=400,
                    detail="Pair 3 status is LOCKED during activation! Deactivate the bot first to enable/disable."
                )
        bot_instance.config.pair3.symbol = p3_sym
        bot_instance.config.pair3.fixed_lot_size = payload.pair3.fixed_lot_size
        bot_instance.config.pair3.fixed_sl_pips = payload.pair3.fixed_sl_pips
        bot_instance.config.pair3.enabled = payload.pair3.enabled

    # Handle multi-pair independent configurations (pair_configs)
    if payload.pair_configs is not None:
        if bot_instance.is_active:
            raise HTTPException(
                status_code=400,
                detail="Pair configurations are LOCKED during activation! Deactivate the bot first to modify Lot Size or SL."
            )
        from core.config import PairSettings
        if not hasattr(bot_instance.config, "pair_configs"):
            bot_instance.config.pair_configs = {}
        for k, v in payload.pair_configs.items():
            sym = {"xauusd": "XAUUSD", "eurusd": "EURUSD", "gbpusd": "GBPUSD", "btcusd": "BTCUSD", "ethusd": "ETHUSD"}.get(str(k).lower(), str(k).upper())
            if sym not in bot_instance.config.pair_configs:
                bot_instance.config.pair_configs[sym] = PairSettings(symbol=sym)
            if isinstance(v, dict):
                if "lot" in v and v["lot"] is not None:
                    bot_instance.config.pair_configs[sym].fixed_lot_size = round(float(v["lot"]), 2)
                elif "fixed_lot_size" in v and v["fixed_lot_size"] is not None:
                    bot_instance.config.pair_configs[sym].fixed_lot_size = round(float(v["fixed_lot_size"]), 2)
                if "sl" in v and v["sl"] is not None:
                    bot_instance.config.pair_configs[sym].fixed_sl_pips = round(float(v["sl"]), 1)
                elif "fixed_sl_pips" in v and v["fixed_sl_pips"] is not None:
                    bot_instance.config.pair_configs[sym].fixed_sl_pips = round(float(v["fixed_sl_pips"]), 1)
                if "enabled" in v:
                    bot_instance.config.pair_configs[sym].enabled = bool(v["enabled"])

                # Sync back to pair1, pair2, pair3 if matches
                if sym == bot_instance.config.pair1.symbol:
                    if bot_instance.config.pair_configs[sym].fixed_lot_size is not None:
                        bot_instance.config.pair1.fixed_lot_size = bot_instance.config.pair_configs[sym].fixed_lot_size
                    if bot_instance.config.pair_configs[sym].fixed_sl_pips is not None:
                        bot_instance.config.pair1.fixed_sl_pips = bot_instance.config.pair_configs[sym].fixed_sl_pips
                elif sym == bot_instance.config.pair2.symbol:
                    if bot_instance.config.pair_configs[sym].fixed_lot_size is not None:
                        bot_instance.config.pair2.fixed_lot_size = bot_instance.config.pair_configs[sym].fixed_lot_size
                    if bot_instance.config.pair_configs[sym].fixed_sl_pips is not None:
                        bot_instance.config.pair2.fixed_sl_pips = bot_instance.config.pair_configs[sym].fixed_sl_pips
                elif sym == bot_instance.config.pair3.symbol:
                    if bot_instance.config.pair_configs[sym].fixed_lot_size is not None:
                        bot_instance.config.pair3.fixed_lot_size = bot_instance.config.pair_configs[sym].fixed_lot_size
                    if bot_instance.config.pair_configs[sym].fixed_sl_pips is not None:
                        bot_instance.config.pair3.fixed_sl_pips = bot_instance.config.pair_configs[sym].fixed_sl_pips

    # Handle enabled strategies update
    if payload.enabled_strategies is not None:
        valid_strats = [s for s in payload.enabled_strategies if s in ("SMC", "SMC_SCALP_5M", "ICT", "ORDER_FLOW", "TREND_REVERSAL")]
        if not valid_strats:
            raise HTTPException(status_code=400, detail="At least 1 strategy must be enabled.")
        if bot_instance.is_active and set(valid_strats) != set(bot_instance.config.enabled_strategies):
            raise HTTPException(
                status_code=400,
                detail="Strategy configuration is LOCKED during activation! Deactivate the bot first to toggle strategies."
            )
        bot_instance.config.enabled_strategies = valid_strats
        bot_instance.strategy.set_enabled_strategies(valid_strats)
        bot_instance.config.reversal_strategy_enabled = ("TREND_REVERSAL" in valid_strats)

    # Handle selected_symbols (Multi-Pair Concurrent Scanning)
    if payload.selected_symbols is not None:
        valid_symbols = [s.strip().upper() for s in payload.selected_symbols if s and s != "NONE"]
        if valid_symbols:
            if bot_instance.is_active and set(valid_symbols) != set(bot_instance.config.selected_symbols):
                raise HTTPException(
                    status_code=400,
                    detail="Market type changes are LOCKED during activation! Deactivate the bot first to switch symbols."
                )
            bot_instance.config.selected_symbols = valid_symbols
            if not payload.pair1 and not payload.pair2 and not payload.pair3:
                bot_instance.config.pair1.symbol = valid_symbols[0]
                bot_instance.config.pair1.enabled = True
                if len(valid_symbols) >= 2:
                    bot_instance.config.pair2.symbol = valid_symbols[1]
                    bot_instance.config.pair2.enabled = True
                else:
                    bot_instance.config.pair2.enabled = False
                if len(valid_symbols) >= 3:
                    bot_instance.config.pair3.symbol = valid_symbols[2]
                    bot_instance.config.pair3.enabled = True
                else:
                    bot_instance.config.pair3.enabled = False
        else:
            bot_instance.config.selected_symbols = []
    # Ensure all enabled pairs (Pair 1, Pair 2, Pair 3) are present in selected_symbols
    active_syms = list(bot_instance.config.selected_symbols) if bot_instance.config.selected_symbols else []
    for p_cfg in (bot_instance.config.pair1, bot_instance.config.pair2, bot_instance.config.pair3):
        if p_cfg and getattr(p_cfg, "enabled", False) and p_cfg.symbol:
            p_s = p_cfg.symbol.strip().upper()
            if p_s and p_s != "NONE" and p_s not in active_syms:
                active_syms.append(p_s)
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
        if bot_instance.is_active and payload.fixed_lot_size != bot_instance.config.fixed_lot_size:
            raise HTTPException(
                status_code=400,
                detail="Lot size is LOCKED during activation! Deactivate the bot first to modify lot size."
            )
        bot_instance.config.fixed_lot_size = payload.fixed_lot_size

    if payload.fixed_sl_pips is not None:
        if bot_instance.is_active and payload.fixed_sl_pips != bot_instance.config.fixed_sl_pips:
            raise HTTPException(
                status_code=400,
                detail="Stop Loss (SL) is LOCKED during activation! Deactivate the bot first to modify SL."
            )
        bot_instance.config.fixed_sl_pips = payload.fixed_sl_pips

    if payload.max_open_positions is not None:
        if bot_instance.is_active and payload.max_open_positions != bot_instance.config.risk.max_open_positions:
            raise HTTPException(
                status_code=400,
                detail="Max open positions limit is LOCKED during activation! Deactivate the bot first to modify trade limits."
            )
        bot_instance.config.risk.max_open_positions = payload.max_open_positions

    if payload.night_limit_enabled is not None:
        if bot_instance.is_active and payload.night_limit_enabled != getattr(bot_instance.config.risk, 'night_limit_enabled', True):
            raise HTTPException(
                status_code=400,
                detail="Night trade limit is LOCKED during activation! Deactivate the bot first to modify."
            )
        bot_instance.config.risk.night_limit_enabled = payload.night_limit_enabled

    if payload.night_max_open_positions is not None:
        if bot_instance.is_active and payload.night_max_open_positions != getattr(bot_instance.config.risk, 'night_max_open_positions', 2):
            raise HTTPException(
                status_code=400,
                detail="Night trade limit capacity is LOCKED during activation! Deactivate the bot first to modify."
            )
        bot_instance.config.risk.night_max_open_positions = payload.night_max_open_positions

    if payload.night_start_hour is not None:
        if bot_instance.is_active and payload.night_start_hour != getattr(bot_instance.config.risk, 'night_start_hour', 23):
            raise HTTPException(
                status_code=400,
                detail="Night start hour is LOCKED during activation! Deactivate the bot first to modify."
            )
        bot_instance.config.risk.night_start_hour = payload.night_start_hour

    if payload.night_end_hour is not None:
        if bot_instance.is_active and payload.night_end_hour != getattr(bot_instance.config.risk, 'night_end_hour', 8):
            raise HTTPException(
                status_code=400,
                detail="Night end hour is LOCKED during activation! Deactivate the bot first to modify."
            )
        bot_instance.config.risk.night_end_hour = payload.night_end_hour

    if payload.night_timezone_mode is not None:
        if bot_instance.is_active and payload.night_timezone_mode != getattr(bot_instance.config.risk, 'night_timezone_mode', 'Asia/Kolkata'):
            raise HTTPException(
                status_code=400,
                detail="Night timezone mode is LOCKED during activation! Deactivate the bot first to modify."
            )
        bot_instance.config.risk.night_timezone_mode = payload.night_timezone_mode

    if payload.directional_loss_cooldown_enabled is not None:
        bot_instance.config.risk.directional_loss_cooldown_enabled = payload.directional_loss_cooldown_enabled

    if payload.ai_confirmation_enabled is not None:
        if bot_instance.is_active and payload.ai_confirmation_enabled != bot_instance.config.ai_confirmation_enabled:
            raise HTTPException(
                status_code=400,
                detail="AI Confirmation setting is LOCKED during activation! Deactivate the bot first to toggle AI gate."
            )
        bot_instance.config.ai_confirmation_enabled = payload.ai_confirmation_enabled

    if payload.ml_gating_enabled is not None:
        if bot_instance.is_active and payload.ml_gating_enabled != getattr(bot_instance.config, 'ml_gating_enabled', True):
            raise HTTPException(
                status_code=400,
                detail="ML Trap Gate setting is LOCKED during activation! Deactivate the bot first to toggle ML filter."
            )
        bot_instance.config.ml_gating_enabled = payload.ml_gating_enabled

    if payload.ml_max_sl_probability is not None:
        if bot_instance.is_active and payload.ml_max_sl_probability != getattr(bot_instance.config, 'ml_max_sl_probability', 0.50):
            raise HTTPException(
                status_code=400,
                detail="ML Max Stop Loss Risk threshold is LOCKED during activation! Deactivate the bot first to modify threshold."
            )
        bot_instance.config.ml_max_sl_probability = payload.ml_max_sl_probability
        if hasattr(bot_instance, "trap_svc") and bot_instance.trap_svc:
            bot_instance.trap_svc.cfg.max_sl_probability = payload.ml_max_sl_probability
            bot_instance.trap_svc.cfg.p_genuine_threshold = 1.0 - payload.ml_max_sl_probability

    if payload.ml_shadow_mode is not None:
        bot_instance.config.ml_shadow_mode = payload.ml_shadow_mode
        if hasattr(bot_instance, "trap_svc") and bot_instance.trap_svc:
            bot_instance.trap_svc.cfg.shadow_mode = payload.ml_shadow_mode

    if payload.temporal_ml_enabled is not None:
        if bot_instance.is_active and payload.temporal_ml_enabled != getattr(bot_instance.config, 'temporal_ml_enabled', True):
            raise HTTPException(
                status_code=400,
                detail="Temporal ML setting is LOCKED during activation! Deactivate the bot first to toggle Temporal ML filter."
            )
        bot_instance.config.temporal_ml_enabled = payload.temporal_ml_enabled
        if hasattr(bot_instance, "temporal_svc") and bot_instance.temporal_svc:
            bot_instance.temporal_svc.cfg.active_gating = payload.temporal_ml_enabled

    if payload.reversal_strategy_enabled is not None:
        bot_instance.config.reversal_strategy_enabled = payload.reversal_strategy_enabled

    # Persist updated configuration to bot_settings.json
    bot_instance.save_settings()

    bot_instance.log(
        f"⚙️ Configuration updated: "
        f"Pair 1={bot_instance.config.pair1.symbol} (lot={bot_instance.config.pair1.fixed_lot_size or 'Dyn'}, SL={bot_instance.config.pair1.fixed_sl_pips or 'Dyn'}) | "
        f"Pair 2={bot_instance.config.pair2.symbol} (lot={bot_instance.config.pair2.fixed_lot_size or 'Dyn'}, SL={bot_instance.config.pair2.fixed_sl_pips or 'Dyn'}) | "
        f"Pair 3={bot_instance.config.pair3.symbol} (lot={bot_instance.config.pair3.fixed_lot_size or 'Dyn'}, SL={bot_instance.config.pair3.fixed_sl_pips or 'Dyn'}) | "
        f"Strategies={bot_instance.config.enabled_strategies} | "
        f"Max Open Positions={bot_instance.config.risk.max_open_positions} | "
        f"AI Confirmation={'ON' if bot_instance.config.ai_confirmation_enabled else 'OFF'} | "
        f"ML Trap Filter={'ON' if getattr(bot_instance.config, 'ml_gating_enabled', True) else 'OFF'} (max_sl={getattr(bot_instance.config, 'ml_max_sl_probability', 0.50):.2f}) | "
        f"Temporal ML={'ON' if getattr(bot_instance.config, 'temporal_ml_enabled', True) else 'OFF'}"
    )

    return {
        "status": "success",
        "pair1": bot_instance.config.pair1.model_dump(),
        "pair2": bot_instance.config.pair2.model_dump(),
        "pair3": bot_instance.config.pair3.model_dump(),
        "enabled_strategies": bot_instance.config.enabled_strategies,
        "selected_symbols": bot_instance.config.selected_symbols,
        "strategy_type": bot_instance.config.strategy_type,
        "fixed_lot_size": bot_instance.config.fixed_lot_size,
        "fixed_sl_pips": bot_instance.config.fixed_sl_pips,
        "max_open_positions": bot_instance.config.risk.max_open_positions,
        "night_limit_enabled": getattr(bot_instance.config.risk, 'night_limit_enabled', True),
        "night_max_open_positions": getattr(bot_instance.config.risk, 'night_max_open_positions', 2),
        "night_start_hour": getattr(bot_instance.config.risk, 'night_start_hour', 23),
        "night_end_hour": getattr(bot_instance.config.risk, 'night_end_hour', 8),
        "night_timezone_mode": getattr(bot_instance.config.risk, 'night_timezone_mode', 'Asia/Kolkata'),
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


@app.get("/api/ml/temporal/summary")
async def get_temporal_ml_summary():
    """Return top profitable and losing days, hours, and market sessions from ML Model 2."""
    if not bot_instance or not hasattr(bot_instance, "temporal_svc"):
        raise HTTPException(status_code=500, detail="Temporal ML service not initialized")
    return bot_instance.temporal_svc.get_summary()


@app.get("/api/ml/temporal/breakdown")
async def get_temporal_ml_breakdown():
    """Return detailed Day, Hour, Session, and DayxSession cross-matrices."""
    if not bot_instance or not hasattr(bot_instance, "temporal_svc"):
        raise HTTPException(status_code=500, detail="Temporal ML service not initialized")
    return bot_instance.temporal_svc.get_full_breakdown()


@app.post("/api/ml/temporal/retrain")
async def retrain_temporal_ml_model():
    """Trigger on-demand retraining of Temporal & Session ML Model from SQLite trade database."""
    if not bot_instance or not hasattr(bot_instance, "temporal_svc"):
        raise HTTPException(status_code=500, detail="Temporal ML service not initialized")
    summary = bot_instance.temporal_svc.train_and_update()
    bot_instance.log("🧠 [TEMPORAL ML] Model retrained on latest trade records and event logs.")
    return {"status": "success", "message": "Temporal ML model retrained successfully", "summary": summary}


# =============================================================================
#  5-Window Daytime Interval & October Shadow Mode Telemetry Endpoints
# =============================================================================

@app.get("/api/windows/current")
async def get_current_window_status():
    """Return active 3-hour daytime window or night guard window with shadow sizing recommendations."""
    from ml.temporal_window_recorder import window_recorder
    win_info = window_recorder.get_window_info()
    return {
        "status": "ok",
        "current_window": win_info,
        "shadow_mode": getattr(bot_instance.config if bot_instance else None, 'temporal_ml_shadow_mode', True),
    }


@app.get("/api/windows/summary")
async def get_windows_summary():
    """Return cumulative win rates, trade counts, and PnL for each of the 5 daytime windows + night guard."""
    from ml.temporal_window_recorder import window_recorder
    return {
        "status": "ok",
        "windows": window_recorder.get_summary_by_window(),
        "total_records": len(window_recorder.read_all_records()),
        "csv_path": str(window_recorder.csv_path),
    }


@app.get("/api/windows/export_csv")
async def export_windows_csv():
    """Download the October 5-window CSV dataset for lot size manipulation and audit."""
    from ml.temporal_window_recorder import window_recorder, CSV_FILE_PATH
    if not CSV_FILE_PATH.exists():
        window_recorder.backfill_from_database()
    return FileResponse(
        path=CSV_FILE_PATH,
        filename="temporal_windows_october.csv",
        media_type="text/csv",
    )


@app.post("/api/windows/sync")
async def sync_windows_from_database():
    """Trigger incremental backfill of any unsynced closed trades into the October CSV."""
    from ml.temporal_window_recorder import window_recorder
    added = window_recorder.backfill_from_database()
    return {
        "status": "success",
        "new_intervals_added": added,
        "total_intervals": len(window_recorder.read_all_records()),
    }



@app.get("/api/ml/shadow-stats")
async def get_ml_shadow_stats(threshold: float = 0.50):
    """Return AI Shadow Mode prediction accuracy and evaluation metrics."""
    if not bot_instance or not hasattr(bot_instance, "trap_svc") or not bot_instance.trap_svc:
        return {"status": "inactive", "message": "ML Trap Detector not initialized"}
    try:
        stats = bot_instance.trap_svc.shadow_prediction_stats(threshold=threshold)
        return {"status": "success", **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/ml/partial_tp/status")
async def get_smart_partial_tp_status():
    """Return model health, trained sample counts, and execution statistics."""
    svc = None
    if bot_instance and hasattr(bot_instance, "position_manager") and getattr(bot_instance.position_manager, "smart_partial_tp_service", None):
        svc = bot_instance.position_manager.smart_partial_tp_service
    elif SmartPartialTPService is not None:
        svc = SmartPartialTPService.get_instance()

    if not svc:
        return {"status": "disabled", "message": "Smart Partial TP service unavailable"}
    return {"status": "active", **svc.get_status()}


@app.get("/api/ml/partial_tp/events")
async def get_smart_partial_tp_events(limit: int = 50):
    """Return recent partial profit booking evaluations and executions."""
    if not bot_instance or not hasattr(bot_instance, "state"):
        return {"events": []}
    events = bot_instance.state.get_recent_partial_tp_events(limit=limit)
    return {"events": events, "count": len(events)}


@app.post("/api/ml/partial_tp/retrain")
async def retrain_smart_partial_tp_model():
    """Trigger retraining or baseline recalibration for the Smart Partial TP Model."""
    svc = None
    if bot_instance and hasattr(bot_instance, "position_manager") and getattr(bot_instance.position_manager, "smart_partial_tp_service", None):
        svc = bot_instance.position_manager.smart_partial_tp_service
    elif SmartPartialTPService is not None:
        svc = SmartPartialTPService.get_instance()

    if not svc:
        raise HTTPException(status_code=500, detail="Smart Partial TP service unavailable")

    svc.model._train_synthetic_baseline()
    if bot_instance:
        bot_instance.log("🧠 [SMART PARTIAL TP ML] Baseline model retrained and recalibrated.")
    return {"status": "success", "message": "Smart Partial TP model retrained successfully", "status_info": svc.get_status()}



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
    """Stream trade ledger as a CSV file on-the-fly without large memory allocations."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    def iter_trades_csv():
        fieldnames = [
            "id", "timestamp", "closed_at", "symbol", "direction", "entry_price",
            "stop_loss", "take_profit", "lot_size", "realized_pnl",
            "status", "duration_seconds", "strategy_name", "magic_number"
        ]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        with bot_instance.state._lock:
            cursor = bot_instance.state.conn.execute(
                "SELECT id, timestamp, closed_at, symbol, direction, entry_price, "
                "stop_loss, take_profit, lot_size, realized_pnl, status, "
                "duration_seconds, strategy_name, magic_number FROM trade_log ORDER BY id ASC"
            )
            for row in cursor:
                writer.writerow(dict(row))
                yield output.getvalue()
                output.seek(0)
                output.truncate(0)

    return StreamingResponse(
        iter_trades_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trades_history.csv"}
    )


@app.get("/api/sessions/export_csv")
async def export_sessions_csv():
    """Stream bot sessions history as a CSV file on-the-fly."""
    if not bot_instance:
        raise HTTPException(status_code=500, detail="Bot not initialized")

    def iter_sessions_csv():
        fieldnames = [
            "id", "activation_time", "deactivation_time", "formatted_duration",
            "symbols", "lot_size", "trigger_source", "notes", "is_active"
        ]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        with bot_instance.state._lock:
            cursor = bot_instance.state.conn.execute(
                "SELECT id, activation_time, deactivation_time, duration_seconds, "
                "symbols, lot_size, trigger_source, deactivation_reason, status FROM bot_sessions ORDER BY id ASC"
            )
            for row in cursor:
                dur_sec = row["duration_seconds"]
                dur_fmt = format_duration(dur_sec) if dur_sec else ""
                writer.writerow({
                    "id": row["id"],
                    "activation_time": row["activation_time"] or "",
                    "deactivation_time": row["deactivation_time"] or "",
                    "formatted_duration": dur_fmt,
                    "symbols": row["symbols"] or "",
                    "lot_size": row["lot_size"] or "",
                    "trigger_source": row["trigger_source"] or "",
                    "notes": row["deactivation_reason"] or "",
                    "is_active": "YES" if row["status"] == "ACTIVE" else "NO",
                })
                yield output.getvalue()
                output.seek(0)
                output.truncate(0)

    return StreamingResponse(
        iter_sessions_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sessions_history.csv"}
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


# ─────────────────────────────────────────────
#  Execution Quality Metrics Endpoints
# ─────────────────────────────────────────────

@app.get("/api/execution/summary")
async def get_execution_summary(window_hours: int = 24):
    """Return top-level execution quality KPIs (slippage, latency, fill rate)."""
    if not bot_instance or not hasattr(bot_instance, "eqm_aggregator"):
        raise HTTPException(status_code=500, detail="Execution metrics subsystem not initialized")
    return bot_instance.eqm_aggregator.get_summary(window_hours=window_hours)


@app.get("/api/execution/breakdowns")
async def get_execution_breakdowns(window_hours: int = 168):
    """Return breakdowns by symbol, strategy engine, KillZone, and reject reasons."""
    if not bot_instance or not hasattr(bot_instance, "eqm_aggregator"):
        raise HTTPException(status_code=500, detail="Execution metrics subsystem not initialized")
    return bot_instance.eqm_aggregator.get_breakdowns(window_hours=window_hours)


@app.get("/api/execution/latency_histogram")
async def get_execution_latency_histogram(window_hours: int = 168, bins: int = 10):
    """Return latency histogram distribution."""
    if not bot_instance or not hasattr(bot_instance, "eqm_aggregator"):
        raise HTTPException(status_code=500, detail="Execution metrics subsystem not initialized")
    return bot_instance.eqm_aggregator.get_latency_histogram(window_hours=window_hours, bins=bins)


@app.get("/api/execution/slippage_distribution")
async def get_execution_slippage_distribution(window_hours: int = 168):
    """Return slippage distribution breakdown (favorable vs adverse)."""
    if not bot_instance or not hasattr(bot_instance, "eqm_aggregator"):
        raise HTTPException(status_code=500, detail="Execution metrics subsystem not initialized")
    return bot_instance.eqm_aggregator.get_slippage_distribution(window_hours=window_hours)


@app.get("/api/execution/orders")
async def get_execution_orders(limit: int = 100):
    """Return recent raw order execution records."""
    if not bot_instance or not hasattr(bot_instance, "eqm_aggregator"):
        raise HTTPException(status_code=500, detail="Execution metrics subsystem not initialized")
    with bot_instance.eqm_aggregator._get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM execution_orders
            ORDER BY signal_time DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/execution/export_csv")
async def export_execution_metrics_csv():
    """Export execution orders ledger to CSV."""
    if not bot_instance or not hasattr(bot_instance, "eqm_aggregator"):
        raise HTTPException(status_code=500, detail="Execution metrics subsystem not initialized")
    
    import io
    import csv
    with bot_instance.eqm_aggregator._get_connection() as conn:
        rows = conn.execute("SELECT * FROM execution_orders ORDER BY signal_time DESC").fetchall()
    
    if not rows:
        return Response(content="order_id,symbol,strategy_name,status\n", media_type="text/csv", headers={"Content-Disposition": "attachment; filename=execution_metrics.csv"})
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    for r in rows:
        writer.writerow(dict(r))
        
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=execution_metrics.csv"}
    )


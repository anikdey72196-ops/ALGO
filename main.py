

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

from core.config import TradingConfig, DEFAULT_CONFIG, get_instrument, Direction, MarketBias, normalize_strategy_key
from core.state import StateManager, TradeRecord
from core.news_filter import NewsFilter
from core.risk_engine import RiskEngine
from core.conflict_resolver import ConflictResolver
from core.ai_analyst import AIAnalyst, AIDecision
from core.market_regime import MarketRegimeDetector, MarketRegime

from strategies.strategy import StrategyEngine, TradeSignal, clamp_tp_to_rr
from strategies.trend_reversal import (
    TrendReversalDetector, TrendReversalAnalysis, CHoCHType, ReversalStage
)

from execution.execution import (
    BrokerAdapter, MockBrokerAdapter, MT5Adapter,
    BracketOrder, PriceQuote,
)
from execution.execution_metrics import ExecutionMetricsCollector, ExecutionOrderRecord
from execution.metrics_aggregator import MetricsAggregator
from execution.spread_guard import DynamicSpreadGuard
from execution.order_manager import OrderManager, generate_idempotency_key, OrderState
from execution.reconciliation import ReconciliationEngine
from execution.position_manager import PositionManager

from ml.trap_detector import TrapDetectorConfig, TrapDetectorService, EventKind
from ml.temporal_analyzer import TemporalModelConfig, TemporalEdgeService, EdgeTier
from ml.temporal_window_recorder import window_recorder, TemporalWindowRecorder


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

    def __init__(self, config: TradingConfig | None = None, load_saved_settings: bool = True):
        self.config = config.model_copy(deep=True) if config is not None else DEFAULT_CONFIG.model_copy(deep=True)
        self._last_utc_day: int | None = None
        self.is_active: bool = False
        self.recent_logs: list[str] = []
        self.ml_logs: list[str] = []
        self._tick_lock = threading.Lock()
        self._load_persisted_logs()

        # Initialize components
        self.state = StateManager(db_path=self.config.db_path)
        self.news_filter = NewsFilter(
            calendar_path=self.config.news_calendar_path,
            blackout_minutes=self.config.news_blackout_minutes,
        )
        self.strategy = StrategyEngine(self.config)
        self.conflict_resolver = ConflictResolver(self.config.risk)
        self.risk_engine = RiskEngine(self.config, self.state)
        self.ai_analyst = AIAnalyst(self.config)
        self.broker = create_broker(self.config)

        # ML Trap Detector — veto gate for high Stop Loss risk and retail traps
        trap_cfg = TrapDetectorConfig(
            db_path=self.config.db_path,
            model_path="ml/artifacts/trap_detector.joblib",
            p_genuine_threshold=1.0 - getattr(self.config, 'ml_max_sl_probability', 0.50),
            max_sl_probability=getattr(self.config, 'ml_max_sl_probability', 0.50),
            shadow_until_samples=getattr(self.config, 'ml_shadow_until_samples', 0),
            shadow_mode=getattr(self.config, 'ml_shadow_mode', False),
        )
        self.trap_svc = TrapDetectorService(trap_cfg)

        # ML Model 2: Temporal & Session Edge Analyzer (Day/Time/Session Profitability & Hazard Gate)
        temporal_cfg = TemporalModelConfig(
            db_path=self.config.db_path,
            model_path="ml/artifacts/temporal_edge_model.joblib",
            stats_path="ml/artifacts/temporal_edge_stats.json",
            active_gating=getattr(self.config, 'temporal_ml_enabled', True),
            veto_toxic=getattr(self.config, 'temporal_veto_toxic', True),
            toxic_loss_threshold=getattr(self.config, 'temporal_max_loss_probability', 0.65),
        )
        self.temporal_svc = TemporalEdgeService(temporal_cfg)
        self.window_recorder = window_recorder
        self._current_window_id: str | None = None


        # Execution Quality Metrics Subsystem
        self.eqm = ExecutionMetricsCollector(db_path=self.config.db_path)
        self.eqm_aggregator = MetricsAggregator(db_path=self.config.db_path)

        # Dynamic Spread Guard, Idempotent Order Manager, Reconciler, and Position Manager
        self.spread_guard = DynamicSpreadGuard(window_size=300, percentile_cutoff=95.0)
        self.order_manager = OrderManager(broker=self.broker, max_retries=3, base_backoff_sec=0.5)
        self.position_manager = PositionManager(
            broker=self.broker,
            state=self.state,
            poll_interval_sec=5.0,
            history_provider=lambda sym, tf, n: self._get_ohlcv(sym, tf, count=n),
            breakeven_sideways_only=True,
            config=self.config,
        )
        self.reconciler = ReconciliationEngine(
            broker=self.broker,
            state=self.state,
            sync_interval_sec=30.0,
            position_manager=self.position_manager,
        )

        # Trend Reversal & CHoCH Subsystem (1H Candlestick Analysis)
        self.reversal_detector = TrendReversalDetector(swing_lookback=3)
        self.trend_reversal_status: dict[str, TrendReversalAnalysis] = {}
        self._last_analyzed_1h_bar: dict[str, str] = {}

        # Price data cache (in production, fetch from broker or data provider)
        self._htf_cache: dict[str, object] = {}
        self._ltf_cache: dict[str, object] = {}
        self._daily_bias_cache: dict[str, dict] = {}

        # Load persisted settings if present
        if load_saved_settings:
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
                    self.config.pair1.avg_spread_points = p1.get("avg_spread_points")
                    self.config.pair1.enabled = p1.get("enabled", True)
                
                if "pair2" in saved and isinstance(saved["pair2"], dict):
                    p2 = saved["pair2"]
                    self.config.pair2.symbol = p2.get("symbol", self.config.pair2.symbol)
                    self.config.pair2.fixed_lot_size = p2.get("fixed_lot_size")
                    self.config.pair2.fixed_sl_pips = p2.get("fixed_sl_pips")
                    self.config.pair2.avg_spread_points = p2.get("avg_spread_points")
                    self.config.pair2.enabled = p2.get("enabled", True)

                if "pair3" in saved and isinstance(saved["pair3"], dict):
                    p3 = saved["pair3"]
                    self.config.pair3.symbol = p3.get("symbol", self.config.pair3.symbol)
                    self.config.pair3.fixed_lot_size = p3.get("fixed_lot_size")
                    self.config.pair3.fixed_sl_pips = p3.get("fixed_sl_pips")
                    self.config.pair3.avg_spread_points = p3.get("avg_spread_points")
                    self.config.pair3.enabled = p3.get("enabled", True)

                # Multi-pair independent settings (pair_configs)
                if "pair_configs" in saved and isinstance(saved["pair_configs"], dict):
                    from core.config import PairSettings
                    for s_k, s_v in saved["pair_configs"].items():
                        if isinstance(s_v, dict):
                            sym_u = str(s_k).upper().strip()
                            self.config.pair_configs[sym_u] = PairSettings(
                                symbol=s_v.get("symbol", sym_u),
                                fixed_lot_size=s_v.get("fixed_lot_size"),
                                fixed_sl_pips=s_v.get("fixed_sl_pips"),
                                avg_spread_points=s_v.get("avg_spread_points"),
                                enabled=s_v.get("enabled", True),
                            )

                # Enabled strategies
                if "enabled_strategies" in saved and isinstance(saved["enabled_strategies"], list):
                    self.config.enabled_strategies = saved["enabled_strategies"]
                    self.strategy.set_enabled_strategies(self.config.enabled_strategies)

                if "selected_symbols" in saved and isinstance(saved["selected_symbols"], list):
                    raw_symbols = saved["selected_symbols"]
                    filtered_symbols = []
                    for s in raw_symbols:
                        sym_u = str(s).strip().upper()
                        if not sym_u or sym_u == "NONE":
                            continue
                        if hasattr(self.config, "pair_configs") and sym_u in self.config.pair_configs:
                            if not getattr(self.config.pair_configs[sym_u], "enabled", True):
                                continue
                        if sym_u not in filtered_symbols:
                            filtered_symbols.append(sym_u)
                    self.config.selected_symbols = filtered_symbols
                else:
                    self.config.selected_symbols = [
                        s for s in [self.config.pair1.symbol, self.config.pair2.symbol, self.config.pair3.symbol]
                        if s and s != "NONE"
                    ]

                if "strategy_type" in saved and isinstance(saved["strategy_type"], str):
                    self.config.strategy_type = saved["strategy_type"]
                if "fixed_lot_size" in saved:
                    self.config.fixed_lot_size = saved["fixed_lot_size"]
                if "fixed_sl_pips" in saved:
                    self.config.fixed_sl_pips = saved["fixed_sl_pips"]
                if "max_open_positions" in saved and isinstance(saved["max_open_positions"], int):
                    self.config.risk.max_open_positions = saved["max_open_positions"]
                if "ai_confirmation_enabled" in saved:
                    self.config.ai_confirmation_enabled = bool(saved["ai_confirmation_enabled"])
                if "night_limit_enabled" in saved:
                    self.config.risk.night_limit_enabled = bool(saved["night_limit_enabled"])
                if "night_start_hour" in saved and isinstance(saved["night_start_hour"], int):
                    self.config.risk.night_start_hour = saved["night_start_hour"]
                if "night_end_hour" in saved and isinstance(saved["night_end_hour"], int):
                    self.config.risk.night_end_hour = saved["night_end_hour"]
                if "night_max_open_positions" in saved and isinstance(saved["night_max_open_positions"], int):
                    self.config.risk.night_max_open_positions = saved["night_max_open_positions"]
                if "night_timezone_mode" in saved and isinstance(saved["night_timezone_mode"], str):
                    self.config.risk.night_timezone_mode = saved["night_timezone_mode"]
                if "ml_gating_enabled" in saved:
                    self.config.ml_gating_enabled = bool(saved["ml_gating_enabled"])
                if "ml_max_sl_probability" in saved and isinstance(saved["ml_max_sl_probability"], (int, float)):
                    self.config.ml_max_sl_probability = float(saved["ml_max_sl_probability"])
                    if hasattr(self, "trap_svc") and self.trap_svc:
                        self.trap_svc.cfg.max_sl_probability = self.config.ml_max_sl_probability
                        self.trap_svc.cfg.p_genuine_threshold = 1.0 - self.config.ml_max_sl_probability
                if "temporal_ml_enabled" in saved:
                    self.config.temporal_ml_enabled = bool(saved["temporal_ml_enabled"])
                    if hasattr(self, "temporal_svc") and self.temporal_svc:
                        target_cfg = getattr(self.temporal_svc, "config", getattr(self.temporal_svc, "cfg", None))
                        if target_cfg:
                            target_cfg.active_gating = self.config.temporal_ml_enabled
                if "reversal_strategy_enabled" in saved:
                    self.config.reversal_strategy_enabled = bool(saved["reversal_strategy_enabled"])
                    if self.config.reversal_strategy_enabled and "TREND_REVERSAL" not in self.config.enabled_strategies:
                        self.config.enabled_strategies.append("TREND_REVERSAL")
                        self.strategy.set_enabled_strategies(self.config.enabled_strategies)
                if "is_active" in saved and isinstance(saved["is_active"], bool):
                    self.is_active = saved["is_active"]
                if "ml_shadow_mode" in saved:
                    self.config.ml_shadow_mode = bool(saved["ml_shadow_mode"])
                    if hasattr(self, "trap_svc") and self.trap_svc:
                        self.trap_svc.cfg.shadow_mode = self.config.ml_shadow_mode

                logger.info(
                    f"Loaded persisted settings from {settings_path}: "
                    f"Strategies={self.config.enabled_strategies} | "
                    f"MaxOpen={self.config.risk.max_open_positions} (NightLimit={self.config.risk.night_max_open_positions if self.config.risk.night_limit_enabled else 'OFF'}) | "
                    f"Pair1={self.config.pair1.symbol} (lot={self.config.pair1.fixed_lot_size}, sl={self.config.pair1.fixed_sl_pips}) | "
                    f"Pair2={self.config.pair2.symbol} (lot={self.config.pair2.fixed_lot_size}, sl={self.config.pair2.fixed_sl_pips}) | "
                    f"Pair3={self.config.pair3.symbol} (lot={self.config.pair3.fixed_lot_size}, sl={self.config.pair3.fixed_sl_pips}) | "
                    f"MLGate={'ON' if getattr(self.config, 'ml_gating_enabled', True) else 'OFF'} (max_sl={getattr(self.config, 'ml_max_sl_probability', 0.50):.2f}, shadow={'ON' if getattr(self.config, 'ml_shadow_mode', False) else 'OFF'}) | "
                    f"TemporalML={'ON' if getattr(self.config, 'temporal_ml_enabled', True) else 'OFF'}"
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
                    "avg_spread_points": getattr(self.config.pair1, "avg_spread_points", None),
                    "enabled": self.config.pair1.enabled,
                },
                "pair2": {
                    "symbol": self.config.pair2.symbol,
                    "fixed_lot_size": self.config.pair2.fixed_lot_size,
                    "fixed_sl_pips": self.config.pair2.fixed_sl_pips,
                    "avg_spread_points": getattr(self.config.pair2, "avg_spread_points", None),
                    "enabled": self.config.pair2.enabled,
                },
                "pair3": {
                    "symbol": self.config.pair3.symbol,
                    "fixed_lot_size": self.config.pair3.fixed_lot_size,
                    "fixed_sl_pips": self.config.pair3.fixed_sl_pips,
                    "avg_spread_points": getattr(self.config.pair3, "avg_spread_points", None),
                    "enabled": self.config.pair3.enabled,
                },
                "pair_configs": {
                    sym: {
                        "symbol": p_cfg.symbol,
                        "fixed_lot_size": p_cfg.fixed_lot_size,
                        "fixed_sl_pips": p_cfg.fixed_sl_pips,
                        "avg_spread_points": getattr(p_cfg, "avg_spread_points", None),
                        "enabled": p_cfg.enabled,
                    }
                    for sym, p_cfg in getattr(self.config, "pair_configs", {}).items()
                },
                "enabled_strategies": self.config.enabled_strategies,
                "selected_symbols": [
                    s for s in self.config.selected_symbols
                    if not (hasattr(self.config, "pair_configs") and s in self.config.pair_configs and not getattr(self.config.pair_configs[s], "enabled", True))
                ],
                "strategy_type": self.config.strategy_type,
                "fixed_lot_size": self.config.fixed_lot_size,
                "fixed_sl_pips": self.config.fixed_sl_pips,
                "max_open_positions": self.config.risk.max_open_positions,
                "night_limit_enabled": getattr(self.config.risk, "night_limit_enabled", True),
                "night_start_hour": getattr(self.config.risk, "night_start_hour", 23),
                "night_end_hour": getattr(self.config.risk, "night_end_hour", 8),
                "night_max_open_positions": getattr(self.config.risk, "night_max_open_positions", 2),
                "night_timezone_mode": getattr(self.config.risk, "night_timezone_mode", "LOCAL"),
                "ai_confirmation_enabled": self.config.ai_confirmation_enabled,
                "ml_gating_enabled": getattr(self.config, "ml_gating_enabled", True),
                "ml_max_sl_probability": getattr(self.config, "ml_max_sl_probability", 0.50),
                "ml_shadow_mode": getattr(self.config, "ml_shadow_mode", True),
                "temporal_ml_enabled": getattr(self.config, "temporal_ml_enabled", True),
                "reversal_strategy_enabled": getattr(self.config, "reversal_strategy_enabled", False),
                "is_active": self.is_active,
            }
            with open("bot_settings.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save bot settings to bot_settings.json: {e}")


    def _load_persisted_logs(self, max_lines: int = 1000) -> None:
        """Hydrate recent_logs and ml_logs from existing log file if available."""
        try:
            log_file = Path(self.config.log_path)
            if not log_file.exists():
                return
            import json
            loaded_recent = []
            loaded_ml = []
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in lines[-max_lines:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    msg = data.get("record", {}).get("message", "")
                    t_str = data.get("record", {}).get("time", {}).get("repr", "")
                    if "T" in t_str:
                        time_part = t_str.split("T")[1].split("+")[0].split(".")[0]
                    elif " " in t_str:
                        time_part = t_str.split()[1].split(".")[0]
                    else:
                        time_part = t_str[:8]
                    entry = f"[{time_part}] {msg}"
                    loaded_recent.append(entry)
                    if any(k in msg for k in ["ML", "TRAP", "Trap", "🪤", "🔬", "🤖", "AI", "TEMPORAL", "DECISION", "vetoed"]):
                        loaded_ml.append(entry)
                except Exception:
                    continue
            if loaded_recent:
                self.recent_logs = loaded_recent[-1000:]
            if loaded_ml:
                self.ml_logs = loaded_ml[-500:]
        except Exception as e:
            logger.debug(f"Could not load persisted logs: {e}")

    def log(self, message: str, level: str = "INFO") -> None:
        """Helper to append to recent logs and send to logger."""
        try:
            tz = self.config.risk.get_timezone()
            timestamp = datetime.now(tz).strftime("%H:%M:%S")
        except Exception:
            timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.recent_logs.append(entry)
        if len(self.recent_logs) > 1000:
            self.recent_logs.pop(0)

        # Dedicated ring buffer for ML & AI logs so they are never drowned out by market ticks
        if any(k in message for k in ["ML", "TRAP", "Trap", "🪤", "🔬", "🤖", "AI", "TEMPORAL", "DECISION", "vetoed"]):
            self.ml_logs.append(entry)
            if len(self.ml_logs) > 500:
                self.ml_logs.pop(0)
        
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
        self.reconciler.shutdown()
        self.position_manager.shutdown()
        self.eqm.shutdown()
        self.trap_svc.close()
        self.broker.disconnect()
        self.state.close()
        logger.info("Shutdown complete.")

    def close_all_open_positions(self, reason: str = "Emergency HALT") -> int:
        """
        Emergency method: Immediately close all open positions across broker and database state.
        Returns count of positions closed.
        """
        count = 0
        try:
            # 1. Close all broker open positions
            if hasattr(self.broker, "close_all_positions"):
                count = self.broker.close_all_positions()
            elif hasattr(self.broker, "get_open_positions"):
                live_positions = self.broker.get_open_positions()
                for p in live_positions:
                    t = p.get("ticket") or p.get("order_id")
                    if t and hasattr(self.broker, "close_position"):
                        if self.broker.close_position(t):
                            count += 1

            # 2. Sync database trade log with broker deals
            self._sync_open_positions()

            # 3. For any remaining open trades in the state table, mark them closed
            open_trades = self.state.get_open_positions()
            for trade in open_trades:
                if trade.id is not None:
                    self.state.update_trade_pnl(trade.id, 0.0, "CLOSED_MANUAL")
                    if hasattr(self, "position_manager") and self.position_manager:
                        self.position_manager.on_trade_closed(trade, 0.0, "CLOSED_MANUAL")
                    self.log(f"🛑 [HALT] Closed trade #{trade.id} ({trade.symbol}) in database ({reason}).")

            self.log(f"🚨 [EMERGENCY HALT] Closed {count} live broker position(s) and synchronized state ({reason}).")
        except Exception as e:
            logger.error(f"Error during close_all_open_positions: {e}")
        return count

    def halt(self, close_trades: bool = True, reason: str = "Emergency HALT") -> dict:
        """
        Emergency halt the bot: deactivates trading engine and closes all open trades.
        """
        self.is_active = False
        self.save_settings()
        closed_session_id = self.state.record_deactivation(reason)

        closed_trades_count = 0
        if close_trades:
            closed_trades_count = self.close_all_open_positions(reason=reason)

        self.log(f"🔴 BOT HALTED by user (Session #{closed_session_id or '---'}). Closed {closed_trades_count} trade(s).")
        return {
            "status": "success",
            "botStatus": "HALTED",
            "session_id": closed_session_id,
            "closed_trades": closed_trades_count,
            "message": f"Bot halted successfully. Closed {closed_trades_count} open trade(s).",
        }


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
                        if hasattr(self, "position_manager") and self.position_manager:
                            self.position_manager.on_trade_closed(trade, pnl, status)
                        self.log(
                            f"🔔 Position #{trade.id} ({trade.symbol}) CLOSED in broker: {status} | "
                            f"Realized PnL: ${pnl:+.2f}"
                        )
                        # Record exit fill in Execution Quality Metrics
                        try:
                            inst = get_instrument(self.config, trade.symbol)
                            pip_sz = 10 ** -inst.digits
                        except Exception:
                            pip_sz = 0.0001
                        exit_target = trade.take_profit if status == "CLOSED_TP" else trade.stop_loss
                        self.eqm.record_exit_fill(
                            trade_id=trade.id,
                            exit_type=status,
                            intended_price=exit_target or trade.entry_price,
                            actual_price=exit_target or trade.entry_price,
                            volume=trade.lot_size,
                            pip_size=pip_sz,
                        )
                    else:
                        self.state.update_trade_pnl(trade.id, 0.0, "CLOSED")
                        if hasattr(self, "position_manager") and self.position_manager:
                            self.position_manager.on_trade_closed(trade, 0.0, "CLOSED")
                        self.log(
                            f"🔔 Position #{trade.id} ({trade.symbol}) no longer active in broker. Marked CLOSED in state.",
                            level="INFO"
                        )
                else:
                    self.state.update_trade_pnl(trade.id, 0.0, "CLOSED")
                    if hasattr(self, "position_manager") and self.position_manager:
                        self.position_manager.on_trade_closed(trade, 0.0, "CLOSED")
                    self.log(
                        f"🔔 Position #{trade.id} ({trade.symbol}) no longer active in broker. Marked CLOSED in state.",
                        level="INFO"
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

        # Step 1b: Max concurrent open positions guard (auto capped to 2 between 11 PM and 8 AM, 15 daytime)
        open_trades = self.state.get_open_positions()
        if hasattr(self.config.risk, 'get_effective_max_open_positions'):
            max_open = self.config.risk.get_effective_max_open_positions(now_utc)
            is_night = self.config.risk.is_night_window(now_utc)
        else:
            max_open = getattr(self.config.risk, 'max_open_positions', 15)
            is_night = False

        if len(open_trades) >= max_open:
            limit_name = "NIGHT LIMIT (11 PM - 8 AM)" if is_night else "MAX OPEN TRADES"
            self.log(
                f"⏸️ {limit_name} ACTIVE ({len(open_trades)}/{max_open} positions open). "
                f"Skipping new trade execution until an existing position closes.",
                level="INFO"
            )
            return

        # Build active pairs list for concurrent multi-pair scanning
        target_symbols: list[str] = []
        if self.config.selected_symbols:
            for s in self.config.selected_symbols:
                s_clean = s.strip().upper()
                if s_clean and s_clean != "NONE" and s_clean not in target_symbols:
                    target_symbols.append(s_clean)
        else:
            target_symbols = [inst.symbol for inst in self.config.instruments]

        # Ensure pair1, pair2, and pair3 are included if enabled
        if getattr(self.config, 'pair1', None) and self.config.pair1.enabled and self.config.pair1.symbol:
            p1_s = self.config.pair1.symbol.strip().upper()
            if p1_s and p1_s != "NONE" and p1_s not in target_symbols:
                target_symbols.insert(0, p1_s)
        if getattr(self.config, 'pair2', None) and self.config.pair2.enabled and self.config.pair2.symbol:
            p2_s = self.config.pair2.symbol.strip().upper()
            if p2_s and p2_s != "NONE" and p2_s not in target_symbols:
                target_symbols.append(p2_s)
        if getattr(self.config, 'pair3', None) and self.config.pair3.enabled and self.config.pair3.symbol:
            p3_s = self.config.pair3.symbol.strip().upper()
            if p3_s and p3_s != "NONE" and p3_s not in target_symbols:
                target_symbols.append(p3_s)

        # Include enabled symbols from pair_configs
        if hasattr(self.config, 'pair_configs') and isinstance(self.config.pair_configs, dict):
            for s_name, p_cfg in self.config.pair_configs.items():
                if getattr(p_cfg, 'enabled', True):
                    s_clean = str(s_name).strip().upper()
                    if s_clean and s_clean != "NONE" and s_clean not in target_symbols:
                        target_symbols.append(s_clean)

        # Exclude pair if explicitly disabled in pair1 / pair2 / pair3 or pair_configs
        enabled_pair_symbols = set()
        for p_name in ('pair1', 'pair2', 'pair3'):
            p_cfg = getattr(self.config, p_name, None)
            if p_cfg and p_cfg.enabled and p_cfg.symbol:
                s_name = p_cfg.symbol.strip().upper()
                if s_name and s_name != "NONE":
                    enabled_pair_symbols.add(s_name)

        if hasattr(self.config, 'pair_configs') and isinstance(self.config.pair_configs, dict):
            for s_name, p_cfg in self.config.pair_configs.items():
                if getattr(p_cfg, 'enabled', True):
                    s_clean = str(s_name).strip().upper()
                    if s_clean and s_clean != "NONE":
                        enabled_pair_symbols.add(s_clean)

        for p_name in ('pair1', 'pair2', 'pair3'):
            p_cfg = getattr(self.config, p_name, None)
            if p_cfg and not p_cfg.enabled and p_cfg.symbol:
                p_s = p_cfg.symbol.strip().upper()
                if p_s in target_symbols and p_s not in enabled_pair_symbols:
                    target_symbols.remove(p_s)

        # Remove explicitly disabled pair_configs
        if hasattr(self.config, 'pair_configs') and isinstance(self.config.pair_configs, dict):
            for s_name, p_cfg in self.config.pair_configs.items():
                if not getattr(p_cfg, 'enabled', True):
                    p_s = str(s_name).strip().upper()
                    if p_s in target_symbols:
                        target_symbols.remove(p_s)

        active_pairs = []
        for sym in target_symbols:
            sym_clean = sym.strip().upper()
            # Double check: skip if explicitly disabled in pair_configs
            if hasattr(self.config, 'pair_configs') and sym_clean in self.config.pair_configs:
                if not getattr(self.config.pair_configs[sym_clean], 'enabled', True):
                    continue

            pair_lot = self.config.fixed_lot_size
            pair_sl = self.config.fixed_sl_pips
            pair_spread = None

            # Custom override from pair1 / pair2 / pair3 or pair_configs if symbol matches
            if getattr(self.config, 'pair1', None) and self.config.pair1.enabled and self.config.pair1.symbol.strip().upper() == sym_clean:
                if self.config.pair1.fixed_lot_size is not None:
                    pair_lot = self.config.pair1.fixed_lot_size
                if self.config.pair1.fixed_sl_pips is not None:
                    pair_sl = self.config.pair1.fixed_sl_pips
                if getattr(self.config.pair1, 'avg_spread_points', None) is not None:
                    pair_spread = self.config.pair1.avg_spread_points
            elif getattr(self.config, 'pair2', None) and self.config.pair2.enabled and self.config.pair2.symbol.strip().upper() == sym_clean:
                if self.config.pair2.fixed_lot_size is not None:
                    pair_lot = self.config.pair2.fixed_lot_size
                if self.config.pair2.fixed_sl_pips is not None:
                    pair_sl = self.config.pair2.fixed_sl_pips
                if getattr(self.config.pair2, 'avg_spread_points', None) is not None:
                    pair_spread = self.config.pair2.avg_spread_points
            elif getattr(self.config, 'pair3', None) and self.config.pair3.enabled and self.config.pair3.symbol.strip().upper() == sym_clean:
                if self.config.pair3.fixed_lot_size is not None:
                    pair_lot = self.config.pair3.fixed_lot_size
                if self.config.pair3.fixed_sl_pips is not None:
                    pair_sl = self.config.pair3.fixed_sl_pips
                if getattr(self.config.pair3, 'avg_spread_points', None) is not None:
                    pair_spread = self.config.pair3.avg_spread_points
            elif hasattr(self.config, 'pair_configs') and sym_clean in self.config.pair_configs:
                p_item = self.config.pair_configs[sym_clean]
                if p_item.fixed_lot_size is not None:
                    pair_lot = p_item.fixed_lot_size
                if p_item.fixed_sl_pips is not None:
                    pair_sl = p_item.fixed_sl_pips
                if getattr(p_item, 'avg_spread_points', None) is not None:
                    pair_spread = p_item.avg_spread_points

            active_pairs.append({
                "pair_num": len(active_pairs) + 1,
                "symbol": sym_clean,
                "fixed_lot_size": pair_lot,
                "fixed_sl_pips": pair_sl,
                "avg_spread_points": pair_spread,
            })

        strats_str = ", ".join(self.config.enabled_strategies)
        pairs_str = ", ".join(f"Pair {p['pair_num']}: {p['symbol']}" for p in active_pairs)
        tz = self.config.risk.get_timezone() if hasattr(self.config.risk, 'get_timezone') else timezone.utc
        local_time_str = now_utc.astimezone(tz).strftime('%Y-%m-%d %H:%M:%S')
        tz_name = getattr(self.config.risk, 'night_timezone_mode', 'IST')
        self.log(f"─── TICK @ {local_time_str} ({tz_name}) | Pairs: [{pairs_str}] | Active Strategies: [{strats_str}] ───")

        for pair_info in active_pairs:
            symbol = pair_info["symbol"]
            pair_num = pair_info["pair_num"]
            pair_lot = pair_info["fixed_lot_size"]
            pair_sl = pair_info["fixed_sl_pips"]

            htf_tf = "1H"
            ltf_tf = "5m" if ("SMC_SCALP_5M" in self.config.enabled_strategies or "ICT" in self.config.enabled_strategies or "ORDER_FLOW" in self.config.enabled_strategies) else self.config.timeframes.ltf

            self.log(f"Analyzing Pair {pair_num} ({symbol}) | Sizing: lot={pair_lot or 'Dynamic'}, SL={pair_sl or 'Dynamic'} pips | HTF: {htf_tf} | LTF: {ltf_tf}...")

            # Dynamic check: Re-fetch open positions so that as trades execute on earlier pairs,
            # subsequent pairs immediately see the updated open count within the exact same scan cycle!
            open_trades = self.state.get_open_positions()
            if hasattr(self.config.risk, 'get_effective_max_open_positions'):
                max_open = self.config.risk.get_effective_max_open_positions(now_utc)
                is_night = self.config.risk.is_night_window(now_utc)
            else:
                max_open = getattr(self.config.risk, 'max_open_positions', 15)
                is_night = False

            if len(open_trades) >= max_open:
                limit_name = "NIGHT LIMIT (11 PM - 8 AM)" if is_night else "MAX OPEN TRADES"
                self.log(
                    f"  ⏸️ {limit_name} ACTIVE ({len(open_trades)}/{max_open} positions open). "
                    f"Skipping Pair {pair_num} ({symbol}).",
                    level="INFO"
                )
                continue

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

            # ── Step 2a: Spread check (Static Multiple + Rolling Percentile Guard) ──
            point_size = 10 ** -instrument.digits
            effective_avg_spread_pts = (
                pair_info.get("avg_spread_points")
                if pair_info.get("avg_spread_points") is not None
                else instrument.avg_spread_points
            )
            spread_result = NewsFilter.check_spread(
                current_spread=current_spread,
                avg_spread=effective_avg_spread_pts * point_size,
                max_spread_multiple=self.config.risk.max_spread_multiple,
            )
            if spread_result.blocked:
                self.log(f"  ⛔ SPREAD EXCESSIVE: {spread_result.reason}", level="WARNING")
                continue

            # Rolling Percentile Spread Guard
            dyn_spread_ok, dyn_spread_msg, _ = self.spread_guard.evaluate_spread(symbol, current_spread)
            if not dyn_spread_ok:
                self.log(f"  ⛔ DYNAMIC SPREAD GUARD: {dyn_spread_msg}", level="WARNING")
                continue

            # ── Step 2b: Fetch OHLCV data ──
            htf_data = self._get_ohlcv(symbol, htf_tf)
            ltf_data = self._get_ohlcv(symbol, ltf_tf)
            if htf_data is None or ltf_data is None:
                logger.warning(f"  No OHLCV data for {symbol} ({htf_tf}/{ltf_tf}). Skipping.")
                continue

            # ── Market Regime Analysis (Sideways vs Trending Risk Sizing) ──
            regime_analysis = MarketRegimeDetector.analyze(ltf_data, adx_threshold=20.0, chop_threshold=61.8, sideways_risk_multiplier=0.5)
            if regime_analysis.is_sideways:
                self.log(
                    f"  🌊 [SIDEWAYS REGIME] {symbol} is Ranging/Consolidating "
                    f"(ADX={regime_analysis.adx:.1f}, CHOP={regime_analysis.choppiness:.1f}, BBW={regime_analysis.bb_width_pct:.1f}%) "
                    f"-> Scaling risk sizing by {regime_analysis.risk_multiplier*100:.0f}%",
                    level="INFO"
                )

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

            # Update daily bias cache & database for this symbol
            try:
                self.compute_daily_bias(symbol)
            except Exception as e:
                logger.debug(f"Daily bias tick update error for {symbol}: {e}")

            if htf_analysis.fib_50 is not None:
                pos_str = "DISCOUNT (<0.5 Fib)" if htf_analysis.is_discount else "PREMIUM (>=0.5 Fib)"
                fail_str = " 🚨 [ALL ZONES FAILED -> BIAS FLIPPED]" if htf_analysis.all_zones_failed else ""
                zone_str = f" | 1st Zone: [{htf_analysis.first_zone_type}]" if htf_analysis.first_zone_type != "NONE" else ""
                self.log(
                    f"  📐 [DEALING RANGE] {symbol}: Bias={htf_analysis.bias.value} | "
                    f"Range=[{htf_analysis.dealing_range_low:.5f} - {htf_analysis.dealing_range_high:.5f}] | "
                    f"0.5 Fib={htf_analysis.fib_50:.5f} ({pos_str}){zone_str}{fail_str}",
                    level="DEBUG"
                )

            # ── Isolate Completed/Closed 1-Hour Candles ──
            # In live MT5 data, the final row of htf_data is often the currently-forming open hour.
            # If current time is before bar_time + 1h, drop the forming candle so only closed bars are evaluated.
            closed_htf_data = htf_data
            if 'time' in htf_data.columns and len(htf_data) > 25:
                last_bar_time = pd.to_datetime(htf_data['time'].iloc[-1], utc=True)
                if now_utc < last_bar_time + pd.Timedelta(hours=1):
                    closed_htf_data = htf_data.iloc[:-1]

            latest_closed_1h_time = str(closed_htf_data['time'].iloc[-1]) if 'time' in closed_htf_data.columns else str(closed_htf_data.index[-1])
            is_new_1h_candle = (
                symbol not in self._last_analyzed_1h_bar
                or self._last_analyzed_1h_bar[symbol] != latest_closed_1h_time
            )

            current_mid_price = (quote.bid + quote.ask) / 2.0 if quote else None

            # ── Institutional Multi-TF Trend Reversal Analysis (SD, Fib 0.5-0.6, FVG in 4H/1H/Daily) ──
            is_reversal_enabled = (
                getattr(self.config, "reversal_strategy_enabled", False)
                or "TREND_REVERSAL" in self.config.enabled_strategies
            )

            df_4h = self._get_ohlcv(symbol, "4H")
            df_1d = self._get_ohlcv(symbol, "1D")
            multitf_dict = {"1H": closed_htf_data}
            if df_4h is not None and len(df_4h) >= 20:
                multitf_dict["4H"] = df_4h
            if df_1d is not None and len(df_1d) >= 15:
                multitf_dict["1D"] = df_1d

            reversal_analysis = None
            if is_reversal_enabled:
                reversal_analysis = self.reversal_detector.analyze_multitf(
                    dfs=multitf_dict,
                    trend=htf_analysis.bias,
                    symbol=symbol,
                    htf_analysis=htf_analysis,
                    current_price=current_mid_price,
                )
                self.trend_reversal_status[symbol] = reversal_analysis

                if is_new_1h_candle:
                    self.log(
                        f"  🕐 [1H CANDLE CLOSE] {symbol}: 1-Hour candle closed @ {latest_closed_1h_time}. "
                        f"Analyzed Reversal Zones (SD / Fib 0.5-0.6 / FVG): Trend={reversal_analysis.trend.value} | "
                        f"Reversal={reversal_analysis.choch_type.value} | Reversal Risk={reversal_analysis.reversal_risk} ({reversal_analysis.reversal_probability:.0f}%)",
                        level="INFO"
                    )
                    self._last_analyzed_1h_bar[symbol] = latest_closed_1h_time

                # ── Always Log Chart Trend Reversal Analysis ──
                if reversal_analysis.is_trending:
                    if reversal_analysis.choch_detected:
                        confluence_str = ", ".join(reversal_analysis.confluence.details) if reversal_analysis.confluence.details else "Reversal Zone Active"
                        priority_str = f" | Priority: [{reversal_analysis.priority_zone_type}] (Rank {reversal_analysis.priority_rank})"
                        if reversal_analysis.priority_order_list:
                            priority_str += f" | Order: [{' -> '.join(reversal_analysis.priority_order_list)}]"
                        self.log(
                            f"  🚨 [{reversal_analysis.timeframe} REVERSAL ZONE ALERT] {symbol} ({reversal_analysis.trend.value}): {reversal_analysis.choch_type.value} REVERSAL DETECTED! "
                            f"Zone Type: {reversal_analysis.reversal_zone_type}{priority_str} | Stage: {reversal_analysis.stage.value} | "
                            f"Reversal Prob: {reversal_analysis.reversal_probability:.0f}% ({reversal_analysis.reversal_risk}) | "
                            f"Confluence: [{confluence_str}]",
                            level="WARNING"
                        )
                    elif reversal_analysis.stage == ReversalStage.PRE_REVERSAL_SWEEP:
                        self.log(
                            f"  ⚠️ [{reversal_analysis.timeframe} REVERSAL EARLY WARNING] {symbol} ({reversal_analysis.trend.value}): Liquidity sweep / extension at extreme ({reversal_analysis.trend_extreme_level:.5f})! "
                            f"Monitoring for Reversal Zone (SD / Fib 0.5-0.6 / FVG).",
                            level="INFO"
                        )
                    else:
                        self.log(
                            f"  📈 [{reversal_analysis.timeframe} TREND HEALTHY] {symbol} ({reversal_analysis.trend.value}): Trend structure intact. "
                            f"No active reversal zone triggered. Reversal Risk: LOW ({reversal_analysis.reversal_probability:.0f}%).",
                            level="DEBUG"
                        )
                else:
                    self.log(
                        f"  ⚖️ [1H MARKET NEUTRAL] {symbol}: No dominant trend structure active. Ranging conditions.",
                        level="DEBUG"
                    )

                # ── Active Trade Protection Against Reversals ──
                if reversal_analysis.choch_detected and reversal_analysis.reversal_probability >= 50.0:
                    opposing_dir = Direction.BUY if reversal_analysis.choch_type == CHoCHType.BEARISH else Direction.SELL
                    for open_t in open_symbol_trades:
                        if open_t.direction == opposing_dir:
                            self.log(
                                f"  🛡️ [REVERSAL SHIELD] Active {open_t.direction.value} trade #{open_t.id} on {symbol} is vulnerable to {reversal_analysis.choch_type.value} Reversal! "
                                f"Securing position with Stop Loss protection.",
                                level="WARNING"
                            )
                            if self.position_manager and open_t.id:
                                self.position_manager.protect_against_reversal(open_t, reversal_analysis)
            else:
                self.trend_reversal_status.pop(symbol, None)

            signals = self.strategy.evaluate_all(
                symbol=symbol,
                htf_data=htf_data,
                ltf_data=ltf_data,
                instrument=instrument,
                current_spread=current_spread,
                fixed_sl_pips=pair_sl,
                enabled_strategies=eval_strats,
                htf_analysis=htf_analysis,
                dfs_by_tf=multitf_dict,
            )

            # ── Reversal Filter: Prevent Entering Trades Against Active High-Probability Reversal Zone or Early Sweep ──
            if is_reversal_enabled and reversal_analysis is not None:
                should_block_reversal = False
                blocked_direction = None

                if reversal_analysis.choch_detected and reversal_analysis.reversal_probability >= 50.0:
                    should_block_reversal = True
                    if htf_analysis.bias == MarketBias.BULLISH and not getattr(htf_analysis, 'all_zones_failed', False):
                        blocked_direction = Direction.SELL
                    elif htf_analysis.bias == MarketBias.BEARISH and not getattr(htf_analysis, 'all_zones_failed', False):
                        blocked_direction = Direction.BUY
                    else:
                        blocked_direction = Direction.BUY if reversal_analysis.choch_type == CHoCHType.BEARISH else Direction.SELL
                elif (reversal_analysis.stage == ReversalStage.PRE_REVERSAL_SWEEP or reversal_analysis.reversal_risk in ("HIGH", "CRITICAL")) and reversal_analysis.reversal_probability >= 45.0:
                    should_block_reversal = True
                    if htf_analysis.bias == MarketBias.BULLISH and not getattr(htf_analysis, 'all_zones_failed', False):
                        blocked_direction = Direction.SELL
                    elif htf_analysis.bias == MarketBias.BEARISH and not getattr(htf_analysis, 'all_zones_failed', False):
                        blocked_direction = Direction.BUY
                    else:
                        blocked_direction = Direction.SELL if reversal_analysis.trend == MarketBias.BEARISH else (Direction.BUY if reversal_analysis.trend == MarketBias.BULLISH else None)

                if should_block_reversal and blocked_direction is not None:
                    prior_len = len(signals)
                    signals = [s for s in signals if s.direction != blocked_direction]
                    if len(signals) < prior_len:
                        reason_desc = f"{reversal_analysis.choch_type.value} Reversal ({reversal_analysis.reversal_zone_type})" if reversal_analysis.choch_detected else f"Pre-Reversal Extreme Sweep ({reversal_analysis.reversal_risk} Risk)"
                        self.log(
                            f"  🚫 [REVERSAL GUARD] Blocked {prior_len - len(signals)} {blocked_direction.value} signal(s) on {symbol}: "
                            f"Trend reversal warning: {reason_desc} (Prob: {reversal_analysis.reversal_probability:.0f}%).",
                            level="INFO"
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

            # Candidate signals: supports multiple distinct strategy signals if both pass gates
            candidate_signals = list(getattr(filter_result, 'accepted_signals', None) or [])
            if not candidate_signals and filter_result.accepted_signal:
                candidate_signals = [filter_result.accepted_signal]

            for best_signal in candidate_signals:
                # Dynamic re-check of limits before each execution
                open_trades = self.state.get_open_positions()
                if len(open_trades) >= max_open:
                    self.log(
                        f"  ⏸️ Global max open positions reached ({len(open_trades)}/{max_open}). "
                        f"Rejecting/queuing extra signal [{best_signal.strategy_name}] on {symbol}.",
                        level="INFO"
                    )
                    break

                open_symbol_trades = [t for t in open_trades if t.symbol == symbol]
                if len(open_symbol_trades) >= max_per_symbol:
                    self.log(
                        f"  ⏸️ Max per-pair open positions reached for {symbol} ({len(open_symbol_trades)}/{max_per_symbol}). "
                        f"Rejecting/queuing signal [{best_signal.strategy_name}]."
                    )
                    break

                # Ensure strategy does not already have an open trade on this symbol
                cur_active_strats = {
                    normalize_strategy_key(t.strategy_name, t.magic_number)
                    for t in open_symbol_trades
                }
                sig_strat_key = normalize_strategy_key(getattr(best_signal, 'strategy_id', None) or best_signal.strategy_name, getattr(best_signal, 'magic_number', None))
                if sig_strat_key in cur_active_strats:
                    self.log(
                        f"  ⏸️ Duplicate trade forbidden: strategy '{best_signal.strategy_name}' already active on {symbol}. Skipping."
                    )
                    continue

                conf_str = f" | {best_signal.candle_confirmation}" if getattr(best_signal, "candle_confirmation", "") else ""
                self.log(
                    f"  ✅ [{best_signal.strategy_name}] SIGNAL DETECTED: {best_signal.direction.value} {symbol} "
                    f"| Entry={best_signal.entry_price:.5f} "
                    f"| SL={best_signal.stop_loss:.5f} "
                    f"| TP={best_signal.take_profit:.5f} "
                    f"| R:R={best_signal.rr_ratio:.2f} "
                    f"| Score={best_signal.quality_score:.1f} ({best_signal.ltf_confirmation.value}){conf_str}"
                )

                # Initialize EQM lifecycle tracking for this candidate order
                point_sz = 10 ** -instrument.digits
                atr_val = getattr(best_signal, "atr_14", 0.0) or (abs(best_signal.entry_price - best_signal.stop_loss) * 0.5)
                eqm_order = self.eqm.start_order(
                    symbol=symbol,
                    strategy_name=best_signal.strategy_name,
                    direction=best_signal.direction.value,
                    requested_price=best_signal.entry_price,
                    atr_14=atr_val,
                    pip_size=point_sz,
                    session_killzone=getattr(best_signal, "session_killzone", "OFF_HOURS") or "LONDON_OPEN",
                    conflict_score=best_signal.quality_score,
                    spread_at_signal=current_spread,
                    magic=best_signal.magic_number,
                )

                # ── Step 2c-1.5: ML Trap Detector Gate (High Stop Loss Risk Veto) ──
                p_tp_val = 0.50
                if getattr(self.config, 'ml_gating_enabled', True):
                    trap_kind = self._signal_to_trap_kind(best_signal)
                    if trap_kind is not None and ltf_data is not None:
                        try:
                            trap_dir = "long" if best_signal.direction == Direction.BUY else "short"
                            ml_df = self._prepare_df_for_trap_detector(ltf_data)
                            strat_key = normalize_strategy_key(best_signal.strategy_name)

                            # Institutional confluence features (Fib 0.5-0.6, Discount <0.5, SD -4.0 to -4.5, HTF FVG/OB)
                            extra_confluence: dict[str, float] = {}
                            rev_analysis = self.trend_reversal_status.get(symbol)
                            if rev_analysis and rev_analysis.confluence:
                                c = rev_analysis.confluence
                                if c.is_below_fib_50:
                                    extra_confluence["is_below_fib_50"] = 1.0
                                    extra_confluence["priority_discount"] = 1.0
                                if getattr(c, "is_first_zone_below_50", False) or getattr(c, "is_first_zone_above_50", False):
                                    extra_confluence["first_zone_priority"] = 1.0
                                if getattr(c, "sd_4_to_4_5_hit", False):
                                    extra_confluence["sd_exhaustion_4x_45x"] = 1.0
                                    extra_confluence["priority_discount"] = 1.0
                                if c.in_fib_50_60_zone:
                                    extra_confluence["in_fib_50_60"] = 1.0
                                if c.standard_deviation_hit and c.sd_multiple and abs(c.sd_multiple) >= 4.0:
                                    extra_confluence["sd_exhaustion_4x"] = 1.0
                                    extra_confluence["priority_discount"] = 1.0
                                if c.liquidity_sweep:
                                    extra_confluence["trendline_liq"] = 1.0
                                if c.fvg_present:
                                    extra_confluence["order_block_test"] = 1.0

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
                                strategy=strat_key,
                                extra_features=extra_confluence,
                            )
                            p_tp = trap_ev.p_genuine if trap_ev.p_genuine is not None else 0.50
                            p_tp_val = p_tp
                            p_sl = 1.0 - p_tp
                            max_sl_thr = getattr(self.trap_svc.cfg, 'max_sl_probability', 0.50)
                            eqm_order.ml_p_tp = p_tp
                            eqm_order.ml_p_sl = p_sl

                            is_shadow = getattr(self.config, 'ml_shadow_mode', False) or getattr(self.trap_svc.cfg, 'shadow_mode', False)
                            priority_tag = " [HIGH PRIORITY DISCOUNT <0.5 FIB / SD -4.5]" if trap_ev.features.get("priority_discount", 0.0) == 1.0 else ""

                            if is_shadow:
                                would_veto_str = " (WOULD VETO)" if (p_sl > max_sl_thr or not trap_ev.allowed) else " (WOULD ALLOW)"
                                self.log(
                                    f"  🪤 [ML SHADOW MODE]{priority_tag} Trap Analysis: P(SL)={p_sl*100:.1f}%, P(TP)={p_tp*100:.1f}%{would_veto_str} "
                                    f"| Strategy '{best_signal.strategy_name}' [{strat_key}] on {symbol} "
                                    f"| Executing trade for live market observation & continuous reinforcement learning.",
                                    level="INFO",
                                )
                            elif not trap_ev.allowed:
                                self.log(
                                    f"  🪤 [ML DECISION OVERRIDE] Trade SKIPPED: High chance of Stop Loss "
                                    f"(P(SL)={p_sl*100:.1f}% > {max_sl_thr*100:.1f}%, P(TP)={p_tp*100:.1f}%) "
                                    f"| Strategy '{best_signal.strategy_name}' [{strat_key}] on {symbol} vetoed by ML Model ({trap_ev.model_version})",
                                    level="WARNING",
                                )
                                self.eqm.mark_rejected(eqm_order.order_id, rejection_code=4001, rejection_reason=f"ML Veto P(SL)={p_sl:.2f}")
                                continue
                            else:
                                self.log(
                                    f"  🔬 [ML MODEL APPROVED] Setup verified genuine [{strat_key}] (P(TP)={p_tp*100:.1f}%, "
                                    f"P(SL)={p_sl*100:.1f}% <= {max_sl_thr*100:.1f}%) | "
                                    f"Model={trap_ev.model_version} | Proceeding to AI & Risk validation"
                                )
                        except Exception as e:
                            self.log(f"  ⚠️ Trap Gate error (failing open): {e}", level="WARNING")

                # ── Step 2c-1.8: ML Model 2 — Temporal & Session Edge Analyzer ──
                temporal_risk_mult = 1.0
                if getattr(self.config, 'temporal_ml_enabled', True):
                    try:
                        strat_key = normalize_strategy_key(best_signal.strategy_name)
                        temp_verdict = self.temporal_svc.evaluate(
                            timestamp=getattr(best_signal, 'timestamp', None) or datetime.now(timezone.utc),
                            symbol=symbol,
                            strategy=strat_key,
                            planned_rr=best_signal.rr_ratio,
                        )
                        win_ctx = self.window_recorder.get_window_info(getattr(best_signal, 'timestamp', None) or datetime.now(timezone.utc))
                        is_temp_shadow = getattr(self.config, 'temporal_ml_shadow_mode', True)
                        if is_temp_shadow:
                            temporal_risk_mult = 1.0
                            would_veto_str = " (WOULD VETO)" if not temp_verdict.allowed else (f" (WOULD SCALE {temp_verdict.risk_multiplier:.2f}x)" if temp_verdict.risk_multiplier != 1.0 else "")
                            self.log(
                                f"  ⏰ [OCTOBER SHADOW MODE]{would_veto_str} {temp_verdict.edge_tier.value} in {win_ctx['id']} ({win_ctx['time_range']}) "
                                f"(P(Win)={temp_verdict.p_win*100:.1f}%, P(SL)={temp_verdict.p_loss*100:.1f}%, ExpRet={temp_verdict.expected_r:+.2f}R) | "
                                f"Executing at full 1.00x base lot.",
                                level="INFO",
                            )
                        else:
                            temporal_risk_mult = temp_verdict.risk_multiplier

                            if not temp_verdict.allowed and getattr(self.config, 'temporal_veto_toxic', False):
                                self.log(
                                    f"  ⏰ [TEMPORAL ML VETO] Trade SKIPPED: High-hazard losing window "
                                    f"({temp_verdict.day_name} {temp_verdict.session} P(SL)={temp_verdict.p_loss*100:.1f}%) | "
                                    f"Expected Return: {temp_verdict.expected_r:+.2f}R | {temp_verdict.reason}",
                                    level="WARNING",
                                )
                                self.eqm.mark_rejected(eqm_order.order_id, rejection_code=4004, rejection_reason=f"Temporal Veto {temp_verdict.edge_tier.value}")
                                continue
                            elif not temp_verdict.allowed:
                                temporal_risk_mult = max(0.25, temporal_risk_mult)
                                self.log(
                                    f"  ⚠️ [TEMPORAL HAZARD WARNING] {temp_verdict.edge_tier.value} in {temp_verdict.session} "
                                    f"(P(SL)={temp_verdict.p_loss*100:.1f}%). Sizing scaled down to {temporal_risk_mult:.2f}x."
                                )
                            else:
                                self.log(
                                    f"  ⏰ [TEMPORAL ML APPROVED] {temp_verdict.edge_tier.value} in {temp_verdict.session} "
                                    f"(P(Win)={temp_verdict.p_win*100:.1f}%, Expected Return={temp_verdict.expected_r:+.2f}R) | "
                                    f"Risk Scale: {temporal_risk_mult:.2f}x"
                                )
                    except Exception as e:
                        self.log(f"  ⚠️ Temporal ML Gate error (failing open): {e}", level="WARNING")

                # ── Step 2c-2: AI Second-Opinion Confirmation Gate ──
                ai_verdict = self.ai_analyst.evaluate_setup(best_signal, htf_analysis, current_spread)
                eqm_order.ai_conviction = ai_verdict.confidence

                if not ai_verdict.confirmed:
                    self.log(f"  🤖 AI GATE REJECTED ({ai_verdict.confidence:.1f}%): {ai_verdict.reason}", level="WARNING")
                    self.eqm.mark_rejected(eqm_order.order_id, rejection_code=4002, rejection_reason=f"AI Rejected: {ai_verdict.reason}")
                    continue

                self.log(f"  🤖 AI CONFIRMED ({ai_verdict.confidence:.1f}%): {ai_verdict.reason}")

                # ── Step 2c-3: Risk authorization (Pair-specific sizing, Market Regime & Temporal scaling) ──
                combined_risk_multiplier = regime_analysis.risk_multiplier * temporal_risk_mult
                equity = self.broker.get_account_equity()
                auth = self.risk_engine.authorize_trade(
                    best_signal, equity, fixed_lot_size=pair_lot, risk_multiplier=combined_risk_multiplier, current_time=now_utc
                )

                if not auth.authorized:
                    self.log(f"  🛑 RISK REJECTED: {auth.rejection_reason}", level="WARNING")
                    self.eqm.mark_rejected(eqm_order.order_id, rejection_code=4003, rejection_reason=f"Risk Rejected: {auth.rejection_reason}")
                    continue

                self.log(
                    f"  💰 AUTHORIZED: {auth.lot_size} lots (Risk Mult: {regime_analysis.risk_multiplier:.2f}) "
                    f"| Risk: ${auth.risk_amount:.2f} "
                    f"| Equity: ${auth.account_equity:.2f}"
                )

                # Enforce strict R:R bounds [min 1:1.5, max 1:3.0] prior to submission
                min_rr_limit = getattr(self.config.risk, 'min_rr_ratio', 1.5)
                max_rr_limit = getattr(self.config.risk, 'max_rr_ratio', 3.0)
                clamped_tp, eff_rr = clamp_tp_to_rr(
                    best_signal.entry_price,
                    best_signal.stop_loss,
                    best_signal.take_profit,
                    best_signal.direction,
                    min_rr=min_rr_limit,
                    max_rr=max_rr_limit,
                )
                best_signal.take_profit = clamped_tp
                best_signal.rr_ratio = eff_rr

                # Mark order submission in EQM
                self.eqm.mark_submission(
                    order_id=eqm_order.order_id,
                    requested_lot=auth.lot_size,
                    stop_loss=best_signal.stop_loss,
                    take_profit=best_signal.take_profit,
                    spread_at_submit=current_spread,
                    risk_pct=self.config.account.risk_pct * 100.0 * regime_analysis.risk_multiplier,
                    risk_amount=auth.risk_amount,
                    account_equity=equity,
                )

                # ── Step 2d: Idempotent order dispatch with retries ──
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

                client_order_id = generate_idempotency_key(
                    symbol=symbol,
                    strategy=best_signal.strategy_name,
                    direction=best_signal.direction.value,
                    entry_price=bracket.entry_price,
                    bar_time_iso=now_utc.isoformat()[:16],
                    magic=best_signal.magic_number,
                )

                order_state, order_result = self.order_manager.submit_bracket_order(bracket, client_order_id)

                if order_result and order_result.success:
                    self.log(
                        f"  ✅ ORDER FILLED ({order_state.value}): ID={order_result.order_id} "
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
                        broker_name=os.getenv("MT5_SERVER", "XMGlobal-MT5 6") or "XMGlobal-MT5 6",
                        session=getattr(best_signal, "session_killzone", "OFF_HOURS") or "LONDON_OPEN",
                    )
                    self.state.record_trade(trade_record)

                    # Mark fill in Execution Quality Metrics
                    if order_result.t1_submit_ns:
                        eqm_order.latency.t1_submit_ns = order_result.t1_submit_ns
                    if order_result.t2_ack_ns:
                        eqm_order.latency.t2_ack_ns = order_result.t2_ack_ns
                    if order_result.t3_fill_ns:
                        eqm_order.latency.t3_fill_ns = order_result.t3_fill_ns
                    if hasattr(order_result, 'filling_mode'):
                        eqm_order.filling_mode = order_result.filling_mode

                    filled_rec = self.eqm.mark_filled(
                        order_id=eqm_order.order_id,
                        trade_id=order_result.order_id,
                        filled_price=order_result.fill_price or bracket.entry_price,
                        filled_lot=bracket.lot_size,
                        spread_at_fill=current_spread,
                        retries_used=order_result.retries_used,
                    )
                    if filled_rec:
                        self.spread_guard.record_fill_slippage(symbol, filled_rec.slippage_pips)
                        self.log(
                            f"  📊 [EQM] Executed in {filled_rec.latency.total_latency_ms:.1f}ms "
                            f"| Slippage: {filled_rec.slippage_pips:+.2f} pips ({filled_rec.slippage_pct_atr:.1f}% ATR) "
                            f"| Retries: {filled_rec.retries_used}"
                        )
                else:
                    err_msg = order_result.error_message if order_result else "Order Dispatch Failed"
                    err_code = order_result.error_code if order_result else 0
                    retries = order_result.retries_used if order_result else 0
                    self.log(
                        f"  ❌ ORDER FAILED ({order_state.value}): {err_msg} "
                        f"(code={err_code}, retries={retries})",
                        level="ERROR"
                    )
                    if order_result.t1_submit_ns:
                        eqm_order.latency.t1_submit_ns = order_result.t1_submit_ns
                    if order_result.t2_ack_ns:
                        eqm_order.latency.t2_ack_ns = order_result.t2_ack_ns
                    if order_result.t3_fill_ns:
                        eqm_order.latency.t3_fill_ns = order_result.t3_fill_ns

                    self.eqm.mark_rejected(
                        order_id=eqm_order.order_id,
                        rejection_code=order_result.error_code or 0,
                        rejection_reason=order_result.error_message or "Order Failed",
                        retries_used=order_result.retries_used,
                    )

        # Check 5-Window Boundary Transition & Sync October CSV
        try:
            current_win = self.window_recorder.get_window_info(now_utc)
            current_wid = current_win["id"]
            if self._current_window_id is None:
                self._current_window_id = current_wid
            elif self._current_window_id != current_wid:
                prev_wid = self._current_window_id
                self._current_window_id = current_wid
                self.log(
                    f"⏰ [5-WINDOW TRANSITION] Window {prev_wid} completed -> Entering {current_wid} ({current_win['time_range']}) | October Shadow telemetry logging to CSV.",
                    level="INFO"
                )
                self.window_recorder.backfill_from_database()
        except Exception as e:
            logger.debug(f"Window transition error: {e}")

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
                # Immediate fallback check for common broker crypto/metal symbols
                fallback_map = {
                    "BTCUSD": ["BTC", "BTCUSDm", "BTC.USD"],
                    "ETHUSD": ["ETH", "ETHUSDm", "ETH.USD"],
                    "XAUUSD": ["GOLD", "XAUUSDm"],
                }
                candidates = fallback_map.get(symbol.upper(), [])
                for alt in candidates:
                    if alt != broker_symbol and mt5.symbol_select(alt, True):
                        alt_rates = mt5.copy_rates_from_pos(alt, tf_map[timeframe], 0, count)
                        if alt_rates is not None and len(alt_rates) > 0:
                            broker_symbol = alt
                            rates = alt_rates
                            if hasattr(self.broker, '_symbol_cache'):
                                self.broker._symbol_cache[symbol] = alt
                            break

            if rates is None or len(rates) == 0:
                err = mt5.last_error()
                # Rate-limit repetitive warning to once every 120s per symbol & timeframe
                warn_key = f"{broker_symbol}_{timeframe}"
                import time as _t
                now_ts = _t.time()
                if not hasattr(self, "_last_mt5_rate_warn"):
                    self._last_mt5_rate_warn = {}
                if now_ts - self._last_mt5_rate_warn.get(warn_key, 0) > 120:
                    self._last_mt5_rate_warn[warn_key] = now_ts
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

    _SWEEP_CONFIRMATIONS = {
        "LIQUIDITY_SWEEP",
        "OF_LIQUIDITY_TRAP",
        "ICT_JUDAS_SWING",
        "OF_ABSORPTION",
        "OF_DELTA_DIVERGENCE",
        "REVERSAL_ZONE_SD",
        "REVERSAL_ZONE_FIB",
        "REVERSAL_CONFLUENCE",
    }

    def _signal_to_trap_kind(self, sig: TradeSignal) -> EventKind:
        """Map any TradeSignal's LTF confirmation to a TrapDetector EventKind.

        Ensures 100% of strategy setups (SMC Swing, 5M Scalp, ICT Institutional,
        Order Flow, and Trend Reversal) are evaluated by the ML model.
        """
        conf_val = getattr(sig.ltf_confirmation, "value", str(sig.ltf_confirmation))
        is_buy = (sig.direction == Direction.BUY)

        if conf_val in ("REVERSAL_ZONE_FVG", "FVG"):
            return EventKind.FVG_BULL if is_buy else EventKind.FVG_BEAR

        if conf_val in self._SWEEP_CONFIRMATIONS:
            # Sweeps & SD/Fib Reversals:
            # SSL / discount level below 0.5 Fib taken -> potential long;
            # BSL / premium level above 0.5 Fib taken -> potential short
            return EventKind.SWEEP_SSL if is_buy else EventKind.SWEEP_BSL

        # Zone, FVG, Retest, Scalp, and Structural Setups:
        return EventKind.FVG_BULL if is_buy else EventKind.FVG_BEAR

    def set_data(self, symbol: str, timeframe: str, df) -> None:
        """Manually inject OHLCV data (useful for backtesting)."""
        cache_key = f"{symbol}_{timeframe}"
        self._htf_cache[cache_key] = df

    # ── Daily Directional Bias Subsystem ──

    def compute_daily_bias(self, symbol: str) -> dict:
        """
        Compute higher-timeframe daily directional bias for a symbol.
        Synthesizes 200 EMA trend, market structure, institutional dealing range
        (0.5 Fib equilibrium), zone failures, and trend reversal alerts.
        """
        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")
        formatted_date = now.strftime("%A, %b %d")

        # 1. Fetch HTF candle data (1H preferred, fallback 1D)
        htf_data = self._get_ohlcv(symbol, "1H")
        if htf_data is None or len(htf_data) < 20:
            htf_data = self._get_ohlcv(symbol, "1D")

        # Fallback if no data available
        if htf_data is None or len(htf_data) < 10:
            existing = self.state.get_daily_bias(symbol, today_str)
            if existing:
                return existing
            return {
                "symbol": symbol,
                "bias": "NEUTRAL",
                "trend_clarity_score": 10.0,
                "clarity_pct": 33,
                "ema_value": 0.0,
                "current_price": 0.0,
                "above_ema": False,
                "dealing_range_low": 0.0,
                "dealing_range_high": 0.0,
                "fib_50": 0.0,
                "is_discount": False,
                "is_premium": False,
                "zone_status": "EQUILIBRIUM",
                "first_zone_type": "NONE",
                "all_zones_failed": False,
                "reversal_risk": "LOW",
                "choch_detected": False,
                "reversal_stage": "TRENDING",
                "day_change_pct": 0.0,
                "date": today_str,
                "formatted_date": formatted_date,
                "summary": f"Awaiting market streaming data for {symbol}.",
                "updated_at": now.isoformat(),
            }

        # 2. Run HTF analysis
        htf_analysis = self.strategy.htf_analyzer.analyze(htf_data)
        current_close = float(htf_data['close'].iloc[-1])
        current_ema = float(htf_analysis.ema_value)
        is_above_ema = current_close > current_ema
        bias_str = htf_analysis.bias.value

        clarity_pct = min(100, max(15, int(round((htf_analysis.trend_clarity_score / 30.0) * 100))))
        zone_status = "DISCOUNT" if htf_analysis.is_discount else ("PREMIUM" if htf_analysis.is_premium else "EQUILIBRIUM")

        # Day Change % (if time column present)
        day_change_pct = 0.0
        try:
            if 'time' in htf_data.columns:
                t_series = pd.to_datetime(htf_data['time'], utc=True)
                today_rows = htf_data[t_series.dt.date == now.date()]
                if not today_rows.empty:
                    day_open = float(today_rows['open'].iloc[0])
                    if day_open > 0:
                        day_change_pct = round(((current_close - day_open) / day_open) * 100, 2)
        except Exception:
            pass

        # Check Trend Reversal detector if available
        reversal_risk = "LOW"
        choch_detected = False
        reversal_stage = "TRENDING"
        if hasattr(self, "trend_reversal_status") and symbol in self.trend_reversal_status:
            rev = self.trend_reversal_status[symbol]
            reversal_risk = getattr(rev, "reversal_risk", "LOW")
            choch_detected = getattr(rev, "choch_detected", False)
            reversal_stage = getattr(rev.stage, "value", str(rev.stage))

        # Build clean narrative summary
        if bias_str == "BULLISH":
            if htf_analysis.is_discount:
                summary = f"Bullish bias on {symbol}. Price is in Discount (<0.5 Fib Dealing Range), favoring long expansion."
            else:
                summary = f"Bullish bias on {symbol}. Price is above 200 EMA in Premium territory with upward order flow."
        elif bias_str == "BEARISH":
            if htf_analysis.is_premium:
                summary = f"Bearish bias on {symbol}. Price is in Premium (>=0.5 Fib Dealing Range), favoring short expansion."
            else:
                summary = f"Bearish bias on {symbol}. Price is below 200 EMA in Discount territory with downward order flow."
        else:
            summary = f"Neutral / Consolidating bias on {symbol}. Dealing range equilibrium in consolidation."

        res = {
            "symbol": symbol,
            "bias": bias_str,
            "trend_clarity_score": float(htf_analysis.trend_clarity_score),
            "clarity_pct": clarity_pct,
            "ema_value": round(current_ema, 5),
            "current_price": round(current_close, 5),
            "above_ema": is_above_ema,
            "dealing_range_low": round(htf_analysis.dealing_range_low, 5) if htf_analysis.dealing_range_low is not None else None,
            "dealing_range_high": round(htf_analysis.dealing_range_high, 5) if htf_analysis.dealing_range_high is not None else None,
            "fib_50": round(htf_analysis.fib_50, 5) if htf_analysis.fib_50 is not None else None,
            "is_discount": bool(htf_analysis.is_discount),
            "is_premium": bool(htf_analysis.is_premium),
            "zone_status": zone_status,
            "first_zone_type": str(htf_analysis.first_zone_type),
            "all_zones_failed": bool(htf_analysis.all_zones_failed),
            "reversal_risk": reversal_risk,
            "choch_detected": choch_detected,
            "reversal_stage": reversal_stage,
            "day_change_pct": day_change_pct,
            "date": today_str,
            "formatted_date": formatted_date,
            "summary": summary,
            "updated_at": now.isoformat(),
        }

        # Save to database and memory cache
        try:
            self.state.save_daily_bias(res)
        except Exception as e:
            logger.warning(f"Failed to persist daily bias for {symbol}: {e}")
        self._daily_bias_cache[symbol] = res
        return res

    def get_daily_bias_summary(self, target_symbol: str | None = None) -> dict:
        """
        Aggregate today's bias across all active monitored instruments.
        Returns primary bias, active symbols list, and detailed metrics map.
        """
        symbols = list(self.config.selected_symbols) if self.config.selected_symbols else []
        if not symbols:
            symbols = [self.config.pair1.symbol, self.config.pair2.symbol, self.config.pair3.symbol]

        unique_symbols = []
        for s in symbols:
            if s and s not in unique_symbols:
                unique_symbols.append(s)
        if not unique_symbols:
            unique_symbols = ["XAUUSD", "EURUSD", "GBPUSD"]

        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")
        formatted_date = now.strftime("%A, %b %d")

        symbols_map = {}
        for sym in unique_symbols:
            cached = self._daily_bias_cache.get(sym)
            if cached and cached.get("date") == today_str:
                symbols_map[sym] = cached
            else:
                symbols_map[sym] = self.compute_daily_bias(sym)

        primary_sym = (target_symbol or unique_symbols[0]).upper()
        if primary_sym not in symbols_map:
            symbols_map[primary_sym] = self.compute_daily_bias(primary_sym)

        primary_bias = symbols_map[primary_sym]["bias"]

        return {
            "date": today_str,
            "formatted_date": formatted_date,
            "primary_symbol": primary_sym,
            "primary_bias": primary_bias,
            "active_pairs": unique_symbols,
            "symbols": symbols_map,
        }


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

    # In standalone CLI terminal mode, activate trading session by default
    bot.is_active = True
    session_id = bot.state.record_activation(
        symbols=bot.config.selected_symbols,
        lot_size=f"P1:{bot.config.pair1.fixed_lot_size or 'Dyn'} | P2:{bot.config.pair2.fixed_lot_size or 'Dyn'} | P3:{bot.config.pair3.fixed_lot_size or 'Dyn'}",
        trigger_source="Laptop Terminal",
    )
    bot.log(
        f"🟢 BOT ACTIVATED via Laptop Terminal (Session #{session_id}). "
        f"Monitoring pairs: [{', '.join(bot.config.selected_symbols)}] | "
        f"Active Strategies: {bot.config.enabled_strategies}"
    )

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
        if bot.is_active:
            bot.state.record_deactivation("Laptop Terminal Stop")
            bot.is_active = False
        bot.shutdown()


def main() -> None:
    """Entry point."""
    asyncio.run(run_scheduled())


if __name__ == "__main__":
    main()

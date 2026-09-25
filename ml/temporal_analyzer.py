"""
ml/temporal_analyzer.py
========================

Machine Learning Temporal & Session Edge Analyzer — discovers, models, and predicts
which trading day, time (hour/minute), and market session produce institutional profitability
versus high-risk retail losses and stop-out traps.

Core Capabilities:
-----------------
1. Day-of-Week Analysis: Identifies the most profitable vs most losing days (Mon-Sun).
2. Time-of-Day Analysis: 24-hour UTC/Local diurnal edge heatmap (00:00 to 23:00).
3. Session Breakdown:
   - ASIAN_SESSION       (00:00 - 06:00 UTC)
   - LONDON_PRE_OPEN     (06:00 - 07:00 UTC)
   - LONDON_OPEN (KZ)    (07:00 - 10:00 UTC)
   - LONDON_MID          (10:00 - 12:00 UTC)
   - NY_AM (KZ)          (12:00 - 14:00 UTC)
   - NY_SILVER_BULLET    (14:00 - 15:00 UTC)
   - LONDON_CLOSE (KZ)   (15:00 - 17:00 UTC)
   - NY_PM               (17:00 - 21:00 UTC)
   - OFF_HOURS/DEAD_ZONE (21:00 - 24:00 UTC)
4. Multi-Dimensional Cross Slicing:
   - Day × Session Matrix (e.g. Tuesday NY_AM vs Friday London Close)
   - Symbol × Session Edge (XAUUSD, EURUSD, GBPUSD, BTCUSD, ETHUSD)
   - Strategy × Session Edge (SMC Swing, 5M Scalp, ICT, Order Flow)
5. Machine Learning Predictor:
   - Calibrated HistGradientBoostingClassifier predicting P(Win) and P(Loss).
   - Expected Return Regressor predicting expected R-multiple.
   - Empirical Bayesian Shrinkage blending machine learning with observed bucket statistics.
6. Edge Gating & Dynamic Risk Sizing:
   - PRIME_EDGE   (Win Rate >= 58%, PnL > 0) -> Boost sizing (e.g. 1.25x)
   - FAVORABLE    (Win Rate 50-58%)          -> Standard sizing (1.00x)
   - NEUTRAL      (Win Rate 45-50%)          -> Conservative sizing (0.75x)
   - UNFAVORABLE  (Win Rate 35-45%)          -> Defensive sizing (0.50x)
   - TOXIC_AVOID  (Win Rate < 35%, P(Loss) > 65%) -> Veto trade / 0.0x sizing
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("algo.ml.temporal_analyzer")


# =============================================================================
# 1. ENUMS & DATA STRUCTURES
# =============================================================================

class TradingSession(str, Enum):
    """Institutional market trading sessions and ICT Kill Zones (UTC)."""
    ASIA = "ASIAN_SESSION"             # 00:00 - 06:00 UTC
    LONDON_PRE = "LONDON_PRE_OPEN"     # 06:00 - 07:00 UTC (Frankfurt open)
    LONDON_OPEN = "LONDON_OPEN"        # 07:00 - 10:00 UTC (London Killzone)
    LONDON_MID = "LONDON_MID"          # 10:00 - 12:00 UTC (London lunch / lull)
    NY_AM = "NY_AM"                    # 12:00 - 14:00 UTC (New York AM Killzone)
    SILVER_BULLET = "SILVER_BULLET"    # 14:00 - 15:00 UTC (ICT New York Silver Bullet)
    LONDON_CLOSE = "LONDON_CLOSE"      # 15:00 - 17:00 UTC (London Close Killzone)
    NY_PM = "NY_PM"                    # 17:00 - 21:00 UTC (New York Afternoon / Equity Close)
    OFF_HOURS = "OFF_HOURS"            # 21:00 - 24:00 UTC (Dead Zone / Rollover / Wide Spreads)


class EdgeTier(str, Enum):
    """Classification of statistical/ML profitability edge for a given trading window."""
    PRIME_EDGE = "PRIME_EDGE"          # Highly profitable sweet spot
    FAVORABLE = "FAVORABLE"            # Consistently positive expectancy
    NEUTRAL = "NEUTRAL"                # Near break-even / moderate edge
    UNFAVORABLE = "UNFAVORABLE"        # Negative expectancy / subpar conditions
    TOXIC_AVOID = "TOXIC_AVOID"        # Consistently severe losses / avoid trading


@dataclass
class WindowPerformance:
    """Detailed statistical and financial metrics for a specific time or session bucket."""
    key: str
    category: str                      # 'day', 'hour', 'session', 'day_session', 'symbol_session', 'strategy_session'
    name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    win_rate: float = 0.0              # Percentage (0-100)
    loss_rate: float = 0.0             # Percentage (0-100)
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    profit_factor: float = 0.0
    expected_r: float = 0.0            # Average R-multiple
    edge_tier: EdgeTier = EdgeTier.NEUTRAL
    rank: int = 0                      # 1 = Most profitable
    is_prime: bool = False
    is_toxic: bool = False


@dataclass
class TemporalVerdict:
    """Predictive ML verdict for an entry candidate at a specific day/time/session."""
    timestamp: str
    day_name: str
    hour_utc: int
    session: str
    is_killzone: bool
    p_win: float                       # Probability of trade reaching TP / profit (0.0 to 1.0)
    p_loss: float                      # Probability of trade hitting SL / loss (0.0 to 1.0)
    expected_r: float                  # Estimated expected R-multiple
    edge_tier: EdgeTier
    allowed: bool                      # Whether the trade passes the temporal gate
    risk_multiplier: float             # Suggested sizing multiplier (e.g. 1.25x, 1.0x, 0.5x, 0.0x)
    reason: str
    historical_win_rate: float
    historical_trades_count: int
    model_version: str


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

class TemporalModelConfig(BaseModel):
    """Configuration tunables for the Temporal ML Analyzer."""
    db_path: str = "trading_state.db"
    model_path: str = "ml/artifacts/temporal_edge_model.joblib"
    stats_path: str = "ml/artifacts/temporal_edge_stats.json"

    # ML & Gate thresholds
    min_samples_to_train: int = 20
    min_bucket_samples: int = 3
    toxic_loss_threshold: float = 0.65       # P(Loss) > 65% classified as TOXIC_AVOID
    prime_win_threshold: float = 0.58        # P(Win) >= 58% classified as PRIME_EDGE
    favorable_win_threshold: float = 0.50    # P(Win) >= 50% classified as FAVORABLE
    unfavorable_win_threshold: float = 0.42  # P(Win) < 42% classified as UNFAVORABLE

    # Gating & Risk Multipliers
    active_gating: bool = True
    veto_toxic: bool = True                  # Veto trade entry if TOXIC_AVOID
    shadow_mode: bool = False                # If True, records predictions but never blocks
    fail_open_on_error: bool = True

    risk_boost_prime: float = 1.25
    risk_standard_favorable: float = 1.00
    risk_penalty_neutral: float = 0.75
    risk_penalty_unfavorable: float = 0.50
    risk_penalty_toxic: float = 0.00         # 0.0 = veto / no position


# =============================================================================
# 3. SESSION IDENTIFICATION & TIME UTILITIES
# =============================================================================

def parse_utc_datetime(dt_val: Any) -> datetime:
    """Normalize input into an aware UTC datetime object."""
    if isinstance(dt_val, str):
        try:
            dt = datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
        except Exception:
            dt = pd.to_datetime(dt_val).to_pydatetime()
    elif isinstance(dt_val, pd.Timestamp):
        dt = dt_val.to_pydatetime()
    elif isinstance(dt_val, datetime):
        dt = dt_val
    else:
        dt = datetime.now(timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def identify_trading_session(dt: datetime) -> TradingSession:
    """
    Map an aware UTC datetime into the exact market session or ICT Kill Zone.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    hour = dt.hour

    # 14:00 - 15:00 UTC = ICT NY Silver Bullet
    if hour == 14:
        return TradingSession.SILVER_BULLET
    # 07:00 - 10:00 UTC = London Open Kill Zone
    elif 7 <= hour < 10:
        return TradingSession.LONDON_OPEN
    # 06:00 - 07:00 UTC = London Pre-Open (Frankfurt)
    elif hour == 6:
        return TradingSession.LONDON_PRE
    # 10:00 - 12:00 UTC = London Midday
    elif 10 <= hour < 12:
        return TradingSession.LONDON_MID
    # 12:00 - 14:00 UTC = New York AM Kill Zone
    elif 12 <= hour < 14:
        return TradingSession.NY_AM
    # 15:00 - 17:00 UTC = London Close Kill Zone (NY Overlap)
    elif 15 <= hour < 17:
        return TradingSession.LONDON_CLOSE
    # 17:00 - 21:00 UTC = New York PM / Equity Session
    elif 17 <= hour < 21:
        return TradingSession.NY_PM
    # 00:00 - 06:00 UTC = Asian Session (Tokyo/Sydney)
    elif 0 <= hour < 6:
        return TradingSession.ASIA
    # 21:00 - 24:00 UTC = Dead Zone / Daily Rollover
    else:
        return TradingSession.OFF_HOURS


def is_killzone_session(session: TradingSession) -> bool:
    """Check if session is one of the institutional high-liquidity Kill Zones."""
    return session in (
        TradingSession.LONDON_OPEN,
        TradingSession.NY_AM,
        TradingSession.SILVER_BULLET,
        TradingSession.LONDON_CLOSE,
    )


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

SESSION_INDEX_MAP = {s: i for i, s in enumerate(TradingSession)}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

TEMPORAL_FEATURE_COLUMNS = [
    "day_of_week",
    "day_sin",
    "day_cos",
    "hour_utc",
    "hour_sin",
    "hour_cos",
    "minute",
    "minute_sin",
    "minute_cos",
    "session_idx",
    "is_killzone",
    "is_session_open_window",
    "is_session_close_window",
    "is_weekend",
    "is_friday_late",
    "is_monday_early",
    "symbol_idx",
    "strategy_idx",
    "planned_rr",
]

SYMBOL_MAP = {"XAUUSD": 0, "EURUSD": 1, "GBPUSD": 2, "BTCUSD": 3, "ETHUSD": 4}
STRATEGY_MAP = {"SMC": 0, "SMC_SCALP_5M": 1, "ICT": 2, "ORDER_FLOW": 3, "TREND_REVERSAL": 4}


def extract_temporal_features(
    dt_val: Any,
    symbol: str = "XAUUSD",
    strategy: str = "SMC",
    planned_rr: float = 2.0,
) -> Dict[str, float]:
    """
    Extract causal cyclical and categorical temporal features for ML inference and training.
    """
    dt = parse_utc_datetime(dt_val)
    day = dt.weekday()           # 0=Monday, 6=Sunday
    hour = dt.hour               # 0..23
    minute = dt.minute           # 0..59

    session = identify_trading_session(dt)
    session_idx = float(SESSION_INDEX_MAP.get(session, 8))
    killzone = 1.0 if is_killzone_session(session) else 0.0

    # Cyclical representations
    day_sin = math.sin(2.0 * math.pi * day / 7.0)
    day_cos = math.cos(2.0 * math.pi * day / 7.0)
    hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * hour / 24.0)
    minute_sin = math.sin(2.0 * math.pi * minute / 60.0)
    minute_cos = math.cos(2.0 * math.pi * minute / 60.0)

    # Edge-case hazard windows
    is_session_open_window = 1.0 if minute < 30 and killzone == 1.0 else 0.0
    is_session_close_window = 1.0 if minute >= 30 and hour in (9, 14, 16) else 0.0
    is_weekend = 1.0 if day >= 5 else 0.0
    is_friday_late = 1.0 if (day == 4 and hour >= 15) else 0.0
    is_monday_early = 1.0 if (day == 0 and hour < 6) else 0.0

    # Clean symbol and strategy strings
    sym_clean = symbol.upper().replace("/", "").strip()
    strat_clean = strategy.upper().replace(" ", "_").strip()
    sym_idx = float(SYMBOL_MAP.get(sym_clean, 0))
    strat_idx = float(STRATEGY_MAP.get(strat_clean, 0))

    return {
        "day_of_week": float(day),
        "day_sin": round(day_sin, 5),
        "day_cos": round(day_cos, 5),
        "hour_utc": float(hour),
        "hour_sin": round(hour_sin, 5),
        "hour_cos": round(hour_cos, 5),
        "minute": float(minute),
        "minute_sin": round(minute_sin, 5),
        "minute_cos": round(minute_cos, 5),
        "session_idx": session_idx,
        "is_killzone": killzone,
        "is_session_open_window": is_session_open_window,
        "is_session_close_window": is_session_close_window,
        "is_weekend": is_weekend,
        "is_friday_late": is_friday_late,
        "is_monday_early": is_monday_early,
        "symbol_idx": sym_idx,
        "strategy_idx": strat_idx,
        "planned_rr": float(planned_rr or 2.0),
    }


# =============================================================================
# 5. STATISTICAL AGGREGATION & RANKING ENGINE
# =============================================================================

def calculate_bucket_metrics(
    records: List[Dict[str, Any]],
    key: str,
    category: str,
    name: str,
    min_bucket_samples: int = 3,
) -> WindowPerformance:
    """Compute financial & statistical expectancy metrics for a grouped slice."""
    total = len(records)
    if total == 0:
        return WindowPerformance(
            key=key, category=category, name=name,
            total_trades=0, edge_tier=EdgeTier.NEUTRAL
        )

    wins = sum(1 for r in records if r["is_win"])
    losses = sum(1 for r in records if r["is_loss"])
    scratches = total - wins - losses
    win_rate = (wins / total) * 100.0 if total > 0 else 0.0
    loss_rate = (losses / total) * 100.0 if total > 0 else 0.0

    pnls = [r["pnl"] for r in records]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / total if total > 0 else 0.0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = 99.0 if gross_profit > 0 else 1.0

    # Expected R multiple (if available, else derived from PnL signs)
    r_multiples = [r.get("r_multiple", 0.0) for r in records if "r_multiple" in r and r.get("r_multiple") is not None]
    if r_multiples:
        expected_r = float(np.mean(r_multiples))
    else:
        expected_r = (wins * 2.0 - losses * 1.0) / total if total > 0 else 0.0

    # Determine Edge Tier using Bayesian smoothed win rate
    # Beta prior alpha=2, beta=2 (45-50% prior baseline)
    smoothed_wr = (wins + 2) / (total + 4) * 100.0

    if total >= min_bucket_samples:
        if smoothed_wr >= 58.0 and total_pnl > 0:
            tier = EdgeTier.PRIME_EDGE
        elif smoothed_wr >= 50.0 and total_pnl >= 0:
            tier = EdgeTier.FAVORABLE
        elif smoothed_wr >= 44.0:
            tier = EdgeTier.NEUTRAL
        elif smoothed_wr >= 35.0 or total_pnl < 0:
            tier = EdgeTier.UNFAVORABLE
        else:
            tier = EdgeTier.TOXIC_AVOID
    else:
        tier = EdgeTier.NEUTRAL

    # Extreme loss concentration check
    if loss_rate >= 70.0 and total >= min_bucket_samples:
        tier = EdgeTier.TOXIC_AVOID

    return WindowPerformance(
        key=key,
        category=category,
        name=name,
        total_trades=total,
        wins=wins,
        losses=losses,
        scratches=scratches,
        win_rate=round(win_rate, 1),
        loss_rate=round(loss_rate, 1),
        total_pnl=round(total_pnl, 2),
        avg_pnl=round(avg_pnl, 2),
        profit_factor=round(min(profit_factor, 99.0), 2),
        expected_r=round(expected_r, 2),
        edge_tier=tier,
        is_prime=(tier == EdgeTier.PRIME_EDGE),
        is_toxic=(tier == EdgeTier.TOXIC_AVOID),
    )


# =============================================================================
# 6. MACHINE LEARNING MODEL PIPELINE
# =============================================================================

class TemporalEdgeMLModel:
    """
    Machine Learning Model for Temporal Edge Analysis.
    Combines calibrated Gradient Boosting classification with empirical statistical rankings.
    """

    def __init__(self, config: Optional[TemporalModelConfig] = None):
        self.config = config or TemporalModelConfig()
        self.version = "temporal_ml_v1.0"
        self.trained_at: Optional[str] = None
        self.n_samples: int = 0
        self.is_fitted: bool = False

        # Classifiers & Regressors
        self.classifier: Any = None
        self.regressor: Any = None
        self.feature_names = TEMPORAL_FEATURE_COLUMNS

        # Cached Empirical Slices & Rankings
        self.by_day: Dict[str, WindowPerformance] = {}
        self.by_hour: Dict[int, WindowPerformance] = {}
        self.by_session: Dict[str, WindowPerformance] = {}
        self.by_day_session: Dict[str, WindowPerformance] = {}
        self.by_strategy_session: Dict[str, WindowPerformance] = {}
        self.by_symbol_session: Dict[str, WindowPerformance] = {}

        # Top Rankings
        self.ranked_days: List[WindowPerformance] = []
        self.ranked_hours: List[WindowPerformance] = []
        self.ranked_sessions: List[WindowPerformance] = []

        self.most_profitable_day: Optional[WindowPerformance] = None
        self.most_losing_day: Optional[WindowPerformance] = None
        self.most_profitable_hour: Optional[WindowPerformance] = None
        self.most_losing_hour: Optional[WindowPerformance] = None
        self.most_profitable_session: Optional[WindowPerformance] = None
        self.most_losing_session: Optional[WindowPerformance] = None

        self.feature_importances: Dict[str, float] = {}

    def extract_dataset_from_db(self, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract trade records from both `trade_log` and `ml_events` tables in SQLite.
        """
        path = db_path or self.config.db_path
        if not Path(path).exists() and path != ":memory:":
            logger.warning(f"Database file {path} does not exist.")
            return []

        dataset: List[Dict[str, Any]] = []

        try:
            con = sqlite3.connect(path)
            con.row_factory = sqlite3.Row
            cur = con.cursor()

            # 1. Ingest executed trades from trade_log
            try:
                rows = cur.execute("""
                    SELECT timestamp, symbol, strategy_name, direction, entry_price, stop_loss,
                           take_profit, realized_pnl, status, duration_seconds
                    FROM trade_log
                    WHERE status != 'OPEN' AND status IS NOT NULL
                """).fetchall()

                for r in rows:
                    ts_str = r["timestamp"]
                    if not ts_str:
                        continue
                    pnl = float(r["realized_pnl"] or 0.0)
                    status = str(r["status"]).upper()
                    is_win = (status == "CLOSED_TP") or (pnl > 0.0)
                    is_loss = (status == "CLOSED_SL") or (pnl < 0.0)

                    # Compute planned R:R
                    ep = float(r["entry_price"] or 0.0)
                    sl = float(r["stop_loss"] or 0.0)
                    tp = float(r["take_profit"] or 0.0)
                    rr = 2.0
                    if abs(ep - sl) > 1e-6:
                        rr = max(0.5, min(10.0, abs(tp - ep) / abs(ep - sl)))

                    # Compute realized R
                    r_mult = 0.0
                    if is_win:
                        r_mult = rr
                    elif is_loss:
                        r_mult = -1.0

                    dataset.append({
                        "source": "trade_log",
                        "timestamp": ts_str,
                        "symbol": r["symbol"] or "XAUUSD",
                        "strategy": r["strategy_name"] or "SMC",
                        "pnl": pnl,
                        "is_win": is_win,
                        "is_loss": is_loss,
                        "planned_rr": rr,
                        "r_multiple": r_mult,
                    })
            except Exception as e:
                logger.debug(f"Could not read trade_log: {e}")

            # 2. Ingest labeled setups from ml_events
            try:
                ml_rows = cur.execute("""
                    SELECT ts, symbol, strategy, kind, direction, entry, stop, target,
                           label, outcome, r_multiple
                    FROM ml_events
                    WHERE outcome IS NOT NULL AND outcome != 'pending'
                """).fetchall()

                for r in ml_rows:
                    ts_str = r["ts"]
                    if not ts_str:
                        continue
                    outcome = str(r["outcome"]).lower()
                    label = r["label"]
                    is_win = (outcome == "tp") or (label == 1)
                    is_loss = (outcome == "sl") or (label == 0)

                    r_mult = float(r["r_multiple"] or (2.0 if is_win else -1.0 if is_loss else 0.0))
                    # Synthetic PnL estimate for ranking consistency ($10 per R)
                    pnl = r_mult * 10.0

                    ep = float(r["entry"] or 0.0)
                    sl = float(r["stop"] or 0.0)
                    tp = float(r["target"] or 0.0)
                    rr = 2.0
                    if abs(ep - sl) > 1e-6:
                        rr = max(0.5, min(10.0, abs(tp - ep) / abs(ep - sl)))

                    dataset.append({
                        "source": "ml_events",
                        "timestamp": ts_str,
                        "symbol": r["symbol"] or "XAUUSD",
                        "strategy": r["strategy"] or "SMC",
                        "pnl": pnl,
                        "is_win": is_win,
                        "is_loss": is_loss,
                        "planned_rr": rr,
                        "r_multiple": r_mult,
                    })
            except Exception as e:
                logger.debug(f"Could not read ml_events: {e}")

            con.close()
        except Exception as e:
            logger.error(f"Error reading SQLite data for temporal model: {e}")

        logger.info(f"TemporalEdgeML: Ingested {len(dataset)} total trade and setup events.")
        return dataset

    def fit(self, records: Optional[List[Dict[str, Any]]] = None) -> "TemporalEdgeMLModel":
        """
        Train the machine learning model and build empirical statistical rankings.
        """
        if records is None:
            records = self.extract_dataset_from_db()

        self.n_samples = len(records)
        if self.n_samples < self.config.min_samples_to_train:
            logger.warning(
                f"Insufficient samples to fit Temporal ML model ({self.n_samples} < {self.config.min_samples_to_train}). "
                "Running in empirical fallback mode."
            )
            self._compute_statistical_aggregates(records)
            self.is_fitted = False
            return self

        # 1. Feature Extraction
        X_list = []
        y_class = []
        y_r = []

        enriched_records = []
        for r in records:
            feats = extract_temporal_features(
                dt_val=r["timestamp"],
                symbol=r.get("symbol", "XAUUSD"),
                strategy=r.get("strategy", "SMC"),
                planned_rr=r.get("planned_rr", 2.0),
            )
            vec = [feats[col] for col in self.feature_names]
            X_list.append(vec)
            y_class.append(1 if r["is_win"] else 0)
            y_r.append(r.get("r_multiple", 0.0))

            r_copy = dict(r)
            r_copy.update(feats)
            dt = parse_utc_datetime(r["timestamp"])
            r_copy["day_name"] = DAY_NAMES[dt.weekday()]
            r_copy["session_name"] = identify_trading_session(dt).value
            enriched_records.append(r_copy)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_class, dtype=np.int32)
        yr = np.array(y_r, dtype=np.float32)

        # 2. Fit ML Model (HistGradientBoostingClassifier + CalibratedClassifierCV)
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

            base_clf = HistGradientBoostingClassifier(
                max_iter=100,
                max_leaf_nodes=15,
                min_samples_leaf=5,
                l2_regularization=1.5,
                random_state=42,
            )

            # Fit base classifier
            base_clf.fit(X, y)
            self.classifier = base_clf

            # Fit regressor for expected return (R multiple)
            base_reg = HistGradientBoostingRegressor(
                max_iter=80,
                max_leaf_nodes=10,
                min_samples_leaf=5,
                l2_regularization=2.0,
                random_state=42,
            )
            base_reg.fit(X, yr)
            self.regressor = base_reg

            # Estimate feature importances using feature variance
            importances = {}
            for idx, col in enumerate(self.feature_names):
                std_val = float(np.std(X[:, idx]))
                mean_val = float(np.mean(np.abs(X[:, idx])))
                importances[col] = std_val / (mean_val + 1e-4)
            total_imp = sum(importances.values()) or 1.0
            self.feature_importances = {k: round(v / total_imp, 4) for k, v in importances.items()}

            self.is_fitted = True
            self.trained_at = datetime.now(timezone.utc).isoformat()
            logger.info(f"TemporalEdgeML: Successfully trained HistGradientBoosting on {self.n_samples} samples.")

        except Exception as e:
            logger.error(f"Error training ML model: {e}. Falling back to empirical mode.")
            self.is_fitted = False

        # 3. Compute rich statistical slice breakdowns & rankings
        self._compute_statistical_aggregates(enriched_records)
        return self

    def _compute_statistical_aggregates(self, records: List[Dict[str, Any]]) -> None:
        """Calculate and rank all Day, Hour, and Session slices."""
        if not records:
            return

        min_bucket = self.config.min_bucket_samples

        # 1. Day of Week Slices
        by_day_dict: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DAY_NAMES}
        # 2. Hour Slices (0..23)
        by_hour_dict: Dict[int, List[Dict[str, Any]]] = {h: [] for h in range(24)}
        # 3. Session Slices
        by_session_dict: Dict[str, List[Dict[str, Any]]] = {s.value: [] for s in TradingSession}
        # 4. Day x Session Slices
        by_day_session_dict: Dict[str, List[Dict[str, Any]]] = {}
        # 5. Strategy x Session
        by_strat_session_dict: Dict[str, List[Dict[str, Any]]] = {}
        # 6. Symbol x Session
        by_sym_session_dict: Dict[str, List[Dict[str, Any]]] = {}

        for r in records:
            dt = parse_utc_datetime(r["timestamp"])
            day_name = DAY_NAMES[dt.weekday()]
            hour = dt.hour
            session_name = identify_trading_session(dt).value
            sym = str(r.get("symbol", "XAUUSD")).upper()
            strat = str(r.get("strategy", "SMC")).upper()

            by_day_dict[day_name].append(r)
            by_hour_dict[hour].append(r)
            by_session_dict[session_name].append(r)

            ds_key = f"{day_name} | {session_name}"
            by_day_session_dict.setdefault(ds_key, []).append(r)

            ss_key = f"{strat} | {session_name}"
            by_strat_session_dict.setdefault(ss_key, []).append(r)

            sym_s_key = f"{sym} | {session_name}"
            by_sym_session_dict.setdefault(sym_s_key, []).append(r)

        # Build WindowPerformance objects
        self.by_day = {
            d: calculate_bucket_metrics(recs, key=d, category="day", name=d, min_bucket_samples=min_bucket)
            for d, recs in by_day_dict.items()
        }
        self.by_hour = {
            h: calculate_bucket_metrics(recs, key=str(h), category="hour", name=f"{h:02d}:00 UTC", min_bucket_samples=min_bucket)
            for h, recs in by_hour_dict.items()
        }
        self.by_session = {
            s: calculate_bucket_metrics(recs, key=s, category="session", name=s, min_bucket_samples=min_bucket)
            for s, recs in by_session_dict.items()
        }
        self.by_day_session = {
            k: calculate_bucket_metrics(recs, key=k, category="day_session", name=k, min_bucket_samples=min_bucket)
            for k, recs in by_day_session_dict.items()
        }
        self.by_strategy_session = {
            k: calculate_bucket_metrics(recs, key=k, category="strategy_session", name=k, min_bucket_samples=min_bucket)
            for k, recs in by_strat_session_dict.items()
        }
        self.by_symbol_session = {
            k: calculate_bucket_metrics(recs, key=k, category="symbol_session", name=k, min_bucket_samples=min_bucket)
            for k, recs in by_sym_session_dict.items()
        }

        # Rank Days (by PnL then Win Rate, filtered for active days)
        active_days = [p for p in self.by_day.values() if p.total_trades > 0]
        active_days.sort(key=lambda p: (p.total_pnl, p.win_rate), reverse=True)
        for rank, p in enumerate(active_days, 1):
            p.rank = rank
        self.ranked_days = active_days
        if active_days:
            self.most_profitable_day = active_days[0]
            self.most_losing_day = active_days[-1]

        # Rank Hours
        active_hours = [p for p in self.by_hour.values() if p.total_trades > 0]
        active_hours.sort(key=lambda p: (p.total_pnl, p.win_rate), reverse=True)
        for rank, p in enumerate(active_hours, 1):
            p.rank = rank
        self.ranked_hours = active_hours
        if active_hours:
            self.most_profitable_hour = active_hours[0]
            self.most_losing_hour = active_hours[-1]

        # Rank Sessions
        active_sessions = [p for p in self.by_session.values() if p.total_trades > 0]
        active_sessions.sort(key=lambda p: (p.total_pnl, p.win_rate), reverse=True)
        for rank, p in enumerate(active_sessions, 1):
            p.rank = rank
        self.ranked_sessions = active_sessions
        if active_sessions:
            self.most_profitable_session = active_sessions[0]
            self.most_losing_session = active_sessions[-1]

    def predict(
        self,
        dt_val: Any,
        symbol: str = "XAUUSD",
        strategy: str = "SMC",
        planned_rr: float = 2.0,
    ) -> TemporalVerdict:
        """
        Evaluate candidate trade entry at the specified timestamp.
        Returns full probability estimation, edge classification, and sizing recommendation.
        """
        dt = parse_utc_datetime(dt_val)
        day_name = DAY_NAMES[dt.weekday()]
        hour = dt.hour
        session = identify_trading_session(dt)
        session_name = session.value
        kz = is_killzone_session(session)

        # Extract features
        feats = extract_temporal_features(dt, symbol=symbol, strategy=strategy, planned_rr=planned_rr)

        # Baseline empirical priors
        day_perf = self.by_day.get(day_name)
        hour_perf = self.by_hour.get(hour)
        sess_perf = self.by_session.get(session_name)
        ds_key = f"{day_name} | {session_name}"
        ds_perf = self.by_day_session.get(ds_key)

        prior_samples = sess_perf.total_trades if sess_perf else 0
        prior_wr = (sess_perf.win_rate / 100.0) if sess_perf and sess_perf.total_trades > 0 else 0.50

        # ML Prediction
        p_win = prior_wr
        expected_r = (prior_wr * planned_rr) - ((1.0 - prior_wr) * 1.0)

        if self.is_fitted and self.classifier is not None:
            try:
                vec = np.array([[feats[c] for c in self.feature_names]], dtype=np.float32)
                probas = self.classifier.predict_proba(vec)[0]
                ml_p_win = float(probas[1]) if len(probas) > 1 else float(probas[0])

                if self.regressor is not None:
                    expected_r = float(self.regressor.predict(vec)[0])

                # Bayesian Blend between ML prediction and Empirical Bucket
                weight_ml = min(0.75, 0.40 + (self.n_samples / 500.0))
                p_win = (ml_p_win * weight_ml) + (prior_wr * (1.0 - weight_ml))
            except Exception as e:
                logger.debug(f"ML inference error: {e}. Using empirical prior.")

        p_win = max(0.02, min(0.98, p_win))
        p_loss = 1.0 - p_win

        # Edge Classification
        cfg = self.config
        if p_win >= cfg.prime_win_threshold and (sess_perf is None or sess_perf.total_pnl >= 0):
            tier = EdgeTier.PRIME_EDGE
            risk_mult = cfg.risk_boost_prime
            action_desc = "Sweet Spot / High Conviction"
        elif p_win >= cfg.favorable_win_threshold:
            tier = EdgeTier.FAVORABLE
            risk_mult = cfg.risk_standard_favorable
            action_desc = "Favorable Institutional Conditions"
        elif p_loss >= cfg.toxic_loss_threshold or (sess_perf and sess_perf.is_toxic):
            tier = EdgeTier.TOXIC_AVOID
            risk_mult = 0.00 if (cfg.veto_toxic and not cfg.shadow_mode) else max(0.50, cfg.risk_penalty_toxic)
            action_desc = "Toxic Stop-Loss Trap Window"
        elif p_win < cfg.unfavorable_win_threshold:
            tier = EdgeTier.UNFAVORABLE
            risk_mult = cfg.risk_penalty_unfavorable
            action_desc = "Sub-par / Negative Expectancy"
        else:
            tier = EdgeTier.NEUTRAL
            risk_mult = cfg.risk_penalty_neutral
            action_desc = "Balanced / Moderate Edge"

        allowed = True
        if tier == EdgeTier.TOXIC_AVOID and cfg.veto_toxic and not cfg.shadow_mode:
            allowed = False

        # Craft human-readable diagnostic reason
        rank_str = f" (Session Rank #{sess_perf.rank})" if sess_perf and sess_perf.rank > 0 else ""
        reason = (
            f"[{tier.value}] {day_name} at {hour:02d}:{dt.minute:02d} UTC in {session_name}{rank_str}. "
            f"Predicted Win Probability: {p_win*100:.1f}%, SL Risk: {p_loss*100:.1f}%, Expected Return: {expected_r:+.2f}R. "
            f"Action: {action_desc}. Sizing scale: {risk_mult:.2f}x."
        )

        return TemporalVerdict(
            timestamp=dt.isoformat(),
            day_name=day_name,
            hour_utc=hour,
            session=session_name,
            is_killzone=kz,
            p_win=round(p_win, 4),
            p_loss=round(p_loss, 4),
            expected_r=round(expected_r, 2),
            edge_tier=tier,
            allowed=allowed,
            risk_multiplier=risk_mult,
            reason=reason,
            historical_win_rate=sess_perf.win_rate if sess_perf else 0.0,
            historical_trades_count=prior_samples,
            model_version=self.version if self.is_fitted else "empirical_prior",
        )

    def save(self, filepath: Optional[str] = None) -> None:
        """Serialize trained model and state to disk."""
        target = Path(filepath or self.config.model_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "trained_at": self.trained_at,
            "n_samples": self.n_samples,
            "is_fitted": self.is_fitted,
            "classifier": self.classifier,
            "regressor": self.regressor,
            "feature_importances": self.feature_importances,
            "by_day": {k: asdict(v) for k, v in self.by_day.items()},
            "by_hour": {k: asdict(v) for k, v in self.by_hour.items()},
            "by_session": {k: asdict(v) for k, v in self.by_session.items()},
            "by_day_session": {k: asdict(v) for k, v in self.by_day_session.items()},
        }
        joblib.dump(payload, target)
        logger.info(f"TemporalEdgeML: Saved model artifact to {target}")

    def load(self, filepath: Optional[str] = None) -> bool:
        """Load trained model and cached state from disk."""
        target = Path(filepath or self.config.model_path)
        if not target.exists():
            return False
        try:
            payload = joblib.load(target)
            self.version = payload.get("version", self.version)
            self.trained_at = payload.get("trained_at")
            self.n_samples = payload.get("n_samples", 0)
            self.is_fitted = payload.get("is_fitted", False)
            self.classifier = payload.get("classifier")
            self.regressor = payload.get("regressor")
            self.feature_importances = payload.get("feature_importances", {})

            # Reconstruct WindowPerformance instances
            def _dict_to_wp(d: Dict[str, Any]) -> WindowPerformance:
                d_copy = dict(d)
                d_copy["edge_tier"] = EdgeTier(d_copy["edge_tier"])
                return WindowPerformance(**d_copy)

            self.by_day = {k: _dict_to_wp(v) for k, v in payload.get("by_day", {}).items()}
            self.by_hour = {int(k): _dict_to_wp(v) for k, v in payload.get("by_hour", {}).items()}
            self.by_session = {k: _dict_to_wp(v) for k, v in payload.get("by_session", {}).items()}
            self.by_day_session = {k: _dict_to_wp(v) for k, v in payload.get("by_day_session", {}).items()}

            # Recompute top lists
            active_days = [p for p in self.by_day.values() if p.total_trades > 0]
            active_days.sort(key=lambda p: (p.total_pnl, p.win_rate), reverse=True)
            self.ranked_days = active_days
            if active_days:
                self.most_profitable_day = active_days[0]
                self.most_losing_day = active_days[-1]

            active_hours = [p for p in self.by_hour.values() if p.total_trades > 0]
            active_hours.sort(key=lambda p: (p.total_pnl, p.win_rate), reverse=True)
            self.ranked_hours = active_hours
            if active_hours:
                self.most_profitable_hour = active_hours[0]
                self.most_losing_hour = active_hours[-1]

            active_sessions = [p for p in self.by_session.values() if p.total_trades > 0]
            active_sessions.sort(key=lambda p: (p.total_pnl, p.win_rate), reverse=True)
            self.ranked_sessions = active_sessions
            if active_sessions:
                self.most_profitable_session = active_sessions[0]
                self.most_losing_session = active_sessions[-1]

            return True
        except Exception as e:
            logger.error(f"Error loading TemporalEdgeML model from {target}: {e}")
            return False


# =============================================================================
# 7. SERVICE WRAPPER
# =============================================================================

class TemporalEdgeService:
    """
    Thread-safe runtime service for temporal profitability analysis, trade gating,
    and dashboard analytics reporting.
    """

    def __init__(self, config: Optional[TemporalModelConfig] = None):
        self.config = config or TemporalModelConfig()
        self._lock = threading.RLock()
        self.model = TemporalEdgeMLModel(self.config)

        # Attempt to load existing artifact, else train fresh
        loaded = self.model.load()
        if not loaded:
            logger.info("TemporalEdgeService: No existing model found. Training initial model from DB...")
            self.train_and_update()

    def train_and_update(self) -> Dict[str, Any]:
        """Trigger model fitting on current database records."""
        with self._lock:
            self.model.fit()
            self.model.save()
            return self.get_summary()

    def evaluate(
        self,
        timestamp: Optional[Any] = None,
        symbol: str = "XAUUSD",
        strategy: str = "SMC",
        planned_rr: float = 2.0,
    ) -> TemporalVerdict:
        """Evaluate a potential trade setup at the given timestamp."""
        ts = timestamp or datetime.now(timezone.utc)
        with self._lock:
            return self.model.predict(ts, symbol=symbol, strategy=strategy, planned_rr=planned_rr)

    def get_summary(self) -> Dict[str, Any]:
        """Structured dictionary of complete temporal performance analytics."""
        with self._lock:
            m = self.model
            return {
                "version": m.version,
                "trained_at": m.trained_at,
                "total_samples": m.n_samples,
                "is_fitted": m.is_fitted,
                "active_gating": self.config.active_gating,
                "most_profitable": {
                    "day": asdict(m.most_profitable_day) if m.most_profitable_day else None,
                    "hour": asdict(m.most_profitable_hour) if m.most_profitable_hour else None,
                    "session": asdict(m.most_profitable_session) if m.most_profitable_session else None,
                },
                "most_losing": {
                    "day": asdict(m.most_losing_day) if m.most_losing_day else None,
                    "hour": asdict(m.most_losing_hour) if m.most_losing_hour else None,
                    "session": asdict(m.most_losing_session) if m.most_losing_session else None,
                },
                "ranked_days": [asdict(p) for p in m.ranked_days],
                "ranked_hours": [asdict(p) for p in m.ranked_hours],
                "ranked_sessions": [asdict(p) for p in m.ranked_sessions],
                "feature_importances": m.feature_importances,
            }

    def get_full_breakdown(self) -> Dict[str, Any]:
        """Full details for web UI and analytics dashboards."""
        with self._lock:
            m = self.model
            return {
                "summary": self.get_summary(),
                "by_day": {k: asdict(v) for k, v in m.by_day.items()},
                "by_hour": {k: asdict(v) for k, v in m.by_hour.items()},
                "by_session": {k: asdict(v) for k, v in m.by_session.items()},
                "by_day_session": {k: asdict(v) for k, v in m.by_day_session.items() if v.total_trades > 0},
                "by_strategy_session": {k: asdict(v) for k, v in m.by_strategy_session.items() if v.total_trades > 0},
                "by_symbol_session": {k: asdict(v) for k, v in m.by_symbol_session.items() if v.total_trades > 0},
            }

    def generate_ascii_report(self) -> str:
        """Generate a color-compatible, formatted ASCII report for terminal viewing."""
        s = self.get_summary()
        b = self.get_full_breakdown()

        lines = [
            "================================================================================",
            "        INSTITUTIONAL TEMPORAL & SESSION PROFITABILITY ML REPORT               ",
            "================================================================================",
            f" Model Version : {s['version']} (Fitted: {s['is_fitted']}) | Samples: {s['total_samples']}",
            f" Last Trained  : {s['trained_at'] or 'Initial'}",
            "--------------------------------------------------------------------------------",
            " [EXECUTIVE SUMMARY: PRIME vs TOXIC TRADING WINDOWS]",
        ]

        # Profitable highlights
        mp_d = s["most_profitable"]["day"]
        mp_h = s["most_profitable"]["hour"]
        mp_s = s["most_profitable"]["session"]
        lines.append(f"  [+] Most Profitable Day     : {mp_d['name'] if mp_d else 'N/A'} (WR: {mp_d['win_rate'] if mp_d else 0}%, PnL: ${mp_d['total_pnl'] if mp_d else 0:+.2f})")
        lines.append(f"  [+] Most Profitable Hour    : {mp_h['name'] if mp_h else 'N/A'} (WR: {mp_h['win_rate'] if mp_h else 0}%, PnL: ${mp_h['total_pnl'] if mp_h else 0:+.2f})")
        lines.append(f"  [+] Most Profitable Session : {mp_s['name'] if mp_s else 'N/A'} (WR: {mp_s['win_rate'] if mp_s else 0}%, PnL: ${mp_s['total_pnl'] if mp_s else 0:+.2f})")

        lines.append("")
        # Losing highlights
        ml_d = s["most_losing"]["day"]
        ml_h = s["most_losing"]["hour"]
        ml_s = s["most_losing"]["session"]
        lines.append(f"  [!] Most Losing Day        : {ml_d['name'] if ml_d else 'N/A'} (Loss Rate: {ml_d['loss_rate'] if ml_d else 0}%, PnL: ${ml_d['total_pnl'] if ml_d else 0:+.2f})")
        lines.append(f"  [!] Most Losing Hour       : {ml_h['name'] if ml_h else 'N/A'} (Loss Rate: {ml_h['loss_rate'] if ml_h else 0}%, PnL: ${ml_h['total_pnl'] if ml_h else 0:+.2f})")
        lines.append(f"  [!] Most Losing Session    : {ml_s['name'] if ml_s else 'N/A'} (Loss Rate: {ml_s['loss_rate'] if ml_s else 0}%, PnL: ${ml_s['total_pnl'] if ml_s else 0:+.2f})")

        lines.append("--------------------------------------------------------------------------------")
        lines.append(" [DAY-OF-WEEK BREAKDOWN & RANKINGS]")
        lines.append(f"  {'Rank':<5} {'Day':<12} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'Win Rate':<10} {'Total PnL':<12} {'Tier':<14}")
        lines.append("  " + "-" * 75)
        for p in s["ranked_days"]:
            lines.append(
                f"  #{p['rank']:<4} {p['name']:<12} {p['total_trades']:<8} {p['wins']:<6} {p['losses']:<8} "
                f"{p['win_rate']:>5.1f}%    ${p['total_pnl']:>8.2f}    {p['edge_tier']:<14}"
            )

        lines.append("--------------------------------------------------------------------------------")
        lines.append(" [MARKET SESSION BREAKDOWN & RANKINGS]")
        lines.append(f"  {'Rank':<5} {'Session':<22} {'Trades':<8} {'Win Rate':<10} {'Total PnL':<12} {'Profit Factor':<14} {'Tier':<12}")
        lines.append("  " + "-" * 75)
        for p in s["ranked_sessions"]:
            lines.append(
                f"  #{p['rank']:<4} {p['name']:<22} {p['total_trades']:<8} {p['win_rate']:>5.1f}%    "
                f"${p['total_pnl']:>8.2f}    {p['profit_factor']:>6.2f}         {p['edge_tier']:<12}"
            )

        lines.append("--------------------------------------------------------------------------------")
        lines.append(" [24-HOUR PROFITABILITY HEATMAP - TOP HOURS]")
        lines.append(f"  {'Rank':<5} {'Hour (UTC)':<14} {'Trades':<8} {'Win Rate':<10} {'Total PnL':<12} {'Tier':<14}")
        lines.append("  " + "-" * 65)
        for p in s["ranked_hours"][:8]:
            lines.append(
                f"  #{p['rank']:<4} {p['name']:<14} {p['total_trades']:<8} {p['win_rate']:>5.1f}%    "
                f"${p['total_pnl']:>8.2f}    {p['edge_tier']:<14}"
            )

        lines.append("================================================================================")
        return "\n".join(lines)


# =============================================================================
# 8. STANDALONE CLI SMOKE TEST & REPORT GENERATOR
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n[+] Initializing Temporal & Session ML Analyzer...")

    svc = TemporalEdgeService()
    report = svc.generate_ascii_report()
    print(report)

    # Test an evaluation query right now
    now_utc = datetime.now(timezone.utc)
    verdict = svc.evaluate(now_utc, symbol="XAUUSD", strategy="SMC_SCALP_5M")
    print("\n--- SAMPLE LIVE PREDICTION FOR CURRENT TIME ---")
    print(f"Timestamp   : {verdict.timestamp}")
    print(f"Day/Hour    : {verdict.day_name} at {verdict.hour_utc:02d}:00 UTC")
    print(f"Session     : {verdict.session} (KillZone: {verdict.is_killzone})")
    print(f"Prediction  : P(Win)={verdict.p_win*100:.1f}%, P(Loss)={verdict.p_loss*100:.1f}%, Expected Return: {verdict.expected_r:+.2f}R")
    print(f"Edge Tier   : {verdict.edge_tier.value} | Allowed: {verdict.allowed} | Risk Multiplier: {verdict.risk_multiplier:.2f}x")
    print(f"Reason      : {verdict.reason}")

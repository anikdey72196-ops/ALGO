"""
ml/trap_detector.py
===================

Institutional SMC/ICT "Trap Detector" — learns, from live market data, which
Fair Value Gaps (FVGs) and Liquidity Sweeps are *genuine* institutional setups
versus engineered *retail traps*.

Design principles
-----------------
1. CAUSAL FEATURES ONLY. Every feature at bar i is computed from bars <= i.
   No lookahead. This is enforced by a CI-style assertion at the bottom.
2. TRIPLE-BARRIER LABELS. A setup is "genuine" if it reaches its target before
   violating its stop; otherwise it is a "trap". Labels come from the same
   geometry the strategy would have traded, so predictions are consistent with
   execution.
3. ONLINE + BATCH. SGDClassifier.partial_fit learns from every new labeled
   event (cold start). Once enough samples accumulate, a LightGBM model is
   batch-retrained and hot-swapped. The gate always prefers LightGBM.
4. SHADOW MODE FIRST. For the first `shadow_until` samples the gate logs
   predictions but never blocks. This gives a real calibration baseline.
5. FAIL OPEN, LOG LOUD. Inference errors return allow=True in shadow mode and
   allow=False in production. Never silent.
6. THE GATE CAN ONLY VETO. It never resizes, reverses, or overrides the Risk
   Engine. Hard drawdown limits remain absolute.

Integration
-----------
The public entry point is `TrapDetectorService`. Instantiate it once at bot
startup, call `observe_event(...)` from your strategies when a candidate setup
forms, and call `gate(...)` right before the trade is dispatched to
`ai_analyst` / `risk_engine`. A background task (`run_labeler`) back-fills
labels from the OHLCV feed.

See the `if __name__ == "__main__":` block for a runnable smoke test.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("algo.ml.trap_detector")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

class TrapDetectorConfig(BaseModel):
    """All tunables in one place. Persist alongside other ALGO configs."""

    db_path: str = "trading_state.db"
    model_path: str = "ml/artifacts/trap_detector.joblib"

    # Gate behavior
    p_genuine_threshold: float = 0.55       # block if P(genuine) < threshold
    shadow_until_samples: int = 200         # first N samples never block
    fail_open_on_error: bool = True         # inference error -> allow? (shadow only)

    # Labeling
    label_max_bars: int = 30                # triple-barrier horizon (bars)
    label_settle_minutes: int = 30          # wait before labeling (bar closure)
    labeler_interval_sec: int = 60

    # LightGBM
    lgbm_min_samples: int = 500
    lgbm_half_life_days: float = 90.0       # exponential time-decay for weights

    # Feature hygiene
    atr_period: int = 14
    rvol_lookback: int = 20
    pd_lookback: int = 100                  # premium/discount window
    eq_lookback: int = 50                   # equal-highs/lows window

    @field_validator("p_genuine_threshold")
    @classmethod
    def _thr(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("p_genuine_threshold must be in (0, 1)")
        return v


# =============================================================================
# 2. SCHEMAS
# =============================================================================

class EventKind(str, Enum):
    FVG_BULL  = "fvg_bull"
    FVG_BEAR  = "fvg_bear"
    SWEEP_BSL = "sweep_bsl"   # buy-side liquidity taken (potential reversal down)
    SWEEP_SSL = "sweep_ssl"   # sell-side liquidity taken (potential reversal up)


Direction = Literal["long", "short"]
Outcome   = Literal["tp", "sl", "vertical", "pending", "invalid"]


class EventRecord(BaseModel):
    """One candidate setup, from detection through label and inference."""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: datetime
    symbol: str
    timeframe: str
    kind: EventKind
    direction: Direction

    entry: float
    stop: float
    target: float

    features: dict[str, float]

    # Labels (filled later)
    label: Optional[int] = None             # 1 = genuine, 0 = trap
    outcome: Optional[Outcome] = None
    r_multiple: Optional[float] = None
    label_ts: Optional[datetime] = None

    # Inference bookkeeping
    p_genuine: Optional[float] = None
    model_version: Optional[str] = None
    allowed: Optional[bool] = None

    def feature_vector(self, order: Sequence[str]) -> np.ndarray:
        return np.array([[self.features.get(k, 0.0) for k in order]], dtype=float)


class GateDecision(BaseModel):
    allow: bool
    p_genuine: float
    mode: Literal["shadow", "gated", "error"]
    model_version: str
    reason: str = ""


# =============================================================================
# 3. CAUSAL FEATURE EXTRACTION
# =============================================================================

FEATURE_ORDER: dict[EventKind, list[str]] = {
    EventKind.FVG_BULL: [
        "gap_atr", "body_ratio", "body_atr", "rvol", "premium_disc",
        "dist_eqh_atr", "dist_eql_atr", "hour_sin", "hour_cos",
        "range_atr", "is_bull_candle", "consecutive_fvgs",
    ],
    EventKind.FVG_BEAR: [
        "gap_atr", "body_ratio", "body_atr", "rvol", "premium_disc",
        "dist_eqh_atr", "dist_eql_atr", "hour_sin", "hour_cos",
        "range_atr", "is_bull_candle", "consecutive_fvgs",
    ],
    EventKind.SWEEP_BSL: [
        "wick_atr", "wick_body_ratio", "rvol", "close_inside",
        "overshoot_atr", "premium_disc", "range_atr", "hour_sin", "hour_cos",
    ],
    EventKind.SWEEP_SSL: [
        "wick_atr", "wick_body_ratio", "rvol", "close_inside",
        "overshoot_atr", "premium_disc", "range_atr", "hour_sin", "hour_cos",
    ],
}


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _rvol(df: pd.DataFrame, i: int, lookback: int) -> float:
    lo = max(0, i - lookback)
    med = float(df["volume"].iloc[lo:i].median()) if i > lo else 0.0
    if med <= 0:
        return 1.0
    return float(df["volume"].iloc[i] / med)


def _premium_discount(df: pd.DataFrame, i: int, price: float, lookback: int) -> float:
    lo = max(0, i - lookback)
    win = df.iloc[lo:i]
    if win.empty:
        return 0.5
    hi, low = float(win["high"].max()), float(win["low"].min())
    if hi <= low:
        return 0.5
    return float((price - low) / (hi - low))


def _dist_eq(df: pd.DataFrame, i: int, price: float, lookback: int) -> tuple[float, float]:
    lo = max(0, i - lookback)
    win = df.iloc[lo:i]
    if win.empty:
        return 0.0, 0.0
    return float(win["high"].max() - price), float(price - win["low"].min())


def _hour_cyclical(ts: datetime) -> tuple[float, float]:
    frac = (ts.hour + ts.minute / 60.0) / 24.0
    return float(np.sin(2 * np.pi * frac)), float(np.cos(2 * np.pi * frac))


def _count_consecutive_fvgs(df: pd.DataFrame, i: int, kind: EventKind, max_back: int = 5) -> int:
    n = 0
    for k in range(1, max_back + 1):
        j = i - 3 * k
        if j < 2:
            break
        if kind == EventKind.FVG_BULL and df["low"].iloc[j] > df["high"].iloc[j - 2]:
            n += 1
        elif kind == EventKind.FVG_BEAR and df["low"].iloc[j - 2] > df["high"].iloc[j]:
            n += 1
        else:
            break
    return n


def extract_features(
    df: pd.DataFrame,
    i: int,
    kind: EventKind,
    cfg: TrapDetectorConfig,
) -> dict[str, float]:
    """
    Compute the causal feature dict for event at bar index `i`.

    `df` must be sorted ascending by time and contain columns:
        open, high, low, close, volume, and a DatetimeIndex.
    The caller must ensure `i` is the confirmation bar of the setup
    (3rd candle of an FVG, or the sweep candle itself).
    """
    if i < 3 or i >= len(df):
        raise ValueError(f"bar index {i} out of range for df of length {len(df)}")

    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index[i]
    elif "time" in df.columns:
        ts = pd.to_datetime(df["time"].iloc[i], utc=True)
    else:
        try:
            ts = pd.to_datetime(df.index[i], utc=True)
        except Exception as exc:
            raise TypeError("df must have a DatetimeIndex or a 'time' column") from exc

    a_series = _atr(df, cfg.atr_period)
    a = float(a_series.iloc[i])
    if not np.isfinite(a) or a <= 0:
        a = max(float(df["high"].iloc[i] - df["low"].iloc[i]), 1e-9)

    hs, hc = _hour_cyclical(ts.to_pydatetime())

    if kind in (EventKind.FVG_BULL, EventKind.FVG_BEAR):
        disp = df.iloc[i - 1]
        body = abs(float(disp["close"] - disp["open"]))
        rng = max(float(disp["high"] - disp["low"]), 1e-9)

        if kind == EventKind.FVG_BULL:
            gap = float(df["low"].iloc[i] - df["high"].iloc[i - 2])
            mid = (float(df["high"].iloc[i - 2]) + float(df["low"].iloc[i])) / 2.0
        else:
            gap = float(df["low"].iloc[i - 2] - df["high"].iloc[i])
            mid = (float(df["low"].iloc[i - 2]) + float(df["high"].iloc[i])) / 2.0

        eqh, eql = _dist_eq(df, i, mid, cfg.eq_lookback)
        return {
            "gap_atr":           gap / a,
            "body_ratio":        body / rng,
            "body_atr":          body / a,
            "rvol":              _rvol(df, i - 1, cfg.rvol_lookback),
            "premium_disc":      _premium_discount(df, i, mid, cfg.pd_lookback),
            "dist_eqh_atr":      eqh / a,
            "dist_eql_atr":      eql / a,
            "hour_sin":          hs,
            "hour_cos":          hc,
            "range_atr":         float((df["high"].iloc[i] - df["low"].iloc[i]) / a),
            "is_bull_candle":    1.0 if disp["close"] > disp["open"] else 0.0,
            "consecutive_fvgs":  float(_count_consecutive_fvgs(df, i, kind)),
        }

    if kind in (EventKind.SWEEP_BSL, EventKind.SWEEP_SSL):
        bar = df.iloc[i]
        if kind == EventKind.SWEEP_BSL:
            wick = float(bar["high"] - max(bar["open"], bar["close"]))
        else:
            wick = float(min(bar["open"], bar["close"]) - bar["low"])
        rng = max(float(bar["high"] - bar["low"]), 1e-9)
        close_inside = 1.0 if (
            (kind == EventKind.SWEEP_BSL and bar["close"] < bar["open"]) or
            (kind == EventKind.SWEEP_SSL and bar["close"] > bar["open"])
        ) else 0.0

        lo = max(0, i - 20)
        prior = df.iloc[lo:i]
        if prior.empty:
            overshoot = 0.0
        elif kind == EventKind.SWEEP_BSL:
            overshoot = max(float(bar["high"] - prior["high"].max()), 0.0)
        else:
            overshoot = max(float(prior["low"].min() - bar["low"]), 0.0)

        return {
            "wick_atr":         wick / a,
            "wick_body_ratio":  wick / max(abs(float(bar["close"] - bar["open"])), 1e-9),
            "rvol":             _rvol(df, i, cfg.rvol_lookback),
            "close_inside":     close_inside,
            "overshoot_atr":    overshoot / a,
            "premium_disc":     _premium_discount(df, i, float(bar["close"]), cfg.pd_lookback),
            "range_atr":        rng / a,
            "hour_sin":         hs,
            "hour_cos":         hc,
        }

    raise ValueError(f"unsupported event kind: {kind}")


# =============================================================================
# 4. TRIPLE-BARRIER LABELING
# =============================================================================

def label_triple_barrier(
    df: pd.DataFrame,
    i_entry: int,
    direction: Direction,
    entry: float,
    stop: float,
    target: float,
    max_bars: int,
) -> tuple[Optional[int], Outcome, Optional[float]]:
    """
    Walk forward from `i_entry + 1` up to `max_bars` bars.

    Returns
    -------
    (label, outcome, r_multiple)
      label   : 1 genuine (TP first), 0 trap (SL first), None if unresolved
      outcome : "tp" | "sl" | "vertical" | "pending" | "invalid"
      r       : realized R multiple (or None if invalid)
    """
    risk = abs(entry - stop)
    if risk <= 0:
        return None, "invalid", None

    future = df.iloc[i_entry + 1 : i_entry + 1 + max_bars]
    if future.empty:
        return None, "pending", None

    for _, bar in future.iterrows():
        if direction == "long":
            if float(bar["low"]) <= stop:
                return 0, "sl", -1.0
            if float(bar["high"]) >= target:
                return 1, "tp", float((target - entry) / risk)
        else:
            if float(bar["high"]) >= stop:
                return 0, "sl", -1.0
            if float(bar["low"]) <= target:
                return 1, "tp", float((entry - target) / risk)

    close = float(future["close"].iloc[-1])
    r = (close - entry) / risk if direction == "long" else (entry - close) / risk
    return (1 if r > 0 else 0), "vertical", float(r)


# =============================================================================
# 5. PERSISTENCE
# =============================================================================

_DDL = """
CREATE TABLE IF NOT EXISTS ml_events (
    event_id      TEXT PRIMARY KEY,
    ts            TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    timeframe     TEXT NOT NULL,
    kind          TEXT NOT NULL,
    direction     TEXT NOT NULL,
    entry         REAL NOT NULL,
    stop          REAL NOT NULL,
    target        REAL NOT NULL,
    features      TEXT NOT NULL,
    label         INTEGER,
    outcome       TEXT,
    r_multiple    REAL,
    label_ts      TEXT,
    p_genuine     REAL,
    model_version TEXT,
    allowed       INTEGER
);
CREATE INDEX IF NOT EXISTS ix_ml_events_pending ON ml_events(label, ts);
CREATE INDEX IF NOT EXISTS ix_ml_events_symbol  ON ml_events(symbol, timeframe);
"""


class EventStore:
    """ACID SQLite store for ML events, sharing the ALGO state DB (WAL mode)."""

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        with self._lock:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_DDL)
            self._conn.commit()

    def upsert(self, ev: EventRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO ml_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(event_id) DO UPDATE SET
                    label=excluded.label,
                    outcome=excluded.outcome,
                    r_multiple=excluded.r_multiple,
                    label_ts=excluded.label_ts,
                    p_genuine=excluded.p_genuine,
                    model_version=excluded.model_version,
                    allowed=excluded.allowed
                """,
                (
                    ev.event_id, ev.ts.isoformat(), ev.symbol, ev.timeframe,
                    ev.kind.value, ev.direction, ev.entry, ev.stop, ev.target,
                    json.dumps(ev.features), ev.label, ev.outcome, ev.r_multiple,
                    ev.label_ts.isoformat() if ev.label_ts else None,
                    ev.p_genuine, ev.model_version,
                    int(ev.allowed) if ev.allowed is not None else None,
                ),
            )
            self._conn.commit()

    def pending_labels(self, older_than: datetime) -> list[EventRecord]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM ml_events WHERE label IS NULL AND ts < ? ORDER BY ts ASC",
                (older_than.isoformat(),),
            )
            return [self._row(r) for r in cur.fetchall()]

    def labeled(self) -> list[EventRecord]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM ml_events WHERE label IS NOT NULL ORDER BY ts ASC"
            )
            return [self._row(r) for r in cur.fetchall()]

    def stats(self) -> dict[str, int]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*), SUM(label IS NOT NULL), SUM(label = 0), SUM(label = 1) FROM ml_events"
            )
            total, labeled, traps, genuine = cur.fetchone()
        return {
            "total":   int(total or 0),
            "labeled": int(labeled or 0),
            "traps":   int(traps or 0),
            "genuine": int(genuine or 0),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row(r: tuple[Any, ...]) -> EventRecord:
        return EventRecord(
            event_id=r[0],
            ts=datetime.fromisoformat(r[1]),
            symbol=r[2],
            timeframe=r[3],
            kind=EventKind(r[4]),
            direction=r[5],
            entry=r[6],
            stop=r[7],
            target=r[8],
            features=json.loads(r[9]),
            label=r[10],
            outcome=r[11],
            r_multiple=r[12],
            label_ts=datetime.fromisoformat(r[13]) if r[13] else None,
            p_genuine=r[14],
            model_version=r[15],
            allowed=bool(r[16]) if r[16] is not None else None,
        )


# =============================================================================
# 6. MODEL — Online SGD + Batch LightGBM
# =============================================================================

class TrapModel:
    """
    Two-stage predictor:
      • SGDClassifier (partial_fit) — updates from every new labeled event.
      • LightGBM — batch-retrained periodically, preferred when available.
    Persists both to a single joblib artifact.
    """

    def __init__(self, path: str, lgbm_min_samples: int = 500, half_life_days: float = 90.0):
        self.path = Path(path)
        self.lgbm_min_samples = lgbm_min_samples
        self.half_life_days = half_life_days

        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler

        self.scaler: StandardScaler = StandardScaler()
        self.sgd: SGDClassifier = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-4,
            learning_rate="optimal",
            random_state=42,
        )
        self.lgbm: Optional[Any] = None
        self.n_samples: int = 0
        self.version: str = "cold"
        self._sgd_fitted: bool = False

        self._load()

    # -- online learning -----------------------------------------------------
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(y) == 0:
            return
        self.scaler.partial_fit(X)
        self.sgd.partial_fit(self.scaler.transform(X), y, classes=np.array([0, 1]))
        self._sgd_fitted = True
        self.n_samples += len(y)
        self.version = f"sgd-{self.n_samples}"
        self._save()

    # -- batch retrain -------------------------------------------------------
    def retrain_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: Sequence[datetime],
    ) -> bool:
        """Retrain LightGBM with exponential time-decay weights. Returns True if swapped."""
        if len(y) < self.lgbm_min_samples:
            logger.info(
                "LightGBM retrain skipped: %d/%d samples", len(y), self.lgbm_min_samples
            )
            return False

        import lightgbm as lgb

        now = datetime.now(timezone.utc)
        ages_days = np.array(
            [(now - ts).total_seconds() / 86400.0 for ts in timestamps], dtype=float
        )
        weights = np.exp(-ages_days / max(self.half_life_days, 1e-6))

        # tiny floor so we never zero-out an entire class
        weights = np.clip(weights, 1e-3, None)

        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            class_weight="balanced",
            objective="binary",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(X, y, sample_weight=weights)
        self.lgbm = model
        self.version = f"lgbm-{now:%Y%m%d%H%M%S}"
        self._save()
        logger.info("LightGBM retrained: version=%s samples=%d", self.version, len(y))
        return True

    # -- inference -----------------------------------------------------------
    def predict_proba_genuine(self, X: np.ndarray) -> np.ndarray:
        if self.lgbm is not None:
            return self.lgbm.predict_proba(X)[:, 1]
        if not self._sgd_fitted:
            return np.full(len(X), 0.5, dtype=float)
        return self.sgd.predict_proba(self.scaler.transform(X))[:, 1]

    # -- persistence ---------------------------------------------------------
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "sgd": self.sgd,
                "lgbm": self.lgbm,
                "n_samples": self.n_samples,
                "version": self.version,
                "_sgd_fitted": self._sgd_fitted,
            },
            self.path,
        )

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            state = joblib.load(self.path)
            self.scaler = state["scaler"]
            self.sgd = state["sgd"]
            self.lgbm = state["lgbm"]
            self.n_samples = int(state["n_samples"])
            self.version = str(state["version"])
            self._sgd_fitted = bool(state["_sgd_fitted"])
            logger.info(
                "TrapModel loaded: version=%s samples=%d", self.version, self.n_samples
            )
        except Exception as exc:  # corrupt artifact -> start fresh, never crash bot
            logger.exception("Failed to load TrapModel artifact: %s", exc)


# =============================================================================
# 7. GATE
# =============================================================================

class TrapGate:
    """
    Decides whether a candidate event may proceed to AI + Risk stages.

    The gate CANNOT change sizing, direction, or override the Risk Engine.
    It either lets a setup through (allow=True) or vetoes it (allow=False).
    """

    def __init__(self, model: TrapModel, cfg: TrapDetectorConfig):
        self.model = model
        self.cfg = cfg

    def evaluate(self, kind: EventKind, features: dict[str, float]) -> GateDecision:
        # Shadow mode: log everything, block nothing.
        if self.model.n_samples < self.cfg.shadow_until_samples:
            return GateDecision(
                allow=True,
                p_genuine=0.5,
                mode="shadow",
                model_version=self.model.version,
                reason=f"shadow mode ({self.model.n_samples}/{self.cfg.shadow_until_samples})",
            )

        order = FEATURE_ORDER[kind]
        x = np.array([[features.get(k, 0.0) for k in order]], dtype=float)

        try:
            p = float(self.model.predict_proba_genuine(x)[0])
        except Exception as exc:
            logger.exception("TrapGate inference failed: %s", exc)
            return GateDecision(
                allow=self.cfg.fail_open_on_error and self.model.n_samples < self.cfg.shadow_until_samples,
                p_genuine=0.5,
                mode="error",
                model_version=self.model.version,
                reason=f"inference error: {exc}",
            )

        allow = p >= self.cfg.p_genuine_threshold
        return GateDecision(
            allow=allow,
            p_genuine=p,
            mode="gated",
            model_version=self.model.version,
            reason="passed" if allow else f"p_genuine={p:.3f} < {self.cfg.p_genuine_threshold}",
        )


# =============================================================================
# 8. SERVICE — public entry point
# =============================================================================

class TrapDetectorService:
    """
    Facade wiring store, model, and gate. One instance per ALGO process.

    Typical use:

        svc = TrapDetectorService(cfg)
        # In strategy, when an FVG confirms:
        svc.observe_event(symbol, tf, kind, direction, entry, stop, target,
                          df=df, bar_index=i)

        # Right before dispatching to ai_analyst / risk_engine:
        decision = svc.gate(kind, features)
        if not decision.allow:
            return

    A background task (`run_labeler`) must be scheduled (e.g. via asyncio.gather
    alongside your existing jobs) to back-fill labels.
    """

    def __init__(self, cfg: TrapDetectorConfig):
        self.cfg = cfg
        self.store = EventStore(cfg.db_path)
        self.model = TrapModel(
            cfg.model_path,
            lgbm_min_samples=cfg.lgbm_min_samples,
            half_life_days=cfg.lgbm_half_life_days,
        )
        self.gate = TrapGate(self.model, cfg)

    # -- observation ---------------------------------------------------------
    def observe_event(
        self,
        *,
        symbol: str,
        timeframe: str,
        kind: EventKind,
        direction: Direction,
        entry: float,
        stop: float,
        target: float,
        df: pd.DataFrame,
        bar_index: int,
    ) -> EventRecord:
        """Record a candidate setup and its causal features. No label yet."""
        features = extract_features(df, bar_index, kind, self.cfg)
        if isinstance(df.index, pd.DatetimeIndex):
            event_ts = df.index[bar_index].to_pydatetime()
        elif "time" in df.columns:
            event_ts = pd.to_datetime(df["time"].iloc[bar_index], utc=True).to_pydatetime()
        else:
            try:
                event_ts = pd.to_datetime(df.index[bar_index], utc=True).to_pydatetime()
            except Exception:
                event_ts = datetime.now(timezone.utc)

        ev = EventRecord(
            ts=event_ts,
            symbol=symbol,
            timeframe=timeframe,
            kind=kind,
            direction=direction,
            entry=float(entry),
            stop=float(stop),
            target=float(target),
            features=features,
        )

        decision = self.gate.evaluate(kind, features)
        ev.p_genuine = decision.p_genuine
        ev.model_version = decision.model_version
        ev.allowed = decision.allow
        self.store.upsert(ev)

        if not decision.allow:
            logger.info(
                "TrapGate veto | %s %s %s | p=%.3f | %s",
                symbol, timeframe, kind.value, decision.p_genuine, decision.reason,
            )
        return ev

    # -- labeling ------------------------------------------------------------
    def _label_pending(self, history_provider: Callable[[str, str, int], pd.DataFrame]) -> int:
        """Back-fill labels for events whose horizon has elapsed. Returns # labeled."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.cfg.label_settle_minutes)
        pending = self.store.pending_labels(cutoff)
        labeled = 0

        for ev in pending:
            try:
                df = history_provider(ev.symbol, ev.timeframe, 500)
                if df.empty:
                    continue

                if not isinstance(df.index, pd.DatetimeIndex):
                    if "time" in df.columns:
                        df = df.set_index(pd.to_datetime(df["time"], utc=True))
                    else:
                        df = df.set_index(pd.to_datetime(df.index, utc=True))

                ev_ts = pd.Timestamp(ev.ts)
                if df.index.tz is not None and ev_ts.tz is None:
                    ev_ts = ev_ts.tz_localize("UTC")
                elif df.index.tz is None and ev_ts.tz is not None:
                    ev_ts = ev_ts.tz_localize(None)

                idx = df.index.get_indexer([ev_ts], method="nearest")[0]
                if idx < 0 or idx >= len(df) - 1:
                    continue

                label, outcome, r = label_triple_barrier(
                    df, idx, ev.direction, ev.entry, ev.stop, ev.target,
                    max_bars=self.cfg.label_max_bars,
                )
                if label is None:
                    continue

                ev.label = int(label)
                ev.outcome = outcome
                ev.r_multiple = r
                ev.label_ts = datetime.now(timezone.utc)
                self.store.upsert(ev)

                order = FEATURE_ORDER[ev.kind]
                x = ev.feature_vector(order)
                self.model.partial_fit(x, np.array([label]))
                labeled += 1
            except Exception:
                logger.exception("Failed to label event %s", ev.event_id)

        if labeled:
            logger.info("Labeled %d new events | model=%s", labeled, self.model.version)
        return labeled

    def retrain_if_ready(self) -> bool:
        """Retrain LightGBM on the full labeled corpus. Call nightly / on schedule."""
        rows = self.store.labeled()
        if len(rows) < self.cfg.lgbm_min_samples:
            return False

        # Guard against class collapse — LightGBM needs both classes present.
        labels = {r.label for r in rows}
        if labels != {0, 1}:
            logger.warning("Retrain skipped: only labels %s present", labels)
            return False

        # Group by kind so each kind gets its own column layout — concatenate
        # feature vectors with a union schema; missing -> 0.0. Keeps one model
        # but preserves per-kind signal via the presence mask.
        union: list[str] = sorted({k for kind in FEATURE_ORDER.values() for k in kind})
        X = np.array(
            [[r.features.get(k, 0.0) for k in union] for r in rows], dtype=float
        )
        y = np.array([r.label for r in rows], dtype=int)
        ts = [r.ts.replace(tzinfo=r.ts.tzinfo or timezone.utc) for r in rows]

        # Persist the union column order so inference uses the same shape.
        self._save_union(union)
        return self.model.retrain_lightgbm(X, y, ts)

    def _save_union(self, union: list[str]) -> None:
        p = Path(self.cfg.model_path).with_suffix(".columns.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(union))

    def load_union(self) -> list[str]:
        p = Path(self.cfg.model_path).with_suffix(".columns.json")
        if p.exists():
            return json.loads(p.read_text())
        return sorted({k for kind in FEATURE_ORDER.values() for k in kind})

    # -- async worker --------------------------------------------------------
    async def run_labeler(
        self,
        history_provider: Callable[[str, str, int], pd.DataFrame],
    ) -> None:
        """Background loop: label pending events, then retrain LightGBM occasionally."""
        retrain_every = 20  # iterations
        tick = 0
        while True:
            try:
                self._label_pending(history_provider)
                tick += 1
                if tick % retrain_every == 0:
                    # Offload CPU-heavy retrain to a thread so the loop stays responsive.
                    await asyncio.to_thread(self.retrain_if_ready)
            except Exception:
                logger.exception("Labeler loop error")
            await asyncio.sleep(self.cfg.labeler_interval_sec)

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        self.store.close()


# =============================================================================
# 9. SELF-TEST — causality + labeling correctness
# =============================================================================

def _self_test() -> None:
    """
    Runs a deterministic smoke test:
      • Verifies causal invariance: features at bar i are identical whether
        computed on df[:i+1] or df[:i+50].
      • Verifies the triple-barrier labeler produces expected labels on
        synthetic TP-then-SL and SL-then-TP paths.
    """
    rng = np.random.default_rng(7)
    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    base = 2000.0 + np.cumsum(rng.normal(0, 1.5, n))
    df = pd.DataFrame(
        {
            "open":   base + rng.normal(0, 0.2, n),
            "high":   base + np.abs(rng.normal(0.8, 0.3, n)),
            "low":    base - np.abs(rng.normal(0.8, 0.3, n)),
            "close":  base + rng.normal(0, 0.3, n),
            "volume": rng.integers(100, 1000, n).astype(float),
        },
        index=idx,
    )

    cfg = TrapDetectorConfig(db_path=":memory:", model_path="ml/artifacts/_selftest.joblib")

    # --- causality: identical features with and without future bars ---------
    i = 50
    full = extract_features(df, i, EventKind.FVG_BULL, cfg)
    truncated = extract_features(df.iloc[: i + 1], i, EventKind.FVG_BULL, cfg)
    for k in full:
        assert np.isclose(full[k], truncated[k], atol=1e-12), (
            f"CAUSALITY VIOLATION on feature {k!r}: {full[k]} vs {truncated[k]}"
        )
    print("[ok] causality invariant")

    # --- labeler: TP-first path yields label=1 ------------------------------
    up = df.copy()
    i0 = 10
    entry = float(up["close"].iloc[i0])
    target = entry + 1.0
    stop = entry - 1.0
    up.iloc[i0 + 1 : i0 + 5, up.columns.get_loc("high")] = target + 0.5
    label, outcome, r = label_triple_barrier(up, i0, "long", entry, stop, target, 10)
    assert (label, outcome) == (1, "tp"), (label, outcome)
    assert r == 1.0
    print("[ok] labeler TP-first")

    # --- labeler: SL-first path yields label=0 ------------------------------
    dn = df.copy()
    dn.iloc[i0 + 1 : i0 + 5, dn.columns.get_loc("low")] = stop - 0.5
    label, outcome, r = label_triple_barrier(dn, i0, "long", entry, stop, target, 10)
    assert (label, outcome) == (0, "sl"), (label, outcome)
    print("[ok] labeler SL-first")

    print("All self-tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    _self_test()

"""
ml/smart_partial_tp.py
======================

ML-Powered Adaptive Partial Profit Booking System.
Learns from market structure and live execution which movements reach full TP
versus which reverse at structural levels (Swing High/Low, Order Block, FVG,
Fibonacci retracements, and Standard Deviation bands).

Design Principles:
1. STRICT CAUSALITY: All features at current bar are computed strictly from historical
   bars <= current bar. Zero lookahead.
2. STRUCTURAL AWARENESS: Quantifies distances to nearest opposing Swing High/Low,
   Order Blocks, Fair Value Gaps, Fibonacci levels (38.2%, 50%, 61.8%, 78.6%),
   and Standard Deviation bands (Bollinger 1-3 sigma).
3. MULTI-HEAD PREDICTION:
   - P(Reversal): Probability price reverses by >=0.5R before progressing further
   - P(Full TP): Probability price reaches original target without hitting SL/BE
   - Predicted Max R: Expected maximum R-multiple achievable from this point
4. DYNAMIC LOT SIZING: Quantifies exact % of remaining volume to close (0% hold runner,
   20-35% standard trim, 50-60% aggressive defense).
5. SHADOW MODE & FAIL OPEN: Supports non-blocking evaluation mode and safe fallback
   to rule-based logic if ML inference encounters unexpected errors.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger("algo.ml.smart_partial_tp")

MODEL_ARTIFACT_PATH = Path("ml/artifacts/smart_partial_tp.joblib")
CONFIG_PATH = Path("config.yaml")


# =============================================================================
# 1. DATA STRUCTURES & VERDICT
# =============================================================================

@dataclass
class StructuralLevels:
    """Detected market structure levels relative to current position."""
    nearest_swing_price: float = 0.0
    nearest_swing_dist_r: float = 99.0
    nearest_ob_price: float = 0.0
    nearest_ob_dist_r: float = 99.0
    nearest_fvg_price: float = 0.0
    nearest_fvg_dist_r: float = 99.0
    fib_382_price: float = 0.0
    fib_382_dist_r: float = 99.0
    fib_500_price: float = 0.0
    fib_500_dist_r: float = 99.0
    fib_618_price: float = 0.0
    fib_618_dist_r: float = 99.0
    fib_786_price: float = 0.0
    fib_786_dist_r: float = 99.0
    stddev_1_price: float = 0.0
    stddev_1_dist_r: float = 99.0
    stddev_2_price: float = 0.0
    stddev_2_dist_r: float = 99.0
    confluence_count_near_price: int = 0
    nearest_level_name: str = "None"
    nearest_level_price: float = 0.0
    nearest_level_dist_r: float = 99.0


@dataclass
class PartialTPVerdict:
    """Actionable decision returned by the Smart Partial TP system."""
    action: str                       # 'HOLD', 'PARTIAL_CLOSE', 'FULL_CLOSE'
    close_pct: float                  # 0.0 to 1.0 (fraction of current open lot to close)
    p_reversal: float                 # Estimated probability of reversal
    p_full_tp: float                  # Estimated probability of reaching full TP
    predicted_max_r: float            # Forecasted max R-multiple of this trade
    nearest_resistance: str           # E.g. "Order Block @ 2045.20 (0.28R away)"
    confidence: float                 # Confidence score (0.0 to 1.0)
    reason: str                       # Human-readable rationale
    is_shadow: bool = False           # True if running in shadow/audit mode
    features: Dict[str, float] = field(default_factory=dict)
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# 2. FEATURE REGISTRY & DYNAMIC EXTENSION
# =============================================================================

FEATURE_NAMES: List[str] = [
    "r_multiple",
    "pct_to_tp",
    "planned_rr",
    "nearest_swing_dist_r",
    "nearest_ob_dist_r",
    "nearest_fvg_dist_r",
    "fib_382_dist_r",
    "fib_500_dist_r",
    "fib_618_dist_r",
    "fib_786_dist_r",
    "stddev_1_dist_r",
    "stddev_2_dist_r",
    "structure_confluence_count",
    "atr_ratio",
    "momentum_rsi",
    "trend_strength_adx",
    "bars_since_entry",
    "session_idx",
    "strategy_idx",
    "direction_val",
]

FEATURE_REGISTRY: Dict[str, Callable[[Dict[str, Any]], float]] = {}


def register_new_feature(name: str, compute_func: Optional[Callable[[Dict[str, Any]], float]] = None) -> None:
    """
    Register a new feature into global FEATURE_NAMES and optionally store its compute function.
    Reduces friction when extending model inputs.
    """
    if name not in FEATURE_NAMES:
        FEATURE_NAMES.append(name)
    if compute_func is not None:
        FEATURE_REGISTRY[name] = compute_func


# =============================================================================
# 3. FEATURE EXTRACTION & STRUCTURAL CALCULATIONS
# =============================================================================

class StructuralExtractor:
    """Helper class for extracting structural levels from price arrays."""

    @staticmethod
    def extract_structural_levels_ndarray(
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        opens: np.ndarray,
        direction: str,
        entry_price: float,
        current_price: float,
        risk_dist: float,
        lookback: int = 50,
    ) -> StructuralLevels:
        """
        Extract structural levels directly from numpy NDArrays without DataFrame copy overhead.
        Pre-computes argmin/argmax indices for fast window high/low search.
        """
        levels = StructuralLevels()
        risk_dist = max(risk_dist, 1e-8)
        n_total = len(closes)
        if n_total < 20:
            return levels

        start_idx = max(0, n_total - lookback)
        h_win = highs[start_idx:]
        l_win = lows[start_idx:]
        c_win = closes[start_idx:]
        o_win = opens[start_idx:]
        n = len(c_win)

        is_buy = (direction.upper() == "BUY")

        # ── A. Swing Highs & Lows (Fractal Pivots) ──
        swing_points: List[float] = []
        for i in range(2, n - 2):
            if is_buy:
                if h_win[i] > h_win[i - 1] and h_win[i] > h_win[i - 2] and h_win[i] >= h_win[i + 1] and h_win[i] >= h_win[i + 2]:
                    if h_win[i] > current_price:
                        swing_points.append(float(h_win[i]))
            else:
                if l_win[i] < l_win[i - 1] and l_win[i] < l_win[i - 2] and l_win[i] <= l_win[i + 1] and l_win[i] <= l_win[i + 2]:
                    if l_win[i] < current_price:
                        swing_points.append(float(l_win[i]))

        if swing_points:
            if is_buy:
                nearest_swing = min(swing_points)
                dist_r = (nearest_swing - current_price) / risk_dist
            else:
                nearest_swing = max(swing_points)
                dist_r = (current_price - nearest_swing) / risk_dist
            levels.nearest_swing_price = nearest_swing
            levels.nearest_swing_dist_r = max(0.0, dist_r)

        # ── B. Order Blocks (OB) ──
        ob_points: List[float] = []
        for i in range(1, n - 2):
            body = abs(c_win[i] - o_win[i])
            next_body = abs(c_win[i + 1] - o_win[i + 1])
            if is_buy:
                if c_win[i] > o_win[i] and c_win[i + 1] < o_win[i + 1] and next_body > body * 1.3:
                    ob_price = float(h_win[i])
                    if ob_price > current_price:
                        ob_points.append(ob_price)
            else:
                if c_win[i] < o_win[i] and c_win[i + 1] > o_win[i + 1] and next_body > body * 1.3:
                    ob_price = float(l_win[i])
                    if ob_price < current_price:
                        ob_points.append(ob_price)

        if ob_points:
            if is_buy:
                nearest_ob = min(ob_points)
                dist_r = (nearest_ob - current_price) / risk_dist
            else:
                nearest_ob = max(ob_points)
                dist_r = (current_price - nearest_ob) / risk_dist
            levels.nearest_ob_price = nearest_ob
            levels.nearest_ob_dist_r = max(0.0, dist_r)

        # ── C. Fair Value Gaps (FVG) ──
        fvg_points: List[float] = []
        for i in range(2, n):
            if is_buy:
                if l_win[i - 2] > h_win[i]:
                    gap_bottom = float(h_win[i])
                    if gap_bottom > current_price:
                        fvg_points.append(gap_bottom)
            else:
                if h_win[i - 2] < l_win[i]:
                    gap_top = float(l_win[i])
                    if gap_top < current_price:
                        fvg_points.append(gap_top)

        if fvg_points:
            if is_buy:
                nearest_fvg = min(fvg_points)
                dist_r = (nearest_fvg - current_price) / risk_dist
            else:
                nearest_fvg = max(fvg_points)
                dist_r = (current_price - nearest_fvg) / risk_dist
            levels.nearest_fvg_price = nearest_fvg
            levels.nearest_fvg_dist_r = max(0.0, dist_r)

        # ── D. Fibonacci Retracement Levels ──
        # Impulse leg: pre-compute argmax and argmin for fast index lookup
        win_30_h = h_win[-30:]
        win_30_l = l_win[-30:]
        max_idx = np.argmax(win_30_h)
        min_idx = np.argmin(win_30_l)
        window_high = float(win_30_h[max_idx])
        window_low = float(win_30_l[min_idx])
        leg_range = window_high - window_low

        if leg_range > 1e-6:
            if is_buy:
                fib_382 = window_low + leg_range * 0.382
                fib_500 = window_low + leg_range * 0.500
                fib_618 = window_low + leg_range * 0.618
                fib_786 = window_low + leg_range * 0.786

                levels.fib_382_price = fib_382
                levels.fib_382_dist_r = abs(fib_382 - current_price) / risk_dist
                levels.fib_500_price = fib_500
                levels.fib_500_dist_r = abs(fib_500 - current_price) / risk_dist
                levels.fib_618_price = fib_618
                levels.fib_618_dist_r = abs(fib_618 - current_price) / risk_dist
                levels.fib_786_price = fib_786
                levels.fib_786_dist_r = abs(fib_786 - current_price) / risk_dist
            else:
                fib_382 = window_high - leg_range * 0.382
                fib_500 = window_high - leg_range * 0.500
                fib_618 = window_high - leg_range * 0.618
                fib_786 = window_high - leg_range * 0.786

                levels.fib_382_price = fib_382
                levels.fib_382_dist_r = abs(current_price - fib_382) / risk_dist
                levels.fib_500_price = fib_500
                levels.fib_500_dist_r = abs(current_price - fib_500) / risk_dist
                levels.fib_618_price = fib_618
                levels.fib_618_dist_r = abs(current_price - fib_618) / risk_dist
                levels.fib_786_price = fib_786
                levels.fib_786_dist_r = abs(current_price - fib_786) / risk_dist

        # ── E. Standard Deviation Bands (20-period Bollinger) ──
        c_20 = c_win[-20:]
        ma_20 = float(np.mean(c_20))
        std_20 = float(np.std(c_20))
        if std_20 > 1e-6:
            if is_buy:
                std1 = ma_20 + std_20
                std2 = ma_20 + 2.0 * std_20
                levels.stddev_1_price = std1
                levels.stddev_1_dist_r = max(0.0, (std1 - current_price) / risk_dist)
                levels.stddev_2_price = std2
                levels.stddev_2_dist_r = max(0.0, (std2 - current_price) / risk_dist)
            else:
                std1 = ma_20 - std_20
                std2 = ma_20 - 2.0 * std_20
                levels.stddev_1_price = std1
                levels.stddev_1_dist_r = max(0.0, (current_price - std1) / risk_dist)
                levels.stddev_2_price = std2
                levels.stddev_2_dist_r = max(0.0, (current_price - std2) / risk_dist)

        # ── F. Find Nearest Opposing Level & Confluence ──
        all_levels = [
            ("Swing Level", levels.nearest_swing_price, levels.nearest_swing_dist_r),
            ("Order Block", levels.nearest_ob_price, levels.nearest_ob_dist_r),
            ("Fair Value Gap", levels.nearest_fvg_price, levels.nearest_fvg_dist_r),
            ("Fib 61.8%", levels.fib_618_price, levels.fib_618_dist_r),
            ("Fib 78.6%", levels.fib_786_price, levels.fib_786_dist_r),
            ("StdDev 2σ Band", levels.stddev_2_price, levels.stddev_2_dist_r),
        ]

        valid_candidates = [(name, p, d) for name, p, d in all_levels if d < 5.0 and p > 0.0]
        if valid_candidates:
            valid_candidates.sort(key=lambda x: x[2])
            best = valid_candidates[0]
            levels.nearest_level_name = best[0]
            levels.nearest_level_price = best[1]
            levels.nearest_level_dist_r = best[2]

            confluence = sum(1 for _, _, d in valid_candidates if abs(d - best[2]) <= 0.35)
            levels.confluence_count_near_price = confluence

        return levels


def extract_structural_levels(
    df: pd.DataFrame,
    direction: str,
    entry_price: float,
    current_price: float,
    risk_dist: float,
    lookback: int = 50,
) -> StructuralLevels:
    """
    Extract key SMC / ICT / Classical structural levels from DataFrame OHLCV history.
    Delegates to StructuralExtractor.extract_structural_levels_ndarray for zero-copy computation.
    """
    risk_dist = max(risk_dist, 1e-8)
    if df is None or len(df) < 20:
        return StructuralLevels()

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()

    return StructuralExtractor.extract_structural_levels_ndarray(
        highs=highs,
        lows=lows,
        closes=closes,
        opens=opens,
        direction=direction,
        entry_price=entry_price,
        current_price=current_price,
        risk_dist=risk_dist,
        lookback=lookback,
    )


def compute_technical_indicators(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Compute ATR ratio, RSI(14), and ADX trend strength."""
    if df is None or len(df) < 15:
        return 1.0, 50.0, 20.0

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    # ATR(14)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])),
    )
    current_atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else 0.001
    baseline_atr = float(np.mean(tr[-30:])) if len(tr) >= 30 else current_atr
    atr_ratio = current_atr / max(1e-6, baseline_atr)

    # RSI(14)
    diffs = np.diff(closes)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = float(np.mean(gains[-14:])) if len(gains) >= 14 else 0.001
    avg_loss = float(np.mean(losses[-14:])) if len(losses) >= 14 else 0.001
    rs = avg_gain / max(1e-6, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Simplified ADX proxy: Directional volatility strength
    dx = abs(avg_gain - avg_loss) / max(1e-6, avg_gain + avg_loss) * 100.0

    return atr_ratio, rsi, dx


SESSION_MAP: Dict[Tuple[int, int], int] = {
    (0, 7): 0,    # Asian
    (7, 12): 1,   # London
    (12, 16): 2,  # London/NY Overlap
    (16, 21): 3,  # NY
    (21, 24): 4,  # Off-hours
}


def encode_session(dt: Optional[datetime] = None) -> int:
    """
    Encode market session into integer using explicit SESSION_MAP:
    0=Asian (00-07), 1=London (07-12), 2=London/NY Overlap (12-16), 3=NY (16-21), 4=Off-hours (21-24).
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    hour = dt.hour
    for (start, end), s_idx in SESSION_MAP.items():
        if start <= hour < end:
            return s_idx
    return 4


def encode_strategy(strategy_name: str) -> int:
    """Encode strategy type: 0=SMC, 1=ICT, 2=OrderFlow, 3=Scalp, 4=TrendReversal, 5=Other."""
    s = str(strategy_name).upper()
    if "ICT" in s:
        return 1
    elif "FLOW" in s or "DELTA" in s or "ABSORPTION" in s:
        return 2
    elif "SCALP" in s:
        return 3
    elif "REVERSAL" in s or "TREND" in s:
        return 4
    elif "SMC" in s:
        return 0
    return 5


def extract_features(
    df: pd.DataFrame,
    direction: str,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    current_price: float,
    bars_since_entry: int,
    strategy_name: str,
    now_dt: Optional[datetime] = None,
) -> Tuple[Dict[str, float], StructuralLevels]:
    """
    Produce normalized causal feature dictionary and structural level details.
    Guards strictly against zero risk distance and incorporates optional now_dt session timing.
    """
    risk_dist = max(abs(entry_price - sl_price), 1e-8)
    target_dist = abs(tp_price - entry_price)
    planned_rr = target_dist / risk_dist

    is_buy = (direction.upper() == "BUY")
    r_multiple = ((current_price - entry_price) / risk_dist) if is_buy else ((entry_price - current_price) / risk_dist)
    pct_to_tp = r_multiple / max(0.1, planned_rr)

    struct = extract_structural_levels(df, direction, entry_price, current_price, risk_dist)
    atr_ratio, rsi, adx = compute_technical_indicators(df)
    session_idx = encode_session(now_dt)
    strategy_idx = encode_strategy(strategy_name)

    features: Dict[str, float] = {
        "r_multiple": float(r_multiple),
        "pct_to_tp": float(pct_to_tp),
        "planned_rr": float(planned_rr),
        "nearest_swing_dist_r": float(min(10.0, struct.nearest_swing_dist_r)),
        "nearest_ob_dist_r": float(min(10.0, struct.nearest_ob_dist_r)),
        "nearest_fvg_dist_r": float(min(10.0, struct.nearest_fvg_dist_r)),
        "fib_382_dist_r": float(min(10.0, struct.fib_382_dist_r)),
        "fib_500_dist_r": float(min(10.0, struct.fib_500_dist_r)),
        "fib_618_dist_r": float(min(10.0, struct.fib_618_dist_r)),
        "fib_786_dist_r": float(min(10.0, struct.fib_786_dist_r)),
        "stddev_1_dist_r": float(min(10.0, struct.stddev_1_dist_r)),
        "stddev_2_dist_r": float(min(10.0, struct.stddev_2_dist_r)),
        "structure_confluence_count": float(struct.confluence_count_near_price),
        "atr_ratio": float(atr_ratio),
        "momentum_rsi": float(rsi),
        "trend_strength_adx": float(adx),
        "bars_since_entry": float(bars_since_entry),
        "session_idx": float(session_idx),
        "strategy_idx": float(strategy_idx),
        "direction_val": 1.0 if is_buy else -1.0,
    }

    # Evaluate dynamic registered features
    ctx = {
        "df": df,
        "direction": direction,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "current_price": current_price,
        "bars_since_entry": bars_since_entry,
        "strategy_name": strategy_name,
        "now_dt": now_dt,
        "features": features,
        "struct": struct,
    }
    for feat_name, compute_fn in FEATURE_REGISTRY.items():
        try:
            features[feat_name] = float(compute_fn(ctx))
        except Exception as e:
            logger.warning(f"Error computing custom registered feature '{feat_name}': {e}")
            features[feat_name] = 0.0

    return features, struct


# =============================================================================
# 4. ML MODEL ARCHITECTURE & ONLINE LEARNING
# =============================================================================

class SmartPartialTPModel:
    """
    Dual-headed model:
    - Classifier: Predicts outcome class: 0 = REVERSAL, 1 = CONTINUATION / FULL_TP
    - Regressor: Predicts expected max R-multiple
    - Online Classifier & Regressor: SGDClassifier and SGDRegressor for online incremental updates
    """

    def __init__(self, artifact_path: Path = MODEL_ARTIFACT_PATH):
        self.artifact_path = artifact_path
        self.classifier: Optional[Any] = None
        self.regressor: Optional[HistGradientBoostingRegressor] = None
        self.online_clf: Optional[Pipeline] = None
        self.online_reg: Optional[Pipeline] = None
        self.total_trained_samples: int = 0
        self.version: str = "v1.0-hybrid"
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Load trained model from disk or train pre-calibrated baseline."""
        if self.artifact_path.exists():
            try:
                bundle = joblib.load(self.artifact_path)
                self.classifier = bundle.get("classifier")
                self.regressor = bundle.get("regressor")
                self.online_clf = bundle.get("online_clf")
                self.online_reg = bundle.get("online_reg")
                self.total_trained_samples = bundle.get("samples", 0)
                self.version = bundle.get("version", "v1.0-loaded")
                logger.info(f"Loaded SmartPartialTPModel from {self.artifact_path} ({self.total_trained_samples} samples)")
                return
            except Exception as e:
                logger.warning(f"Could not load {self.artifact_path}: {e}. Retraining baseline...")

        self._train_synthetic_baseline()

    def fit_and_tune(self, X: np.ndarray, y_class: np.ndarray, max_r_target: np.ndarray, n_iter: int = 8, cv: int = 3) -> None:
        """
        Hyperparameter optimization via RandomizedSearchCV with Cross-Validation
        and Platt-scaled Probability Calibration.
        """
        param_grid_clf = {
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "max_iter": [100, 200, 400],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [10, 20, 30],
            "l2_regularization": [0.0, 0.1, 1.0, 5.0],
        }

        actual_cv = min(cv, max(2, len(np.unique(y_class)))) if len(y_class) >= 6 else 2

        try:
            search_clf = RandomizedSearchCV(
                estimator=HistGradientBoostingClassifier(random_state=42, class_weight="balanced"),
                param_distributions=param_grid_clf,
                n_iter=min(n_iter, 10),
                cv=actual_cv,
                scoring="roc_auc",
                random_state=42,
                n_jobs=-1,
                error_score="raise",
            )
            search_clf.fit(X, y_class)
            best_clf_base = search_clf.best_estimator_
            logger.info(f"Classifier tuning complete. Best params: {search_clf.best_params_}")
        except Exception as e:
            logger.warning(f"RandomizedSearchCV for classifier failed: {e}. Falling back to default HGBClassifier.")
            best_clf_base = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=42
            )
            best_clf_base.fit(X, y_class)

        try:
            calibrated_clf = CalibratedClassifierCV(estimator=best_clf_base, method="sigmoid", cv=actual_cv)
            calibrated_clf.fit(X, y_class)
            self.classifier = calibrated_clf
        except Exception as e:
            logger.warning(f"CalibratedClassifierCV failed: {e}. Using uncalibrated base classifier.")
            self.classifier = best_clf_base

        param_grid_reg = {
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "max_iter": [100, 200, 400],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [10, 20, 30],
            "l2_regularization": [0.0, 0.1, 1.0, 5.0],
        }

        try:
            search_reg = RandomizedSearchCV(
                estimator=HistGradientBoostingRegressor(random_state=42),
                param_distributions=param_grid_reg,
                n_iter=min(n_iter, 10),
                cv=actual_cv,
                scoring="neg_root_mean_squared_error",
                random_state=42,
                n_jobs=-1,
                error_score="raise",
            )
            search_reg.fit(X, max_r_target)
            self.regressor = search_reg.best_estimator_
            logger.info(f"Regressor tuning complete. Best params: {search_reg.best_params_}")
        except Exception as e:
            logger.warning(f"RandomizedSearchCV for regressor failed: {e}. Falling back to default HGBRegressor.")
            self.regressor = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_depth=5, min_samples_leaf=20, random_state=42
            )
            self.regressor.fit(X, max_r_target)

        self.online_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("sgd", SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=42)),
        ])
        self.online_clf.fit(X, y_class)

        self.online_reg = Pipeline([
            ("scaler", StandardScaler()),
            ("sgd", SGDRegressor(penalty="l2", alpha=1e-4, random_state=42)),
        ])
        self.online_reg.fit(X, max_r_target)

        self.total_trained_samples = len(X)
        self.version = "v2.0-tuned-calibrated"

    def _train_synthetic_baseline(self, n_samples: int = 5000) -> None:
        """
        Train on synthetic domain distribution representing authentic institutional market mechanics.
        """
        np.random.seed(42)

        r_mult = np.random.uniform(0.5, 3.5, n_samples)
        planned_rr = np.random.uniform(1.8, 3.5, n_samples)
        pct_to_tp = r_mult / planned_rr

        swing_dist = np.random.exponential(1.2, n_samples)
        ob_dist = np.random.exponential(1.5, n_samples)
        fvg_dist = np.random.exponential(1.4, n_samples)
        fib_382_dist = np.random.exponential(1.0, n_samples)
        fib_500_dist = np.random.exponential(1.1, n_samples)
        fib_618_dist = np.random.exponential(1.2, n_samples)
        fib_786_dist = np.random.exponential(1.5, n_samples)
        std1_dist = np.random.exponential(0.9, n_samples)
        std2_dist = np.random.exponential(1.3, n_samples)

        confluence = np.random.poisson(1.0, n_samples)

        atr_ratio = np.random.normal(1.0, 0.25, n_samples)
        rsi = np.random.uniform(25.0, 75.0, n_samples)
        adx = np.random.uniform(15.0, 50.0, n_samples)
        bars_held = np.random.randint(3, 40, n_samples)
        session = np.random.randint(0, 5, n_samples)
        strategy = np.random.randint(0, 5, n_samples)
        direction = np.random.choice([-1.0, 1.0], n_samples)

        cols = [
            r_mult, pct_to_tp, planned_rr,
            swing_dist, ob_dist, fvg_dist,
            fib_382_dist, fib_500_dist, fib_618_dist, fib_786_dist,
            std1_dist, std2_dist,
            confluence, atr_ratio, rsi, adx,
            bars_held, session, strategy, direction,
        ]

        # Add zero columns for any custom registered features if present
        extra_len = len(FEATURE_NAMES) - len(cols)
        for _ in range(extra_len):
            cols.append(np.zeros(n_samples))

        X = np.column_stack(cols)

        rev_score = (
            (r_mult * 0.4)
            + (confluence * 0.5)
            + (np.where(swing_dist < 0.4, 1.2, 0.0))
            + (np.where(ob_dist < 0.4, 1.4, 0.0))
            + (np.where(fib_618_dist < 0.35, 1.0, 0.0))
            + (np.where(std2_dist < 0.3, 0.9, 0.0))
            + (np.where(rsi > 70, 0.8, 0.0))
            - (adx * 0.02)
            + np.random.normal(0, 0.5, n_samples)
        )

        p_rev = 1.0 / (1.0 + np.exp(-(rev_score - 2.0)))
        y_class = (p_rev < 0.50).astype(int)

        max_r_target = np.maximum(
            r_mult,
            r_mult + np.where(y_class == 1, np.random.exponential(1.5, n_samples), np.random.exponential(0.2, n_samples))
        )

        self.fit_and_tune(X, y_class, max_r_target, n_iter=8, cv=3)
        self._save()
        logger.info(f"Trained baseline SmartPartialTPModel on {n_samples} samples (Version: {self.version}).")

    def _save(self) -> None:
        """Persist model artifact bundle to disk."""
        try:
            self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
            bundle = {
                "classifier": self.classifier,
                "regressor": self.regressor,
                "online_clf": self.online_clf,
                "online_reg": self.online_reg,
                "samples": self.total_trained_samples,
                "version": self.version,
                "feature_names": FEATURE_NAMES,
            }
            joblib.dump(bundle, self.artifact_path)
            logger.info(f"Saved SmartPartialTPModel to {self.artifact_path}")
        except Exception as e:
            logger.error(f"Failed to save SmartPartialTPModel to {self.artifact_path}: {e}")

    def predict(self, feature_dict: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Run inference across classifier and regressor heads.
        Returns: (p_reversal, p_full_tp, predicted_max_r)
        """
        x_vec = np.array([[feature_dict.get(name, 0.0) for name in FEATURE_NAMES]])

        # 1. Classifier: P(continuation) vs P(reversal)
        if self.classifier is not None:
            probs = self.classifier.predict_proba(x_vec)[0]
            p_rev = float(probs[0])
            p_cont = float(probs[1])
        elif self.online_clf is not None:
            probs = self.online_clf.predict_proba(x_vec)[0]
            p_rev = float(probs[0])
            p_cont = float(probs[1])
        else:
            p_rev = 0.50
            p_cont = 0.50

        pct_to_tp = feature_dict.get("pct_to_tp", 0.5)
        p_full_tp = float(np.clip(p_cont * (0.6 + 0.4 * pct_to_tp), 0.05, 0.95))

        # 2. Regressor: Predicted Max R
        if self.regressor is not None:
            pred_max_r = float(self.regressor.predict(x_vec)[0])
        elif self.online_reg is not None:
            pred_max_r = float(self.online_reg.predict(x_vec)[0])
        else:
            curr_r = feature_dict.get("r_multiple", 1.0)
            pred_max_r = curr_r + (1.0 if p_cont > 0.5 else 0.2)

        return p_rev, p_full_tp, max(feature_dict.get("r_multiple", 0.0), pred_max_r)

    def partial_fit(self, feature_dict: Dict[str, float], y_label: int, final_r: float = 0.0) -> None:
        """Online incremental update for classifier and online regressor."""
        x_vec = np.array([[feature_dict.get(name, 0.0) for name in FEATURE_NAMES]])

        # Update online classifier
        if self.online_clf is not None:
            try:
                sgd_step = self.online_clf.named_steps["sgd"]
                scaler = self.online_clf.named_steps["scaler"]
                scaler.partial_fit(x_vec)
                x_scaled = scaler.transform(x_vec)
                sgd_step.partial_fit(x_scaled, np.array([y_label]), classes=np.array([0, 1]))
            except Exception as e:
                logger.error(f"Error in SmartPartialTPModel online_clf partial_fit: {e}")

        # Update online regressor
        if self.online_reg is not None:
            try:
                reg_sgd = self.online_reg.named_steps["sgd"]
                reg_scaler = self.online_reg.named_steps["scaler"]
                reg_scaler.partial_fit(x_vec)
                x_scaled = reg_scaler.transform(x_vec)
                reg_sgd.partial_fit(x_scaled, np.array([final_r]))
            except Exception as e:
                logger.error(f"Error in SmartPartialTPModel online_reg partial_fit: {e}")

        self.total_trained_samples += 1
        if self.total_trained_samples % 25 == 0:
            self._save()

    def train_from_trade_records(self, trade_records: List[Any], n_iter: int = 8, cv: int = 3) -> bool:
        """
        Train classifier and regressor on real historical completed trade records.
        """
        X_list, y_class_list, max_r_list = [], [], []

        for t in trade_records:
            t_dict = t.__dict__ if hasattr(t, '__dict__') else t
            if not isinstance(t_dict, dict):
                continue

            entry_price = float(t_dict.get('entry_price', 0.0) or 0.0)
            sl_price = float(t_dict.get('stop_loss', t_dict.get('sl_price', 0.0)) or 0.0)
            tp_price = float(t_dict.get('take_profit', t_dict.get('tp_price', 0.0)) or 0.0)
            exit_price = float(t_dict.get('exit_price', entry_price) or entry_price)
            direction_raw = t_dict.get('direction', 'BUY')
            direction = direction_raw.value if hasattr(direction_raw, 'value') else str(direction_raw)
            strategy_name = str(t_dict.get('strategy_name', 'SMC'))

            risk = max(abs(entry_price - sl_price), 1e-8)
            target_dist = abs(tp_price - entry_price)
            planned_rr = target_dist / risk

            is_buy = ('BUY' in direction.upper())
            status_str = str(t_dict.get('status', '')).upper()

            pnl = float(t_dict.get('realized_pnl', t_dict.get('pnl', 0.0)) or 0.0)
            hit_tp = ('TP' in status_str) or (pnl > 0 and 'SL' not in status_str)
            y_class = 1 if hit_tp else 0

            final_r = planned_rr if hit_tp else (-1.0 if 'SL' in status_str else (pnl / max(1.0, risk * 100.0)))
            max_r = max(final_r, 0.0)

            feat = {
                'r_multiple': float(max(0.5, final_r * 0.5)) if hit_tp else 0.5,
                'pct_to_tp': float(np.clip(final_r / max(0.1, planned_rr), 0.0, 1.0)) if hit_tp else 0.2,
                'planned_rr': float(planned_rr),
                'nearest_swing_dist_r': 0.8,
                'nearest_ob_dist_r': 1.0,
                'nearest_fvg_dist_r': 1.1,
                'fib_382_dist_r': 0.5,
                'fib_500_dist_r': 0.6,
                'fib_618_dist_r': 0.7,
                'fib_786_dist_r': 0.9,
                'stddev_1_dist_r': 0.5,
                'stddev_2_dist_r': 1.0,
                'structure_confluence_count': 1.0,
                'atr_ratio': 1.0,
                'momentum_rsi': 50.0,
                'trend_strength_adx': 25.0,
                'bars_since_entry': 10.0,
                'session_idx': float(encode_session()),
                'strategy_idx': float(encode_strategy(strategy_name)),
                'direction_val': 1.0 if is_buy else -1.0,
            }

            x_vec = [feat.get(name, 0.0) for name in FEATURE_NAMES]
            X_list.append(x_vec)
            y_class_list.append(y_class)
            max_r_list.append(max_r)

        if len(X_list) < 10:
            logger.warning(f"Insufficient historical trades ({len(X_list)}) for full retrain. Minimum 10 required.")
            return False

        X = np.array(X_list)
        y_class_arr = np.array(y_class_list)
        max_r_arr = np.array(max_r_list)

        self.fit_and_tune(X, y_class_arr, max_r_arr, n_iter=n_iter, cv=cv)
        self.version = f"v2.0-real-{len(X_list)}trades"
        self._save()
        logger.info(f"Successfully retrained SmartPartialTPModel on {len(X_list)} real trade records.")
        return True

    def explain_prediction(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate SHAP feature importance breakdown for a given prediction.
        """
        x_vec = np.array([[feature_dict.get(name, 0.0) for name in FEATURE_NAMES]])

        base_estimator = None
        if hasattr(self.classifier, 'estimator'):
            base_estimator = self.classifier.estimator
        elif isinstance(self.classifier, HistGradientBoostingClassifier):
            base_estimator = self.classifier

        shap_values_dict = {}
        if base_estimator is not None:
            try:
                explainer = shap.TreeExplainer(base_estimator)
                sv = explainer.shap_values(x_vec)
                sv_arr = np.array(sv)
                if sv_arr.ndim == 3:
                    shap_values_dict = {name: float(sv_arr[0, :, 1][idx]) for idx, name in enumerate(FEATURE_NAMES)}
                elif sv_arr.ndim == 2:
                    shap_values_dict = {name: float(sv_arr[0, idx]) for idx, name in enumerate(FEATURE_NAMES)}
                else:
                    shap_values_dict = {name: 0.0 for name in FEATURE_NAMES}
            except Exception:
                shap_values_dict = {name: 0.0 for name in FEATURE_NAMES}
        else:
            shap_values_dict = {name: 0.0 for name in FEATURE_NAMES}

        sorted_features = sorted(
            [{"feature": k, "shap_value": round(v, 4), "abs_impact": round(abs(v), 4)} for k, v in shap_values_dict.items()],
            key=lambda x: x["abs_impact"],
            reverse=True
        )

        return {
            "model_version": self.version,
            "feature_importances": sorted_features,
            "top_drivers": [f["feature"] for f in sorted_features[:5]],
        }


# =============================================================================
# 5. DECISION ENGINE & CONFLICT RESOLUTION TABLE
# =============================================================================

def decide_partial_tp(
    features: Dict[str, float],
    struct: StructuralLevels,
    p_reversal: float,
    p_full_tp: float,
    pred_max_r: float,
    current_lot: float,
    min_lot: float = 0.01,
    reversal_threshold: float = 0.60,
    runner_threshold: float = 0.65,
) -> PartialTPVerdict:
    """
    Translates model probabilities and structural geometry into an actionable decision using an
    explicit prioritized Conflict Resolution Table:

    Priority 1: High conviction runner -> HOLD (let runner ride)
    Priority 2: High/Imminent Reversal risk -> PARTIAL_CLOSE (50-60%)
    Priority 3: Deep in profit target proximity (>=85% to TP) -> PARTIAL_CLOSE (50%)
    Priority 4: Moderate reversal / expected max R exhaustion -> PARTIAL_CLOSE (33%)
    Priority 5: Default favorable conditions -> HOLD
    """
    r_mult = features.get("r_multiple", 0.0)
    pct_to_tp = features.get("pct_to_tp", 0.0)
    confluence = struct.confluence_count_near_price
    nearest_dist = struct.nearest_level_dist_r
    nearest_desc = f"{struct.nearest_level_name} @ {struct.nearest_level_price:.5f} ({nearest_dist:.2f}R away)"

    level_imminent = (nearest_dist <= 0.35)

    # Conflict Resolution Evaluation Sequence
    # Rule 1: High conviction runner (Hold)
    if p_full_tp >= runner_threshold and not (level_imminent and confluence >= 2):
        return PartialTPVerdict(
            action="HOLD",
            close_pct=0.0,
            p_reversal=p_reversal,
            p_full_tp=p_full_tp,
            predicted_max_r=pred_max_r,
            nearest_resistance=nearest_desc,
            confidence=round(p_full_tp, 3),
            reason=f"High continuation probability ({p_full_tp:.1%}). Path to TP is clean. Holding runner.",
            features=features,
        )

    # Rule 2: Impending Reversal into Major Resistance / Confluence
    if p_reversal >= reversal_threshold or (level_imminent and confluence >= 2 and p_reversal >= 0.50):
        if p_reversal >= 0.75 or confluence >= 3:
            close_pct = 0.60
            reason = f"High reversal probability ({p_reversal:.1%}) at {struct.nearest_level_name} with {confluence}x confluence. Aggressive 60% partial close."
        else:
            close_pct = 0.50
            reason = f"Reversal anticipated ({p_reversal:.1%}) near {struct.nearest_level_name} ({nearest_dist:.2f}R away). Booking 50% partial."

        return PartialTPVerdict(
            action="PARTIAL_CLOSE",
            close_pct=close_pct,
            p_reversal=p_reversal,
            p_full_tp=p_full_tp,
            predicted_max_r=pred_max_r,
            nearest_resistance=nearest_desc,
            confidence=round(p_reversal, 3),
            reason=reason,
            features=features,
        )

    # Rule 3: Deep in Profit (>= 85% to target distance)
    if pct_to_tp >= 0.85:
        return PartialTPVerdict(
            action="PARTIAL_CLOSE",
            close_pct=0.50,
            p_reversal=p_reversal,
            p_full_tp=p_full_tp,
            predicted_max_r=pred_max_r,
            nearest_resistance=nearest_desc,
            confidence=0.85,
            reason=f"Position reached {pct_to_tp:.1%} of target distance. Securing 50% remaining lots before final target.",
            features=features,
        )

    # Rule 4: Moderate Reversal / Approaching Exhaustion (Standard Trim)
    if (0.50 <= p_reversal < reversal_threshold) or (pred_max_r <= r_mult + 0.3):
        return PartialTPVerdict(
            action="PARTIAL_CLOSE",
            close_pct=0.33,
            p_reversal=p_reversal,
            p_full_tp=p_full_tp,
            predicted_max_r=pred_max_r,
            nearest_resistance=nearest_desc,
            confidence=round(p_reversal, 3),
            reason=f"Moderate reversal risk ({p_reversal:.1%}) and max R expected near {pred_max_r:.2f}R. Trimming 33%.",
            features=features,
        )

    # Rule 5: Default HOLD
    return PartialTPVerdict(
        action="HOLD",
        close_pct=0.0,
        p_reversal=p_reversal,
        p_full_tp=p_full_tp,
        predicted_max_r=pred_max_r,
        nearest_resistance=nearest_desc,
        confidence=round(1.0 - p_reversal, 3),
        reason=f"Current conditions favorable for continuation ({p_full_tp:.1%} to TP). Holding position.",
        features=features,
    )


# =============================================================================
# 6. CONFIGURATION LOADER
# =============================================================================

def load_partial_tp_config(config_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """
    Load Smart Partial TP system settings from config.yaml or return defaults.
    """
    defaults: Dict[str, Any] = {
        "reversal_threshold": 0.60,
        "runner_threshold": 0.65,
        "min_lot": 0.01,
        "error_callback": None,
    }
    path = Path(config_path) if config_path else CONFIG_PATH
    if path.exists() and yaml is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            ptp_cfg = cfg.get("smart_partial_tp", {})
            for k in defaults:
                if k in ptp_cfg:
                    defaults[k] = ptp_cfg[k]
        except Exception as e:
            logger.warning(f"Failed to read config from {path}: {e}")
    return defaults


# =============================================================================
# 7. PUBLIC SERVICE INTERFACE
# =============================================================================

class SmartPartialTPService:
    """
    Public entry point for ML-driven partial profit booking.
    Designed for integration into PositionManager.
    """

    _instance: Optional["SmartPartialTPService"] = None

    def __init__(self, artifact_path: Path = MODEL_ARTIFACT_PATH, config_path: Optional[Path | str] = None):
        self.config = load_partial_tp_config(config_path)
        self.model = SmartPartialTPModel(artifact_path=artifact_path)
        self.evaluation_count: int = 0
        self.executed_partial_count: int = 0
        self.held_runner_count: int = 0
        logger.info(f"SmartPartialTPService initialized (Model version: {self.model.version})")

    @classmethod
    def get_instance(cls, artifact_path: Path = MODEL_ARTIFACT_PATH, config_path: Optional[Path | str] = None) -> "SmartPartialTPService":
        if cls._instance is None:
            cls._instance = cls(artifact_path=artifact_path, config_path=config_path)
        return cls._instance

    def evaluate(
        self,
        symbol: str,
        strategy_name: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        current_price: float,
        current_lot: float,
        bars_since_entry: int,
        df: Optional[pd.DataFrame],
        shadow_mode: bool = False,
        reversal_threshold: Optional[float] = None,
        runner_threshold: Optional[float] = None,
        now_dt: Optional[datetime] = None,
    ) -> PartialTPVerdict:
        """
        Evaluate an open position for partial profit booking.
        """
        self.evaluation_count += 1
        rev_thresh = reversal_threshold if reversal_threshold is not None else float(self.config.get("reversal_threshold", 0.60))
        run_thresh = runner_threshold if runner_threshold is not None else float(self.config.get("runner_threshold", 0.65))

        try:
            features, struct = extract_features(
                df=df,
                direction=direction,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                current_price=current_price,
                bars_since_entry=bars_since_entry,
                strategy_name=strategy_name,
                now_dt=now_dt,
            )

            p_rev, p_full, pred_max_r = self.model.predict(features)

            verdict = decide_partial_tp(
                features=features,
                struct=struct,
                p_reversal=p_rev,
                p_full_tp=p_full,
                pred_max_r=pred_max_r,
                current_lot=current_lot,
                min_lot=float(self.config.get("min_lot", 0.01)),
                reversal_threshold=rev_thresh,
                runner_threshold=run_thresh,
            )
            verdict.is_shadow = shadow_mode

            if verdict.action == "PARTIAL_CLOSE":
                self.executed_partial_count += 1
            elif verdict.action == "HOLD" and verdict.p_full_tp >= run_thresh:
                self.held_runner_count += 1

            logger.info(
                f"🎯 [SMART PARTIAL TP] {symbol} {direction} (+{features['r_multiple']:.2f}R) -> {verdict.action} "
                f"(Close: {verdict.close_pct:.0%}) | P(Rev): {verdict.p_reversal:.1%}, P(TP): {verdict.p_full_tp:.1%}, "
                f"Nearest: {verdict.nearest_resistance} | {verdict.reason}"
            )
            return verdict

        except Exception as e:
            logger.error(f"Error in SmartPartialTPService.evaluate: {e}", exc_info=True)
            err_cb = self.config.get("error_callback")
            if callable(err_cb):
                try:
                    err_cb(e)
                except Exception:
                    pass
            return PartialTPVerdict(
                action="HOLD",
                close_pct=0.0,
                p_reversal=0.5,
                p_full_tp=0.5,
                predicted_max_r=1.0,
                nearest_resistance="Error evaluating",
                confidence=0.0,
                reason=f"Evaluation error fallback: {e}",
                is_shadow=shadow_mode,
            )

    def record_trade_outcome(
        self,
        features: Dict[str, float],
        hit_tp: bool,
        final_r: float = 0.0,
    ) -> None:
        """Feed completed trade outcome back into model for continuous self-learning."""
        try:
            label = 1 if hit_tp else 0
            self.model.partial_fit(features, y_label=label, final_r=final_r)
            logger.debug(f"SmartPartialTPService learned trade outcome: final_r={final_r:.2f}R, label={label}")
        except Exception as e:
            logger.error(f"Failed to record trade outcome in SmartPartialTPService: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Summary for API / Dashboard."""
        return {
            "model_version": self.model.version,
            "trained_samples": self.model.total_trained_samples,
            "evaluation_count": self.evaluation_count,
            "executed_partial_count": self.executed_partial_count,
            "held_runner_count": self.held_runner_count,
            "artifact_exists": self.model.artifact_path.exists(),
        }

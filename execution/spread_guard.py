"""
spread_guard.py — Percentile-based Dynamic Spread Guard & Slippage Estimator.
"""
from __future__ import annotations
import collections
import threading
from typing import Dict, Deque, Tuple
import numpy as np
from loguru import logger


class DynamicSpreadGuard:
    """Tracks tick spreads across rolling windows to filter out anomalous spread spikes."""

    def __init__(self, window_size: int = 300, percentile_cutoff: float = 95.0):
        self.window_size = window_size
        self.percentile_cutoff = percentile_cutoff
        self._spread_history: Dict[str, Deque[float]] = {}
        self._slippage_history: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def record_tick(self, symbol: str, spread: float) -> None:
        if spread <= 0:
            return
        with self._lock:
            buf = self._spread_history.setdefault(symbol, collections.deque(maxlen=self.window_size))
            buf.append(spread)

    def record_fill_slippage(self, symbol: str, slippage_pips: float) -> None:
        with self._lock:
            buf = self._slippage_history.setdefault(symbol, collections.deque(maxlen=50))
            buf.append(abs(slippage_pips))

    def evaluate_spread(self, symbol: str, current_spread: float, session_killzone: str = "OFF_HOURS",
                        multiplier_map: dict | None = None) -> Tuple[bool, str | None, float]:
        """Check if current spread is acceptable based on rolling percentile distribution."""
        self.record_tick(symbol, current_spread)
        mult = (multiplier_map or {}).get(session_killzone, 1.0)

        with self._lock:
            buf = self._spread_history.get(symbol)
            if not buf or len(buf) < 15:
                return True, None, current_spread

            p_threshold = float(np.percentile(list(buf), self.percentile_cutoff)) * mult

        if current_spread > p_threshold:
            msg = f"Spread spike detected for {symbol}: current {current_spread:.5f} exceeds P{self.percentile_cutoff:.0f} threshold ({p_threshold:.5f}) in {session_killzone}"
            return False, msg, p_threshold

        return True, None, p_threshold

    def estimate_slippage_pips(self, symbol: str) -> float:
        with self._lock:
            buf = self._slippage_history.get(symbol)
            if not buf or len(buf) < 3:
                return 0.2  # Default baseline
            return float(np.median(list(buf)))

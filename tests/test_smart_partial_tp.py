"""
tests/test_smart_partial_tp.py
==============================

Comprehensive test suite for the ML-powered adaptive partial profit booking system.
Verifies:
1. Causal feature extraction and structural level detection (Swing, OB, FVG, Fib, StdDev).
2. Model inference (P(reversal), P(full_tp), predicted max R).
3. Decision engine logic (HOLD runner, PARTIAL_CLOSE scaling).
4. Incremental online learning via partial_fit.
5. Integration with PositionManager and StateManager.
"""

import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from core.config import Direction, PositionManagementRuleConfig, TradingConfig
from core.state import StateManager, TradeRecord
from execution.position_manager import PositionManager
from ml.smart_partial_tp import (
    SmartPartialTPModel,
    SmartPartialTPService,
    StructuralLevels,
    decide_partial_tp,
    extract_features,
    extract_structural_levels,
)


def _generate_synthetic_ohlcv(n: int = 100, trend: str = "bullish") -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data."""
    np.random.seed(123)
    base = 2000.0
    drift = 0.5 if trend == "bullish" else -0.5
    closes = [base]
    for _ in range(n - 1):
        step = np.random.normal(drift, 1.2)
        closes.append(closes[-1] + step)

    closes = np.array(closes)
    highs = closes + np.random.uniform(0.3, 1.5, n)
    lows = closes - np.random.uniform(0.3, 1.5, n)
    opens = closes + np.random.uniform(-0.5, 0.5, n)
    volumes = np.random.uniform(100, 1000, n)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return df


class MockQuote:
    def __init__(self, bid: float, ask: float):
        self.bid = bid
        self.ask = ask


class MockBroker:
    def __init__(self, current_price: float = 2015.0):
        self.price = current_price
        self.closed_positions = []

    def get_current_price(self, symbol: str):
        return MockQuote(bid=self.price, ask=self.price + 0.1)


def test_structural_level_extraction():
    """Verify Swing, OB, FVG, Fib, and StdDev extraction on sample data."""
    df = _generate_synthetic_ohlcv(100, "bullish")
    entry = 2000.0
    current_price = 2010.0
    risk_dist = 5.0

    struct = extract_structural_levels(
        df=df,
        direction="BUY",
        entry_price=entry,
        current_price=current_price,
        risk_dist=risk_dist,
        lookback=50,
    )

    assert isinstance(struct, StructuralLevels)
    assert struct.nearest_swing_dist_r >= 0.0
    assert struct.fib_382_dist_r >= 0.0
    assert struct.fib_618_dist_r >= 0.0
    assert struct.stddev_1_dist_r >= 0.0
    assert struct.stddev_2_dist_r >= 0.0
    assert struct.nearest_level_name != ""


def test_causal_feature_extraction():
    """Verify that feature dictionary is strictly valid and contains no NaNs or infs."""
    df = _generate_synthetic_ohlcv(80, "bullish")
    entry = 2000.0
    sl = 1995.0
    tp = 2015.0
    current_price = 2005.0

    features, struct = extract_features(
        df=df,
        direction="BUY",
        entry_price=entry,
        sl_price=sl,
        tp_price=tp,
        current_price=current_price,
        bars_since_entry=12,
        strategy_name="SMC",
    )

    assert features["r_multiple"] == pytest.approx(1.0, 0.01)
    assert features["planned_rr"] == pytest.approx(3.0, 0.01)
    assert 0.0 <= features["pct_to_tp"] <= 1.0
    for k, v in features.items():
        assert not math.isnan(v), f"Feature {k} is NaN"
        assert not math.isinf(v), f"Feature {k} is Inf"


def test_decision_engine_rules():
    """Verify decision policies for high reversal, runner holding, and trim."""
    base_features = {
        "r_multiple": 1.2,
        "pct_to_tp": 0.4,
        "planned_rr": 3.0,
    }
    struct = StructuralLevels(
        nearest_level_name="Order Block",
        nearest_level_price=2012.0,
        nearest_level_dist_r=0.20,
        confluence_count_near_price=2,
    )

    # 1. High reversal risk at resistance -> PARTIAL_CLOSE
    v_rev = decide_partial_tp(
        features=base_features,
        struct=struct,
        p_reversal=0.72,
        p_full_tp=0.25,
        pred_max_r=1.5,
        current_lot=1.0,
    )
    assert v_rev.action == "PARTIAL_CLOSE"
    assert v_rev.close_pct >= 0.50

    # 2. High continuation conviction with clean structure -> HOLD runner
    clean_struct = StructuralLevels(
        nearest_level_name="None",
        nearest_level_price=0.0,
        nearest_level_dist_r=99.0,
        confluence_count_near_price=0,
    )
    v_runner = decide_partial_tp(
        features=base_features,
        struct=clean_struct,
        p_reversal=0.25,
        p_full_tp=0.75,
        pred_max_r=3.2,
        current_lot=1.0,
    )
    assert v_runner.action == "HOLD"
    assert v_runner.close_pct == 0.0


def test_model_inference_and_online_fit():
    """Verify SmartPartialTPModel inference and online incremental fitting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_path = Path(tmpdir) / "test_model.joblib"
        model = SmartPartialTPModel(artifact_path=tmp_model_path)
        assert model.classifier is not None
        assert model.regressor is not None

        df = _generate_synthetic_ohlcv(50)
        feats, _ = extract_features(
            df=df,
            direction="BUY",
            entry_price=2000.0,
            sl_price=1995.0,
            tp_price=2015.0,
            current_price=2006.0,
            bars_since_entry=5,
            strategy_name="ICT",
        )

        p_rev, p_full, pred_max_r = model.predict(feats)
        assert 0.0 <= p_rev <= 1.0
        assert 0.0 <= p_full <= 1.0
        assert pred_max_r >= feats["r_multiple"]

        # Test incremental fit
        prev_samples = model.total_trained_samples
        model.partial_fit(feats, y_label=1)
        assert model.total_trained_samples == prev_samples + 1


def test_position_manager_integration():
    """Verify that PositionManager executes partial TP and logs to state DB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_trading.db"
        state = StateManager(db_path=str(db_path))
        pm = None
        try:
            # Seed an open trade at +1.2R
            trade = TradeRecord(
                id=101,
                timestamp=datetime.now(timezone.utc),
                symbol="EURUSD",
                direction=Direction.BUY,
                entry_price=1.1000,
                stop_loss=1.0950,    # 50 pips risk
                take_profit=1.1150,  # 150 pips target (3R)
                lot_size=0.10,
                realized_pnl=0.0,
                status="OPEN",
                strategy_name="SMC",
            )
            trade_id = state.record_trade(trade)

            # Current price = 1.1080 (+1.6R, >50% of 150-pip target distance)
            mock_broker = MockBroker(current_price=1.1080)
            df_hist = _generate_synthetic_ohlcv(60)

            # Create SmartPartialTPService with temp model
            model_path = Path(tmpdir) / "smart_ptp.joblib"
            svc = SmartPartialTPService(artifact_path=model_path)

            config = TradingConfig()
            rules = config.position_management.get_rules("SMC", "EURUSD")
            rules.smart_partial_tp_enabled = True
            rules.smart_partial_tp_min_r = 0.5
            rules.smart_partial_tp_shadow_mode = False
            rules.smart_partial_tp_cooldown_bars = 0  # immediate execution for test

            pm = PositionManager(
                broker=mock_broker,
                state=state,
                poll_interval_sec=100.0,
                history_provider=lambda sym, tf, n: df_hist,
                config=config,
                smart_partial_tp_service=svc,
            )

            # Process positions
            pm.process_positions()

            # Check that partial TP was recorded in DB
            events = state.get_recent_partial_tp_events(limit=10)
            assert len(events) >= 1
            ev = events[0]
            assert ev["trade_id"] == trade_id
            assert ev["r_multiple"] >= 1.0
            assert ev["action"] in ("PARTIAL_CLOSE", "HOLD")
            assert ev["symbol"] == "EURUSD"

            # If partial close was executed, remaining lot should be reduced
            if ev["action"] == "PARTIAL_CLOSE":
                assert pm._remaining_lots[trade_id] < 0.10
                assert pm._remaining_lots[trade_id] >= 0.01  # runner preserved
        finally:
            if pm is not None:
                pm.shutdown()
            state.close()


def test_target_50_pct_partial_profit_and_cost_to_cost_sl():
    """
    Test user requirement:
    - On small 1-2 candle movement, do NOT book premature profit.
    - When price reaches 50% of target distance, book exactly 50% profit (close half the lot).
    - Move Stop Loss to Cost-to-Cost (entry price / breakeven).
    - Hold remaining 50% lot until full Take Profit (no additional cuts).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_50pct_target.db"
        state = StateManager(db_path=str(db_path))
        pm = None
        try:
            # GBPUSD trade: 0.12 lot, Entry 1.3000, SL 1.2980 (20 pips risk), TP 1.3040 (40 pips target)
            trade = TradeRecord(
                id=999,
                timestamp=datetime.now(timezone.utc),
                symbol="GBPUSD",
                direction=Direction.BUY,
                entry_price=1.3000,
                stop_loss=1.2980,
                take_profit=1.3040,
                lot_size=0.12,
                realized_pnl=0.0,
                status="OPEN",
                strategy_name="SMC",
            )
            state.record_trade(trade)

            # Broker with mutable price and SL tracking
            class TrackingBroker:
                def __init__(self, price: float):
                    self.price = price
                    self.closed_volume = 0.0
                    self.sl_record = None

                def get_current_price(self, symbol: str):
                    return MockQuote(bid=self.price, ask=self.price + 0.0001)

            broker = TrackingBroker(price=1.3003)  # Small 1-2 candle move (only 3 pips, ~8% of target)
            config = TradingConfig()

            pm = PositionManager(
                broker=broker,
                state=state,
                poll_interval_sec=100.0,
                config=config,
            )
            # Track SL modifications
            sl_modified_values = []
            pm._modify_mt5_sl_tp = lambda ticket, sym, sl, tp: sl_modified_values.append(sl) or True
            # Track lot closures
            closed_lots = []
            pm._close_mt5_position = lambda ticket, sym, lot, dir: closed_lots.append(lot) or True

            # ── Step 1: Small move (+3 pips) -> MUST NOT BOOK PROFIT ──
            pm.process_positions()
            assert len(closed_lots) == 0, "Premature profit booking must NOT occur on small 1-2 candle moves"
            assert pm._remaining_lots.get(999, 0.12) == 0.12
            assert 999 not in pm._partial_tp_applied

            # ── Step 2: Price reaches 50% of target distance (+20 pips -> 1.3020) ──
            broker.price = 1.3020
            pm.process_positions()

            # Verify 50% of 0.12 lots (0.06 lots) was booked
            assert len(closed_lots) == 1, "50% profit booking MUST trigger at 50% target"
            assert closed_lots[0] == 0.06, f"Expected 0.06 lots closed (50% of 0.12), got {closed_lots[0]}"
            assert pm._remaining_lots[999] == 0.06, f"Expected 0.06 lots remaining, got {pm._remaining_lots[999]}"
            assert 999 in pm._partial_tp_applied

            # Verify Stop Loss was moved to Cost-to-Cost (entry price: 1.3000)
            assert len(sl_modified_values) >= 1
            assert abs(sl_modified_values[-1] - 1.3000) < 1e-4, f"SL must be moved to entry price 1.3000, got {sl_modified_values[-1]}"

            # ── Step 3: Price moves further (+30 pips -> 1.3030) -> NO MORE PARTIAL CUTS ──
            broker.price = 1.3030
            pm.process_positions()
            assert len(closed_lots) == 1, "Remaining position must be held for full TP without further partial cuts"
            assert pm._remaining_lots[999] == 0.06

            # Verify recorded event in state database
            events = state.get_recent_partial_tp_events(limit=10)
            assert len(events) >= 1
            assert events[0]["trade_id"] == 999
            assert events[0]["closed_lot"] == 0.06
            assert events[0]["remaining_lot"] == 0.06

        finally:
            if pm is not None:
                pm.shutdown()
            state.close()


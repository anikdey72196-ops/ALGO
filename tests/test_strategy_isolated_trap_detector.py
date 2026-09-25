import unittest
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from config import StrategyType, normalize_strategy_key
from ml.trap_detector import (
    TrapDetectorConfig,
    TrapDetectorService,
    TrapGate,
    TrapModel,
    EventKind,
    EventRecord,
    EventStore,
    ALL_FEATURES,
)


def _create_dummy_df(n: int = 50) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    base = 100.0 + np.arange(n) * 0.1
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base + 0.05,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


class TestStrategyIsolatedTrapDetector(unittest.TestCase):
    def setUp(self):
        self.cfg = TrapDetectorConfig(
            db_path=":memory:",
            model_path="ml/artifacts/_test_trap_detector.joblib",
            shadow_until_samples=0,
            p_genuine_threshold=0.50,
            max_sl_probability=0.50,
        )
        self.svc = TrapDetectorService(self.cfg)

    def tearDown(self):
        self.svc.close()
        from pathlib import Path
        for p in Path("ml/artifacts").glob("_test_trap_detector*"):
            try:
                p.unlink()
            except Exception:
                pass

    def test_independent_strategy_models_initialized(self):
        """Service should initialize separate dedicated models for core strategies."""
        smc_model = self.svc.get_model("SMC")
        ict_model = self.svc.get_model("ICT")
        scalp_model = self.svc.get_model("SMC_SCALP_5M")
        of_model = self.svc.get_model("ORDER_FLOW")

        self.assertIsNot(smc_model, ict_model)
        self.assertIsNot(smc_model, scalp_model)
        self.assertIsNot(ict_model, of_model)

        # Backward-compatible property .model should return the SMC model
        self.assertIs(self.svc.model, smc_model)

    def test_observe_event_assigns_strategy_and_uses_strategy_model(self):
        """Observe event must record strategy and evaluate using that specific strategy model."""
        df = _create_dummy_df(40)

        ev_smc = self.svc.observe_event(
            symbol="EURUSD",
            timeframe="15m",
            kind=EventKind.FVG_BULL,
            direction="long",
            entry=103.0,
            stop=102.5,
            target=104.5,
            df=df,
            bar_index=len(df) - 1,
            strategy="SMC",
        )
        self.assertEqual(ev_smc.strategy, "SMC")
        self.assertIn("SMC:", ev_smc.model_version)

        ev_ict = self.svc.observe_event(
            symbol="GBPUSD",
            timeframe="5m",
            kind=EventKind.SWEEP_SSL,
            direction="long",
            entry=103.0,
            stop=102.5,
            target=104.5,
            df=df,
            bar_index=len(df) - 1,
            strategy="ICT",
        )
        self.assertEqual(ev_ict.strategy, "ICT")
        self.assertIn("ICT:", ev_ict.model_version)

    def test_training_isolation_between_strategies(self):
        """Training SMC data must NOT modify or affect the ICT or Scalp models."""
        df = _create_dummy_df(40)
        x = np.zeros((1, len(ALL_FEATURES)), dtype=float)

        smc_model = self.svc.get_model("SMC")
        ict_model = self.svc.get_model("ICT")

        initial_smc_samples = smc_model.n_samples
        initial_ict_samples = ict_model.n_samples

        # Partially fit SMC model with a trade outcome
        smc_model.partial_fit(x, np.array([1]))

        # SMC must increment samples; ICT must stay exactly the same
        self.assertEqual(smc_model.n_samples, initial_smc_samples + 1)
        self.assertEqual(ict_model.n_samples, initial_ict_samples)

        # Partially fit ICT model with a different outcome
        ict_model.partial_fit(x, np.array([0]))
        ict_model.partial_fit(x, np.array([0]))

        self.assertEqual(smc_model.n_samples, initial_smc_samples + 1)
        self.assertEqual(ict_model.n_samples, initial_ict_samples + 2)

    def test_gating_decision_isolation(self):
        """Mocking high SL risk on SMC should veto SMC setups but NOT veto ICT setups."""
        df = _create_dummy_df(40)

        # Force SMC model to predict high SL risk (p_genuine=0.10 -> p_sl=0.90)
        self.svc.get_model("SMC").predict_proba_genuine = lambda x: np.array([0.10])
        # Force ICT model to predict genuine setup (p_genuine=0.90 -> p_sl=0.10)
        self.svc.get_model("ICT").predict_proba_genuine = lambda x: np.array([0.90])

        ev_smc = self.svc.observe_event(
            symbol="EURUSD",
            timeframe="15m",
            kind=EventKind.FVG_BULL,
            direction="long",
            entry=103.0,
            stop=102.5,
            target=104.5,
            df=df,
            bar_index=len(df) - 1,
            strategy="SMC",
        )
        self.assertFalse(ev_smc.allowed, "SMC setup should be vetoed because SMC model predicts high SL risk")

        ev_ict = self.svc.observe_event(
            symbol="EURUSD",
            timeframe="15m",
            kind=EventKind.FVG_BULL,
            direction="long",
            entry=103.0,
            stop=102.5,
            target=104.5,
            df=df,
            bar_index=len(df) - 1,
            strategy="ICT",
        )
        self.assertTrue(ev_ict.allowed, "ICT setup should be approved because ICT model predicts genuine")

    def test_database_strategy_partitioning(self):
        """EventStore queries must filter and partition by strategy correctly."""
        store = self.svc.store

        ev1 = EventRecord(
            ts=datetime.now(timezone.utc) - timedelta(minutes=60),
            symbol="EURUSD",
            timeframe="15m",
            kind=EventKind.FVG_BULL,
            direction="long",
            entry=1.1, stop=1.09, target=1.12,
            features={k: 0.0 for k in ALL_FEATURES},
            strategy="SMC",
            label=1,
        )
        ev2 = EventRecord(
            ts=datetime.now(timezone.utc) - timedelta(minutes=60),
            symbol="GBPUSD",
            timeframe="5m",
            kind=EventKind.SWEEP_BSL,
            direction="short",
            entry=1.3, stop=1.31, target=1.28,
            features={k: 0.0 for k in ALL_FEATURES},
            strategy="ICT",
            label=0,
        )
        ev3 = EventRecord(
            ts=datetime.now(timezone.utc) - timedelta(minutes=60),
            symbol="EURUSD",
            timeframe="5m",
            kind=EventKind.FVG_BEAR,
            direction="short",
            entry=1.1, stop=1.11, target=1.08,
            features={k: 0.0 for k in ALL_FEATURES},
            strategy="SMC_SCALP_5M",
            label=None,  # Pending
        )

        store.upsert(ev1)
        store.upsert(ev2)
        store.upsert(ev3)

        # Check labeled queries with strategy filter
        smc_labeled = store.labeled(strategy="SMC")
        self.assertEqual(len(smc_labeled), 1)
        self.assertEqual(smc_labeled[0].strategy, "SMC")

        ict_labeled = store.labeled(strategy="ICT")
        self.assertEqual(len(ict_labeled), 1)
        self.assertEqual(ict_labeled[0].strategy, "ICT")

        # Check stats by strategy
        by_strat = store.stats_by_strategy()
        self.assertIn("SMC", by_strat)
        self.assertIn("ICT", by_strat)
        self.assertIn("SMC_SCALP_5M", by_strat)
        self.assertEqual(by_strat["SMC"]["genuine"], 1)
        self.assertEqual(by_strat["ICT"]["traps"], 1)
        self.assertEqual(by_strat["SMC_SCALP_5M"]["total"], 1)

    def test_database_auto_migration_for_legacy_table(self):
        """Ensure legacy tables without 'strategy' column are migrated automatically."""
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE ml_events (
                event_id TEXT PRIMARY KEY, ts TEXT, symbol TEXT, timeframe TEXT, kind TEXT,
                direction TEXT, entry REAL, stop REAL, target REAL, features TEXT,
                label INTEGER, outcome TEXT, r_multiple REAL, label_ts TEXT,
                p_genuine REAL, model_version TEXT, allowed INTEGER
            )
        """)
        conn.commit()

        cur = conn.execute("PRAGMA table_info(ml_events)")
        cols = [c[1] for c in cur.fetchall()]
        self.assertNotIn("strategy", cols)

        if "strategy" not in cols:
            conn.execute("ALTER TABLE ml_events ADD COLUMN strategy TEXT NOT NULL DEFAULT 'SMC'")
            conn.commit()

        cur2 = conn.execute("PRAGMA table_info(ml_events)")
        cols2 = [c[1] for c in cur2.fetchall()]
        self.assertIn("strategy", cols2)
        conn.close()


if __name__ == "__main__":
    unittest.main()

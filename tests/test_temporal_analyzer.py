"""
test_temporal_analyzer.py
==========================

Unit and Integration Tests for the Temporal & Session ML Model and Service.

Verifies:
1. Temporal feature extraction (cyclical day, hour, minute encodings, session indices).
2. Institutional trading session and Kill Zone identification.
3. EdgeTier assignment and risk scaling logic.
4. ML Classifier training and probability prediction.
5. Toxic window trade veto mechanism.
6. Service initialization, serialization, and report generation.
7. Web API temporal endpoints.
"""

import os
os.environ["USE_MOCK_BROKER"] = "true"
import unittest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from ml.temporal_analyzer import (
    TradingSession,
    EdgeTier,
    TemporalModelConfig,
    TemporalEdgeMLModel,
    TemporalEdgeService,
    TemporalVerdict,
    identify_trading_session,
    is_killzone_session,
    extract_temporal_features,
    calculate_bucket_metrics,
)
from web_app import app, bot_instance


class TestTemporalAnalyzer(unittest.TestCase):
    def setUp(self):
        self.config = TemporalModelConfig(
            db_path=":memory:",
            min_samples_to_train=10,
            min_bucket_samples=2,
            active_gating=True,
            veto_toxic=True,
        )

    def test_session_identification(self):
        """Verify UTC hour mappings to trading sessions."""
        # 03:00 UTC -> Asian Session
        dt_asia = datetime(2026, 9, 21, 3, 0, tzinfo=timezone.utc)
        self.assertEqual(identify_trading_session(dt_asia), TradingSession.ASIA)
        self.assertFalse(is_killzone_session(TradingSession.ASIA))

        # 06:30 UTC -> London Pre-Open
        dt_pre = datetime(2026, 9, 21, 6, 30, tzinfo=timezone.utc)
        self.assertEqual(identify_trading_session(dt_pre), TradingSession.LONDON_PRE)

        # 08:15 UTC -> London Open Killzone
        dt_london = datetime(2026, 9, 21, 8, 15, tzinfo=timezone.utc)
        self.assertEqual(identify_trading_session(dt_london), TradingSession.LONDON_OPEN)
        self.assertTrue(is_killzone_session(TradingSession.LONDON_OPEN))

        # 11:00 UTC -> London Midday
        dt_mid = datetime(2026, 9, 21, 11, 0, tzinfo=timezone.utc)
        self.assertEqual(identify_trading_session(dt_mid), TradingSession.LONDON_MID)

        # 13:00 UTC -> NY AM Killzone
        dt_ny_am = datetime(2026, 9, 21, 13, 0, tzinfo=timezone.utc)
        self.assertEqual(identify_trading_session(dt_ny_am), TradingSession.NY_AM)
        self.assertTrue(is_killzone_session(TradingSession.NY_AM))

        # 14:30 UTC -> NY Silver Bullet
        dt_sb = datetime(2026, 9, 21, 14, 30, tzinfo=timezone.utc)
        self.assertEqual(identify_trading_session(dt_sb), TradingSession.SILVER_BULLET)
        self.assertTrue(is_killzone_session(TradingSession.SILVER_BULLET))

        # 15:30 UTC -> London Close Killzone
        dt_close = datetime(2026, 9, 21, 15, 30, tzinfo=timezone.utc)
        self.assertEqual(identify_trading_session(dt_close), TradingSession.LONDON_CLOSE)
        self.assertTrue(is_killzone_session(TradingSession.LONDON_CLOSE))

        # 18:00 UTC -> NY PM
        dt_pm = datetime(2026, 9, 21, 18, 0, tzinfo=timezone.utc)
        self.assertEqual(identify_trading_session(dt_pm), TradingSession.NY_PM)

        # 22:00 UTC -> Off Hours
        dt_off = datetime(2026, 9, 21, 22, 0, tzinfo=timezone.utc)
        self.assertEqual(identify_trading_session(dt_off), TradingSession.OFF_HOURS)

    def test_feature_extraction(self):
        """Verify cyclical and categorical feature extraction."""
        dt = datetime(2026, 9, 21, 14, 15, tzinfo=timezone.utc)  # Monday 14:15 UTC
        feats = extract_temporal_features(dt, symbol="XAUUSD", strategy="SMC", planned_rr=2.5)

        self.assertEqual(feats["day_of_week"], 0.0)  # Monday
        self.assertEqual(feats["hour_utc"], 14.0)
        self.assertEqual(feats["minute"], 15.0)
        self.assertEqual(feats["is_killzone"], 1.0)
        self.assertEqual(feats["planned_rr"], 2.5)
        self.assertIn("day_sin", feats)
        self.assertIn("day_cos", feats)
        self.assertIn("hour_sin", feats)
        self.assertIn("hour_cos", feats)

    def test_bucket_metrics_calculation(self):
        """Verify financial metrics calculation across trade samples."""
        recs = [
            {"is_win": True, "is_loss": False, "pnl": 50.0, "r_multiple": 2.0},
            {"is_win": True, "is_loss": False, "pnl": 50.0, "r_multiple": 2.0},
            {"is_win": False, "is_loss": True, "pnl": -25.0, "r_multiple": -1.0},
        ]
        perf = calculate_bucket_metrics(recs, key="test", category="session", name="Test Session", min_bucket_samples=2)

        self.assertEqual(perf.total_trades, 3)
        self.assertEqual(perf.wins, 2)
        self.assertEqual(perf.losses, 1)
        self.assertAlmostEqual(perf.win_rate, 66.7, places=1)
        self.assertEqual(perf.total_pnl, 75.0)
        self.assertEqual(perf.profit_factor, 4.0)
        # With 3 samples (2 wins, 1 loss), Bayesian shrinkage (4/7 = 57.1%) is FAVORABLE
        self.assertEqual(perf.edge_tier, EdgeTier.FAVORABLE)

        # With 5 samples (4 wins, 1 loss), Bayesian shrinkage (6/9 = 66.7%) is PRIME_EDGE
        recs_prime = recs + [
            {"is_win": True, "is_loss": False, "pnl": 50.0, "r_multiple": 2.0},
            {"is_win": True, "is_loss": False, "pnl": 50.0, "r_multiple": 2.0},
        ]
        perf_prime = calculate_bucket_metrics(recs_prime, key="test_prime", category="session", name="Prime Session", min_bucket_samples=2)
        self.assertEqual(perf_prime.edge_tier, EdgeTier.PRIME_EDGE)

    def test_model_training_and_prediction(self):
        """Test model training on synthetic records and predictable edge tiers."""
        model = TemporalEdgeMLModel(self.config)

        # Generate synthetic records (15 winning in NY Silver Bullet, 15 losing in London Mid)
        synthetic_records = []
        for i in range(15):
            # Tuesday 14:15 UTC (Silver bullet win)
            synthetic_records.append({
                "timestamp": "2026-09-22T14:15:00+00:00",
                "symbol": "XAUUSD",
                "strategy": "SMC",
                "pnl": 20.0,
                "is_win": True,
                "is_loss": False,
                "planned_rr": 2.0,
                "r_multiple": 2.0,
            })
            # Tuesday 10:30 UTC (London mid loss)
            synthetic_records.append({
                "timestamp": "2026-09-22T10:30:00+00:00",
                "symbol": "XAUUSD",
                "strategy": "SMC",
                "pnl": -10.0,
                "is_win": False,
                "is_loss": True,
                "planned_rr": 2.0,
                "r_multiple": -1.0,
            })

        model.fit(synthetic_records)
        self.assertTrue(model.is_fitted)

        # Predict for Tuesday 14:15 (should be favorable or prime edge)
        v_win = model.predict("2026-09-22T14:15:00+00:00", symbol="XAUUSD", strategy="SMC")
        self.assertGreater(v_win.p_win, 0.50)
        self.assertTrue(v_win.allowed)
        self.assertGreaterEqual(v_win.risk_multiplier, 1.0)

        # Predict for Tuesday 10:30 (should be toxic avoid or unfavorable)
        v_loss = model.predict("2026-09-22T10:30:00+00:00", symbol="XAUUSD", strategy="SMC")
        self.assertGreater(v_loss.p_loss, 0.50)
        self.assertIn(v_loss.edge_tier, (EdgeTier.TOXIC_AVOID, EdgeTier.UNFAVORABLE))

    def test_service_ascii_report(self):
        """Test TemporalEdgeService report formatting."""
        svc = TemporalEdgeService(self.config)
        report = svc.generate_ascii_report()
        self.assertIn("INSTITUTIONAL TEMPORAL & SESSION PROFITABILITY ML REPORT", report)
        self.assertIn("DAY-OF-WEEK BREAKDOWN", report)
        self.assertIn("MARKET SESSION BREAKDOWN", report)

    def test_web_api_endpoints(self):
        """Test FastAPI endpoints for temporal ML analytics."""
        client = TestClient(app)

        # Test /api/ml/temporal/summary
        resp_summary = client.get("/api/ml/temporal/summary")
        self.assertEqual(resp_summary.status_code, 200)
        data = resp_summary.json()
        self.assertIn("most_profitable", data)
        self.assertIn("most_losing", data)

        # Test /api/ml/temporal/breakdown
        resp_breakdown = client.get("/api/ml/temporal/breakdown")
        self.assertEqual(resp_breakdown.status_code, 200)
        data_b = resp_breakdown.json()
        self.assertIn("by_day", data_b)
        self.assertIn("by_session", data_b)

        # Test /api/state contains temporal_ml_model
        resp_state = client.get("/api/state")
        self.assertEqual(resp_state.status_code, 200)
        state_data = resp_state.json()
        self.assertIn("temporal_ml_model", state_data)


if __name__ == "__main__":
    unittest.main()

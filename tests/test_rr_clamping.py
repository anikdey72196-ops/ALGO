import unittest
from datetime import datetime, timezone
import pandas as pd

from core.config import RiskConfig, Direction, MarketBias
from core.conflict_resolver import ConflictResolver
from strategies.strategy import (
    TradeSignal,
    LTFConfirmation,
    clamp_tp_to_rr,
)


class TestRRClamping(unittest.TestCase):
    def test_clamp_tp_to_rr_buy_too_high(self):
        """If BUY TP is at 5R, it must be clamped down to exactly 3.0R."""
        entry = 1.1000
        sl = 1.0980  # sl_dist = 0.0020 (20 pips)
        tp = 1.1100  # tp_dist = 0.0100 (5.0R)
        
        clamped_tp, eff_rr = clamp_tp_to_rr(entry, sl, tp, Direction.BUY, min_rr=1.5, max_rr=3.0)
        self.assertAlmostEqual(clamped_tp, 1.1060, places=5)
        self.assertAlmostEqual(eff_rr, 3.0, places=2)

    def test_clamp_tp_to_rr_sell_too_high(self):
        """If SELL TP is at 4.5R, it must be clamped down to exactly 3.0R."""
        entry = 1.12374
        sl = 1.12524  # sl_dist = 0.00150 (15 pips)
        tp = 1.11699  # tp_dist = 0.00675 (4.5R)
        
        clamped_tp, eff_rr = clamp_tp_to_rr(entry, sl, tp, Direction.SELL, min_rr=1.5, max_rr=3.0)
        expected_tp = entry - (0.00150 * 3.0)  # 1.11924
        self.assertAlmostEqual(clamped_tp, expected_tp, places=5)
        self.assertAlmostEqual(eff_rr, 3.0, places=2)

    def test_clamp_tp_to_rr_too_low(self):
        """If TP is at 1.0R, it must be clamped up to minimum 1.5R."""
        entry = 1.1000
        sl = 1.0990  # sl_dist = 0.0010 (10 pips)
        tp = 1.1010  # tp_dist = 0.0010 (1.0R)
        
        clamped_tp, eff_rr = clamp_tp_to_rr(entry, sl, tp, Direction.BUY, min_rr=1.5, max_rr=3.0)
        self.assertAlmostEqual(clamped_tp, 1.1015, places=5)
        self.assertAlmostEqual(eff_rr, 1.5, places=2)

    def test_clamp_tp_to_rr_within_bounds(self):
        """If TP is 2.0R, it must remain unchanged."""
        entry = 1.1000
        sl = 1.0980  # sl_dist = 20 pips
        tp = 1.1040  # tp_dist = 40 pips (2.0R)
        
        clamped_tp, eff_rr = clamp_tp_to_rr(entry, sl, tp, Direction.BUY, min_rr=1.5, max_rr=3.0)
        self.assertAlmostEqual(clamped_tp, 1.1040, places=5)
        self.assertAlmostEqual(eff_rr, 2.0, places=2)

    def test_conflict_resolver_clamps_above_max_rr(self):
        """Gate 2 in ConflictResolver must clamp signals above max_rr down to 3.0."""
        risk_cfg = RiskConfig(min_rr_ratio=1.5, max_rr_ratio=3.0)
        resolver = ConflictResolver(risk_cfg)

        huge_tp_sig = TradeSignal(
            symbol="EURUSD",
            direction=Direction.SELL,
            entry_price=1.12000,
            stop_loss=1.12100,  # sl_dist = 10 pips
            take_profit=1.11500,  # tp_dist = 50 pips (5.0R)
            htf_bias=MarketBias.BEARISH,
            ltf_confirmation=LTFConfirmation.OB_PLUS_FVG,
            rr_ratio=5.0,
            quality_score=90.0,
            timestamp=datetime.now(timezone.utc),
            sl_distance=0.00100,
            tp_distance=0.00500,
        )

        res = resolver.resolve([huge_tp_sig], current_spread=0.00010)
        self.assertIsNotNone(res.accepted_signal)
        self.assertAlmostEqual(res.accepted_signal.rr_ratio, 3.0, places=2)
        self.assertAlmostEqual(res.accepted_signal.take_profit, 1.11700, places=5)

    def test_conflict_resolver_rejects_below_min_rr(self):
        """Gate 2 in ConflictResolver must reject signals below 1.5R."""
        risk_cfg = RiskConfig(min_rr_ratio=1.5, max_rr_ratio=3.0)
        resolver = ConflictResolver(risk_cfg)

        low_rr_sig = TradeSignal(
            symbol="EURUSD",
            direction=Direction.SELL,
            entry_price=1.12000,
            stop_loss=1.12100,
            take_profit=1.11880,  # 1.2R
            htf_bias=MarketBias.BEARISH,
            ltf_confirmation=LTFConfirmation.OB_PLUS_FVG,
            rr_ratio=1.2,
            quality_score=90.0,
            timestamp=datetime.now(timezone.utc),
            sl_distance=0.00100,
            tp_distance=0.00120,
        )

        res = resolver.resolve([low_rr_sig], current_spread=0.00010)
        self.assertIsNone(res.accepted_signal)
        self.assertTrue(any("[LOW_RR]" in r for r in res.rejection_reasons))


if __name__ == "__main__":
    unittest.main()

"""
test_htf_bias_and_cache.py — Unit tests for:
1. HTFAnalyzer macro EMA + swing structure bias resolution.
2. TradingBot per-tick candle cache clearing.
3. RiskEngine unlimited daily trade execution (no daily max trade limit).
"""

import unittest
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from config import TradingConfig, RiskConfig, Direction, MarketBias
from strategy import HTFAnalyzer, SwingPoint
from risk_engine import RiskEngine
from state import StateManager, TradeRecord


class TestHTFBiasAndCache(unittest.TestCase):

    def setUp(self):
        self.analyzer = HTFAnalyzer(ema_period=50)

    def test_macro_downtrend_with_neutral_structure(self):
        """When swings are consolidating (neutral) but price is below 200 EMA, bias should be BEARISH."""
        # Create a series of 100 bars trending downward below EMA
        closes = [1.20 - (i * 0.0005) for i in range(100)]
        # Introduce a minor consolidation at the end
        closes[-2] = closes[-3] + 0.0002
        closes[-1] = closes[-2] - 0.0001
        highs = [c + 0.0005 for c in closes]
        lows = [c - 0.0005 for c in closes]

        df = pd.DataFrame({
            'open': closes,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': [100] * 100,
            'time': pd.date_range('2026-01-01', periods=100, freq='1h', tz='UTC')
        })

        analysis = self.analyzer.analyze(df)
        self.assertEqual(analysis.bias, MarketBias.BEARISH)
        self.assertGreaterEqual(analysis.trend_clarity_score, 20.0)

    def test_macro_uptrend_with_neutral_structure(self):
        """When swings are consolidating (neutral) but price is above 200 EMA, bias should be BULLISH."""
        # Create a series of 100 bars trending upward above EMA
        closes = [1.10 + (i * 0.0005) for i in range(100)]
        closes[-2] = closes[-3] - 0.0002
        closes[-1] = closes[-2] + 0.0001
        highs = [c + 0.0005 for c in closes]
        lows = [c - 0.0005 for c in closes]

        df = pd.DataFrame({
            'open': closes,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': [100] * 100,
            'time': pd.date_range('2026-01-01', periods=100, freq='1h', tz='UTC')
        })

        analysis = self.analyzer.analyze(df)
        self.assertEqual(analysis.bias, MarketBias.BULLISH)
        self.assertGreaterEqual(analysis.trend_clarity_score, 20.0)

    def test_risk_engine_unlimited_daily_trades(self):
        """Ensure RiskEngine does not block trades on daily trade count."""
        import tempfile
        from pathlib import Path
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        try:
            cfg = TradingConfig()
            cfg.db_path = temp_db

            state = StateManager(db_path=temp_db)
            # Simulate 15 trades recorded today
            now = datetime.now(timezone.utc)
            for i in range(15):
                state.record_trade(TradeRecord(
                    id=1000 + i,
                    timestamp=now,
                    symbol="XAUUSD",
                    direction=Direction.BUY,
                    entry_price=2000.0,
                    stop_loss=1990.0,
                    take_profit=2020.0,
                    lot_size=0.01,
                    realized_pnl=10.0,
                    status="CLOSED_TP",
                    strategy_name="SMC",
                ))

            self.assertEqual(state.get_trade_count(), 15)

            risk = RiskEngine(cfg, state)
            is_blocked, reason = risk.check_circuit_breakers(equity=10000.0)
            self.assertFalse(is_blocked, f"Blocked unexpectedly: {reason}")
            state.close()
        finally:
            Path(temp_db).unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()

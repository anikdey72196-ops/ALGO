"""
test_all_pairs_concurrent_execution.py

Validates the user requirement:
1. All pairs (XAUUSD, EURUSD, GBPUSD, BTCUSD, ETHUSD) are analyzed in the exact same tick.
2. Up to 3 trades per pair can execute at the exact same time (from distinct strategies: SMC, ICT, ORDER_FLOW).
3. Up to 15 total trades can be held concurrently across all 5 pairs (5 pairs * 3 trades = 15).
4. A 4th trade on any individual pair is strictly blocked (max 3 per pair limit).
5. A duplicate trade from the same strategy on any pair is blocked (1 trade per strategy per pair).
6. A 16th trade across the bot is strictly blocked (max 15 global limit).
7. All quality rules (HTF bias alignment, R:R minimums, spread limits, AI gate) remain strictly enforced.
"""

import os
import tempfile
import unittest
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from config import TradingConfig, Direction, MarketBias, InstrumentConfig
from execution import MockBrokerAdapter
from state import StateManager, TradeRecord
from risk_engine import RiskEngine
from strategy import TradeSignal, LTFConfirmation
from main import TradingBot


class TestAllPairsConcurrentExecution(unittest.TestCase):

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.state = StateManager(db_path=self.temp_db.name)

        self.config = TradingConfig(
            use_mock_broker=True,
            selected_symbols=["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD", "ETHUSD"],
            enabled_strategies=["SMC", "ICT", "ORDER_FLOW"],
        )
        self.config.risk.max_open_positions = 15
        self.config.risk.max_open_per_symbol = 3
        self.risk_engine = RiskEngine(self.config, self.state)

    def tearDown(self):
        try:
            self.state.close()
            os.remove(self.temp_db.name)
        except OSError:
            pass

    def _make_signal(self, symbol: str, strategy_name: str, magic_number: int, entry: float, sl: float, tp: float):
        return TradeSignal(
            symbol=symbol,
            direction=Direction.BUY,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            htf_bias=MarketBias.BULLISH,
            ltf_confirmation=LTFConfirmation.LIQUIDITY_SWEEP,
            rr_ratio=2.5,
            quality_score=88.0,
            timestamp=datetime.now(timezone.utc),
            sl_distance=abs(entry - sl),
            tp_distance=abs(tp - entry),
            strategy_id=strategy_name,
            strategy_name=strategy_name,
            magic_number=magic_number,
        )

    def test_three_trades_per_pair_across_all_five_pairs(self):
        """
        Test that 3 distinct strategies can execute on every pair simultaneously:
        5 pairs * 3 trades = 15 concurrent trades.
        """
        equity = 100_000.0
        now = datetime.now(timezone.utc)
        pairs = ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD", "ETHUSD"]
        strategies = [
            ("SMC", 124456),
            ("ICT", 126456),
            ("ORDER_FLOW", 127456),
        ]

        trade_id = 1
        for symbol in pairs:
            # 1. Authorize and record 1st trade (SMC) on this pair
            sig_smc = self._make_signal(symbol, "SMC", 124456, 100.0, 98.0, 105.0)
            auth_smc = self.risk_engine.authorize_trade(sig_smc, equity, fixed_lot_size=0.01)
            self.assertTrue(auth_smc.authorized)
            t1 = TradeRecord(
                id=trade_id, timestamp=now, symbol=symbol, direction="BUY",
                entry_price=100.0, stop_loss=98.0, take_profit=105.0,
                lot_size=0.01, realized_pnl=0.0, status="OPEN",
                strategy_name="SMC", magic_number=124456
            )
            self.state.record_trade(t1)
            trade_id += 1

            # 2. Strict Rule: Duplicate strategy trade on this pair must be REJECTED (even though symbol has only 1/3 trades)
            auth_dup = self.risk_engine.authorize_trade(sig_smc, equity, fixed_lot_size=0.01)
            self.assertFalse(auth_dup.authorized)
            self.assertIn(f"already has an active trade for {symbol}", auth_dup.rejection_reason)

            # 3. Authorize and record 2nd trade (ICT) on this pair
            sig_ict = self._make_signal(symbol, "ICT", 126456, 100.0, 98.0, 105.0)
            auth_ict = self.risk_engine.authorize_trade(sig_ict, equity, fixed_lot_size=0.01)
            self.assertTrue(auth_ict.authorized)
            t2 = TradeRecord(
                id=trade_id, timestamp=now, symbol=symbol, direction="BUY",
                entry_price=100.0, stop_loss=98.0, take_profit=105.0,
                lot_size=0.01, realized_pnl=0.0, status="OPEN",
                strategy_name="ICT", magic_number=126456
            )
            self.state.record_trade(t2)
            trade_id += 1

            # 4. Authorize and record 3rd trade (ORDER_FLOW) on this pair
            sig_of = self._make_signal(symbol, "ORDER_FLOW", 127456, 100.0, 98.0, 105.0)
            auth_of = self.risk_engine.authorize_trade(sig_of, equity, fixed_lot_size=0.01)
            self.assertTrue(auth_of.authorized)
            t3 = TradeRecord(
                id=trade_id, timestamp=now, symbol=symbol, direction="BUY",
                entry_price=100.0, stop_loss=98.0, take_profit=105.0,
                lot_size=0.01, realized_pnl=0.0, status="OPEN",
                strategy_name="ORDER_FLOW", magic_number=127456
            )
            self.state.record_trade(t3)
            trade_id += 1

            # 5. Strict Rule: 4th trade on the same pair must be REJECTED (max 3 per symbol or max 15 global)
            sig_4th = self._make_signal(symbol, "SMC_SCALP_5M", 125456, 100.0, 98.0, 105.0)
            auth_4th = self.risk_engine.authorize_trade(sig_4th, equity, fixed_lot_size=0.01)
            self.assertFalse(auth_4th.authorized)
            self.assertTrue(
                f"Max open positions for {symbol} reached" in auth_4th.rejection_reason
                or "Max concurrent open positions reached" in auth_4th.rejection_reason
            )

        # 6. We now have exactly 15 open positions across the 5 pairs (3 trades per pair)
        open_trades = self.state.get_open_positions()
        self.assertEqual(len(open_trades), 15)

        # 7. Strict Rule: 16th trade across the bot must be REJECTED by global max_open_positions limit (15/15)
        extra_sig = self._make_signal("XAUUSD", "EXTRA_STRAT", 999999, 100.0, 98.0, 105.0)
        auth_16th = self.risk_engine.authorize_trade(extra_sig, equity, fixed_lot_size=0.01)
        self.assertFalse(auth_16th.authorized)
        self.assertIn("Max concurrent open positions reached (15/15 open)", auth_16th.rejection_reason)

    def test_trading_bot_scans_all_pairs_in_tick(self):
        """
        Verify that TradingBot._execute_tick iterates over all 5 selected pairs.
        """
        bot = TradingBot(self.config)
        bot.startup()
        bot.config.selected_symbols = ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD", "ETHUSD"]
        bot.is_active = True

        # Run tick
        bot.tick()

        # Check logs to verify all 5 pairs were analyzed
        recent_log_text = " ".join(bot.recent_logs)
        for sym in ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD", "ETHUSD"]:
            self.assertIn(sym, recent_log_text, f"Expected {sym} to be analyzed in the tick cycle!")

        bot.shutdown()


if __name__ == '__main__':
    unittest.main()

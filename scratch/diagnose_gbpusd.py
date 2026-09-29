import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import TradingBot
from core.config import TradingConfig

print("=== RUNNING TRADINGBOT TICK SCAN ON GBPUSD ===")
config = TradingConfig()
bot = TradingBot(config)
bot.broker.connect()

# Set up to analyze GBPUSD
print("Active pairs in config:")
for p in ["pair1", "pair2", "pair3"]:
    cfg = getattr(bot.config, p, None)
    print(f"  {p}: {cfg.symbol} (enabled={cfg.enabled}, lot={cfg.fixed_lot_size}, sl={cfg.fixed_sl_pips})")

print(f"selected_symbols: {bot.config.selected_symbols}")

# Test _get_ohlcv for GBPUSD
df_1h = bot._get_ohlcv("GBPUSD", "1H", 200)
df_15m = bot._get_ohlcv("GBPUSD", "15m", 200)
df_5m = bot._get_ohlcv("GBPUSD", "5m", 200)
print(f"Candles fetched for GBPUSD: 1H={len(df_1h) if df_1h is not None else 0}, 15m={len(df_15m) if df_15m is not None else 0}, 5m={len(df_5m) if df_5m is not None else 0}")

# Test quote
quote = bot.broker.get_current_price("GBPUSD")
print(f"Quote for GBPUSD: {quote}")

# Test strategy evaluation
from core.config import get_instrument
inst = get_instrument(bot.config, "GBPUSD")
htf_analysis = bot.strategy.htf_analyzer.analyze(df_1h)
print(f"HTF Analysis for GBPUSD: Bias={htf_analysis.bias}, EMA={htf_analysis.ema_value}")

signals = bot.strategy.evaluate_all(
    symbol="GBPUSD",
    htf_data=df_1h,
    ltf_data=df_15m,
    bars_15m=df_15m,
    bars_1h=df_1h,
    bars_5m=df_5m,
    instrument=inst,
    current_spread=quote.spread if quote else 0.00015,
    fixed_sl_pips=bot.config.pair3.fixed_sl_pips,
    htf_analysis=htf_analysis,
    enabled_strategies=bot.config.enabled_strategies,
)
print(f"Signals for GBPUSD across enabled strategies {bot.config.enabled_strategies}: {len(signals)}")
for s in signals:
    print(f"  Signal: {s.strategy_name} | {s.direction.value} | Entry={s.entry_price} | SL={s.stop_loss} | TP={s.take_profit} | R:R={s.rr_ratio:.2f}")

bot.broker.disconnect()
bot.state.close()

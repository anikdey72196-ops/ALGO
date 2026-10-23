import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

try:
    from core.config import InstrumentConfig
except ImportError:
    try:
        from config import InstrumentConfig
    except ImportError:
        pass

def generate_trending_ohlcv(
    symbol: str,
    timeframe: str,
    bars: int = 300,
    start_price: float = 1.1000,
    trend_direction: str = "bullish",  # or "bearish"
    volatility: float = 0.0003,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame with a clear trend and realistic
    micro-structure (pullbacks, swing highs/lows, occasional wicks).
    
    Returns DataFrame with columns: ['time', 'open', 'high', 'low', 'close', 'volume']
    - time: datetime in UTC
    - OHLC: float prices
    - volume: int tick volume
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Parse timeframe
    tf_map = {
        '1m': timedelta(minutes=1),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '1H': timedelta(hours=1),
        '4H': timedelta(hours=4),
        '1D': timedelta(days=1),
    }
    bar_duration = tf_map.get(timeframe, timedelta(hours=1))
    
    # Generate times
    end_time = datetime.now(timezone.utc)
    start_time = end_time - (bars - 1) * bar_duration
    times = [start_time + i * bar_duration for i in range(bars)]
    
    drift = volatility * 0.3 if trend_direction == "bullish" else -volatility * 0.3
    
    # State tracking
    current_price = start_price
    data = []
    
    # Pullback parameters
    bars_until_pullback = np.random.randint(20, 41)
    pullback_remaining = 0
    
    for i in range(bars):
        # Determine current trend
        is_pullback = False
        if pullback_remaining > 0:
            is_pullback = True
            pullback_remaining -= 1
        else:
            bars_until_pullback -= 1
            if bars_until_pullback <= 0:
                pullback_remaining = np.random.randint(5, 11)
                bars_until_pullback = np.random.randint(20, 41)
        
        # Effective drift
        current_drift = -drift if is_pullback else drift
        
        # Open price
        if i == 0:
            open_price = current_price
        else:
            # Previous close with occasional small gap
            open_price = data[-1]['close']
            if np.random.random() < 0.1: # 10% chance of small gap
                open_price += np.random.normal(0, volatility * 0.2)
        
        # Direction of candle
        # 60% of candles align with effective drift direction
        prob_up = 0.6 if current_drift > 0 else 0.4
        is_up_candle = np.random.random() < prob_up
        
        body_size = abs(np.random.normal(0, volatility))
        
        if is_up_candle:
            close_price = open_price + body_size
        else:
            close_price = open_price - body_size
            
        # Add wicks
        high_price = max(open_price, close_price) + abs(np.random.normal(0, volatility * 0.5))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, volatility * 0.5))
        
        # Base volume
        volume = np.random.randint(100, 5000)
        
        # Increase volume near reversals (pullback start/end)
        if pullback_remaining == 0 and bars_until_pullback <= 2:
            volume = int(volume * 1.5)
        elif pullback_remaining > 0 and pullback_remaining <= 2:
            volume = int(volume * 1.5)
            
        data.append({
            'time': times[i],
            'open': round(open_price, 5),
            'high': round(high_price, 5),
            'low': round(low_price, 5),
            'close': round(close_price, 5),
            'volume': volume
        })
        
    df = pd.DataFrame(data)
    return df

def generate_mock_price_quote(
    symbol: str,
    base_price: float = 1.10000,
    spread_points: float = 1.2,
    pip_size: float = 0.0001,
) -> dict:
    """Return a dict with keys: symbol, bid, ask, spread, timestamp."""
    spread_value = spread_points * (pip_size / 10) # assuming spread in points (pipettes)
    half_spread = spread_value / 2
    
    bid = base_price - half_spread
    ask = base_price + half_spread
    
    return {
        'symbol': symbol,
        'bid': round(bid, 5),
        'ask': round(ask, 5),
        'spread': spread_points,
        'timestamp': datetime.now(timezone.utc)
    }

def get_demo_datasets() -> dict[str, dict[str, pd.DataFrame]]:
    """
    Return pre-built datasets for testing.
    Returns: {
        'XAUUSD': {'1H': df_htf, '15m': df_ltf},
    }
    XAUUSD is generated with a bullish trending bias (gold uptrend).
    Volatility is scaled for gold's typical $15-25/hour moves.
    """
    datasets = {}

    # XAUUSD Bullish — Gold typically trends well on 1H/15m
    xauusd_1h = generate_trending_ohlcv(
        'XAUUSD', '1H', bars=300, start_price=2550.00,
        trend_direction='bullish', volatility=2.5, seed=42,
    )
    xauusd_15m = generate_trending_ohlcv(
        'XAUUSD', '15m', bars=1200, start_price=2550.00,
        trend_direction='bullish', volatility=1.2, seed=43,
    )
    xauusd_4h = generate_trending_ohlcv('XAUUSD', '4H', bars=150, start_price=2550.00, trend_direction='bullish', volatility=4.0, seed=42)
    xauusd_1d = generate_trending_ohlcv('XAUUSD', '1D', bars=100, start_price=2550.00, trend_direction='bullish', volatility=8.0, seed=42)
    datasets['XAUUSD'] = {'1H': xauusd_1h, '15m': xauusd_15m, '4H': xauusd_4h, '1D': xauusd_1d}

    # EURUSD Bullish
    eurusd_1h = generate_trending_ohlcv('EURUSD', '1H', bars=300, start_price=1.0850, trend_direction='bullish', volatility=0.0003, seed=44)
    eurusd_15m = generate_trending_ohlcv('EURUSD', '15m', bars=1200, start_price=1.0850, trend_direction='bullish', volatility=0.00015, seed=45)
    eurusd_4h = generate_trending_ohlcv('EURUSD', '4H', bars=150, start_price=1.0850, trend_direction='bullish', volatility=0.0006, seed=44)
    eurusd_1d = generate_trending_ohlcv('EURUSD', '1D', bars=100, start_price=1.0850, trend_direction='bullish', volatility=0.0012, seed=44)
    datasets['EURUSD'] = {'1H': eurusd_1h, '15m': eurusd_15m, '4H': eurusd_4h, '1D': eurusd_1d}

    # GBPUSD Bearish
    gbpusd_1h = generate_trending_ohlcv('GBPUSD', '1H', bars=300, start_price=1.2850, trend_direction='bearish', volatility=0.0004, seed=46)
    gbpusd_15m = generate_trending_ohlcv('GBPUSD', '15m', bars=1200, start_price=1.2850, trend_direction='bearish', volatility=0.0002, seed=47)
    gbpusd_4h = generate_trending_ohlcv('GBPUSD', '4H', bars=150, start_price=1.2850, trend_direction='bearish', volatility=0.0008, seed=46)
    gbpusd_1d = generate_trending_ohlcv('GBPUSD', '1D', bars=100, start_price=1.2850, trend_direction='bearish', volatility=0.0016, seed=46)
    datasets['GBPUSD'] = {'1H': gbpusd_1h, '15m': gbpusd_15m, '4H': gbpusd_4h, '1D': gbpusd_1d}

    # BTCUSD Bullish
    btcusd_1h = generate_trending_ohlcv('BTCUSD', '1H', bars=300, start_price=65000.0, trend_direction='bullish', volatility=150.0, seed=48)
    btcusd_15m = generate_trending_ohlcv('BTCUSD', '15m', bars=1200, start_price=65000.0, trend_direction='bullish', volatility=80.0, seed=49)
    btcusd_4h = generate_trending_ohlcv('BTCUSD', '4H', bars=150, start_price=65000.0, trend_direction='bullish', volatility=250.0, seed=48)
    btcusd_1d = generate_trending_ohlcv('BTCUSD', '1D', bars=100, start_price=65000.0, trend_direction='bullish', volatility=500.0, seed=48)
    datasets['BTCUSD'] = {'1H': btcusd_1h, '15m': btcusd_15m, '4H': btcusd_4h, '1D': btcusd_1d}

    # ETHUSD Bullish
    ethusd_1h = generate_trending_ohlcv('ETHUSD', '1H', bars=300, start_price=3200.0, trend_direction='bullish', volatility=12.0, seed=50)
    ethusd_15m = generate_trending_ohlcv('ETHUSD', '15m', bars=1200, start_price=3200.0, trend_direction='bullish', volatility=6.0, seed=51)
    ethusd_4h = generate_trending_ohlcv('ETHUSD', '4H', bars=150, start_price=3200.0, trend_direction='bullish', volatility=20.0, seed=50)
    ethusd_1d = generate_trending_ohlcv('ETHUSD', '1D', bars=100, start_price=3200.0, trend_direction='bullish', volatility=40.0, seed=50)
    datasets['ETHUSD'] = {'1H': ethusd_1h, '15m': ethusd_15m, '4H': ethusd_4h, '1D': ethusd_1d}

    # AUDJPY Bullish
    audjpy_1h = generate_trending_ohlcv('AUDJPY', '1H', bars=300, start_price=98.50, trend_direction='bullish', volatility=0.15, seed=52)
    audjpy_15m = generate_trending_ohlcv('AUDJPY', '15m', bars=1200, start_price=98.50, trend_direction='bullish', volatility=0.08, seed=53)
    audjpy_4h = generate_trending_ohlcv('AUDJPY', '4H', bars=150, start_price=98.50, trend_direction='bullish', volatility=0.25, seed=52)
    audjpy_1d = generate_trending_ohlcv('AUDJPY', '1D', bars=100, start_price=98.50, trend_direction='bullish', volatility=0.50, seed=52)
    datasets['AUDJPY'] = {'1H': audjpy_1h, '15m': audjpy_15m, '4H': audjpy_4h, '1D': audjpy_1d}

    # EURJPY Bullish
    eurjpy_1h = generate_trending_ohlcv('EURJPY', '1H', bars=300, start_price=163.20, trend_direction='bullish', volatility=0.20, seed=54)
    eurjpy_15m = generate_trending_ohlcv('EURJPY', '15m', bars=1200, start_price=163.20, trend_direction='bullish', volatility=0.10, seed=55)
    eurjpy_4h = generate_trending_ohlcv('EURJPY', '4H', bars=150, start_price=163.20, trend_direction='bullish', volatility=0.35, seed=54)
    eurjpy_1d = generate_trending_ohlcv('EURJPY', '1D', bars=100, start_price=163.20, trend_direction='bullish', volatility=0.60, seed=54)
    datasets['EURJPY'] = {'1H': eurjpy_1h, '15m': eurjpy_15m, '4H': eurjpy_4h, '1D': eurjpy_1d}

    # USDCHF Bearish
    usdchf_1h = generate_trending_ohlcv('USDCHF', '1H', bars=300, start_price=0.8950, trend_direction='bearish', volatility=0.0003, seed=56)
    usdchf_15m = generate_trending_ohlcv('USDCHF', '15m', bars=1200, start_price=0.8950, trend_direction='bearish', volatility=0.00015, seed=57)
    usdchf_4h = generate_trending_ohlcv('USDCHF', '4H', bars=150, start_price=0.8950, trend_direction='bearish', volatility=0.0006, seed=56)
    usdchf_1d = generate_trending_ohlcv('USDCHF', '1D', bars=100, start_price=0.8950, trend_direction='bearish', volatility=0.0012, seed=56)
    datasets['USDCHF'] = {'1H': usdchf_1h, '15m': usdchf_15m, '4H': usdchf_4h, '1D': usdchf_1d}

    # CADCHF Bearish
    cadchf_1h = generate_trending_ohlcv('CADCHF', '1H', bars=300, start_price=0.6550, trend_direction='bearish', volatility=0.00025, seed=58)
    cadchf_15m = generate_trending_ohlcv('CADCHF', '15m', bars=1200, start_price=0.6550, trend_direction='bearish', volatility=0.00012, seed=59)
    cadchf_4h = generate_trending_ohlcv('CADCHF', '4H', bars=150, start_price=0.6550, trend_direction='bearish', volatility=0.0005, seed=58)
    cadchf_1d = generate_trending_ohlcv('CADCHF', '1D', bars=100, start_price=0.6550, trend_direction='bearish', volatility=0.0010, seed=58)
    datasets['CADCHF'] = {'1H': cadchf_1h, '15m': cadchf_15m, '4H': cadchf_4h, '1D': cadchf_1d}

    # NZDCAD Bullish
    nzdcad_1h = generate_trending_ohlcv('NZDCAD', '1H', bars=300, start_price=0.8350, trend_direction='bullish', volatility=0.0003, seed=60)
    nzdcad_15m = generate_trending_ohlcv('NZDCAD', '15m', bars=1200, start_price=0.8350, trend_direction='bullish', volatility=0.00015, seed=61)
    nzdcad_4h = generate_trending_ohlcv('NZDCAD', '4H', bars=150, start_price=0.8350, trend_direction='bullish', volatility=0.0006, seed=60)
    nzdcad_1d = generate_trending_ohlcv('NZDCAD', '1D', bars=100, start_price=0.8350, trend_direction='bullish', volatility=0.0012, seed=60)
    datasets['NZDCAD'] = {'1H': nzdcad_1h, '15m': nzdcad_15m, '4H': nzdcad_4h, '1D': nzdcad_1d}

    # NZDCHF Bearish
    nzdchf_1h = generate_trending_ohlcv('NZDCHF', '1H', bars=300, start_price=0.5480, trend_direction='bearish', volatility=0.00025, seed=62)
    nzdchf_15m = generate_trending_ohlcv('NZDCHF', '15m', bars=1200, start_price=0.5480, trend_direction='bearish', volatility=0.00012, seed=63)
    nzdchf_4h = generate_trending_ohlcv('NZDCHF', '4H', bars=150, start_price=0.5480, trend_direction='bearish', volatility=0.0005, seed=62)
    nzdchf_1d = generate_trending_ohlcv('NZDCHF', '1D', bars=100, start_price=0.5480, trend_direction='bearish', volatility=0.0010, seed=62)
    datasets['NZDCHF'] = {'1H': nzdchf_1h, '15m': nzdchf_15m, '4H': nzdchf_4h, '1D': nzdchf_1d}

    # NZDUSD Bullish
    nzdusd_1h = generate_trending_ohlcv('NZDUSD', '1H', bars=300, start_price=0.6120, trend_direction='bullish', volatility=0.0003, seed=64)
    nzdusd_15m = generate_trending_ohlcv('NZDUSD', '15m', bars=1200, start_price=0.6120, trend_direction='bullish', volatility=0.00015, seed=65)
    nzdusd_4h = generate_trending_ohlcv('NZDUSD', '4H', bars=150, start_price=0.6120, trend_direction='bullish', volatility=0.0006, seed=64)
    nzdusd_1d = generate_trending_ohlcv('NZDUSD', '1D', bars=100, start_price=0.6120, trend_direction='bullish', volatility=0.0012, seed=64)
    datasets['NZDUSD'] = {'1H': nzdusd_1h, '15m': nzdusd_15m, '4H': nzdusd_4h, '1D': nzdusd_1d}

    return datasets


def get_demo_dataset_for_symbol(symbol: str, timeframe: str = "1H", bars: int = 300) -> pd.DataFrame:
    """Dynamically return or synthesize OHLCV demo data for any requested symbol."""
    sym_u = str(symbol).strip().upper()
    datasets = get_demo_datasets()
    if sym_u in datasets and timeframe in datasets[sym_u]:
        return datasets[sym_u][timeframe]

    is_jpy = "JPY" in sym_u
    is_gold = "XAU" in sym_u or "GOLD" in sym_u
    is_crypto = "BTC" in sym_u or "ETH" in sym_u
    start_p = 100.0 if is_jpy else (2500.0 if is_gold else (50000.0 if is_crypto else 1.0))
    vol = 0.15 if is_jpy else (2.0 if is_gold else (100.0 if is_crypto else 0.0003))

    return generate_trending_ohlcv(sym_u, timeframe, bars=bars, start_price=start_p, volatility=vol)



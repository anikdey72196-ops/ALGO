import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

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
    datasets['XAUUSD'] = {'1H': xauusd_1h, '15m': xauusd_15m}

    # EURUSD Bullish
    eurusd_1h = generate_trending_ohlcv('EURUSD', '1H', bars=300, start_price=1.0850, trend_direction='bullish', volatility=0.0003, seed=44)
    eurusd_15m = generate_trending_ohlcv('EURUSD', '15m', bars=1200, start_price=1.0850, trend_direction='bullish', volatility=0.00015, seed=45)
    datasets['EURUSD'] = {'1H': eurusd_1h, '15m': eurusd_15m}

    # GBPUSD Bearish
    gbpusd_1h = generate_trending_ohlcv('GBPUSD', '1H', bars=300, start_price=1.2850, trend_direction='bearish', volatility=0.0004, seed=46)
    gbpusd_15m = generate_trending_ohlcv('GBPUSD', '15m', bars=1200, start_price=1.2850, trend_direction='bearish', volatility=0.0002, seed=47)
    datasets['GBPUSD'] = {'1H': gbpusd_1h, '15m': gbpusd_15m}

    # BTCUSD Bullish
    btcusd_1h = generate_trending_ohlcv('BTCUSD', '1H', bars=300, start_price=65000.0, trend_direction='bullish', volatility=150.0, seed=48)
    btcusd_15m = generate_trending_ohlcv('BTCUSD', '15m', bars=1200, start_price=65000.0, trend_direction='bullish', volatility=80.0, seed=49)
    datasets['BTCUSD'] = {'1H': btcusd_1h, '15m': btcusd_15m}

    # ETHUSD Bullish
    ethusd_1h = generate_trending_ohlcv('ETHUSD', '1H', bars=300, start_price=3200.0, trend_direction='bullish', volatility=12.0, seed=50)
    ethusd_15m = generate_trending_ohlcv('ETHUSD', '15m', bars=1200, start_price=3200.0, trend_direction='bullish', volatility=6.0, seed=51)
    datasets['ETHUSD'] = {'1H': ethusd_1h, '15m': ethusd_15m}

    return datasets


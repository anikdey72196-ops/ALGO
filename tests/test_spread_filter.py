import pytest
from core.news_filter import NewsFilter, SpreadFilterResult
from core.config import TradingConfig, get_instrument, PairSettings


def test_spread_filter_normal_broker_spread():
    """Verify that normal XM broker spreads (e.g. 2.4 pips / 0.00024) pass without being blocked."""
    config = TradingConfig()
    eurusd = get_instrument(config, "EURUSD")
    assert eurusd.avg_spread_points >= 15.0  # Should be points, not pips
    
    point_size = 10 ** -eurusd.digits
    avg_spread = eurusd.avg_spread_points * point_size
    current_spread = 0.00023999999999979593  # User's exact live XM spread (~2.4 pips)
    
    res = NewsFilter.check_spread(
        current_spread=current_spread,
        avg_spread=avg_spread,
        max_spread_multiple=2.0,
    )
    assert not res.blocked
    assert res.current_spread == current_spread


def test_spread_filter_excessive_spread_blocking_and_formatting():
    """Verify that truly excessive spreads (e.g. spike to 10 pips) are blocked with clean formatting."""
    config = TradingConfig()
    eurusd = get_instrument(config, "EURUSD")
    point_size = 10 ** -eurusd.digits
    avg_spread = eurusd.avg_spread_points * point_size
    spike_spread = 0.00100  # 10 pips
    
    res = NewsFilter.check_spread(
        current_spread=spike_spread,
        avg_spread=avg_spread,
        max_spread_multiple=2.0,
    )
    assert res.blocked
    assert "0.0000" not in res.reason  # Reason must not report 0.0000
    assert "exceeds maximum allowed" in res.reason


def test_spread_filter_zero_or_negative_avg_spread_fallback():
    """Verify that non-positive avg_spread doesn't crash or block trades unconditionally."""
    res = NewsFilter.check_spread(
        current_spread=0.00024,
        avg_spread=0.0,
        max_spread_multiple=2.0,
    )
    assert not res.blocked


def test_pair_settings_avg_spread_points_override():
    """Verify PairSettings avg_spread_points can override instrument defaults."""
    pair_cfg = PairSettings(symbol="EURUSD", avg_spread_points=25.0)
    assert pair_cfg.avg_spread_points == 25.0

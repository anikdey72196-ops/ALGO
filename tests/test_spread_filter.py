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
    # Zero avg_spread
    res_zero = NewsFilter.check_spread(
        current_spread=0.00024,
        avg_spread=0.0,
        max_spread_multiple=2.0,
    )
    assert not res_zero.blocked
    assert res_zero.current_spread == 0.00024
    assert res_zero.avg_spread == 0.0
    assert res_zero.reason is None

    # Negative avg_spread
    res_neg = NewsFilter.check_spread(
        current_spread=0.00024,
        avg_spread=-0.00010,
        max_spread_multiple=2.0,
    )
    assert not res_neg.blocked
    assert res_neg.current_spread == 0.00024
    assert res_neg.avg_spread == -0.00010
    assert res_neg.reason is None


def test_spread_filter_non_positive_avg_spread_skips_even_on_extreme_spike():
    """Verify that when avg_spread <= 0, the check is skipped and does NOT block even on huge spread spikes."""
    res_spike_zero = NewsFilter.check_spread(
        current_spread=10.0,  # Massive 100,000-pip spike on forex
        avg_spread=0.0,
        max_spread_multiple=1.5,
    )
    assert not res_spike_zero.blocked
    assert res_spike_zero.current_spread == 10.0
    assert res_spike_zero.avg_spread == 0.0
    assert res_spike_zero.reason is None

    res_spike_neg = NewsFilter.check_spread(
        current_spread=5.0,
        avg_spread=-0.5,
        max_spread_multiple=0.5,
    )
    assert not res_spike_neg.blocked
    assert res_spike_neg.current_spread == 5.0
    assert res_spike_neg.avg_spread == -0.5
    assert res_spike_neg.reason is None


@pytest.mark.parametrize("avg_spread", [0.0, -0.0, -0.00001, -1.0, -50.0])
def test_spread_filter_various_non_positive_avg_spread_values(avg_spread):
    """Verify various non-positive avg_spread edge values consistently skip check and return blocked=False."""
    res = NewsFilter.check_spread(
        current_spread=0.00030,
        avg_spread=avg_spread,
        max_spread_multiple=2.0,
    )
    assert res.blocked is False
    assert res.current_spread == 0.00030
    assert res.avg_spread == avg_spread
    assert res.reason is None


def test_pair_settings_avg_spread_points_override():
    """Verify PairSettings avg_spread_points can override instrument defaults."""
    pair_cfg = PairSettings(symbol="EURUSD", avg_spread_points=25.0)
    assert pair_cfg.avg_spread_points == 25.0

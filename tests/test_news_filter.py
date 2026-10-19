"""
Unit tests for core/news_filter.py.

Covers:
1. Calendar loading (_load_calendar):
   - Missing calendar file fallback
   - Malformed / corrupted JSON fallback
   - Both {"events": [...]} and flat list [...] structures
   - ISO datetime parsing (trailing 'Z', UTC offsets)
   - Impact enum mapping and invalid impact filtering
2. Currency extraction (_get_instrument_currencies):
   - Standard 6-char Forex pairs (EURUSD, GBPUSD, USDJPY)
   - Special cases: GOLD, XAUUSD, XAUUSD.m
   - Whitespace and casing handling
   - Symbols shorter than 6 characters
3. News blackout filtering (check_news_blackout):
   - High impact base currency event blocking
   - High impact quote currency event blocking
   - Gold blackout on USD events
   - Medium and Low impact event exclusion
   - Irrelevant currency event exclusion
   - Exact boundary testing (inside window, boundary, outside window)
   - Custom blackout_minutes windows
   - Default current_time fallback (datetime.now(timezone.utc))
   - Invalid / unextractable symbols handling
4. Spread filtering (check_spread):
   - Normal spread pass
   - Spread spike block with descriptive reason
   - Non-positive avg_spread fallback
   - 5-digit (forex) vs 2-digit (metals/indices) format check
   - Custom max_spread_multiple parameter
5. Dataclass immutability (frozen=True)
"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import FrozenInstanceError

from core.config import Impact
from core.news_filter import (
    NewsEvent,
    NewsFilter,
    NewsFilterResult,
    SpreadFilterResult,
)


class TestNewsFilterCalendarLoading(unittest.TestCase):
    """Tests for NewsFilter._load_calendar and calendar initialization."""

    def test_missing_file_returns_empty_list(self):
        """When calendar JSON does not exist, return an empty list gracefully."""
        filter_instance = NewsFilter(calendar_path="non_existent_calendar_12345.json")
        self.assertEqual(filter_instance.events, [])

    def test_malformed_json_returns_empty_list(self):
        """When calendar JSON is corrupted, return an empty list gracefully."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            tmp.write("{ invalid json content: true ")
            tmp_path = tmp.name

        try:
            filter_instance = NewsFilter(calendar_path=tmp_path)
            self.assertEqual(filter_instance.events, [])
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_dict_format_events_parsing(self):
        """Parses events when wrapped in {'events': [...]} dict structure."""
        data = {
            "events": [
                {
                    "title": "US Non-Farm Payrolls",
                    "currency": "USD",
                    "impact": "HIGH",
                    "datetime_utc": "2026-10-19T12:30:00Z",
                },
                {
                    "title": "ECB Interest Rate Decision",
                    "currency": "EUR",
                    "impact": "HIGH",
                    "datetime_utc": "2026-10-19T11:45:00+00:00",
                },
            ]
        }
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            json.dump(data, tmp)
            tmp_path = tmp.name

        try:
            filter_instance = NewsFilter(calendar_path=tmp_path)
            self.assertEqual(len(filter_instance.events), 2)
            nfp = filter_instance.events[0]
            self.assertEqual(nfp.title, "US Non-Farm Payrolls")
            self.assertEqual(nfp.currency, "USD")
            self.assertEqual(nfp.impact, Impact.HIGH)
            self.assertEqual(nfp.datetime_utc, datetime(2026, 10, 19, 12, 30, tzinfo=timezone.utc))

            ecb = filter_instance.events[1]
            self.assertEqual(ecb.currency, "EUR")
            self.assertEqual(ecb.impact, Impact.HIGH)
            self.assertEqual(ecb.datetime_utc, datetime(2026, 10, 19, 11, 45, tzinfo=timezone.utc))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_flat_list_format_events_parsing(self):
        """Parses events when JSON root is a flat list [...]."""
        data = [
            {
                "title": "BoE Rate Vote",
                "currency": "GBP",
                "impact": "HIGH",
                "datetime_utc": "2026-10-19T11:00:00Z",
            }
        ]
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            json.dump(data, tmp)
            tmp_path = tmp.name

        try:
            filter_instance = NewsFilter(calendar_path=tmp_path)
            self.assertEqual(len(filter_instance.events), 1)
            self.assertEqual(filter_instance.events[0].title, "BoE Rate Vote")
            self.assertEqual(filter_instance.events[0].currency, "GBP")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_impact_levels_and_invalid_impact_skipped(self):
        """Ensures valid Impact enums are loaded and invalid ones skipped without crashing."""
        data = [
            {"title": "High Event", "currency": "USD", "impact": "HIGH", "datetime_utc": "2026-10-19T10:00:00Z"},
            {"title": "Medium Event", "currency": "USD", "impact": "MEDIUM", "datetime_utc": "2026-10-19T11:00:00Z"},
            {"title": "Low Event", "currency": "USD", "impact": "LOW", "datetime_utc": "2026-10-19T12:00:00Z"},
            {"title": "Unknown Impact Event", "currency": "USD", "impact": "SUPER_CRITICAL", "datetime_utc": "2026-10-19T13:00:00Z"},
        ]
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            json.dump(data, tmp)
            tmp_path = tmp.name

        try:
            filter_instance = NewsFilter(calendar_path=tmp_path)
            self.assertEqual(len(filter_instance.events), 3)
            self.assertEqual(filter_instance.events[0].impact, Impact.HIGH)
            self.assertEqual(filter_instance.events[1].impact, Impact.MEDIUM)
            self.assertEqual(filter_instance.events[2].impact, Impact.LOW)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestNewsFilterCurrencies(unittest.TestCase):
    """Tests for NewsFilter._get_instrument_currencies."""

    def setUp(self):
        self.news_filter = NewsFilter(calendar_path="non_existent.json")

    def test_standard_forex_pairs(self):
        """Extracts base and quote currencies for 6-char pairs."""
        self.assertEqual(self.news_filter._get_instrument_currencies("EURUSD"), ("EUR", "USD"))
        self.assertEqual(self.news_filter._get_instrument_currencies("GBPUSD"), ("GBP", "USD"))
        self.assertEqual(self.news_filter._get_instrument_currencies("USDJPY"), ("USD", "JPY"))
        self.assertEqual(self.news_filter._get_instrument_currencies("AUDNZD"), ("AUD", "NZD"))

    def test_case_insensitivity_and_whitespace(self):
        """Normalizes lowercase and trailing whitespace."""
        self.assertEqual(self.news_filter._get_instrument_currencies("  eurusd  "), ("EUR", "USD"))
        self.assertEqual(self.news_filter._get_instrument_currencies("gbpusd"), ("GBP", "USD"))

    def test_gold_symbols(self):
        """Correctly maps GOLD and XAUUSD variations to XAU and USD."""
        self.assertEqual(self.news_filter._get_instrument_currencies("GOLD"), ("XAU", "USD"))
        self.assertEqual(self.news_filter._get_instrument_currencies("XAUUSD"), ("XAU", "USD"))
        self.assertEqual(self.news_filter._get_instrument_currencies("XAUUSD.m"), ("XAU", "USD"))
        self.assertEqual(self.news_filter._get_instrument_currencies("GOLD_MICRO"), ("XAU", "USD"))

    def test_short_invalid_symbols(self):
        """Symbols with fewer than 6 characters (not gold) return empty strings."""
        self.assertEqual(self.news_filter._get_instrument_currencies("BTC"), ("", ""))
        self.assertEqual(self.news_filter._get_instrument_currencies(""), ("", ""))
        self.assertEqual(self.news_filter._get_instrument_currencies("US30"), ("", ""))


class TestNewsBlackout(unittest.TestCase):
    """Tests for NewsFilter.check_news_blackout."""

    def setUp(self):
        self.news_filter = NewsFilter(calendar_path="non_existent.json", blackout_minutes=30)
        self.base_event_time = datetime(2026, 10, 19, 14, 0, 0, tzinfo=timezone.utc)

    def test_base_currency_high_impact_blocks_trade(self):
        """A HIGH impact event on base currency (EUR) blocks EURUSD during the blackout window."""
        self.news_filter.events = [
            NewsEvent(
                title="ECB Rate Decision",
                currency="EUR",
                impact=Impact.HIGH,
                datetime_utc=self.base_event_time,
            )
        ]
        # Test 15 minutes before event (inside 30m window)
        check_time = self.base_event_time - timedelta(minutes=15)
        result = self.news_filter.check_news_blackout(symbol="EURUSD", current_time=check_time)
        self.assertTrue(result.blocked)
        self.assertIn("Blackout window active for EUR", result.reason)
        self.assertIn("ECB Rate Decision", result.reason)

    def test_quote_currency_high_impact_blocks_trade(self):
        """A HIGH impact event on quote currency (USD) blocks EURUSD during the blackout window."""
        self.news_filter.events = [
            NewsEvent(
                title="US CPI Inflation",
                currency="USD",
                impact=Impact.HIGH,
                datetime_utc=self.base_event_time,
            )
        ]
        # Test 10 minutes after event (inside 30m window)
        check_time = self.base_event_time + timedelta(minutes=10)
        result = self.news_filter.check_news_blackout(symbol="EURUSD", current_time=check_time)
        self.assertTrue(result.blocked)
        self.assertIn("Blackout window active for USD", result.reason)
        self.assertIn("US CPI Inflation", result.reason)

    def test_gold_blackout_on_usd_news(self):
        """Gold (XAUUSD / GOLD) is blocked by USD high-impact news."""
        self.news_filter.events = [
            NewsEvent(
                title="FOMC Statement",
                currency="USD",
                impact=Impact.HIGH,
                datetime_utc=self.base_event_time,
            )
        ]
        check_time = self.base_event_time - timedelta(minutes=5)
        res_gold = self.news_filter.check_news_blackout(symbol="GOLD", current_time=check_time)
        self.assertTrue(res_gold.blocked)

        res_xau = self.news_filter.check_news_blackout(symbol="XAUUSD", current_time=check_time)
        self.assertTrue(res_xau.blocked)

    def test_irrelevant_currency_does_not_block(self):
        """High impact JPY or CAD events do not block EURUSD."""
        self.news_filter.events = [
            NewsEvent(
                title="BoJ Rate Decision",
                currency="JPY",
                impact=Impact.HIGH,
                datetime_utc=self.base_event_time,
            ),
            NewsEvent(
                title="BoC Rate Decision",
                currency="CAD",
                impact=Impact.HIGH,
                datetime_utc=self.base_event_time,
            ),
        ]
        result = self.news_filter.check_news_blackout(symbol="EURUSD", current_time=self.base_event_time)
        self.assertFalse(result.blocked)
        self.assertIsNone(result.reason)

    def test_medium_and_low_impact_events_do_not_block(self):
        """Medium and Low impact events for EUR/USD do not block trades."""
        self.news_filter.events = [
            NewsEvent(
                title="German PPI",
                currency="EUR",
                impact=Impact.MEDIUM,
                datetime_utc=self.base_event_time,
            ),
            NewsEvent(
                title="US Consumer Sentiment",
                currency="USD",
                impact=Impact.LOW,
                datetime_utc=self.base_event_time,
            ),
        ]
        result = self.news_filter.check_news_blackout(symbol="EURUSD", current_time=self.base_event_time)
        self.assertFalse(result.blocked)

    def test_window_boundaries(self):
        """
        Verify exact window boundaries for blackout_minutes=30.
        - Exactly at event (diff = 0m): BLOCKED
        - Exactly at boundary -30m: BLOCKED (<= 30m)
        - Exactly at boundary +30m: BLOCKED (<= 30m)
        - 30m + 1s before: PASSED
        - 30m + 1s after: PASSED
        """
        self.news_filter.events = [
            NewsEvent(
                title="Fed Chair Speaks",
                currency="USD",
                impact=Impact.HIGH,
                datetime_utc=self.base_event_time,
            )
        ]

        # Exactly at event time
        self.assertTrue(
            self.news_filter.check_news_blackout("EURUSD", self.base_event_time).blocked
        )

        # Exactly at 30 minutes before
        t_minus_30 = self.base_event_time - timedelta(minutes=30)
        self.assertTrue(
            self.news_filter.check_news_blackout("EURUSD", t_minus_30).blocked
        )

        # Exactly at 30 minutes after
        t_plus_30 = self.base_event_time + timedelta(minutes=30)
        self.assertTrue(
            self.news_filter.check_news_blackout("EURUSD", t_plus_30).blocked
        )

        # 30 minutes and 1 second before (outside window)
        t_outside_before = self.base_event_time - timedelta(minutes=30, seconds=1)
        self.assertFalse(
            self.news_filter.check_news_blackout("EURUSD", t_outside_before).blocked
        )

        # 30 minutes and 1 second after (outside window)
        t_outside_after = self.base_event_time + timedelta(minutes=30, seconds=1)
        self.assertFalse(
            self.news_filter.check_news_blackout("EURUSD", t_outside_after).blocked
        )

    def test_custom_blackout_minutes(self):
        """Respects custom blackout_minutes window duration."""
        custom_filter = NewsFilter(calendar_path="non_existent.json", blackout_minutes=60)
        custom_filter.events = [
            NewsEvent(
                title="NFP",
                currency="USD",
                impact=Impact.HIGH,
                datetime_utc=self.base_event_time,
            )
        ]
        # 45 minutes before: blocked with 60m window, but allowed with 30m window
        t_45m_before = self.base_event_time - timedelta(minutes=45)
        self.assertTrue(custom_filter.check_news_blackout("EURUSD", t_45m_before).blocked)
        self.assertFalse(self.news_filter.check_news_blackout("EURUSD", t_45m_before).blocked)

    def test_invalid_symbol_passes_cleanly(self):
        """Unextractable symbols (< 6 chars) pass news filter without error."""
        self.news_filter.events = [
            NewsEvent(
                title="NFP",
                currency="USD",
                impact=Impact.HIGH,
                datetime_utc=self.base_event_time,
            )
        ]
        result = self.news_filter.check_news_blackout(symbol="BTC", current_time=self.base_event_time)
        self.assertFalse(result.blocked)

    def test_default_current_time_uses_now_utc(self):
        """When current_time is None, it defaults to datetime.now(timezone.utc)."""
        now = datetime.now(timezone.utc)
        self.news_filter.events = [
            NewsEvent(
                title="Breaking Event",
                currency="USD",
                impact=Impact.HIGH,
                datetime_utc=now,
            )
        ]
        # Call without current_time argument
        result = self.news_filter.check_news_blackout(symbol="EURUSD")
        self.assertTrue(result.blocked)


class TestSpreadFilter(unittest.TestCase):
    """Tests for NewsFilter.check_spread static method."""

    def test_normal_spread_passes(self):
        """Current spread below max multiple passes."""
        result = NewsFilter.check_spread(
            current_spread=0.00015,
            avg_spread=0.00010,
            max_spread_multiple=2.0,
        )
        self.assertFalse(result.blocked)
        self.assertEqual(result.current_spread, 0.00015)
        self.assertEqual(result.avg_spread, 0.00010)
        self.assertIsNone(result.reason)

    def test_spread_spike_blocked(self):
        """Current spread exceeding multiplier is blocked with detailed reason."""
        result = NewsFilter.check_spread(
            current_spread=0.00035,
            avg_spread=0.00010,
            max_spread_multiple=2.0,
        )
        self.assertTrue(result.blocked)
        self.assertIn("exceeds maximum allowed", result.reason)
        self.assertIn("2.0x", result.reason)

    def test_non_positive_avg_spread_skips_check(self):
        """Zero or negative average spread passes without error or false block."""
        res_zero = NewsFilter.check_spread(current_spread=0.0002, avg_spread=0.0)
        self.assertFalse(res_zero.blocked)

        res_negative = NewsFilter.check_spread(current_spread=0.0002, avg_spread=-0.0001)
        self.assertFalse(res_negative.blocked)

    def test_precision_formatting_forex_vs_metals(self):
        """Forex (<0.05) formats with 5 decimals; metals/indices (>=0.05) format with 2 decimals."""
        # Forex pair formatting (< 0.05)
        res_forex = NewsFilter.check_spread(
            current_spread=0.00050,
            avg_spread=0.00020,
            max_spread_multiple=2.0,
        )
        self.assertTrue(res_forex.blocked)
        self.assertIn("0.00050", res_forex.reason)
        self.assertIn("0.00040", res_forex.reason)

        # Metals / Indices formatting (>= 0.05)
        res_metal = NewsFilter.check_spread(
            current_spread=3.50,
            avg_spread=1.00,
            max_spread_multiple=2.0,
        )
        self.assertTrue(res_metal.blocked)
        self.assertIn("3.50", res_metal.reason)
        self.assertIn("2.00", res_metal.reason)

    def test_custom_max_spread_multiple(self):
        """Respects custom max_spread_multiple threshold."""
        # 1.5x threshold -> 0.00016 > 0.00015 (blocked)
        res_strict = NewsFilter.check_spread(
            current_spread=0.00016,
            avg_spread=0.00010,
            max_spread_multiple=1.5,
        )
        self.assertTrue(res_strict.blocked)

        # 3.0x threshold -> 0.00016 <= 0.00030 (passed)
        res_lenient = NewsFilter.check_spread(
            current_spread=0.00016,
            avg_spread=0.00010,
            max_spread_multiple=3.0,
        )
        self.assertFalse(res_lenient.blocked)


class TestDataclassesImmutability(unittest.TestCase):
    """Verifies dataclass immutability (frozen=True) for safety."""

    def test_news_event_is_frozen(self):
        event = NewsEvent(
            title="CPI",
            currency="USD",
            impact=Impact.HIGH,
            datetime_utc=datetime.now(timezone.utc),
        )
        with self.assertRaises(FrozenInstanceError):
            event.title = "Modified"  # type: ignore

    def test_news_filter_result_is_frozen(self):
        result = NewsFilterResult(blocked=False)
        with self.assertRaises(FrozenInstanceError):
            result.blocked = True  # type: ignore

    def test_spread_filter_result_is_frozen(self):
        result = SpreadFilterResult(blocked=False, current_spread=0.0001, avg_spread=0.0001)
        with self.assertRaises(FrozenInstanceError):
            result.blocked = True  # type: ignore


if __name__ == "__main__":
    unittest.main()

"""
test_timezone_and_night_limit.py — Comprehensive tests for timezone resolution
and the automatic night limit window (11 PM - 8 AM).
"""

import unittest
from datetime import datetime, timezone, timedelta
from core.config import TradingConfig, RiskConfig, resolve_timezone


class TestTimezoneAndNightLimit(unittest.TestCase):

    def test_resolve_timezone_formats(self):
        """Verify resolve_timezone parses named zones, abbreviations, and offsets."""
        now = datetime.now()
        tz_ist = resolve_timezone("Asia/Kolkata")
        self.assertEqual(tz_ist.utcoffset(now), timedelta(hours=5, minutes=30))

        tz_ist_abbr = resolve_timezone("IST")
        self.assertEqual(tz_ist_abbr.utcoffset(now), timedelta(hours=5, minutes=30))

        tz_offset = resolve_timezone("+05:30")
        self.assertEqual(tz_offset.utcoffset(now), timedelta(hours=5, minutes=30))

        tz_utc = resolve_timezone("UTC")
        self.assertEqual(tz_utc.utcoffset(now), timedelta(0))

        tz_est = resolve_timezone("EST")
        self.assertEqual(tz_est.utcoffset(now), timedelta(hours=-5))

    def test_daytime_at_1300_ist_is_not_night(self):
        """
        At 13:00 IST (07:30 UTC), is_night_window must return False
        and effective max open positions must be 15 (not 2).
        """
        risk = RiskConfig(
            night_limit_enabled=True,
            night_start_hour=23,
            night_end_hour=8,
            night_max_open_positions=2,
            max_open_positions=15,
            night_timezone_mode="Asia/Kolkata",
        )

        # 07:30 UTC == 13:00 IST (Afternoon)
        dt_1300_ist_in_utc = datetime(2026, 9, 28, 7, 30, tzinfo=timezone.utc)
        self.assertFalse(risk.is_night_window(dt_1300_ist_in_utc))
        self.assertEqual(risk.get_effective_max_open_positions(dt_1300_ist_in_utc), 15)

    def test_morning_at_0830_ist_is_not_night(self):
        """At 08:30 IST (03:00 UTC), night window has ended."""
        risk = RiskConfig(
            night_limit_enabled=True,
            night_start_hour=23,
            night_end_hour=8,
            night_max_open_positions=2,
            max_open_positions=15,
            night_timezone_mode="Asia/Kolkata",
        )
        dt_0830_ist_in_utc = datetime(2026, 9, 28, 3, 0, tzinfo=timezone.utc)
        self.assertFalse(risk.is_night_window(dt_0830_ist_in_utc))
        self.assertEqual(risk.get_effective_max_open_positions(dt_0830_ist_in_utc), 15)

    def test_night_at_2330_ist_is_night(self):
        """At 23:30 IST (18:00 UTC), night window is active."""
        risk = RiskConfig(
            night_limit_enabled=True,
            night_start_hour=23,
            night_end_hour=8,
            night_max_open_positions=2,
            max_open_positions=15,
            night_timezone_mode="Asia/Kolkata",
        )
        dt_2330_ist_in_utc = datetime(2026, 9, 28, 18, 0, tzinfo=timezone.utc)
        self.assertTrue(risk.is_night_window(dt_2330_ist_in_utc))
        self.assertEqual(risk.get_effective_max_open_positions(dt_2330_ist_in_utc), 2)

    def test_overnight_at_0300_ist_is_night(self):
        """At 03:00 IST (21:30 UTC prev day), night window is active."""
        risk = RiskConfig(
            night_limit_enabled=True,
            night_start_hour=23,
            night_end_hour=8,
            night_max_open_positions=2,
            max_open_positions=15,
            night_timezone_mode="Asia/Kolkata",
        )
        dt_0300_ist_in_utc = datetime(2026, 9, 27, 21, 30, tzinfo=timezone.utc)
        self.assertTrue(risk.is_night_window(dt_0300_ist_in_utc))
        self.assertEqual(risk.get_effective_max_open_positions(dt_0300_ist_in_utc), 2)

    def test_disabled_night_limit(self):
        """When night limit is disabled, is_night_window always returns False."""
        risk = RiskConfig(
            night_limit_enabled=False,
            night_start_hour=23,
            night_end_hour=8,
            night_max_open_positions=2,
            max_open_positions=15,
            night_timezone_mode="Asia/Kolkata",
        )
        dt_night = datetime(2026, 9, 28, 18, 0, tzinfo=timezone.utc)
        self.assertFalse(risk.is_night_window(dt_night))
        self.assertEqual(risk.get_effective_max_open_positions(dt_night), 15)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from pathlib import Path
# pyrefly: ignore [missing-import]
from loguru import logger
from core.config import Impact, InstrumentConfig


@dataclass(frozen=True)
class NewsEvent:
    """A single economic calendar event."""
    title: str
    currency: str  # e.g. 'USD', 'EUR', 'GBP'
    impact: Impact
    datetime_utc: datetime


@dataclass(frozen=True)
class NewsFilterResult:
    """Result of a news filter check."""
    blocked: bool
    reason: str | None = None  # Human-readable reason if blocked


@dataclass(frozen=True)
class SpreadFilterResult:
    """Result of a spread filter check."""
    blocked: bool
    current_spread: float = 0.0
    avg_spread: float = 0.0
    reason: str | None = None


class NewsFilter:
    """Gatekeeper that blocks trades during news blackout windows and excessive spreads."""
    
    def __init__(self, calendar_path: str = 'news_calendar.json', blackout_minutes: int = 30):
        """
        Load economic calendar from JSON file.
        blackout_minutes: the ± window around HIGH impact events.
        """
        self.calendar_path = calendar_path
        self.blackout_minutes = blackout_minutes
        self.events: list[NewsEvent] = self._load_calendar(self.calendar_path)
    
    def _load_calendar(self, path: str) -> list[NewsEvent]:
        """Parse the news_calendar.json file into NewsEvent objects."""
        calendar_file = Path(path)
        if not calendar_file.exists():
            logger.warning(f"News calendar file {path} not found. Returning empty list.")
            return []
            
        try:
            with calendar_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
                
            events = []
            # Handle both {"events": [...]} and flat [...] formats
            items = data.get('events', data) if isinstance(data, dict) else data
            for item in items:
                dt_str = item.get('datetime_utc', '')
                if dt_str.endswith('Z'):
                    dt_str = dt_str[:-1] + '+00:00'
                dt = datetime.fromisoformat(dt_str)
                
                # Handling Impact enum mapping gracefully
                impact_val = item.get('impact')
                try:
                    impact = Impact(impact_val) if isinstance(impact_val, str) else impact_val
                except ValueError:
                    # Default/fallback if enum fails, though it depends on config.Impact definition
                    # Assuming it matches exact string for simplicity
                    continue
                
                event = NewsEvent(
                    title=item.get('title', ''),
                    currency=item.get('currency', ''),
                    impact=impact,
                    datetime_utc=dt
                )
                events.append(event)
            return events
        except Exception as e:
            logger.warning(f"Error loading news calendar from {path}: {e}")
            return []
    
    def _get_instrument_currencies(self, symbol: str) -> tuple[str, str]:
        """
        Extract base and quote currencies from a symbol string.
        E.g. 'EURUSD' -> ('EUR', 'USD'), 'GOLD' -> ('XAU', 'USD')
        """
        s = symbol.upper().strip()
        if "GOLD" in s or s.startswith("XAU"):
            return ("XAU", "USD")
        if len(s) < 6:
            logger.warning(f"Symbol '{symbol}' is less than 6 characters. Cannot extract currencies.")
            return ('', '')
        base = s[:3]
        quote = s[3:6]
        return (base, quote)
    
    def check_news_blackout(self, symbol: str, current_time: datetime | None = None) -> NewsFilterResult:
        """
        Check if current_time falls within ±blackout_minutes of any HIGH impact
        news event affecting the base or quote currency of the symbol.
        
        Returns NewsFilterResult with blocked=True and reason if in blackout.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
            
        base, quote = self._get_instrument_currencies(symbol)
        if not base or not quote:
            # If symbol is invalid, let's not block based on news, or maybe we should?
            # Safe default: allow, but the trading engine might fail it elsewhere.
            return NewsFilterResult(blocked=False)
            
        relevant_currencies = {base, quote}
        
        for event in self.events:
            if event.impact == Impact.HIGH and event.currency in relevant_currencies:
                time_diff = abs(current_time - event.datetime_utc)
                if time_diff <= timedelta(minutes=self.blackout_minutes):
                    reason = f"Blackout window active for {event.currency} event: '{event.title}' at {event.datetime_utc}"
                    logger.info(f"NewsFilter blocked {symbol}: {reason}")
                    return NewsFilterResult(blocked=True, reason=reason)
                    
        logger.debug(f"NewsFilter passed {symbol} at {current_time}.")
        return NewsFilterResult(blocked=False)
    
    @staticmethod
    def check_spread(
        current_spread: float,
        avg_spread: float,
        max_spread_multiple: float = 2.0,
    ) -> SpreadFilterResult:
        """
        Check if current spread exceeds max_spread_multiple × avg_spread.
        Returns SpreadFilterResult with blocked=True if excessive.
        """
        threshold = max_spread_multiple * avg_spread
        if current_spread > threshold:
            reason = f"Current spread {current_spread} exceeds maximum allowed ({threshold:.4f}) based on avg {avg_spread:.4f}."
            logger.info(f"SpreadFilter blocked: {reason}")
            return SpreadFilterResult(
                blocked=True,
                current_spread=current_spread,
                avg_spread=avg_spread,
                reason=reason
            )
            
        logger.debug(f"SpreadFilter passed: current_spread={current_spread}, avg_spread={avg_spread}.")
        return SpreadFilterResult(
            blocked=False,
            current_spread=current_spread,
            avg_spread=avg_spread
        )

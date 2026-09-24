"""
order_manager.py — Idempotent Order State Machine, Error Classifier, and Retries.
"""
from __future__ import annotations
import time
import random
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
from loguru import logger
from execution import BrokerAdapter, BracketOrder, OrderResult


class OrderState(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class ErrorClassification(Enum):
    TRANSIENT = "TRANSIENT"      # Safe to retry (requote, busy, connection)
    PERMANENT = "PERMANENT"      # Never retry (invalid volume, market closed, bad stop)
    UNKNOWN = "UNKNOWN"


def classify_error(retcode: int | None, error_message: str | None) -> ErrorClassification:
    """Classify MT5 retcodes and broker error messages."""
    msg = (error_message or "").upper()
    
    # Permanent retcodes
    permanent_codes = {10013, 10014, 10015, 10016, 10017, 10019, 10026, 10027, 10028}
    if retcode in permanent_codes or "INVALID" in msg or "MONEY" in msg or "DISABLED" in msg:
        return ErrorClassification.PERMANENT

    # Transient retcodes
    transient_codes = {10004, 10006, 10018, 10021, 10024, 10025}
    if retcode in transient_codes or "REQUOTE" in msg or "TIMEOUT" in msg or "BUSY" in msg or "CONNECTION" in msg:
        return ErrorClassification.TRANSIENT

    return ErrorClassification.TRANSIENT if retcode is not None else ErrorClassification.UNKNOWN


def generate_idempotency_key(symbol: str, strategy: str, direction: str,
                             entry_price: float, bar_time_iso: str, magic: int) -> str:
    """Generate a deterministic, reproducible client order ID."""
    raw = f"{symbol}_{strategy}_{direction}_{round(entry_price, 4)}_{bar_time_iso}_{magic}"
    digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12].upper()
    return f"ORD-{symbol}-{digest}"


class OrderManager:
    """Orchestrates order submission with idempotency protection and bounded retries."""

    def __init__(self, broker: BrokerAdapter, max_retries: int = 3, base_backoff_sec: float = 0.5):
        self.broker = broker
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        self.active_orders: Dict[str, Dict[str, Any]] = {}

    def submit_bracket_order(self, bracket: BracketOrder, client_order_id: str) -> Tuple[OrderState, OrderResult]:
        if client_order_id in self.active_orders:
            existing = self.active_orders[client_order_id]
            if existing["state"] in (OrderState.FILLED, OrderState.SUBMITTED):
                logger.warning(f"Idempotent order submission ignored: {client_order_id} is already {existing['state'].value}")
                return existing["state"], existing.get("result")

        self.active_orders[client_order_id] = {
            "state": OrderState.SUBMITTED,
            "order": bracket,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "retries": 0,
        }

        last_result = None
        for attempt in range(self.max_retries):
            result = self.broker.send_bracket_order(bracket)
            last_result = result

            if result.success:
                self.active_orders[client_order_id]["state"] = OrderState.FILLED
                self.active_orders[client_order_id]["result"] = result
                self.active_orders[client_order_id]["order_id"] = result.order_id
                return OrderState.FILLED, result

            # Failure classification
            classification = classify_error(result.error_code, result.error_message)
            if classification == ErrorClassification.PERMANENT:
                logger.error(f"Permanent rejection on attempt {attempt+1}: {result.error_message}. Halting retries.")
                self.active_orders[client_order_id]["state"] = OrderState.REJECTED
                return OrderState.REJECTED, result

            # Transient backoff with jitter
            backoff = (self.base_backoff_sec * (2 ** attempt)) + random.uniform(0.05, 0.2)
            logger.warning(f"Transient error on {client_order_id} (attempt {attempt+1}/{self.max_retries}): {result.error_message}. Retrying in {backoff:.2f}s...")
            time.sleep(backoff)

        self.active_orders[client_order_id]["state"] = OrderState.REJECTED
        return OrderState.REJECTED, last_result or OrderResult(success=False, error_message="Max retries exhausted")

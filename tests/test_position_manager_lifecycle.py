"""
tests/test_position_manager_lifecycle.py — Tests for PositionManager thread lifecycle and non-blocking shutdown.
"""
import time
from unittest.mock import MagicMock
import pytest

from execution.position_manager import PositionManager


def test_position_manager_fast_shutdown():
    """Verify that PositionManager terminates immediately on shutdown without blocking for poll_interval_sec."""
    mock_broker = MagicMock()
    mock_state = MagicMock()
    mock_state.get_open_positions.return_value = []

    # Initialize PositionManager with a large poll interval (e.g. 5.0s)
    pm = PositionManager(
        broker=mock_broker,
        state=mock_state,
        poll_interval_sec=5.0,
        smart_partial_tp_service=MagicMock(),
    )

    # Let thread start and enter _run_loop wait
    time.sleep(0.05)
    assert pm._worker.is_alive()

    # Measure shutdown duration
    t0 = time.perf_counter()
    pm.shutdown()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Thread should join in well under 100ms (not wait for 2.0s join timeout or 5.0s sleep)
    assert elapsed_ms < 100.0, f"Shutdown took too long: {elapsed_ms:.2f}ms"
    assert not pm._worker.is_alive(), "Worker thread should be dead after shutdown"


def test_position_manager_loop_execution():
    """Verify that process_positions continues to be called on interval when not stopped."""
    mock_broker = MagicMock()
    mock_state = MagicMock()
    mock_state.get_open_positions.return_value = []

    # Use very short interval for test
    pm = PositionManager(
        broker=mock_broker,
        state=mock_state,
        poll_interval_sec=0.05,
        smart_partial_tp_service=MagicMock(),
    )

    time.sleep(0.15)
    pm.shutdown()

    # Ensure get_open_positions was invoked during the active intervals
    assert mock_state.get_open_positions.call_count >= 1
    assert not pm._worker.is_alive()

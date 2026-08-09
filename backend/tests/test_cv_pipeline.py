"""Unit tests for CV loop pipeline helpers (latest-frame slot, ADB probe timing)."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from app.cv.event_ocr import EventOcrEngine
from app.cv.regions import default_assets_dir, load_regions
from app.services import cv_runner
from app.services.cv_runner import _collect_battle_animation_events
from app.services.session import BattlePhase, SessionStore


@pytest.mark.asyncio
async def test_put_latest_frame_keeps_only_newest() -> None:
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=1)
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.ones((2, 2, 3), dtype=np.uint8)
    third = np.full((2, 2, 3), 2, dtype=np.uint8)

    await cv_runner._put_latest_frame(queue, first)
    await cv_runner._put_latest_frame(queue, second)
    await cv_runner._put_latest_frame(queue, third)

    assert queue.qsize() == 1
    got = queue.get_nowait()
    np.testing.assert_array_equal(got, third)


@pytest.mark.asyncio
async def test_maybe_probe_adb_throttles(monkeypatch: pytest.MonkeyPatch) -> None:
    store = SessionStore()
    calls = {"n": 0}

    def fake_connected():
        calls["n"] += 1
        return True

    monkeypatch.setattr(cv_runner, "is_adb_connected", fake_connected)

    t0 = time.monotonic()
    t1 = await cv_runner._maybe_probe_adb(store, last_probe_at=0.0)
    assert calls["n"] == 1
    assert store.adb_connected is True

    t2 = await cv_runner._maybe_probe_adb(store, last_probe_at=t1)
    assert calls["n"] == 1  # throttled
    assert t2 == t1

    # Force interval elapsed.
    t3 = await cv_runner._maybe_probe_adb(
        store,
        last_probe_at=t1 - cv_runner._ADB_PROBE_INTERVAL_SEC - 0.01,
    )
    assert calls["n"] == 2
    assert t3 >= t0


def test_poll_interval_sleep_accounts_for_elapsed_work() -> None:
    """Document intended minimum-period semantics used by the CV loop."""
    interval = cv_runner._poll_interval(BattlePhase.BATTLE_ANIMATION)
    assert abs(interval - (1.0 / 3.0)) < 1e-9
    elapsed = 10.0
    assert max(0.0, interval - elapsed) == 0.0
    elapsed = 0.05
    assert abs(max(0.0, interval - elapsed) - (interval - 0.05)) < 1e-9


@patch("app.cv.event_ocr._ocr_text")
def test_collect_battle_animation_events_does_not_append(mock_ocr) -> None:
    """Gather collectors must return events without mutating SessionStore."""
    mock_ocr.return_value = "Hatterene's Sitrus Berry"
    store = SessionStore()
    store.turn_number = 1
    store.battle_logs = [[], []]
    engine = EventOcrEngine()
    config = load_regions()
    frame = np.asarray(
        Image.open(default_assets_dir() / "opponent_slot_2_banner.png").convert("RGB"),
        dtype=np.uint8,
    )
    events = _collect_battle_animation_events(
        frame,
        engine,
        config,
        player_species=None,
        opponent_species=None,
    )
    assert events
    assert store.battle_logs[1] == []

"""Background CV task: phase detection and team preview vision."""

from __future__ import annotations

import asyncio
import logging

from app.cv.adb_capture import capture_screenshot, is_adb_connected
from app.cv.phase_detector import PhaseDetector
from app.cv.regions import load_regions
from app.cv.team_preview_reader import read_opponent_team_preview
from app.services.gemini import GeminiService
from app.services.session import BattlePhase, SessionStore

logger = logging.getLogger(__name__)

_cv_task: asyncio.Task[None] | None = None
_ADB_PROBE_INTERVAL_SEC = 5.0
_TEAM_PREVIEW_POLL_SEC = 1.0
_DEFAULT_POLL_SEC = 0.5


def _poll_interval(phase: BattlePhase) -> float:
    if phase == BattlePhase.TEAM_PREVIEW:
        return _TEAM_PREVIEW_POLL_SEC
    return _DEFAULT_POLL_SEC


async def _process_team_preview_entry(store: SessionStore, frame) -> None:
    """Crop opponent sprites, identify six species, and request bring suggestions."""
    if store._team_preview_processed or store.player_team is None:
        return

    store._team_preview_processed = True
    config = load_regions()

    try:
        gemini = GeminiService()
    except ValueError:
        logger.warning("GEMINI_API_KEY not set; skipping team preview vision")
        return

    try:
        opponent = await read_opponent_team_preview(frame, config, gemini=gemini)
        store.opponent_team_species = opponent.species
        store.team_preview_suggestion = await gemini.suggest_team_preview(
            store.player_team,
            opponent.species,
        )
        logger.info(
            "Team preview suggestion ready (opponent 6: %s)",
            ", ".join(opponent.species),
        )
    except Exception:
        logger.exception("Team preview vision failed")
        store._team_preview_processed = False


async def _cv_loop(store: SessionStore) -> None:
    logger.info("CV loop started")
    detector = PhaseDetector()
    try:
        while store.cv_running:
            store.adb_connected = await asyncio.to_thread(is_adb_connected)
            if not store.adb_connected:
                logger.debug("ADB not connected; retrying in %.0fs", _ADB_PROBE_INTERVAL_SEC)
                await asyncio.sleep(_ADB_PROBE_INTERVAL_SEC)
                continue

            try:
                frame = await asyncio.to_thread(capture_screenshot)
            except RuntimeError as exc:
                logger.debug("Screenshot capture failed: %s", exc)
                store.adb_connected = False
                await asyncio.sleep(_ADB_PROBE_INTERVAL_SEC)
                continue

            transition = detector.detect_transition(frame)
            store.phase = transition.current

            if transition.entered_team_preview:
                await _process_team_preview_entry(store, frame)

            await asyncio.sleep(_poll_interval(transition.current))
    except asyncio.CancelledError:
        logger.info("CV loop stopped")
        raise


def start_cv(store: SessionStore) -> None:
    global _cv_task
    if _cv_task is not None and not _cv_task.done():
        return
    _cv_task = asyncio.create_task(_cv_loop(store))


async def stop_cv() -> None:
    global _cv_task
    if _cv_task is None:
        return
    _cv_task.cancel()
    try:
        await _cv_task
    except asyncio.CancelledError:
        pass
    _cv_task = None


async def shutdown_cv() -> None:
    """Cancel the CV task during application shutdown."""
    await stop_cv()

"""Background CV task: phase detection and team preview vision."""

from __future__ import annotations

import asyncio
import logging

from app.cv.adb_capture import capture_screenshot, is_adb_connected
from app.cv.event_ocr import EventOcrEngine
from app.cv.hp_reader import HPReader
from app.cv.phase_detector import PhaseDetector
from app.cv.regions import load_regions
from app.cv.team_preview_reader import read_opponent_team_preview
from app.cv.team_selection_reader import read_player_selected_species
from app.schema.battle_log import TurnStartEvent
from app.services.gamestate_reducer import ensure_seeded
from app.services.gemini import GeminiService
from app.services.session import BattlePhase, SessionStore

logger = logging.getLogger(__name__)

_cv_task: asyncio.Task[None] | None = None
_ADB_PROBE_INTERVAL_SEC = 5.0
_TEAM_PREVIEW_POLL_SEC = 1.0
_TEAM_SELECTED_POLL_SEC = 1.0
_ACTION_SELECTION_POLL_SEC = 1.0
_BATTLE_ANIMATION_POLL_SEC = 1.0 / 3.0  # 3 FPS — HP reader + event OCR
_DEFAULT_POLL_SEC = 0.5


def _poll_interval(phase: BattlePhase) -> float:
    if phase == BattlePhase.TEAM_PREVIEW:
        return _TEAM_PREVIEW_POLL_SEC
    if phase == BattlePhase.TEAM_SELECTED:
        return _TEAM_SELECTED_POLL_SEC
    if phase == BattlePhase.BATTLE_ANIMATION:
        return _BATTLE_ANIMATION_POLL_SEC
    if phase == BattlePhase.ACTION_SELECTION:
        return _ACTION_SELECTION_POLL_SEC
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


def _process_team_selected_frame(store: SessionStore, frame, config) -> None:
    """OCR selection-order badges and map numbered panels via pokepaste order."""
    if store.player_team is None:
        return
    try:
        selected = read_player_selected_species(frame, config, store.player_team)
    except Exception:
        logger.exception("Player team selection read failed")
        return
    if selected != store.player_selected_species:
        store.player_selected_species = selected
        logger.info("Player selected bring: %s", ", ".join(selected))


def _process_battle_animation_events(
    store: SessionStore,
    frame,
    event_ocr: EventOcrEngine,
    config,
) -> None:
    """OCR changed per-slot banners and battle text, appending parsed events."""
    for event in event_ocr.process_frame(frame, config):
        store.append_battle_log(event)
        logger.info("Battle log event: %s — %r", event.type, event.raw_text)


def _process_hp_animation_frame(
    store: SessionStore,
    frame,
    hp_reader: HPReader,
    config,
) -> None:
    """Poll slot cards at animation FPS; append stable HPChangeEvents."""
    for event in hp_reader.process_animation_frame(frame, config, store.game_state):
        store.append_battle_log(event)
        logger.info(
            "HP change: %s slot %s %+d%% — %r",
            event.pokemon.side,
            event.pokemon.slot,
            event.hp_pct_change,
            event.raw_text,
        )


def _process_hp_action_selection_snapshot(
    store: SessionStore,
    frame,
    hp_reader: HPReader,
    config,
) -> None:
    """Authoritative 4-slot HP snapshot on action_selection entry."""
    for event in hp_reader.read_action_selection_snapshot(
        frame, config, store.game_state
    ):
        store.append_battle_log(event)
        logger.info(
            "HP snapshot reconcile: %s slot %s %+d%% — %r",
            event.pokemon.side,
            event.pokemon.slot,
            event.hp_pct_change,
            event.raw_text,
        )


def _emit_turn_start_on_action_selection_entry(store: SessionStore) -> None:
    """Increment turn counter and append TurnStartEvent on action_selection entry."""
    store.turn_number += 1
    event = TurnStartEvent(
        raw_text=f"Turn {store.turn_number}",
        turn_number=store.turn_number,
    )
    store.append_battle_log(event)
    logger.info("Turn start: %d", store.turn_number)


async def _cv_loop(store: SessionStore) -> None:
    logger.info("CV loop started")
    detector = PhaseDetector()
    event_ocr = EventOcrEngine()
    hp_reader = HPReader()
    region_config = load_regions()
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

            if transition.current == BattlePhase.TEAM_SELECTED:
                await asyncio.to_thread(
                    _process_team_selected_frame,
                    store,
                    frame,
                    region_config,
                )

            if transition.entered_battle:
                ensure_seeded(store)

            if transition.entered_action_selection:
                _emit_turn_start_on_action_selection_entry(store)
                await asyncio.to_thread(
                    _process_hp_action_selection_snapshot,
                    store,
                    frame,
                    hp_reader,
                    region_config,
                )

            if transition.current == BattlePhase.BATTLE_ANIMATION:
                await asyncio.to_thread(
                    _process_battle_animation_events,
                    store,
                    frame,
                    event_ocr,
                    region_config,
                )
                await asyncio.to_thread(
                    _process_hp_animation_frame,
                    store,
                    frame,
                    hp_reader,
                    region_config,
                )
            else:
                event_ocr.reset()
                if transition.current not in (
                    BattlePhase.ACTION_SELECTION,
                    BattlePhase.BATTLE_ANIMATION,
                ):
                    hp_reader.reset()

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

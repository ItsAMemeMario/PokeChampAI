"""Background CV task: phase detection, OCR, and Gemini suggestions."""

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
from app.services.gemini import create_gemini_service, previous_turn_battle_log_events
from app.services.session import BattlePhase, SessionStore
from app.services.ws_hub import (
    publish_phase,
    publish_session,
    publish_state,
    publish_team_preview,
    publish_turn_suggestion,
)

logger = logging.getLogger(__name__)

_cv_task: asyncio.Task[None] | None = None
_ADB_PROBE_INTERVAL_SEC = 5.0
_TEAM_PREVIEW_POLL_SEC = 1.0
_TEAM_SELECTED_POLL_SEC = 1.0
_ACTION_SELECTION_POLL_SEC = 1.0 / 5.0  # 5 FPS — catch brief "Communicating..." standby
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
    logger.info("Processing team preview entry")
    if store._team_preview_processed or store.player_team is None:
        return

    store._team_preview_processed = True
    config = load_regions()

    gemini = create_gemini_service()

    try:
        opponent = await read_opponent_team_preview(frame, config, gemini=gemini)
        store.opponent_team_species = opponent.species
        store.team_preview_suggestion = await gemini.suggest_team_preview(
            store.player_team,
            opponent.species,
        )
        store.gemini_interaction_id = gemini.interaction_id
        publish_team_preview(store)
        logger.info(
            "Team preview suggestion ready (opponent 6: %s)",
            ", ".join(opponent.species),
        )
    except Exception:
        logger.exception("Team preview vision failed")
        store._team_preview_processed = False
        store.gemini_interaction_id = gemini.interaction_id


def _process_team_selected_frame(store: SessionStore, frame, config) -> None:
    """OCR selection-order badges and map numbered panels via pokepaste order."""
    if store.player_team is None:
        return
    try:
        selected = read_player_selected_species(frame, config, store.player_team)
    except Exception:
        logger.exception("Player team selection read failed")
        return
    store.player_selected_species = selected
    publish_team_preview(store)
    logger.info("Player selected bring: %s", ", ".join(selected))


def _collect_battle_animation_events(
    frame,
    event_ocr: EventOcrEngine,
    config,
    *,
    player_species,
    opponent_species,
):
    """OCR changed banners/battle text; return events (do not touch SessionStore)."""
    logger.info("Processing battle animation events")
    return event_ocr.process_frame(
        frame,
        config,
        player_species=player_species,
        opponent_species=opponent_species,
    )


def _collect_hp_animation_events(
    frame,
    hp_reader: HPReader,
    config,
    game_state,
    *,
    player_species,
    opponent_species,
):
    """Poll slot cards; return stable HPChangeEvents (do not touch SessionStore)."""
    logger.info("Processing HP animation frame")
    return hp_reader.process_animation_frame(
        frame,
        config,
        game_state,
        player_species=player_species,
        opponent_species=opponent_species,
    )


def _append_battle_animation_events(store: SessionStore, events) -> None:
    for event in events:
        store.append_battle_log(event)
        logger.info("Battle log event: %s — %r", event.type, event.raw_text)


def _append_hp_events(store: SessionStore, events, *, snapshot: bool = False) -> None:
    label = "HP snapshot reconcile" if snapshot else "HP change"
    for event in events:
        store.append_battle_log(event)
        logger.info(
            "%s: %s slot %s %+d%% — %r",
            label,
            event.pokemon.side,
            event.pokemon.slot,
            event.hp_pct_change,
            event.raw_text,
        )


def _process_battle_animation_events(
    store: SessionStore,
    frame,
    event_ocr: EventOcrEngine,
    config,
) -> None:
    """OCR changed regions and append parsed events (sync helper for tests)."""
    events = _collect_battle_animation_events(
        frame,
        event_ocr,
        config,
        player_species=store.player_selected_species,
        opponent_species=store.opponent_team_species,
    )
    _append_battle_animation_events(store, events)


def _process_hp_animation_frame(
    store: SessionStore,
    frame,
    hp_reader: HPReader,
    config,
) -> None:
    """Poll slot cards and append HP events (sync helper for tests)."""
    events = _collect_hp_animation_events(
        frame,
        hp_reader,
        config,
        store.game_state,
        player_species=store.player_selected_species,
        opponent_species=store.opponent_team_species,
    )
    _append_hp_events(store, events)


def _process_hp_action_selection_snapshot(
    store: SessionStore,
    frame,
    hp_reader: HPReader,
    config,
) -> None:
    """Authoritative 4-slot HP snapshot on action_selection entry."""
    logger.info("Processing HP action selection snapshot")
    events = hp_reader.read_action_selection_snapshot(
        frame,
        config,
        store.game_state,
        player_species=store.player_selected_species,
        opponent_species=store.opponent_team_species,
    )
    _append_hp_events(store, events, snapshot=True)


def _emit_turn_start_on_action_selection_entry(store: SessionStore) -> None:
    """Increment turn counter and append TurnStartEvent on action_selection entry."""
    store.turn_number += 1
    event = TurnStartEvent(
        raw_text=f"Turn {store.turn_number}",
        turn_number=store.turn_number,
    )
    store.append_battle_log(event)
    logger.info("Turn start: %d", store.turn_number)


async def _process_turn_suggestion(store: SessionStore) -> None:
    """Request a Gemini turn suggestion once per turn (debounced).

    Requires team-preview opponent species; skips prompting when missing.
    """
    if store.player_team is None or store.game_state is None:
        return
    if store.turn_number < 1:
        return
    if not store.opponent_team_species:
        logger.debug(
            "Skipping turn suggestion for turn %d: opponent team species unknown",
            store.turn_number,
        )
        return
    if store._turn_suggestion_turn == store.turn_number:
        return

    gemini = create_gemini_service(interaction_id=store.gemini_interaction_id)

    try:
        recent = previous_turn_battle_log_events(store.battle_logs, store.turn_number)
        suggestion = await gemini.suggest_turn(
            store.game_state,
            store.player_team,
            recent,
            turn_number=store.turn_number,
            opponent_team_species=store.opponent_team_species,
        )
        store.turn_suggestion = suggestion
        store._turn_suggestion_turn = store.turn_number
        store.gemini_interaction_id = gemini.interaction_id
        publish_turn_suggestion(store)
        logger.info("Turn suggestion ready for turn %d", store.turn_number)
    except Exception:
        logger.exception("Turn suggestion failed for turn %d", store.turn_number)
        store.gemini_interaction_id = gemini.interaction_id


async def _cv_loop(store: SessionStore) -> None:
    logger.info("CV loop started")
    detector = PhaseDetector()
    event_ocr = EventOcrEngine()
    hp_reader = HPReader()
    region_config = load_regions()
    try:
        while store.cv_running:
            previous_adb = store.adb_connected
            store.adb_connected = await asyncio.to_thread(is_adb_connected)
            logger.info("ADB connected: %s", store.adb_connected)
            if store.adb_connected != previous_adb:
                publish_session(store)
            if not store.adb_connected:
                logger.debug("ADB not connected; retrying in %.0fs", _ADB_PROBE_INTERVAL_SEC)
                await asyncio.sleep(_ADB_PROBE_INTERVAL_SEC)
                continue

            try:
                frame = await asyncio.to_thread(capture_screenshot)
            except RuntimeError as exc:
                logger.debug("Screenshot capture failed: %s", exc)
                if store.adb_connected:
                    store.adb_connected = False
                    publish_session(store)
                await asyncio.sleep(_ADB_PROBE_INTERVAL_SEC)
                continue

            transition = detector.detect_transition(frame)
            previous_phase = store.phase
            store.phase = transition.current

            if store.phase != previous_phase:
                publish_phase(store)

            if transition.entered_team_preview:
                store.begin_battle()
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
                publish_state(store)

            if transition.entered_action_selection:
                _emit_turn_start_on_action_selection_entry(store)
                await asyncio.to_thread(
                    _process_hp_action_selection_snapshot,
                    store,
                    frame,
                    hp_reader,
                    region_config,
                )
                await _process_turn_suggestion(store)

            if transition.current == BattlePhase.BATTLE_ANIMATION:
                player_species = store.player_selected_species
                opponent_species = store.opponent_team_species
                game_state = store.game_state
                event_events, hp_events = await asyncio.gather(
                    asyncio.to_thread(
                        _collect_battle_animation_events,
                        frame,
                        event_ocr,
                        region_config,
                        player_species=player_species,
                        opponent_species=opponent_species,
                    ),
                    asyncio.to_thread(
                        _collect_hp_animation_events,
                        frame,
                        hp_reader,
                        region_config,
                        game_state,
                        player_species=player_species,
                        opponent_species=opponent_species,
                    ),
                )
                # Append on the event loop thread so SessionStore stays single-threaded.
                _append_battle_animation_events(store, event_events)
                _append_hp_events(store, hp_events)
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

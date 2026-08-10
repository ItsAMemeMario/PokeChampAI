from __future__ import annotations

from enum import Enum

from app.schema.battle_log import BattleLogEvent, LeadInEvent, MoveUsedEvent, TurnStartEvent
from app.schema.gamestate import GameState
from app.schema.suggestions import TeamPreviewSuggestion, TurnSuggestion
from app.schema.team import PlayerTeam

import logging

logger = logging.getLogger(__name__)


class BattlePhase(str, Enum):
    IDLE = "idle"
    TEAM_PREVIEW = "team_preview"
    TEAM_SELECTED = "team_selected"
    BATTLE_ANIMATION = "battle_animation"
    ACTION_SELECTION = "action_selection"


class SessionStore:
    """In-memory session state for the single-user local app."""

    def __init__(self) -> None:
        self.player_team: PlayerTeam | None = None
        self.phase: BattlePhase = BattlePhase.IDLE
        self.cv_running: bool = False
        self.adb_connected: bool = False
        self.game_state: GameState | None = None
        # Indexed by turn number (1-based). Index 0 is unused.
        # Each turn list starts with a TurnStartEvent.
        self.battle_logs: list[list[BattleLogEvent]] = [[]]
        self.opponent_team_species: list[str] | None = None
        self.player_selected_species: list[str] | None = None
        self.team_preview_suggestion: TeamPreviewSuggestion | None = None
        self.turn_suggestion: TurnSuggestion | None = None
        self.turn_number: int = 0
        self._team_preview_processed: bool = False
        # Debounce: turn number for which we already stored a turn suggestion.
        self._turn_suggestion_turn: int | None = None
        # Gemini Interactions API conversation id for the current battle.
        self.gemini_interaction_id: str | None = None

    @property
    def team_loaded(self) -> bool:
        return self.player_team is not None

    def set_team(self, team: PlayerTeam) -> None:
        self.player_team = team

    def begin_battle(self) -> None:
        """Reset per-battle state when a new team preview starts.

        Clears the Gemini conversation so the next prompt creates a fresh
        interaction rather than continuing the previous match.
        """
        self.turn_number = 0
        self.game_state = None
        self.battle_logs = [[]]
        self.opponent_team_species = None
        self.player_selected_species = None
        self.team_preview_suggestion = None
        self.turn_suggestion = None
        self._team_preview_processed = False
        self._turn_suggestion_turn = None
        self.gemini_interaction_id = None

    def start_monitoring(self) -> None:
        if self.player_team is None:
            raise ValueError("Team must be saved before starting monitoring")
        self.cv_running = True
        self.phase = BattlePhase.IDLE
        self.begin_battle()

    def stop_monitoring(self) -> None:
        self.cv_running = False
        self.adb_connected = False
        self.phase = BattlePhase.IDLE

    def append_battle_log(self, event: BattleLogEvent) -> list[tuple[int, int]]:
        """Append a parsed CV event into the current turn's log.

        ``TurnStartEvent`` opens ``battle_logs[turn_number]`` as a new turn
        list (starting with that event). All other events append to the
        active turn. Events that arrive before the first ``TurnStartEvent``
        (opening lead switch-ins during battle animation) append to
        ``battle_logs[0]``. After append: completer patches partial fields,
        then the reducer applies the (possibly patched) event to ``game_state``.
        Returns ``(turn, index)`` pairs patched by the completer.
        """
        logger.info("Appending battle log event: %s", event)
        if isinstance(event, TurnStartEvent):
            turn = event.turn_number
            if turn < 1:
                raise ValueError(f"TurnStartEvent turn_number must be >= 1, got {turn}")
            self.turn_number = turn
            while len(self.battle_logs) <= turn:
                self.battle_logs.append([])
            self.battle_logs[turn] = [event]
        else:
            turn = self.turn_number
            if turn < 1:
                # Pre-turn / lead-in bucket (before first action_selection).
                turn = 0
                while len(self.battle_logs) <= 0:
                    self.battle_logs.append([])
            else:
                while len(self.battle_logs) <= turn:
                    self.battle_logs.append([])
                if not self.battle_logs[turn]:
                    raise ValueError(
                        f"Turn {turn} is not open; append a TurnStartEvent first"
                    )

            turn_logs = self.battle_logs[turn]
            event_key = _semantic_key(event)
            if turn_logs and _is_ocr_reread(turn_logs[-1], event):
                turn_logs[-1] = event
            elif any(_semantic_key(existing) == event_key for existing in turn_logs):
                return []
            else:
                turn_logs.append(event)

        # Local imports avoid circular dependencies at module load time.
        from app.services.battle_log_completer import complete_battle_logs
        from app.services.gamestate_reducer import apply_event_to_store

        patched = complete_battle_logs(self)
        # Apply the (possibly completer-patched) event at the end of this turn.
        apply_event_to_store(self, self.battle_logs[turn][-1])

        # Live dashboard: push log patches, the appended event, and latest state.
        from app.services.ws_hub import (
            publish_log,
            publish_log_patched,
            publish_state,
        )

        for patch_turn, patch_index in patched:
            publish_log_patched(
                patch_turn,
                patch_index,
                self.battle_logs[patch_turn][patch_index],
            )
        publish_log(self.battle_logs[turn][-1])
        publish_state(self)
        return patched


def _is_ocr_reread(previous: BattleLogEvent, new: BattleLogEvent) -> bool:
    """True when ``new`` is a cleaner OCR of the same on-screen message as ``previous``."""
    if previous.type != new.type:
        return False
    if isinstance(new, MoveUsedEvent) and isinstance(previous, MoveUsedEvent):
        # Same side's move text often re-OCRs with different spellings before clear.
        return previous.actor.side == new.actor.side
    if isinstance(new, LeadInEvent) and isinstance(previous, LeadInEvent):
        return previous.side == new.side

    prev_mon = getattr(previous, "pokemon", None)
    new_mon = getattr(new, "pokemon", None)
    if prev_mon is None or new_mon is None:
        return False
    if prev_mon.side != new_mon.side:
        return False
    # Dual switch-ins are different species; only collapse same-species jitter.
    if new.type in {"switch_in", "switch_out", "faint", "item_used"}:
        return prev_mon.species == new_mon.species
    if new.type in {"stat_change", "status_applied", "volatile_applied"}:
        return prev_mon.species == new_mon.species
    return False


def _semantic_key(event: BattleLogEvent) -> tuple:
    """Turn-level identity used to drop late re-emits of the same CV event."""
    key: list[object] = [event.type]
    if isinstance(event, LeadInEvent):
        key.extend([event.side, event.slot_1.species, event.slot_2.species])
        return tuple(key)
    pokemon = getattr(event, "pokemon", None) or getattr(event, "actor", None)
    if pokemon is not None:
        # Match OCR fingerprints: switch identity is species+side, not slot.
        if event.type in {"switch_in", "switch_out"}:
            key.extend([pokemon.species, pokemon.side])
        else:
            key.extend([pokemon.species, pokemon.side, pokemon.slot])
    if event.type == "stat_change":
        key.extend(
            [getattr(event, "stat", None), getattr(event, "stages_delta", None)]
        )
    if event.type in {"move_used", "move_failed"}:
        key.append(getattr(event, "move", None))
    if event.type == "item_used":
        key.append(getattr(event, "item", None))
    if event.type == "ability_triggered":
        key.append(getattr(event, "ability", None))
    if event.type in {"status_applied", "volatile_applied"}:
        key.append(getattr(event, "status", None) or getattr(event, "volatile", None))
    return tuple(key)


session_store = SessionStore()

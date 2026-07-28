from __future__ import annotations

from enum import Enum

from app.schema.battle_log import BattleLogEvent, TurnStartEvent
from app.schema.gamestate import GameState
from app.schema.suggestions import TeamPreviewSuggestion, TurnSuggestion
from app.schema.team import PlayerTeam


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

    @property
    def team_loaded(self) -> bool:
        return self.player_team is not None

    def set_team(self, team: PlayerTeam) -> None:
        self.player_team = team

    def start_monitoring(self) -> None:
        if self.player_team is None:
            raise ValueError("Team must be saved before starting monitoring")
        self.cv_running = True
        self.phase = BattlePhase.IDLE
        self.turn_number = 0
        self.game_state = None
        self.battle_logs = [[]]
        self.opponent_team_species = None
        self.player_selected_species = None
        self.team_preview_suggestion = None
        self.turn_suggestion = None
        self._team_preview_processed = False
        self._turn_suggestion_turn = None

    def stop_monitoring(self) -> None:
        self.cv_running = False
        self.adb_connected = False
        self.phase = BattlePhase.IDLE

    def append_battle_log(self, event: BattleLogEvent) -> list[tuple[int, int]]:
        """Append a parsed CV event into the current turn's log.

        ``TurnStartEvent`` opens ``battle_logs[turn_number]`` as a new turn
        list (starting with that event). All other events append to the
        active turn. After append: completer patches partial fields, then the
        reducer applies the (possibly patched) event to ``game_state``.
        Returns ``(turn, index)`` pairs patched by the completer.
        """
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
                raise ValueError(
                    "Cannot append battle log event before a TurnStartEvent "
                    f"(turn_number={self.turn_number})"
                )
            while len(self.battle_logs) <= turn:
                self.battle_logs.append([])
            if not self.battle_logs[turn]:
                raise ValueError(
                    f"Turn {turn} is not open; append a TurnStartEvent first"
                )
            self.battle_logs[turn].append(event)

        # Local imports avoid circular dependencies at module load time.
        from app.services.battle_log_completer import complete_battle_logs
        from app.services.gamestate_reducer import apply_event_to_store

        patched = complete_battle_logs(self)
        # Apply the (possibly completer-patched) event at the end of this turn.
        apply_event_to_store(self, self.battle_logs[turn][-1])
        return patched


session_store = SessionStore()

from __future__ import annotations

from enum import Enum

from app.schema.battle_log import BattleLogEvent
from app.schema.gamestate import GameState
from app.schema.suggestions import TeamPreviewSuggestion, TurnSuggestion
from app.schema.team import PlayerTeam


class BattlePhase(str, Enum):
    IDLE = "idle"
    TEAM_PREVIEW = "team_preview"
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
        self.battle_logs: list[BattleLogEvent] = []
        self.team_preview_suggestion: TeamPreviewSuggestion | None = None
        self.turn_suggestion: TurnSuggestion | None = None
        self.turn_number: int = 0

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
        self.battle_logs.clear()
        self.team_preview_suggestion = None
        self.turn_suggestion = None

    def stop_monitoring(self) -> None:
        self.cv_running = False
        self.adb_connected = False
        self.phase = BattlePhase.IDLE


session_store = SessionStore()

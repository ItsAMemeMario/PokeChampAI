"""Mock Gemini service for local testing without an API key."""

from __future__ import annotations

import logging
import random
from typing import Literal

import numpy as np

from app.schema.battle_log import BattleLogEvent
from app.schema.common import Pokemon, Slot
from app.schema.gamestate import GameState, SideState
from app.schema.suggestions import Move, Switch, TeamPreviewSuggestion, TurnAction, TurnSuggestion
from app.schema.team import PlayerTeam
from app.services.gemini import validate_turn_suggestion_moves

logger = logging.getLogger(__name__)

FILLER_TEXT = "Mock Gemini filler text."

# Fixed legal species used when vision would otherwise identify the opponent six.
MOCK_OPPONENT_SPECIES: list[str] = [
    "Mawile",
    "Musharna",
    "Metagross",
    "Torkoal",
    "Dragapult",
    "Meowscarada",
]

_MOCK_INTERACTION_ID = "mock-interaction"


class MockGeminiService:
    """Drop-in stand-in for ``GeminiService`` that returns deterministic filler data.

    Used when ``GEMINI_API_KEY`` is unset so the CV / suggestion pipeline can be
    exercised without calling Google's API.
    """

    def __init__(self, *, interaction_id: str | None = None) -> None:
        self._interaction_id = interaction_id or _MOCK_INTERACTION_ID
        logger.info("MockGeminiService active (no GEMINI_API_KEY)")

    @property
    def interaction_id(self) -> str | None:
        return self._interaction_id

    async def identify_opponent_species(self, image: np.ndarray) -> list[str]:
        """Return a fixed six-species filler team (image is ignored)."""
        del image  # unused — no vision in mock mode
        return list(MOCK_OPPONENT_SPECIES)

    async def suggest_team_preview(
        self,
        player_team: PlayerTeam,
        opponent_species: list[str],
    ) -> TeamPreviewSuggestion:
        """Suggest the first four Pokemon from each side as bring / lead."""
        player_bring = [mon.species for mon in player_team.pokemon[:4]]
        opponent_bring = list(opponent_species[:4])
        if len(player_bring) != 4:
            raise ValueError(
                f"Mock team preview requires 4+ player Pokemon, got {len(player_bring)}"
            )
        if len(opponent_bring) != 4:
            raise ValueError(
                f"Mock team preview requires 4+ opponent species, got {len(opponent_bring)}"
            )
        return TeamPreviewSuggestion(
            predicted_opponent_bring=opponent_bring,
            predicted_opponent_lead_pair=(opponent_bring[0], opponent_bring[1]),
            suggested_player_bring=player_bring,
            suggested_player_lead_pair=(player_bring[0], player_bring[1]),
            reasoning=FILLER_TEXT,
        )

    async def suggest_turn(
        self,
        game_state: GameState,
        player_team: PlayerTeam,
        recent_events: list[BattleLogEvent] | list[dict],
        *,
        turn_number: int | None = None,
        opponent_team_species: list[str] | None = None,
    ) -> TurnSuggestion:
        """For each active slot, randomly suggest a legal move or switch."""
        del recent_events  # unused — mock does not reason about battle log
        if not opponent_team_species:
            raise ValueError("opponent_team_species is required for turn suggestions")

        resolved_turn = turn_number if turn_number is not None else game_state.turn_number
        moves_by_species = {
            mon.species.casefold(): list(mon.moves) for mon in player_team.pokemon
        }
        opponent_targets = _active_targets(game_state.opponent)

        claimed_switch_ins: set[str] = set()
        actions: list[TurnAction] = [
            _random_slot_action(
                game_state=game_state,
                slot=slot,
                moves_by_species=moves_by_species,
                opponent_targets=opponent_targets,
                claimed_switch_ins=claimed_switch_ins,
            )
            for slot in (1, 2)
        ]

        suggestion = TurnSuggestion(
            turn_number=resolved_turn,
            actions=actions,
            overall_reasoning=FILLER_TEXT,
        )
        validate_turn_suggestion_moves(suggestion, player_team)
        return suggestion


def _active_targets(side: SideState) -> list[Pokemon]:
    targets: list[Pokemon] = []
    for slot, active in ((1, side.slot_1), (2, side.slot_2)):
        if active is None:
            continue
        targets.append(Pokemon(species=active.species, side="opponent", slot=slot))
    return targets


def _random_slot_action(
    *,
    game_state: GameState,
    slot: Slot,
    moves_by_species: dict[str, list[str]],
    opponent_targets: list[Pokemon],
    claimed_switch_ins: set[str],
) -> TurnAction:
    active = game_state.player.slot_1 if slot == 1 else game_state.player.slot_2
    if active is None:
        raise ValueError(f"Player slot {slot} is empty; cannot mock a turn action")

    actor = Pokemon(species=active.species, side="player", slot=slot)
    known_moves = moves_by_species.get(active.species.casefold(), [])
    available_bench = [
        mon
        for mon in game_state.player.benched
        if mon.species.casefold() not in claimed_switch_ins
    ]

    choices: list[Literal["move", "switch"]] = []
    if known_moves:
        choices.append("move")
    if available_bench:
        choices.append("switch")
    if not choices:
        raise ValueError(
            f"No legal mock actions for {active.species} in slot {slot} "
            "(no known moves and no available bench)"
        )

    kind = random.choice(choices)
    if kind == "move":
        move_name = random.choice(known_moves)
        targets = [random.choice(opponent_targets)] if opponent_targets else []
        action: Move | Switch = Move(
            actor=actor,
            mega=False,
            move=move_name,
            targets=targets,
        )
    else:
        switch_in_mon = random.choice(available_bench)
        claimed_switch_ins.add(switch_in_mon.species.casefold())
        action = Switch(
            switch_out=actor,
            switch_in=Pokemon(
                species=switch_in_mon.species,
                side="player",
                slot=slot,
            ),
        )

    return TurnAction(action=action, reasoning=FILLER_TEXT)

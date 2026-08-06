"""Gemini API integration for team preview vision and suggestions."""

from __future__ import annotations

import base64
import json
import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np
from google import genai
from PIL import Image

from app.schema.battle_log import BattleLogEvent
from app.schema.gamestate import GameState
from app.schema.suggestions import Move, Switch, TeamPreviewSuggestion, TurnSuggestion
from app.schema.team import OpponentTeamPreview, PlayerTeam
from app.data.species import REGULATION_MB_SPECIES
from app.util.legal_snap import snap_to_legal

if TYPE_CHECKING:
    from app.services.mock_gemini import MockGeminiService

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.6-flash"


def create_gemini_service(
    *,
    api_key: str | None = None,
    model: str | None = None,
    client: genai.Client | None = None,
    interaction_id: str | None = None,
) -> GeminiService | MockGeminiService:
    """Return a real Gemini client, or ``MockGeminiService`` when no API key is set.

    Pass an explicit ``client`` (as tests do) to force the real service without a key.
    """
    from app.services.mock_gemini import MockGeminiService

    resolved_key = (
        api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
    ).strip()
    if client is None and not resolved_key:
        return MockGeminiService(interaction_id=interaction_id)
    return GeminiService(
        api_key=api_key,
        model=model,
        client=client,
        interaction_id=interaction_id,
    )


class GeminiService:
    """Thin wrapper around google-genai Interactions API for battle-assist prompts.

    Keeps a stateful multi-turn conversation via ``previous_interaction_id`` for
    the duration of one battle (team preview → battle end). Pass ``interaction_id``
    from session state on construction; callers persist the updated ID after use.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: genai.Client | None = None,
        interaction_id: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        if client is None and not resolved_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self._model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self._client = client or genai.Client(api_key=resolved_key)
        self._interaction_id = interaction_id

    @property
    def interaction_id(self) -> str | None:
        """Latest interaction ID in the current battle conversation, if any."""
        return self._interaction_id

    @staticmethod
    def _rgb_to_png_bytes(image: np.ndarray) -> bytes:
        pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _json_response_format(schema: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            }
        ]

    async def _create_interaction(
        self,
        *,
        input: str | list[dict[str, Any]],
        response_schema: dict[str, Any],
    ) -> str:
        """Create an interaction, chaining to the prior turn when available."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": input,
            "response_format": self._json_response_format(response_schema),
        }
        if self._interaction_id:
            kwargs["previous_interaction_id"] = self._interaction_id
        interaction = await self._client.aio.interactions.create(**kwargs)
        self._interaction_id = interaction.id
        return interaction.output_text or "{}"

    async def identify_opponent_species(self, image: np.ndarray) -> list[str]:
        """Identify six opponent species from team-preview sprite crops."""
        prompt = (
            "You are analyzing a Pokemon Champions team preview screen crop. "
            "The image shows six opponent Pokemon sprites stacked vertically, top to bottom. "
            "No species names are visible — identify each Pokemon from its sprite and any "
            "visible type icons. Return exactly six species names in top-to-bottom order "
            "using standard English Showdown names. Use hyphenated form names when the "
            "sprite is a regional or alternate form (e.g. 'Goodra-Hisui', 'Lycanroc-Dusk'). "
            "Do not use mega or G-Max names. "
            # I kid you not this makes Gemini stop hallucinating on Floette-Eternal
            "Be very accurate, no mistake can be afforded."
        )
        image_b64 = base64.b64encode(self._rgb_to_png_bytes(image)).decode("utf-8")
        output_text = await self._create_interaction(
            input=[
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "mime_type": "image/png",
                    "data": image_b64,
                },
            ],
            response_schema=OpponentTeamPreview.model_json_schema(),
        )
        parsed = OpponentTeamPreview.model_validate_json(output_text)
        return [
            snap_to_legal(name, REGULATION_MB_SPECIES) or name.strip()
            for name in parsed.species
        ]

    async def suggest_team_preview(
        self,
        player_team: PlayerTeam,
        opponent_species: list[str],
    ) -> TeamPreviewSuggestion:
        """Suggest bring-4 and lead pairs given the player's team and opponent's six."""
        player_payload = [mon.model_dump() for mon in player_team.pokemon]
        prompt = (
            "You are a Pokemon Champions Regulation M-B doubles VGC expert.\n"
            f"Player team: {json.dumps(player_payload)}\n"
            f"Opponent team: {json.dumps(opponent_species)}\n"
            "Predict which 4 the opponent will bring and suggest which 4 the player should "
            "bring, including selection order (first two entries in each bring list are the "
            "lead pair). Return JSON matching the TeamPreviewSuggestion schema."
        )
        output_text = await self._create_interaction(
            input=prompt,
            response_schema=TeamPreviewSuggestion.model_json_schema(),
        )
        return TeamPreviewSuggestion.model_validate_json(output_text)

    async def suggest_turn(
        self,
        game_state: GameState,
        player_team: PlayerTeam,
        recent_events: list[BattleLogEvent] | list[dict],
        *,
        turn_number: int | None = None,
        opponent_team_species: list[str] | None = None,
    ) -> TurnSuggestion:
        """Suggest actions for both active player Pokemon this turn.

        Uses structured JSON output matching ``TurnSuggestion``. Player move
        choices are constrained to each species' known pokepaste moveset and
        validated after the model responds. ``opponent_team_species`` (team-preview
        vision) is required so the model can reason about unrevealed bench threats.
        """
        if not opponent_team_species:
            raise ValueError(
                "opponent_team_species is required for turn suggestions"
            )
        resolved_turn = turn_number if turn_number is not None else game_state.turn_number
        known_moves = {mon.species: list(mon.moves) for mon in player_team.pokemon}
        player_payload = [mon.model_dump() for mon in player_team.pokemon]
        event_payload = _serialize_events(recent_events)
        opponent_six = list(opponent_team_species)
        revealed_opponent = _revealed_opponent_species(game_state)
        leftover_opponent = [
            species
            for species in opponent_six
            if species.casefold() not in {s.casefold() for s in revealed_opponent}
        ]
        bring_complete = len(revealed_opponent) >= 4
        if bring_complete:
            leftover_label = "Opponent species not brought to battle"
            leftover_guidance = (
                "All four opponent brings are known; species not brought cannot switch in. "
                "Only target Pokemon currently on the field.\n"
            )
        else:
            leftover_label = (
                "Opponent species still unrevealed (possible remaining bring / bench)"
            )
            leftover_guidance = (
                "Factor unrevealed opponent species into risk assessment (switches, "
                "coverage, speed control), but only target Pokemon currently on the field.\n"
            )
        prompt = (
            "You are a Pokemon Champions Regulation M-B doubles VGC expert.\n"
            f"Current game state: {json.dumps(game_state.model_dump(mode='json', by_alias=True))}\n"
            f"Player full team (known): {json.dumps(player_payload)}\n"
            f"Opponent team from preview (6): {json.dumps(opponent_six)}\n"
            f"Opponent species already revealed in battle: "
            f"{json.dumps(sorted(revealed_opponent))}\n"
            f"{leftover_label}: {json.dumps(leftover_opponent)}\n"
            f"{leftover_guidance}"
            f"Known legal moves by species (MUST use only these for player Move actions): "
            f"{json.dumps(known_moves)}\n"
            f"Previous turn battle log ({len(event_payload)} events): {json.dumps(event_payload)}\n"
            f"Suggest actions for both active player Pokemon for turn {resolved_turn}.\n"
            "Return exactly two TurnAction entries — one for player slot 1 and one for "
            "player slot 2. Each action is either a Move (actor must be the player Pokemon "
            "in that slot; move must be from that species' known legal moves; set mega=true "
            "only if mega evolving this turn) or a Switch (switch_out = active player mon, "
            "switch_in = a benched player species). Do not invent illegal moves.\n"
            "Return JSON matching the TurnSuggestion schema."
        )
        output_text = await self._create_interaction(
            input=prompt,
            response_schema=TurnSuggestion.model_json_schema(),
        )
        suggestion = TurnSuggestion.model_validate_json(output_text)
        # Prefer the authoritative turn number from the session/game state.
        if suggestion.turn_number != resolved_turn:
            suggestion = suggestion.model_copy(update={"turn_number": resolved_turn})
        validate_turn_suggestion_moves(suggestion, player_team)
        return suggestion


def _serialize_events(
    events: list[BattleLogEvent] | list[dict],
) -> list[dict]:
    serialized: list[dict] = []
    for event in events:
        if isinstance(event, dict):
            serialized.append(event)
        else:
            serialized.append(event.model_dump(mode="json"))
    return serialized


def _revealed_opponent_species(game_state: GameState) -> list[str]:
    """Species currently known on the opponent side (active + bench), original casing."""
    revealed: list[str] = []
    seen: set[str] = set()
    side = game_state.opponent
    for slot in (side.slot_1, side.slot_2):
        if slot is None:
            continue
        key = slot.species.casefold()
        if key not in seen:
            seen.add(key)
            revealed.append(slot.species)
    for mon in side.benched:
        key = mon.species.casefold()
        if key not in seen:
            seen.add(key)
            revealed.append(mon.species)
    return revealed


def validate_turn_suggestion_moves(
    suggestion: TurnSuggestion,
    player_team: PlayerTeam,
) -> None:
    """Raise ValueError if any player Move uses a move not on the pokepaste set."""
    moves_by_species = {
        mon.species.casefold(): set(mon.moves) for mon in player_team.pokemon
    }
    team_species = {mon.species.casefold() for mon in player_team.pokemon}

    for turn_action in suggestion.actions:
        action = turn_action.action
        if isinstance(action, Move):
            if action.actor.side != "player":
                raise ValueError(
                    f"Turn suggestion Move actor must be player side, got {action.actor.side}"
                )
            known = moves_by_species.get(action.actor.species.casefold())
            if known is None:
                raise ValueError(
                    f"Turn suggestion Move actor species '{action.actor.species}' "
                    "is not on the player team"
                )
            if action.move not in known:
                raise ValueError(
                    f"Illegal move '{action.move}' for {action.actor.species}; "
                    f"known moves: {sorted(known)}"
                )
        elif isinstance(action, Switch):
            if action.switch_out.side != "player" or action.switch_in.side != "player":
                raise ValueError("Turn suggestion Switch must use player-side Pokemon")
            if action.switch_in.species.casefold() not in team_species:
                raise ValueError(
                    f"Switch-in species '{action.switch_in.species}' is not on the player team"
                )


def previous_turn_battle_log_events(
    battle_logs: list[list[BattleLogEvent]],
    turn_number: int,
) -> list[BattleLogEvent]:
    """Return all events from the turn before ``turn_number`` (empty on turn 1)."""
    previous = turn_number - 1
    if previous < 1 or previous >= len(battle_logs):
        return []
    return list(battle_logs[previous])

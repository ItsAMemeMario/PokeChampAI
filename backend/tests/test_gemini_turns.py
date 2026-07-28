"""Tests for Gemini turn suggestion service and CV wiring."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schema.battle_log import MoveUsedEvent, TurnStartEvent
from app.schema.common import Pokemon
from app.schema.gamestate import (
    ActivePokemon,
    BenchedPokemon,
    FieldState,
    GameState,
    Hazards,
    SideState,
    StatStages,
)
from app.schema.suggestions import Move, Switch, TurnAction, TurnSuggestion
from app.schema.team import PlayerPokemon, PlayerTeam
from app.services.gemini import (
    GeminiService,
    previous_turn_battle_log_events,
    validate_turn_suggestion_moves,
)
from app.services.session import SessionStore


def _sample_team() -> PlayerTeam:
    return PlayerTeam(
        pokemon=[
            PlayerPokemon(
                species="Sinistcha",
                item="Sitrus Berry",
                ability="Hospitality",
                evs={"hp": 252},
                nature="Bold",
                moves=["Matcha Gotcha", "Rage Powder", "Trick Room", "Protect"],
            ),
            PlayerPokemon(
                species="Staraptor",
                item="Staraptorite",
                ability="Intimidate",
                evs={"atk": 252},
                nature="Jolly",
                moves=["Brave Bird", "Close Combat", "U-turn", "Protect"],
            ),
            PlayerPokemon(
                species="Garchomp",
                item="Lum Berry",
                ability="Rough Skin",
                evs={"atk": 252},
                nature="Jolly",
                moves=["Earthquake", "Outrage", "Rock Slide", "Protect"],
            ),
            PlayerPokemon(
                species="Incineroar",
                item="Wide Lens",
                ability="Intimidate",
                evs={"hp": 252},
                nature="Careful",
                moves=["Fake Out", "Flare Blitz", "Knock Off", "Parting Shot"],
            ),
            PlayerPokemon(
                species="Rillaboom",
                item="Leftovers",
                ability="Grassy Surge",
                evs={"atk": 252},
                nature="Adamant",
                moves=["Grassy Glide", "Wood Hammer", "U-turn", "Protect"],
            ),
            PlayerPokemon(
                species="Urshifu-Rapid-Strike",
                item="Focus Sash",
                ability="Unseen Fist",
                evs={"atk": 252},
                nature="Jolly",
                moves=["Surging Strikes", "Close Combat", "Aqua Jet", "Protect"],
            ),
        ]
    )


def _active(species: str, hp: int = 100) -> ActivePokemon:
    return ActivePokemon(
        species=species,
        hp_percentage=hp,
        stat_stages=StatStages(),
    )


def _side(
    s1: ActivePokemon | None,
    s2: ActivePokemon | None,
    *,
    benched: list[BenchedPokemon] | None = None,
) -> SideState:
    return SideState(
        slot_1=s1,
        slot_2=s2,
        benched=benched or [],
        hazards=Hazards(spikes=0, toxic_spikes=0, stealth_rocks=0),
    )


def _game_state(
    *,
    turn: int = 1,
    opponent_bench: list[str] | None = None,
) -> GameState:
    bench = [
        BenchedPokemon(species=species, hp_percentage=100)
        for species in (opponent_bench or [])
    ]
    return GameState(
        turn_number=turn,
        field=FieldState(),
        player=_side(_active("Sinistcha"), _active("Staraptor")),
        opponent=_side(_active("Scizor"), _active("Hatterene"), benched=bench),
    )


def _legal_suggestion(turn: int = 1) -> TurnSuggestion:
    return TurnSuggestion(
        turn_number=turn,
        actions=[
            TurnAction(
                action=Move(
                    actor=Pokemon(species="Sinistcha", side="player", slot=1),
                    mega=False,
                    move="Matcha Gotcha",
                    targets=[Pokemon(species="Scizor", side="opponent", slot=1)],
                ),
                reasoning="Chip both foes",
            ),
            TurnAction(
                action=Move(
                    actor=Pokemon(species="Staraptor", side="player", slot=2),
                    mega=True,
                    move="Brave Bird",
                    targets=[Pokemon(species="Hatterene", side="opponent", slot=2)],
                ),
                reasoning="KO threat",
            ),
        ],
        overall_reasoning="Pressure both slots",
    )


def test_validate_turn_suggestion_accepts_legal_moves() -> None:
    validate_turn_suggestion_moves(_legal_suggestion(), _sample_team())


def test_validate_turn_suggestion_rejects_illegal_move() -> None:
    bad = TurnSuggestion(
        turn_number=1,
        actions=[
            TurnAction(
                action=Move(
                    actor=Pokemon(species="Sinistcha", side="player", slot=1),
                    mega=False,
                    move="Hyper Beam",
                    targets=[],
                ),
                reasoning="Bad",
            ),
            TurnAction(
                action=Move(
                    actor=Pokemon(species="Staraptor", side="player", slot=2),
                    mega=False,
                    move="Brave Bird",
                    targets=[],
                ),
                reasoning="Ok",
            ),
        ],
        overall_reasoning="Illegal",
    )
    with pytest.raises(ValueError, match="Illegal move"):
        validate_turn_suggestion_moves(bad, _sample_team())


def test_validate_turn_suggestion_accepts_switch() -> None:
    suggestion = TurnSuggestion(
        turn_number=1,
        actions=[
            TurnAction(
                action=Move(
                    actor=Pokemon(species="Sinistcha", side="player", slot=1),
                    mega=False,
                    move="Protect",
                    targets=[],
                ),
                reasoning="Stall",
            ),
            TurnAction(
                action=Switch(
                    switch_out=Pokemon(species="Staraptor", side="player", slot=2),
                    switch_in=Pokemon(species="Incineroar", side="player", slot=2),
                ),
                reasoning="Bring Intimidate",
            ),
        ],
        overall_reasoning="Pivot",
    )
    validate_turn_suggestion_moves(suggestion, _sample_team())


def test_previous_turn_battle_log_events_returns_full_prior_turn() -> None:
    logs: list[list] = [[]]
    logs.append(
        [
            TurnStartEvent(raw_text="Turn 1", turn_number=1),
            MoveUsedEvent(
                raw_text="used",
                actor=Pokemon(species="Sinistcha", side="player", slot=1),
                move="Protect",
                targets=[],
            ),
            MoveUsedEvent(
                raw_text="used",
                actor=Pokemon(species="Staraptor", side="player", slot=2),
                move="Brave Bird",
                targets=[],
            ),
        ]
    )
    logs.append(
        [
            TurnStartEvent(raw_text="Turn 2", turn_number=2),
            MoveUsedEvent(
                raw_text="used",
                actor=Pokemon(species="Scizor", side="opponent", slot=1),
                move="Bullet Punch",
                targets=[],
            ),
        ]
    )

    assert previous_turn_battle_log_events(logs, turn_number=1) == []
    prior = previous_turn_battle_log_events(logs, turn_number=2)
    assert len(prior) == 3
    assert prior[0].type == "turn_start"
    assert prior[0].turn_number == 1
    assert prior[-1].move == "Brave Bird"
    # Current turn events are not included.
    assert all(
        not (getattr(e, "type", None) == "move_used" and e.move == "Bullet Punch")
        for e in prior
    )


def _opponent_six() -> list[str]:
    return [
        "Scizor",
        "Hatterene",
        "Blaziken",
        "Milotic",
        "Aerodactyl",
        "Grimmsnarl",
    ]


@pytest.mark.asyncio
async def test_suggest_turn_uses_structured_json_and_validates() -> None:
    suggestion = _legal_suggestion(turn=2)
    mock_response = MagicMock()
    mock_response.text = suggestion.model_dump_json()

    mock_aio_models = MagicMock()
    mock_aio_models.generate_content = AsyncMock(return_value=mock_response)
    mock_client = MagicMock()
    mock_client.aio.models = mock_aio_models

    opponent_six = [
        "Scizor",
        "Hatterene",
        "Blaziken",
        "Milotic",
        "Aerodactyl",
        "Grimmsnarl",
    ]
    service = GeminiService(api_key="test-key", client=mock_client)
    result = await service.suggest_turn(
        _game_state(turn=2),
        _sample_team(),
        [],
        turn_number=2,
        opponent_team_species=opponent_six,
    )

    assert result.turn_number == 2
    assert len(result.actions) == 2
    assert isinstance(result.actions[0].action, Move)
    assert result.actions[0].action.move == "Matcha Gotcha"

    mock_aio_models.generate_content.assert_awaited_once()
    call_kwargs = mock_aio_models.generate_content.await_args.kwargs
    config = call_kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert "TurnSuggestion" in json.dumps(config.response_json_schema)
    prompt = call_kwargs["contents"]
    assert "Matcha Gotcha" in prompt
    assert "Known legal moves" in prompt
    assert "turn 2" in prompt
    assert "Opponent team from preview (6)" in prompt
    assert "Blaziken" in prompt
    assert "still unrevealed" in prompt
    assert "not brought to battle" not in prompt
    # Active foes are Scizor/Hatterene; remaining four should be listed as unrevealed.
    assert "Aerodactyl" in prompt
    assert "Grimmsnarl" in prompt


@pytest.mark.asyncio
async def test_suggest_turn_labels_leftover_as_not_brought_when_bring_complete() -> None:
    suggestion = _legal_suggestion(turn=3)
    mock_response = MagicMock()
    mock_response.text = suggestion.model_dump_json()
    mock_aio_models = MagicMock()
    mock_aio_models.generate_content = AsyncMock(return_value=mock_response)
    mock_client = MagicMock()
    mock_client.aio.models = mock_aio_models

    service = GeminiService(api_key="test-key", client=mock_client)
    await service.suggest_turn(
        _game_state(turn=3, opponent_bench=["Blaziken", "Milotic"]),
        _sample_team(),
        [],
        turn_number=3,
        opponent_team_species=_opponent_six(),
    )

    prompt = mock_aio_models.generate_content.await_args.kwargs["contents"]
    assert "Opponent species not brought to battle" in prompt
    assert "still unrevealed" not in prompt
    assert "Aerodactyl" in prompt
    assert "Grimmsnarl" in prompt
    assert "cannot switch in" in prompt


@pytest.mark.asyncio
async def test_suggest_turn_rejects_hallucinated_move_from_model() -> None:
    bad = TurnSuggestion(
        turn_number=1,
        actions=[
            TurnAction(
                action=Move(
                    actor=Pokemon(species="Sinistcha", side="player", slot=1),
                    mega=False,
                    move="Shadow Ball",
                    targets=[],
                ),
                reasoning="Hallucination",
            ),
            TurnAction(
                action=Move(
                    actor=Pokemon(species="Staraptor", side="player", slot=2),
                    mega=False,
                    move="Brave Bird",
                    targets=[],
                ),
                reasoning="Ok",
            ),
        ],
        overall_reasoning="Bad",
    )
    mock_response = MagicMock()
    mock_response.text = bad.model_dump_json()
    mock_aio_models = MagicMock()
    mock_aio_models.generate_content = AsyncMock(return_value=mock_response)
    mock_client = MagicMock()
    mock_client.aio.models = mock_aio_models

    service = GeminiService(api_key="test-key", client=mock_client)
    with pytest.raises(ValueError, match="Illegal move"):
        await service.suggest_turn(
            _game_state(),
            _sample_team(),
            [],
            turn_number=1,
            opponent_team_species=_opponent_six(),
        )


@pytest.mark.asyncio
async def test_suggest_turn_requires_opponent_team_species() -> None:
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock()
    service = GeminiService(api_key="test-key", client=mock_client)

    with pytest.raises(ValueError, match="opponent_team_species is required"):
        await service.suggest_turn(
            _game_state(),
            _sample_team(),
            [],
            turn_number=1,
            opponent_team_species=None,
        )
    with pytest.raises(ValueError, match="opponent_team_species is required"):
        await service.suggest_turn(
            _game_state(),
            _sample_team(),
            [],
            turn_number=1,
            opponent_team_species=[],
        )
    mock_client.aio.models.generate_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_turn_suggestion_stores_and_debounces() -> None:
    from app.services import cv_runner as cv_runner_module

    store = SessionStore()
    store.player_team = _sample_team()
    store.game_state = _game_state(turn=2)
    store.turn_number = 2
    store.opponent_team_species = [
        "Scizor",
        "Hatterene",
        "Blaziken",
        "Milotic",
        "Aerodactyl",
        "Grimmsnarl",
    ]
    prior_events = [
        TurnStartEvent(raw_text="Turn 1", turn_number=1),
        MoveUsedEvent(
            raw_text="used",
            actor=Pokemon(species="Sinistcha", side="player", slot=1),
            move="Protect",
            targets=[],
        ),
    ]
    store.battle_logs = [
        [],
        prior_events,
        [TurnStartEvent(raw_text="Turn 2", turn_number=2)],
    ]

    suggestion = _legal_suggestion(turn=2)
    mock_gemini = MagicMock()
    mock_gemini.suggest_turn = AsyncMock(return_value=suggestion)

    original = cv_runner_module.GeminiService
    cv_runner_module.GeminiService = MagicMock(return_value=mock_gemini)
    try:
        await cv_runner_module._process_turn_suggestion(store)
        await cv_runner_module._process_turn_suggestion(store)
    finally:
        cv_runner_module.GeminiService = original

    assert store.turn_suggestion == suggestion
    assert store._turn_suggestion_turn == 2
    mock_gemini.suggest_turn.assert_awaited_once()
    args = mock_gemini.suggest_turn.await_args.args
    kwargs = mock_gemini.suggest_turn.await_args.kwargs
    assert args[2] == prior_events
    assert kwargs["opponent_team_species"] == store.opponent_team_species
    assert kwargs["turn_number"] == 2


@pytest.mark.asyncio
async def test_process_turn_suggestion_skips_without_opponent_species() -> None:
    from app.services import cv_runner as cv_runner_module

    store = SessionStore()
    store.player_team = _sample_team()
    store.game_state = _game_state(turn=1)
    store.turn_number = 1
    store.opponent_team_species = None

    mock_gemini = MagicMock()
    mock_gemini.suggest_turn = AsyncMock()
    mock_service_cls = MagicMock(return_value=mock_gemini)
    original = cv_runner_module.GeminiService
    cv_runner_module.GeminiService = mock_service_cls
    try:
        await cv_runner_module._process_turn_suggestion(store)
        store.opponent_team_species = []
        await cv_runner_module._process_turn_suggestion(store)
    finally:
        cv_runner_module.GeminiService = original

    assert store.turn_suggestion is None
    mock_gemini.suggest_turn.assert_not_awaited()
    mock_service_cls.assert_not_called()


@pytest.mark.asyncio
async def test_process_turn_suggestion_skips_without_api_key() -> None:
    from app.services import cv_runner as cv_runner_module

    store = SessionStore()
    store.player_team = _sample_team()
    store.game_state = _game_state(turn=1)
    store.turn_number = 1
    store.opponent_team_species = _opponent_six()

    original = cv_runner_module.GeminiService
    cv_runner_module.GeminiService = MagicMock(side_effect=ValueError("no key"))
    try:
        await cv_runner_module._process_turn_suggestion(store)
    finally:
        cv_runner_module.GeminiService = original

    assert store.turn_suggestion is None
    assert store._turn_suggestion_turn is None


@pytest.mark.asyncio
async def test_get_turn_suggestion_endpoint() -> None:
    from httpx import ASGITransport, AsyncClient

    from app.main import app
    from app.api.deps import get_session_store

    store = SessionStore()
    store.player_team = _sample_team()
    store.turn_suggestion = _legal_suggestion(turn=3)

    app.dependency_overrides[get_session_store] = lambda: store
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/suggestions/turn")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["suggestion"]["turn_number"] == 3
    assert len(body["suggestion"]["actions"]) == 2

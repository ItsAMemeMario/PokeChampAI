"""Tests for WebSocket hub and state/logs REST endpoints."""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schema.battle_log import MoveFailedEvent, MoveUsedEvent, TurnStartEvent
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
from app.schema.suggestions import TeamPreviewSuggestion
from app.services.session import BattlePhase, SessionStore, session_store
from app.services.ws_hub import (
    flatten_battle_logs,
    publish_log_patched,
    snapshot_payload,
    ws_hub,
)


@pytest.fixture(autouse=True)
def reset_session() -> None:
    session_store.player_team = None
    session_store.phase = BattlePhase.IDLE
    session_store.cv_running = False
    session_store.adb_connected = False
    session_store.game_state = None
    session_store.battle_logs = [[]]
    session_store.opponent_team_species = None
    session_store.player_selected_species = None
    session_store.team_preview_suggestion = None
    session_store.turn_suggestion = None
    session_store.turn_number = 0
    session_store._team_preview_processed = False
    session_store._turn_suggestion_turn = None
    ws_hub._connections.clear()
    yield
    ws_hub._connections.clear()


def _sample_game_state() -> GameState:
    stages = StatStages()
    active = ActivePokemon(
        species="Sinistcha",
        hp_percentage=80,
        stat_stages=stages,
    )
    return GameState(
        turn_number=1,
        field=FieldState(),
        player=SideState(
            slot_1=active,
            slot_2=None,
            benched=[
                BenchedPokemon(species="Staraptor", hp_percentage=100),
            ],
            hazards=Hazards(),
        ),
        opponent=SideState(
            slot_1=ActivePokemon(
                species="Hatterene",
                hp_percentage=100,
                stat_stages=StatStages(),
            ),
            slot_2=None,
            benched=[],
            hazards=Hazards(),
        ),
    )


def test_get_state_empty() -> None:
    client = TestClient(app)
    response = client.get("/api/state")
    assert response.status_code == 200
    assert response.json() == {"game_state": None}


def test_get_state_with_game() -> None:
    session_store.game_state = _sample_game_state()
    client = TestClient(app)
    response = client.get("/api/state")
    assert response.status_code == 200
    body = response.json()
    assert body["game_state"]["turn_number"] == 1
    assert body["game_state"]["player"]["slot_1"]["species"] == "Sinistcha"
    assert body["game_state"]["player"]["slot_1"]["hp_percentage"] == 80


def test_get_logs_empty() -> None:
    client = TestClient(app)
    response = client.get("/api/logs")
    assert response.status_code == 200
    assert response.json() == {"events": []}


def test_get_logs_returns_recent_events() -> None:
    session_store.turn_number = 1
    session_store.battle_logs = [
        [],
        [
            TurnStartEvent(
                raw_text="Turn 1",
                turn_number=1,
                timestamp=datetime(2026, 1, 1, 12, 0, 0),
            )
        ],
    ]
    client = TestClient(app)
    response = client.get("/api/logs")
    assert response.status_code == 200
    events = response.json()["events"]
    assert len(events) == 1
    assert events[0]["type"] == "turn_start"
    assert events[0]["turn_number"] == 1


def test_flatten_battle_logs_respects_limit() -> None:
    session_store.battle_logs = [[]]
    for turn in range(1, 6):
        session_store.battle_logs.append(
            [
                TurnStartEvent(
                    raw_text=f"Turn {turn}",
                    turn_number=turn,
                )
            ]
        )
    events = flatten_battle_logs(session_store, limit=2)
    assert len(events) == 2
    assert events[0]["turn_number"] == 4
    assert events[1]["turn_number"] == 5


def test_snapshot_payload_includes_suggestions() -> None:
    session_store.phase = BattlePhase.TEAM_PREVIEW
    session_store.opponent_team_species = ["A", "B", "C", "D", "E", "F"]
    session_store.team_preview_suggestion = TeamPreviewSuggestion(
        predicted_opponent_bring=["A", "B", "C", "D"],
        predicted_opponent_lead_pair=("A", "B"),
        suggested_player_bring=["W", "X", "Y", "Z"],
        suggested_player_lead_pair=("W", "X"),
        reasoning="Test",
    )
    snap = snapshot_payload(session_store)
    assert snap["session"]["phase"] == "team_preview"
    assert snap["opponent_team_species"] == ["A", "B", "C", "D", "E", "F"]
    assert snap["team_preview_suggestion"]["reasoning"] == "Test"


def test_websocket_sends_snapshot_on_connect() -> None:
    session_store.phase = BattlePhase.ACTION_SELECTION
    session_store.turn_number = 2
    session_store.cv_running = True
    session_store.game_state = _sample_game_state()

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            message = websocket.receive_json()
            assert message["type"] == "snapshot"
            payload = message["payload"]
            assert payload["session"]["phase"] == "action_selection"
            assert payload["session"]["turn_number"] == 2
            assert payload["game_state"]["player"]["slot_1"]["species"] == "Sinistcha"


@pytest.mark.asyncio
async def test_ws_hub_broadcasts_to_connected_clients() -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        async def send_json(self, message: dict) -> None:
            self.messages.append(message)

    fake = FakeSocket()
    ws_hub._connections.add(fake)  # type: ignore[arg-type]
    try:
        await ws_hub.broadcast({"type": "phase", "payload": {"phase": "idle", "turn_number": 0}})
        assert fake.messages == [
            {"type": "phase", "payload": {"phase": "idle", "turn_number": 0}}
        ]
    finally:
        ws_hub._connections.discard(fake)  # type: ignore[arg-type]


def test_publish_log_patched_includes_event_location(monkeypatch) -> None:
    messages: list[dict] = []
    monkeypatch.setattr(ws_hub, "publish", messages.append)
    event = MoveFailedEvent(
        raw_text="But it failed!",
        actor=Pokemon(species="Staraptor", side="player", slot=2),
        move="Protect",
        reason="failed",
    )

    publish_log_patched(3, 4, event)

    assert len(messages) == 1
    message = messages[0]
    assert message["type"] == "log_patched"
    payload = message["payload"]
    assert payload["turn"] == 3
    assert payload["index"] == 4
    assert payload["event"]["type"] == "move_failed"
    assert payload["event"]["actor"] == {
        "species": "Staraptor",
        "side": "player",
        "slot": 2,
    }
    assert payload["event"]["move"] == "Protect"


def test_session_publishes_completer_patch_with_turn_and_index(monkeypatch) -> None:
    import app.services.ws_hub as ws_hub_module

    patches: list[tuple[int, int, MoveFailedEvent]] = []
    monkeypatch.setattr(
        ws_hub_module,
        "publish_log_patched",
        lambda turn, index, event: patches.append((turn, index, event)),
    )
    monkeypatch.setattr(ws_hub_module, "publish_log", lambda _event: None)
    monkeypatch.setattr(ws_hub_module, "publish_state", lambda _store: None)

    store = SessionStore()
    store.append_battle_log(TurnStartEvent(raw_text="Turn 1", turn_number=1))
    store.append_battle_log(
        MoveUsedEvent(
            raw_text="Staraptor used Protect!",
            actor=Pokemon(species="Staraptor", side="player", slot=1),
            move="Protect",
            targets=[],
        )
    )
    store.append_battle_log(MoveFailedEvent(raw_text="But it failed!"))

    assert len(patches) == 1
    turn, index, event = patches[0]
    assert (turn, index) == (1, 2)
    assert event.type == "move_failed"
    assert event.actor is not None
    assert event.actor.species == "Staraptor"
    assert event.move == "Protect"

"""Tests for retroactive battle log completion."""

from __future__ import annotations

from datetime import datetime

from app.schema.battle_log import (
    AbilityTriggeredEvent,
    FaintEvent,
    HPChangeEvent,
    MoveUsedEvent,
    StatChangeEvent,
    TurnStartEvent,
)
from app.schema.common import Pokemon
from app.schema.gamestate import (
    ActivePokemon,
    FieldState,
    GameState,
    Hazards,
    SideState,
    StatStages,
)
from app.services.battle_log_completer import complete_battle_logs
from app.services.session import SessionStore


def _active(species: str, hp: int = 100) -> ActivePokemon:
    return ActivePokemon(
        species=species,
        hp_percentage=hp,
        stat_stages=StatStages(),
    )


def _game_state(
    *,
    player_slot_1: ActivePokemon | None = None,
    player_slot_2: ActivePokemon | None = None,
    opponent_slot_1: ActivePokemon | None = None,
    opponent_slot_2: ActivePokemon | None = None,
) -> GameState:
    def side(s1: ActivePokemon | None, s2: ActivePokemon | None) -> SideState:
        return SideState(
            slot_1=s1,
            slot_2=s2,
            benched=[],
            hazards=Hazards(spikes=0, toxic_spikes=0, stealth_rocks=0),
        )

    return GameState(
        turn_number=1,
        field=FieldState(),
        player=side(player_slot_1, player_slot_2),
        opponent=side(opponent_slot_1, opponent_slot_2),
    )


def _move(
    species: str,
    move: str,
    *,
    side: str = "player",
    slot: int = 1,
    targets: list[Pokemon] | None = None,
) -> MoveUsedEvent:
    return MoveUsedEvent(
        raw_text=f"{species} used {move}!",
        timestamp=datetime(2026, 1, 1),
        actor=Pokemon(species=species, side=side, slot=slot),  # type: ignore[arg-type]
        move=move,
        targets=targets or [],
    )


def test_fills_move_targets_from_following_hp_changes() -> None:
    store = SessionStore()
    store.append_battle_log(_move("Garchomp", "Earthquake"))
    store.append_battle_log(
        HPChangeEvent(
            raw_text="Incineroar 72%",
            pokemon=Pokemon(species="Incineroar", side="opponent", slot=1),
            hp_pct_change=-28,
        )
    )
    store.append_battle_log(
        HPChangeEvent(
            raw_text="Aerodactyl 81%",
            pokemon=Pokemon(species="Aerodactyl", side="opponent", slot=2),
            hp_pct_change=-19,
        )
    )

    move = store.battle_logs[0]
    assert move.type == "move_used"
    assert {(t.species, t.side, t.slot) for t in move.targets} == {
        ("Incineroar", "opponent", 1),
        ("Aerodactyl", "opponent", 2),
    }


def test_fills_move_targets_from_faint() -> None:
    store = SessionStore()
    store.append_battle_log(_move("Sinistcha", "Matcha Gotcha"))
    store.append_battle_log(
        FaintEvent(
            raw_text="The opposing Flutter Mane fainted!",
            pokemon=Pokemon(species="Flutter Mane", side="opponent", slot=1),
        )
    )

    move = store.battle_logs[0]
    assert move.type == "move_used"
    assert len(move.targets) == 1
    assert move.targets[0].species == "Flutter Mane"
    assert move.targets[0].side == "opponent"


def test_excludes_actor_self_hp_change_from_targets() -> None:
    store = SessionStore()
    store.append_battle_log(_move("Sinistcha", "Matcha Gotcha"))
    store.append_battle_log(
        HPChangeEvent(
            raw_text="Sinistcha healed",
            pokemon=Pokemon(species="Sinistcha", side="player", slot=1),
            hp_pct_change=15,
        )
    )
    store.append_battle_log(
        HPChangeEvent(
            raw_text="Incineroar damaged",
            pokemon=Pokemon(species="Incineroar", side="opponent", slot=1),
            hp_pct_change=-30,
        )
    )

    move = store.battle_logs[0]
    assert move.type == "move_used"
    assert [t.species for t in move.targets] == ["Incineroar"]


def test_earthquake_spread_defaults_from_game_state() -> None:
    store = SessionStore()
    store.game_state = _game_state(
        player_slot_1=_active("Garchomp"),
        player_slot_2=_active("Sinistcha"),
        opponent_slot_1=_active("Incineroar"),
        opponent_slot_2=_active("Aerodactyl"),
    )
    store.append_battle_log(_move("Garchomp", "Earthquake", slot=1))

    move = store.battle_logs[0]
    assert move.type == "move_used"
    assert {(t.species, t.side, t.slot) for t in move.targets} == {
        ("Sinistcha", "player", 2),
        ("Incineroar", "opponent", 1),
        ("Aerodactyl", "opponent", 2),
    }


def test_rock_slide_targets_foes_only() -> None:
    store = SessionStore()
    store.game_state = _game_state(
        player_slot_1=_active("Garchomp"),
        player_slot_2=_active("Sinistcha"),
        opponent_slot_1=_active("Incineroar"),
        opponent_slot_2=_active("Aerodactyl"),
    )
    store.append_battle_log(_move("Garchomp", "Rock Slide", slot=1))

    move = store.battle_logs[0]
    assert move.type == "move_used"
    assert {(t.species, t.side) for t in move.targets} == {
        ("Incineroar", "opponent"),
        ("Aerodactyl", "opponent"),
    }


def test_evidence_preferred_over_spread_defaults() -> None:
    """If only one foe took damage (e.g. Protect), keep evidence targets."""
    store = SessionStore()
    store.game_state = _game_state(
        player_slot_1=_active("Garchomp"),
        player_slot_2=_active("Sinistcha"),
        opponent_slot_1=_active("Incineroar"),
        opponent_slot_2=_active("Aerodactyl"),
    )
    store.append_battle_log(_move("Garchomp", "Earthquake", slot=1))
    store.append_battle_log(
        HPChangeEvent(
            raw_text="Incineroar hit",
            pokemon=Pokemon(species="Incineroar", side="opponent", slot=1),
            hp_pct_change=-40,
        )
    )

    move = store.battle_logs[0]
    assert move.type == "move_used"
    assert [t.species for t in move.targets] == ["Incineroar"]


def test_resolves_actor_slot_from_game_state() -> None:
    store = SessionStore()
    store.game_state = _game_state(
        player_slot_1=_active("Sinistcha"),
        player_slot_2=_active("Garchomp"),
        opponent_slot_1=_active("Incineroar"),
        opponent_slot_2=_active("Aerodactyl"),
    )
    # OCR defaults slot to 1; Garchomp is actually slot 2.
    store.append_battle_log(_move("Garchomp", "Outrage", slot=1))

    move = store.battle_logs[0]
    assert move.type == "move_used"
    assert move.actor.slot == 2
    assert move.actor.species == "Garchomp"


def test_fills_ability_effect_text_from_following_stat_changes() -> None:
    store = SessionStore()
    store.append_battle_log(
        AbilityTriggeredEvent(
            raw_text="Staraptor's Intimidate",
            actor=Pokemon(species="Staraptor", side="player", slot=1),
            ability="Intimidate",
            effect_text="",
        )
    )
    store.append_battle_log(
        StatChangeEvent(
            raw_text="The opposing Incineroar's Attack fell!",
            pokemon=Pokemon(species="Incineroar", side="opponent", slot=1),
            stat="atk",
            stages_delta=-1,
        )
    )
    store.append_battle_log(
        StatChangeEvent(
            raw_text="The opposing Aerodactyl's Attack fell!",
            pokemon=Pokemon(species="Aerodactyl", side="opponent", slot=2),
            stat="atk",
            stages_delta=-1,
        )
    )

    ability = store.battle_logs[0]
    assert ability.type == "ability_triggered"
    assert "Incineroar's Attack fell" in ability.effect_text
    assert "Aerodactyl's Attack fell" in ability.effect_text


def test_does_not_cross_next_move_boundary() -> None:
    store = SessionStore()
    store.append_battle_log(_move("Garchomp", "Outrage"))
    store.append_battle_log(_move("Incineroar", "Flare Blitz", side="opponent"))
    store.append_battle_log(
        HPChangeEvent(
            raw_text="Garchomp damaged",
            pokemon=Pokemon(species="Garchomp", side="player", slot=1),
            hp_pct_change=-50,
        )
    )

    first = store.battle_logs[0]
    second = store.battle_logs[1]
    assert first.type == "move_used"
    assert first.targets == []
    assert second.type == "move_used"
    assert [t.species for t in second.targets] == ["Garchomp"]


def test_turn_start_does_not_clear_prior_completion() -> None:
    store = SessionStore()
    store.game_state = _game_state(
        player_slot_1=_active("Garchomp"),
        player_slot_2=_active("Sinistcha"),
        opponent_slot_1=_active("Incineroar"),
        opponent_slot_2=_active("Aerodactyl"),
    )
    store.append_battle_log(_move("Garchomp", "Earthquake"))
    assert store.battle_logs[0].targets  # type: ignore[union-attr]

    store.append_battle_log(TurnStartEvent(raw_text="Turn 2", turn_number=2))
    assert len(store.battle_logs[0].targets) == 3  # type: ignore[union-attr]


def test_complete_battle_logs_idempotent() -> None:
    store = SessionStore()
    store.append_battle_log(_move("Sinistcha", "Matcha Gotcha"))
    store.append_battle_log(
        FaintEvent(
            raw_text="The opposing Floette fainted!",
            pokemon=Pokemon(species="Floette", side="opponent", slot=1),
        )
    )
    first = complete_battle_logs(store)
    second = complete_battle_logs(store)
    assert first == []  # already completed on append
    assert second == []
    assert len(store.battle_logs[0].targets) == 1  # type: ignore[union-attr]


def test_single_target_move_without_evidence_stays_empty() -> None:
    store = SessionStore()
    store.game_state = _game_state(
        player_slot_1=_active("Garchomp"),
        opponent_slot_1=_active("Incineroar"),
    )
    store.append_battle_log(_move("Garchomp", "Outrage"))

    move = store.battle_logs[0]
    assert move.type == "move_used"
    assert move.targets == []

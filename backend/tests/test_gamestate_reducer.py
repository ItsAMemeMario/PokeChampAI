"""Tests for GameState reducer: event application + turn_start seeding."""

from __future__ import annotations

from datetime import datetime

from app.schema.battle_log import (
    AbilityTriggeredEvent,
    FaintEvent,
    HPChangeEvent,
    ItemUsedEvent,
    MegaEvolutionEvent,
    MoveFailedEvent,
    MoveUsedEvent,
    SideConditionEvent,
    StatChangeEvent,
    StatusAppliedEvent,
    SwitchInEvent,
    SwitchOutEvent,
    TerrainChangeEvent,
    TrickRoomChangeEvent,
    TurnStartEvent,
    VolatileAppliedEvent,
    WeatherChangeEvent,
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
from app.schema.suggestions import TeamPreviewSuggestion
from app.services.gamestate_reducer import (
    apply_event,
    empty_game_state,
    empty_side,
    ensure_seeded,
    seed_from_session,
    seed_game_state,
)
from app.services.session import SessionStore


def _active(
    species: str,
    hp: int = 100,
    *,
    revealed_item: str | None = None,
    revealed_ability: str | None = None,
    revealed_moves: list[str] | None = None,
) -> ActivePokemon:
    return ActivePokemon(
        species=species,
        hp_percentage=hp,
        stat_stages=StatStages(),
        revealed_item=revealed_item,
        revealed_ability=revealed_ability,
        revealed_moves=revealed_moves or [],
    )


def _game_state(
    *,
    turn: int = 1,
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
        turn_number=turn,
        field=FieldState(),
        player=side(player_slot_1, player_slot_2),
        opponent=side(opponent_slot_1, opponent_slot_2),
    )


def _poke(species: str, side: str = "player", slot: int = 1) -> Pokemon:
    return Pokemon(species=species, side=side, slot=slot)  # type: ignore[arg-type]


def test_hp_change_sets_final_percentage() -> None:
    state = _game_state(player_slot_1=_active("Sinistcha", 100))
    event = HPChangeEvent(
        raw_text="Sinistcha 82/178",
        timestamp=datetime(2026, 1, 1),
        pokemon=_poke("Sinistcha"),
        hp_pct_change=46 - 100,
    )
    new_state = apply_event(state, event)
    assert new_state.player.slot_1 is not None
    assert new_state.player.slot_1.hp_percentage == 46
    # Original unchanged (pure).
    assert state.player.slot_1 is not None
    assert state.player.slot_1.hp_percentage == 100


def test_item_used_sets_revealed_item() -> None:
    state = _game_state(opponent_slot_2=_active("Hatterene"))
    event = ItemUsedEvent(
        raw_text="Hatterene's Sitrus Berry",
        pokemon=_poke("Hatterene", "opponent", 2),
        item="Sitrus Berry",
    )
    new_state = apply_event(state, event)
    assert new_state.opponent.slot_2 is not None
    assert new_state.opponent.slot_2.revealed_item == "Sitrus Berry"


def test_ability_triggered_sets_revealed_ability() -> None:
    state = _game_state(player_slot_1=_active("Staraptor"))
    event = AbilityTriggeredEvent(
        raw_text="Staraptor's Intimidate",
        actor=_poke("Staraptor"),
        ability="Intimidate",
        effect_text="",
    )
    new_state = apply_event(state, event)
    assert new_state.player.slot_1 is not None
    assert new_state.player.slot_1.revealed_ability == "Intimidate"


def test_stat_change_clamps_and_updates_def() -> None:
    state = _game_state(player_slot_1=_active("Garchomp"))
    event = StatChangeEvent(
        raw_text="Garchomp's Defense fell!",
        pokemon=_poke("Garchomp"),
        stat="def",
        stages_delta=-1,
    )
    new_state = apply_event(state, event)
    assert new_state.player.slot_1 is not None
    assert new_state.player.slot_1.stat_stages.def_ == -1

    # Clamp at -6
    for _ in range(10):
        new_state = apply_event(
            new_state,
            StatChangeEvent(
                raw_text="fell",
                pokemon=_poke("Garchomp"),
                stat="def",
                stages_delta=-1,
            ),
        )
    assert new_state.player.slot_1.stat_stages.def_ == -6


def test_status_volatile_faint() -> None:
    state = _game_state(opponent_slot_1=_active("Scizor", 40))
    state = apply_event(
        state,
        StatusAppliedEvent(
            raw_text="burned",
            pokemon=_poke("Scizor", "opponent"),
            status="brn",
        ),
    )
    state = apply_event(
        state,
        VolatileAppliedEvent(
            raw_text="confused",
            pokemon=_poke("Scizor", "opponent"),
            volatile="confused",
        ),
    )
    state = apply_event(
        state,
        FaintEvent(raw_text="fainted", pokemon=_poke("Scizor", "opponent")),
    )
    mon = state.opponent.slot_1
    assert mon is not None
    assert mon.status_condition == "brn"
    assert "confused" in mon.volatile_statuses
    assert mon.hp_percentage == 0


def test_switch_out_and_in() -> None:
    state = seed_game_state(
        player_leads=("Sneasler", "Grimmsnarl"),
        player_bench=("Charizard", "Sinistcha"),
    )
    state = apply_event(
        state,
        SwitchOutEvent(raw_text="come back", pokemon=_poke("Sneasler", "player", 1)),
    )
    assert state.player.slot_1 is None
    assert any(m.species == "Sneasler" for m in state.player.benched)

    state = apply_event(
        state,
        SwitchInEvent(raw_text="Go! Charizard!", pokemon=_poke("Charizard", "player", 1)),
    )
    assert state.player.slot_1 is not None
    assert state.player.slot_1.species == "Charizard"
    assert not any(m.species == "Charizard" for m in state.player.benched)


def test_move_used_reveals_and_protect() -> None:
    state = _game_state(player_slot_1=_active("Staraptor"))
    state = apply_event(
        state,
        MoveUsedEvent(
            raw_text="used Protect!",
            actor=_poke("Staraptor"),
            move="Protect",
            targets=[],
        ),
    )
    mon = state.player.slot_1
    assert mon is not None
    assert "Protect" in mon.revealed_moves
    assert mon.is_protected_this_turn is True


def test_move_failed_clears_protect() -> None:
    state = _game_state(player_slot_1=_active("Staraptor"))
    actor = _poke("Staraptor")
    state = apply_event(
        state,
        MoveUsedEvent(
            raw_text="Staraptor used Protect!",
            actor=actor,
            move="Protect",
            targets=[],
        ),
    )
    assert state.player.slot_1 is not None
    assert state.player.slot_1.is_protected_this_turn is True

    state = apply_event(
        state,
        MoveFailedEvent(
            raw_text="But it failed!",
            actor=actor,
            move="Protect",
        ),
    )
    assert state.player.slot_1 is not None
    assert state.player.slot_1.is_protected_this_turn is False
    assert "Protect" in state.player.slot_1.revealed_moves


def test_move_failed_non_protect_leaves_state() -> None:
    state = _game_state(player_slot_1=_active("Garchomp"))
    actor = _poke("Garchomp")
    state = apply_event(
        state,
        MoveUsedEvent(
            raw_text="Garchomp used Outrage!",
            actor=actor,
            move="Outrage",
            targets=[],
        ),
    )
    state = apply_event(
        state,
        MoveFailedEvent(
            raw_text="But it failed!",
            actor=actor,
            move="Outrage",
        ),
    )
    assert state.player.slot_1 is not None
    assert "Outrage" in state.player.slot_1.revealed_moves
    assert state.player.slot_1.is_protected_this_turn is False


def test_turn_start_increments_and_ticks_field() -> None:
    state = _game_state(turn=1, player_slot_1=_active("A", revealed_moves=["Protect"]))
    assert state.player.slot_1 is not None
    state.player.slot_1.is_protected_last_turn = True
    # Re-build with protect set via model
    state = state.model_copy(
        update={
            "player": state.player.model_copy(
                update={
                    "slot_1": state.player.slot_1.model_copy(
                        update={"is_protected_last_turn": True}
                    ),
                    "tailwind_turns": 4,
                    "reflect_turns": 2,
                }
            ),
            "field": FieldState(weather="sun", weather_turns=5),
        }
    )

    # Turn 1 start should not tick durations
    state = apply_event(
        state,
        TurnStartEvent(raw_text="Turn 1", turn_number=1),
    )
    assert state.turn_number == 1
    assert state.field.weather_turns == 5
    assert state.player.tailwind_turns == 4
    assert state.player.reflect_turns == 2
    assert state.player.slot_1 is not None
    assert state.player.slot_1.is_protected_last_turn is False

    state = state.model_copy(
        update={
            "player": state.player.model_copy(
                update={
                    "slot_1": state.player.slot_1.model_copy(
                        update={"is_protected_last_turn": True}
                    )
                }
            )
        }
    )
    state = apply_event(
        state,
        TurnStartEvent(raw_text="Turn 2", turn_number=2),
    )
    assert state.turn_number == 2
    assert state.field.weather_turns == 4
    assert state.player.tailwind_turns == 3
    assert state.player.reflect_turns == 1
    assert state.player.slot_1.is_protected_last_turn is False

    # Inactive extendible counters stay 0 (do not invent an extension).
    empty = empty_game_state(turn_number=1)
    empty = apply_event(empty, TurnStartEvent(raw_text="Turn 2", turn_number=2))
    assert empty.field.weather_turns == 0
    assert empty.field.terrain_turns == 0
    assert empty.field.trick_room_turns == 0
    assert empty.player.reflect_turns == 0
    assert empty.player.light_screen_turns == 0
    assert empty.player.aurora_veil_turns == 0

    # At 1 turn left with no expiry text, assume extender item → 3 turns.
    # Trick Room is not extendible — it ticks to 0.
    extended = empty_game_state(turn_number=1).model_copy(
        update={
            "field": FieldState(
                weather="rain",
                weather_turns=1,
                terrain="electric",
                terrain_turns=1,
                trick_room_turns=1,
            ),
            "player": empty_side().model_copy(update={"reflect_turns": 1}),
        }
    )
    extended = apply_event(extended, TurnStartEvent(raw_text="Turn 2", turn_number=2))
    assert extended.field.weather_turns == 3
    assert extended.field.terrain_turns == 3
    assert extended.field.trick_room_turns == 0
    assert extended.player.reflect_turns == 3


def test_field_and_side_conditions() -> None:
    state = empty_game_state()
    state = apply_event(
        state,
        WeatherChangeEvent(raw_text="sun", weather="sunny"),
    )
    assert state.field.weather == "sun"
    assert state.field.weather_turns == 5

    state = apply_event(
        state,
        TerrainChangeEvent(raw_text="psychic", terrain="psychic_terrain"),
    )
    assert state.field.terrain == "psychic"

    state = apply_event(
        state,
        TrickRoomChangeEvent(raw_text="TR", active=True),
    )
    assert state.field.trick_room_turns == 5

    state = apply_event(
        state,
        SideConditionEvent(
            raw_text="tailwind",
            side="player",
            condition="tailwind",
        ),
    )
    assert state.player.tailwind_turns == 4

    state = apply_event(
        state,
        SideConditionEvent(
            raw_text="rocks",
            side="opponent",
            condition="stealth_rocks",
        ),
    )
    assert state.opponent.hazards.stealth_rocks == 1

    # Expiry is via battle text, not the turn ticker.
    state = apply_event(
        state,
        WeatherChangeEvent(raw_text="The sunlight faded.", weather="none"),
    )
    assert state.field.weather == "none"
    assert state.field.weather_turns == 0

    state = apply_event(
        state,
        TerrainChangeEvent(raw_text="The psychic terrain disappeared.", terrain="none"),
    )
    assert state.field.terrain == "none"
    assert state.field.terrain_turns == 0

    state = apply_event(
        state,
        TrickRoomChangeEvent(raw_text="The twisted dimensions returned to normal!", active=False),
    )
    assert state.field.trick_room_turns == 0


def test_mega_evolution_renames_species() -> None:
    state = _game_state(opponent_slot_1=_active("Scizor"))
    state = apply_event(
        state,
        MegaEvolutionEvent(
            raw_text="Mega Evolved",
            pokemon=_poke("Scizor", "opponent"),
            variant="regular",
        ),
    )
    assert state.opponent.slot_1 is not None
    assert state.opponent.slot_1.species == "Mega Scizor"


def test_mega_evolution_renames_xy_form() -> None:
    state = _game_state(player_slot_1=_active("Charizard"))
    state = apply_event(
        state,
        MegaEvolutionEvent(
            raw_text="Charizardite Y is reacting",
            pokemon=_poke("Charizard"),
            variant="Y",
        ),
    )
    assert state.player.slot_1 is not None
    assert state.player.slot_1.species == "Mega Charizard Y"


def test_seed_game_state_fills_leads_and_bench() -> None:
    state = seed_game_state(
        player_leads=("A", "B"),
        player_bench=("C", "D"),
    )
    assert state.player.slot_1 is not None and state.player.slot_1.species == "A"
    assert state.player.slot_2 is not None and state.player.slot_2.species == "B"
    assert [m.species for m in state.player.benched] == ["C", "D"]
    assert state.opponent.slot_1 is None
    assert state.opponent.benched == []


def test_seed_game_state_allows_unknown_sides() -> None:
    state = seed_game_state(player_leads=("A", "B"), player_bench=("C",))
    assert state.player.slot_1 is not None and state.player.slot_1.species == "A"
    assert state.opponent.slot_1 is None
    assert state.opponent.slot_2 is None
    assert state.opponent.benched == []

    empty = seed_game_state()
    assert empty.player.slot_1 is None
    assert empty.player.benched == []
    assert empty.opponent.slot_1 is None


def test_seed_from_session_uses_player_selection_not_suggestions() -> None:
    store = SessionStore()
    store.team_preview_suggestion = TeamPreviewSuggestion(
        predicted_opponent_bring=["O1", "O2", "O3", "O4"],
        predicted_opponent_lead_pair=("O1", "O2"),
        suggested_player_bring=["Sug1", "Sug2", "Sug3", "Sug4"],
        suggested_player_lead_pair=("Sug1", "Sug2"),
        reasoning="test",
    )
    store.player_selected_species = ["P1", "P2", "P3", "P4"]

    seeded = seed_from_session(store)
    assert seeded.player.slot_1 is not None
    assert seeded.player.slot_1.species == "P1"
    assert seeded.player.slot_2 is not None
    assert seeded.player.slot_2.species == "P2"
    assert [m.species for m in seeded.player.benched] == ["P3", "P4"]
    # Opponent unknown until revealed — never filled from suggestions.
    assert seeded.opponent.slot_1 is None
    assert seeded.opponent.slot_2 is None
    assert seeded.opponent.benched == []


def test_seed_from_session_empty_when_player_selection_missing() -> None:
    store = SessionStore()
    store.team_preview_suggestion = TeamPreviewSuggestion(
        predicted_opponent_bring=["O1", "O2", "O3", "O4"],
        predicted_opponent_lead_pair=("O1", "O2"),
        suggested_player_bring=["Sug1", "Sug2", "Sug3", "Sug4"],
        suggested_player_lead_pair=("Sug1", "Sug2"),
        reasoning="ignored",
    )

    seeded = seed_from_session(store)
    assert seeded.player.slot_1 is None
    assert seeded.player.slot_2 is None
    assert seeded.player.benched == []
    assert seeded.opponent.slot_1 is None
    assert seeded.opponent.benched == []

    ensure_seeded(store)
    assert store.game_state is not None
    assert store.game_state.player.slot_1 is None
    assert store.game_state.opponent.slot_1 is None


def test_append_battle_log_reduces_game_state() -> None:
    store = SessionStore()
    store.game_state = _game_state(player_slot_1=_active("Sinistcha", 100))
    store.append_battle_log(TurnStartEvent(raw_text="Turn 1", turn_number=1))
    store.append_battle_log(
        HPChangeEvent(
            raw_text="dmg",
            pokemon=_poke("Sinistcha"),
            hp_pct_change=-20,
        )
    )
    assert store.game_state is not None
    assert store.game_state.player.slot_1 is not None
    assert store.game_state.player.slot_1.hp_percentage == 80
    assert len(store.battle_logs[1]) == 2


def test_append_lead_in_events_before_turn_start() -> None:
    """Opening animation switch-ins land in battle_logs[0] before TurnStartEvent."""
    store = SessionStore()
    store.begin_battle()
    assert store.turn_number == 0

    store.append_battle_log(
        SwitchInEvent(
            raw_text="sent out Musharna and Dragapult!",
            pokemon=_poke("Musharna", side="opponent", slot=1),
        )
    )
    store.append_battle_log(
        SwitchInEvent(
            raw_text="sent out Musharna and Dragapult!",
            pokemon=_poke("Dragapult", side="opponent", slot=2),
        )
    )

    assert store.turn_number == 0
    assert len(store.battle_logs[0]) == 2
    assert store.battle_logs[0][0].type == "switch_in"
    assert store.game_state is not None
    assert store.game_state.opponent.slot_1 is not None
    assert store.game_state.opponent.slot_1.species == "Musharna"

    store.append_battle_log(TurnStartEvent(raw_text="Turn 1", turn_number=1))
    assert store.turn_number == 1
    assert len(store.battle_logs[0]) == 2
    assert store.battle_logs[1][0].type == "turn_start"


def test_append_turn_start_updates_store_turn_number() -> None:
    store = SessionStore()
    store.game_state = empty_game_state(turn_number=0)
    store.append_battle_log(TurnStartEvent(raw_text="Turn 1", turn_number=1))
    assert store.turn_number == 1
    assert store.game_state is not None
    assert store.game_state.turn_number == 1


def test_seed_player_team_fills_revealed_fields() -> None:
    from app.schema.team import parse_team

    team = parse_team(
        """
Sinistcha @ Sitrus Berry
Ability: Hospitality
Level: 50
EVs: 252 HP / 4 Def / 252 SpD
Bold Nature
- Matcha Gotcha
- Strength Sap
- Rage Powder
- Protect

Staraptor @ Staraptite
Ability: Intimidate
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Brave Bird
- Close Combat
- U-turn
- Protect

Garchomp @ Lum Berry
Ability: Rough Skin
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Earthquake
- Outrage
- Rock Slide
- Protect

Grimmsnarl @ Light Clay
Ability: Prankster
Level: 50
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Reflect
- Light Screen
- Thunder Wave
- Foul Play

Charizard @ Charizardite Y
Ability: Blaze
Level: 50
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Heat Wave
- Solar Beam
- Air Slash
- Protect

Sneasler @ Focus Sash
Ability: Poison Touch
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Dire Claw
- Close Combat
- Fake Out
- Protect
"""
    )
    state = seed_game_state(
        player_leads=("Sinistcha", "Staraptor"),
        player_bench=("Garchomp", "Grimmsnarl"),
        player_team=team,
    )
    assert state.player.slot_1 is not None
    assert state.player.slot_1.revealed_ability == "Hospitality"
    assert state.player.slot_1.revealed_item == "Sitrus Berry"
    assert "Matcha Gotcha" in state.player.slot_1.revealed_moves

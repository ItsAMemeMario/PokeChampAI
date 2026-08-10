"""Apply BattleLogEvents to GameState (pure reducer + session helpers).

Pipeline order after each OCR append: completer → rebuild/replay reducer.
Turn boundaries are emitted as ``turn_start`` when the phase detector enters
``action_selection``.
"""

from __future__ import annotations

from typing import Iterable

from app.schema.battle_log import (
    AbilityTriggeredEvent,
    BattleLogEvent,
    FaintEvent,
    FieldEffectChangedEvent,
    HeldItemChangedEvent,
    HPChangeEvent,
    ItemUsedEvent,
    LeadInEvent,
    MegaEvolutionEvent,
    MoveAvailabilityChangedEvent,
    MoveFailedEvent,
    MoveOutcomeEvent,
    MoveUsedEvent,
    PerishSongStartedEvent,
    SideConditionEvent,
    StatChangeEvent,
    StatStageOperationEvent,
    StatusAppliedEvent,
    StatusCuredEvent,
    SwitchInEvent,
    SwitchLockStartedEvent,
    SwitchOutEvent,
    TerrainEndEvent,
    TerrainStartEvent,
    TrickRoomStartEvent,
    TrickRoomEndEvent,
    TurnStartEvent,
    VolatileAppliedEvent,
    VolatileCuredEvent,
    WeatherEndEvent,
    WeatherStartEvent,
)
from app.schema.common import Pokemon, Side, Slot
from app.schema.gamestate import (
    ActivePokemon,
    BenchedPokemon,
    FieldState,
    GameState,
    Hazards,
    MoveRestrictionState,
    SideState,
    StatStages,
)
from app.schema.team import PlayerTeam
from app.services.session import SessionStore

# Gen 9 doubles defaults when a condition is first applied.
_TAILWIND_TURNS = 4
_SCREEN_TURNS = 5
_SAFEGUARD_TURNS = 5
_WEATHER_TURNS = 5
_TERRAIN_TURNS = 5
_TRICK_ROOM_TURNS = 5
_GRAVITY_TURNS = 5
_MAGIC_ROOM_TURNS = 5
_WONDER_ROOM_TURNS = 5
# Fairy Lock: set to 2 so it covers the next full turn, then clears.
_FAIRY_LOCK_TURNS = 2

_WEATHER_TO_FIELD: dict[str, str] = {
    "sunny": "sun",
    "rain": "rain",
    "sandstorm": "sand",
    "snow": "snow",
}

_TERRAIN_TO_FIELD: dict[str, str] = {
    "electric_terrain": "electric",
    "grassy_terrain": "grassy",
    "misty_terrain": "misty",
    "psychic_terrain": "psychic",
}

_PROTECT_MOVES = frozenset(
    {
        "protect",
        "detect",
        "spiky shield",
        "baneful bunker",
        "king's shield",
        "obstruct",
        "silk trap",
        "burning bulwark",
        "wide guard",
        "quick guard",
    }
)

_STAT_ATTR: dict[str, str] = {
    "atk": "atk",
    "def": "def_",
    "spa": "spa",
    "spd": "spd",
    "spe": "spe",
    "accuracy": "accuracy",
    "evasion": "evasion",
}

_OFFENSIVE_STATS = ("atk", "spa")
_DEFENSIVE_STATS = ("def", "spd")


def empty_side() -> SideState:
    return SideState(
        slot_1=None,
        slot_2=None,
        benched=[],
        hazards=Hazards(),
    )


def empty_game_state(*, turn_number: int = 0) -> GameState:
    return GameState(
        turn_number=turn_number,
        field=FieldState(),
        player=empty_side(),
        opponent=empty_side(),
    )


def seed_game_state(
    *,
    player_leads: tuple[str, str] | list[str] | None = None,
    player_bench: Iterable[str] = (),
    player_team: PlayerTeam | None = None,
    turn_number: int = 0,
) -> GameState:
    """Build initial doubles state from known species only.
    """
    player = empty_side()
    if player_leads is not None:
        p_leads = tuple(player_leads)
        if len(p_leads) != 2:
            raise ValueError("Player leads must be exactly two species when provided")
        player = _side_from_bring(
            leads=p_leads,
            bench=tuple(player_bench),
            player_team=player_team,
            is_player=True,
        )
    opponent = empty_side()

    return GameState(
        turn_number=turn_number,
        field=FieldState(),
        player=player,
        opponent=opponent,
    )


def seed_from_session(store: SessionStore) -> GameState:
    """Seed from CV-known player selection only; opponent starts empty.

    Player slots/bench are filled only when ``player_selected_species`` has at
    least two species (first two = leads, remainder = bench). Opponent slots
    stay ``None`` and bench ``[]`` until switch-in OCR reveals them. Team-preview
    suggestions are never used.
    """
    selected = list(store.player_selected_species or [])
    player_leads: tuple[str, str] | None = None
    player_bench: list[str] = []
    if len(selected) >= 2:
        player_leads = (selected[0], selected[1])
        player_bench = selected[2:]

    return seed_game_state(
        player_leads=player_leads,
        player_bench=player_bench,
        player_team=store.player_team,
        turn_number=store.turn_number,
    )


def apply_event(
    state: GameState,
    event: BattleLogEvent,
    *,
    player_team: PlayerTeam | None = None,
) -> GameState:
    """Pure ``(GameState, BattleLogEvent) → GameState``."""
    if isinstance(event, TurnStartEvent):
        return _apply_turn_start(state, event)
    if isinstance(event, HPChangeEvent):
        return _apply_hp_change(state, event)
    if isinstance(event, ItemUsedEvent):
        return _apply_item_used(state, event)
    if isinstance(event, HeldItemChangedEvent):
        return _apply_held_item_changed(state, event)
    if isinstance(event, AbilityTriggeredEvent):
        return _apply_ability_triggered(state, event)
    if isinstance(event, MoveUsedEvent):
        return _apply_move_used(state, event)
    if isinstance(event, MoveFailedEvent):
        return _apply_move_failed(state, event)
    if isinstance(event, MoveAvailabilityChangedEvent):
        return _apply_move_availability(state, event)
    if isinstance(event, MoveOutcomeEvent):
        return state
    if isinstance(event, StatChangeEvent):
        return _apply_stat_change(state, event)
    if isinstance(event, StatStageOperationEvent):
        return _apply_stat_stage_operation(state, event)
    if isinstance(event, StatusAppliedEvent):
        return _apply_status(state, event)
    if isinstance(event, StatusCuredEvent):
        return _apply_status_cured(state, event)
    if isinstance(event, VolatileAppliedEvent):
        return _apply_volatile(state, event)
    if isinstance(event, VolatileCuredEvent):
        return _apply_volatile_cured(state, event)
    if isinstance(event, FaintEvent):
        return _apply_faint(state, event)
    if isinstance(event, SwitchOutEvent):
        return _apply_switch_out(state, event)
    if isinstance(event, LeadInEvent):
        return _apply_lead_in(state, event, player_team=player_team)
    if isinstance(event, SwitchInEvent):
        return _apply_switch_in(state, event, player_team=player_team)
    if isinstance(event, MegaEvolutionEvent):
        return _apply_mega(state, event)
    if isinstance(event, WeatherStartEvent):
        return _apply_weather_start(state, event)
    if isinstance(event, WeatherEndEvent):
        return _apply_weather_end(state, event)
    if isinstance(event, TerrainStartEvent):
        return _apply_terrain_start(state, event)
    if isinstance(event, TerrainEndEvent):
        return _apply_terrain_end(state, event)
    if isinstance(event, TrickRoomStartEvent) or isinstance(event, TrickRoomEndEvent):
        return _apply_trick_room(state, event)
    if isinstance(event, FieldEffectChangedEvent):
        return _apply_field_effect_changed(state, event)
    if isinstance(event, PerishSongStartedEvent):
        return _apply_perish_song(state, event)
    if isinstance(event, SwitchLockStartedEvent):
        return _apply_switch_lock(state, event)
    if isinstance(event, SideConditionEvent):
        return _apply_side_condition(state, event)
    return state


def apply_events(
    state: GameState,
    events: Iterable[BattleLogEvent],
    *,
    player_team: PlayerTeam | None = None,
) -> GameState:
    for event in events:
        state = apply_event(state, event, player_team=player_team)
    return state


def rebuild_game_state_from_logs(store: SessionStore) -> GameState:
    """Replay every logged event from a captured board origin.

    Origin is the primed/seeded board before log effects (player selection,
    test fixtures, or ``ensure_seeded``). Replaying from that baseline keeps
    completer patches and OCR rereads consistent with ``battle_logs``.
    """
    origin = _ensure_state_origin(store)
    state = origin.model_copy(deep=True)
    for turn_logs in store.battle_logs:
        state = apply_events(state, turn_logs, player_team=store.player_team)
    store.game_state = state
    return state


def _ensure_state_origin(store: SessionStore) -> GameState:
    if store._state_origin is not None:
        return store._state_origin
    if store.game_state is not None:
        store._state_origin = store.game_state.model_copy(deep=True)
    else:
        store._state_origin = seed_from_session(store)
    return store._state_origin


def apply_event_to_store(store: SessionStore, event: BattleLogEvent) -> GameState | None:
    """Update ``store.game_state`` from a (possibly completer-patched) event."""
    state = store.game_state
    if state is None:
        if isinstance(event, (LeadInEvent, SwitchInEvent, TurnStartEvent)):
            state = seed_from_session(store)
        else:
            return None

    new_state = apply_event(state, event, player_team=store.player_team)
    store.game_state = new_state
    if isinstance(event, TurnStartEvent):
        store.turn_number = event.turn_number
    return new_state


def ensure_seeded(store: SessionStore) -> GameState:
    """Seed ``store.game_state`` once when battle begins (known data only)."""
    if store.game_state is not None:
        return store.game_state
    seeded = seed_from_session(store)
    store.game_state = seeded
    if store._state_origin is None:
        store._state_origin = seeded.model_copy(deep=True)
    return seeded


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _apply_turn_start(state: GameState, event: TurnStartEvent) -> GameState:
    ticked = _tick_field_and_sides(state) if event.turn_number > 1 else state
    player = _update_protect_flags(ticked.player)
    opponent = _update_protect_flags(ticked.opponent)
    return ticked.model_copy(
        update={
            "turn_number": event.turn_number,
            "player": player,
            "opponent": opponent,
        }
    )


def _apply_hp_change(state: GameState, event: HPChangeEvent) -> GameState:
    def update(active: ActivePokemon) -> ActivePokemon:
        final = max(0, min(100, active.hp_percentage + event.hp_pct_change))
        return active.model_copy(update={"hp_percentage": final})

    return _update_active(state, event.pokemon, update)


def _apply_item_used(state: GameState, event: ItemUsedEvent) -> GameState:
    def update(active: ActivePokemon) -> ActivePokemon:
        return active.model_copy(
            update={"revealed_item": event.item, "item_state": "held"}
        )

    return _update_active(state, event.pokemon, update)


def _apply_held_item_changed(state: GameState, event: HeldItemChangedEvent) -> GameState:
    change = event.change
    if change == "stolen":
        state = _apply_item_transfer(state, event)
        return state
    if event.pokemon is None:
        return state

    def update(active: ActivePokemon) -> ActivePokemon:
        patch: dict = {}
        if event.item:
            patch["revealed_item"] = event.item
        if change in {"revealed", "activated", "obtained"}:
            patch["item_state"] = "held"
        elif change == "consumed":
            patch["item_state"] = "consumed"
        elif change == "lost":
            patch["item_state"] = "lost"
        return active.model_copy(update=patch) if patch else active

    return _update_active(state, event.pokemon, update)


def _apply_item_transfer(state: GameState, event: HeldItemChangedEvent) -> GameState:
    """Thief/Covet-style steal: recipient obtains, source loses."""
    if event.pokemon is not None:
        def update_recipient(active: ActivePokemon) -> ActivePokemon:
            patch: dict = {"item_state": "held"}
            if event.item:
                patch["revealed_item"] = event.item
            return active.model_copy(update=patch)

        state = _update_active(state, event.pokemon, update_recipient)

    if event.source is not None:
        def update_source(active: ActivePokemon) -> ActivePokemon:
            return active.model_copy(update={"item_state": "lost"})

        state = _update_active(state, event.source, update_source)
    return state


def _apply_ability_triggered(state: GameState, event: AbilityTriggeredEvent) -> GameState:
    def update(active: ActivePokemon) -> ActivePokemon:
        return active.model_copy(update={"revealed_ability": event.ability})

    return _update_active(state, event.actor, update)


def _apply_move_used(state: GameState, event: MoveUsedEvent) -> GameState:
    def update(active: ActivePokemon) -> ActivePokemon:
        moves = list(active.revealed_moves)
        if event.move and event.move not in moves:
            moves.append(event.move)
        patch: dict = {"revealed_moves": moves}
        if event.move.strip().lower() in _PROTECT_MOVES:
            patch["is_protected_this_turn"] = True
        return active.model_copy(update=patch)

    return _update_active(state, event.actor, update)


def _apply_move_failed(state: GameState, event: MoveFailedEvent) -> GameState:
    """Clear protect when a protection move fails (e.g. successive Protect)."""
    if event.actor is None or event.move.strip().lower() not in _PROTECT_MOVES:
        return state

    def update(active: ActivePokemon) -> ActivePokemon:
        if not active.is_protected_this_turn:
            return active
        return active.model_copy(update={"is_protected_this_turn": False})

    return _update_active(state, event.actor, update)


def _apply_move_availability(
    state: GameState,
    event: MoveAvailabilityChangedEvent,
) -> GameState:
    if event.pokemon is None:
        # Leave unresolved until a completer can name the actor with confidence.
        return state

    restriction = MoveRestrictionState(
        restriction=event.restriction,
        move=event.move,
        source_item=event.source_item,
        clears_on_switch=(
            True if event.clears_on_switch is None else event.clears_on_switch
        ),
    )

    def update(active: ActivePokemon) -> ActivePokemon:
        remaining = [
            existing
            for existing in active.move_restrictions
            if not (
                existing.restriction == restriction.restriction
                and existing.move == restriction.move
            )
        ]
        remaining.append(restriction)
        return active.model_copy(update={"move_restrictions": remaining})

    return _update_active(state, event.pokemon, update)


def _apply_stat_change(state: GameState, event: StatChangeEvent) -> GameState:
    attr = _STAT_ATTR.get(event.stat)
    if attr is None:
        return state

    def update(active: ActivePokemon) -> ActivePokemon:
        stages = active.stat_stages
        current = getattr(stages, attr)
        new_val = max(-6, min(6, current + event.stages_delta))
        return active.model_copy(
            update={"stat_stages": stages.model_copy(update={attr: new_val})}
        )

    return _update_active(state, event.pokemon, update)


def _apply_stat_stage_operation(
    state: GameState,
    event: StatStageOperationEvent,
) -> GameState:
    operation = event.operation
    if operation == "clear_all":
        return _clear_all_stat_stages(state)
    if event.pokemon is None:
        return state

    if operation == "clear_one":
        def clear_one(active: ActivePokemon) -> ActivePokemon:
            return active.model_copy(update={"stat_stages": StatStages()})

        return _update_active(state, event.pokemon, clear_one)

    if operation == "invert":
        def invert(active: ActivePokemon) -> ActivePokemon:
            stages = active.stat_stages
            inverted = {
                attr: max(-6, min(6, -getattr(stages, attr)))
                for attr in _STAT_ATTR.values()
            }
            return active.model_copy(
                update={"stat_stages": stages.model_copy(update=inverted)}
            )

        return _update_active(state, event.pokemon, invert)

    if event.target is None:
        return state

    source_side = state.player if event.pokemon.side == "player" else state.opponent
    target_side = state.player if event.target.side == "player" else state.opponent
    _, source_active = _find_active(source_side, event.pokemon)
    _, target_active = _find_active(target_side, event.target)
    if source_active is None or target_active is None:
        return state

    if operation == "copy":
        def copy_boosts(active: ActivePokemon) -> ActivePokemon:
            return active.model_copy(
                update={"stat_stages": target_active.stat_stages.model_copy()}
            )

        return _update_active(state, event.pokemon, copy_boosts)

    stats = list(event.stats)
    if operation == "swap_all":
        stats = list(_STAT_ATTR.keys())
    elif operation == "swap_offensive":
        stats = list(_OFFENSIVE_STATS)
    elif operation == "swap_defensive":
        stats = list(_DEFENSIVE_STATS)
    if not stats:
        return state

    source_patch: dict[str, int] = {}
    target_patch: dict[str, int] = {}
    for stat in stats:
        attr = _STAT_ATTR.get(stat)
        if attr is None:
            continue
        source_patch[attr] = getattr(target_active.stat_stages, attr)
        target_patch[attr] = getattr(source_active.stat_stages, attr)

    def update_source(active: ActivePokemon) -> ActivePokemon:
        return active.model_copy(
            update={
                "stat_stages": active.stat_stages.model_copy(update=source_patch)
            }
        )

    def update_target(active: ActivePokemon) -> ActivePokemon:
        return active.model_copy(
            update={
                "stat_stages": active.stat_stages.model_copy(update=target_patch)
            }
        )

    state = _update_active(state, event.pokemon, update_source)
    return _update_active(state, event.target, update_target)


def _clear_all_stat_stages(state: GameState) -> GameState:
    def clear_side(side: SideState) -> SideState:
        patch: dict = {}
        if side.slot_1 is not None:
            patch["slot_1"] = side.slot_1.model_copy(update={"stat_stages": StatStages()})
        if side.slot_2 is not None:
            patch["slot_2"] = side.slot_2.model_copy(update={"stat_stages": StatStages()})
        return side.model_copy(update=patch) if patch else side

    return state.model_copy(
        update={
            "player": clear_side(state.player),
            "opponent": clear_side(state.opponent),
        }
    )


def _apply_status(state: GameState, event: StatusAppliedEvent) -> GameState:
    def update(active: ActivePokemon) -> ActivePokemon:
        return active.model_copy(update={"status_condition": event.status})

    return _update_active(state, event.pokemon, update)


def _apply_status_cured(state: GameState, event: StatusCuredEvent) -> GameState:
    def update(active: ActivePokemon) -> ActivePokemon:
        current = active.status_condition
        # tox.end redirects to psn.end text in Showdown; clear either poison.
        if current == event.status:
            return active.model_copy(update={"status_condition": "none"})
        if event.status == "psn" and current == "tox":
            return active.model_copy(update={"status_condition": "none"})
        if event.status == "tox" and current == "psn":
            return active.model_copy(update={"status_condition": "none"})
        return active

    return _update_active(state, event.pokemon, update)


def _apply_volatile(state: GameState, event: VolatileAppliedEvent) -> GameState:
    def update(active: ActivePokemon) -> ActivePokemon:
        vols = list(active.volatile_statuses)
        if event.volatile not in vols:
            vols.append(event.volatile)
        return active.model_copy(update={"volatile_statuses": vols})

    return _update_active(state, event.pokemon, update)


def _apply_volatile_cured(state: GameState, event: VolatileCuredEvent) -> GameState:
    def update(active: ActivePokemon) -> ActivePokemon:
        vols = [v for v in active.volatile_statuses if v != event.volatile]
        return active.model_copy(update={"volatile_statuses": vols})

    return _update_active(state, event.pokemon, update)


def _apply_faint(state: GameState, event: FaintEvent) -> GameState:
    # Really a redundant fail-safe, in case rounding errors cause HP to be > 0% when fainting
    def update(active: ActivePokemon) -> ActivePokemon:
        return active.model_copy(update={"hp_percentage": 0})

    return _update_active(state, event.pokemon, update)


def _apply_switch_out(state: GameState, event: SwitchOutEvent) -> GameState:
    side = event.pokemon.side
    side_state = state.player if side == "player" else state.opponent
    slot, active = _find_active(side_state, event.pokemon)
    if active is None or slot is None:
        return state

    benched = list(side_state.benched)
    benched.append(_to_benched(active))
    patch: dict = {"benched": benched}
    if slot == 1:
        patch["slot_1"] = None
    else:
        patch["slot_2"] = None
    new_side = side_state.model_copy(update=patch)
    return state.model_copy(update={side: new_side})


def _apply_lead_in(
    state: GameState,
    event: LeadInEvent,
    *,
    player_team: PlayerTeam | None = None,
) -> GameState:
    """Place both opening leads from a dual send-out message."""
    state = _apply_switch_in(
        state,
        SwitchInEvent(raw_text=event.raw_text, pokemon=event.slot_1),
        player_team=player_team,
    )
    return _apply_switch_in(
        state,
        SwitchInEvent(raw_text=event.raw_text, pokemon=event.slot_2),
        player_team=player_team,
    )


def _apply_switch_in(
    state: GameState,
    event: SwitchInEvent,
    *,
    player_team: PlayerTeam | None = None,
) -> GameState:
    # player_team is optional for pure apply; session seeding fills revealed_* earlier.
    side = event.pokemon.side
    side_state = state.player if side == "player" else state.opponent
    slot = _resolve_switch_in_slot(side_state, event.pokemon)

    incoming, remaining_bench = _take_from_bench(side_state.benched, event.pokemon.species)
    if incoming is None:
        incoming = _new_active(
            event.pokemon.species,
            player_team=player_team if side == "player" else None,
        )
    else:
        incoming = _benched_to_active(incoming)

    patch: dict = {"benched": remaining_bench}
    # If target slot occupied by someone else, bench them first.
    current = side_state.slot_1 if slot == 1 else side_state.slot_2
    if current is not None and not _species_matches(current.species, event.pokemon.species):
        patch["benched"] = [*remaining_bench, _to_benched(current)]

    if slot == 1:
        patch["slot_1"] = incoming
    else:
        patch["slot_2"] = incoming

    new_side = side_state.model_copy(update=patch)
    return state.model_copy(update={side: new_side})


def _apply_mega(state: GameState, event: MegaEvolutionEvent) -> GameState:
    side = event.pokemon.side
    side_state = state.player if side == "player" else state.opponent

    def update(active: ActivePokemon) -> ActivePokemon:
        species = active.species
        if not species.lower().startswith("mega "):
            base = event.pokemon.species
            if base.lower().startswith("mega "):
                species = base
            elif event.variant == "regular":
                species = f"Mega {base}"
            else:
                species = f"Mega {base} {event.variant}"
        return active.model_copy(update={"species": species})

    state = _update_active(state, event.pokemon, update)
    side_state = state.player if side == "player" else state.opponent
    return state.model_copy(
        update={side: side_state.model_copy(update={"mega_used": True})}
    )


def _apply_weather_start(state: GameState, event: WeatherStartEvent) -> GameState:
    weather = _WEATHER_TO_FIELD.get(event.weather, "none")
    return state.model_copy(
        update={
            "field": state.field.model_copy(
                update={
                    "weather": weather,
                    "weather_turns": _WEATHER_TURNS,
                    "weather_suppressed": False,
                }
            )
        }
    )


def _apply_weather_end(state: GameState, event: WeatherEndEvent) -> GameState:
    return state.model_copy(
        update={
            "field": state.field.model_copy(
                update={
                    "weather": "none",
                    "weather_turns": 0,
                    "weather_suppressed": False,
                }
            )
        }
    )


def _apply_terrain_start(state: GameState, event: TerrainStartEvent) -> GameState:
    terrain = _TERRAIN_TO_FIELD.get(event.terrain, "none")
    return state.model_copy(
        update={
            "field": state.field.model_copy(
                update={"terrain": terrain, "terrain_turns": _TERRAIN_TURNS}
            )
        }
    )


def _apply_terrain_end(state: GameState, event: TerrainEndEvent) -> GameState:
    return state.model_copy(
        update={
            "field": state.field.model_copy(
                update={"terrain": "none", "terrain_turns": 0}
            )
        }
    )

def _apply_trick_room(state: GameState, event: TrickRoomStartEvent | TrickRoomEndEvent) -> GameState:
    turns = _TRICK_ROOM_TURNS if isinstance(event, TrickRoomStartEvent) else 0
    return state.model_copy(
        update={
            "field": state.field.model_copy(update={"trick_room_turns": turns})
        }
    )


def _apply_field_effect_changed(
    state: GameState,
    event: FieldEffectChangedEvent,
) -> GameState:
    effect = event.effect
    starting = event.action == "start"
    field = state.field
    if effect == "gravity":
        return state.model_copy(
            update={
                "field": field.model_copy(
                    update={"gravity_turns": _GRAVITY_TURNS if starting else 0}
                )
            }
        )
    if effect == "magic_room":
        return state.model_copy(
            update={
                "field": field.model_copy(
                    update={"magic_room_turns": _MAGIC_ROOM_TURNS if starting else 0}
                )
            }
        )
    if effect == "wonder_room":
        return state.model_copy(
            update={
                "field": field.model_copy(
                    update={"wonder_room_turns": _WONDER_ROOM_TURNS if starting else 0}
                )
            }
        )
    if effect == "weather_suppression":
        # Preserve underlying weather; only toggle suppression.
        return state.model_copy(
            update={
                "field": field.model_copy(update={"weather_suppressed": starting})
            }
        )
    return state


def _apply_perish_song(state: GameState, event: PerishSongStartedEvent) -> GameState:
    targets = list(event.affected)
    if not targets:
        # Completer may fill later; until then mark every current active.
        for side_name, side_state in (("player", state.player), ("opponent", state.opponent)):
            for slot, active in ((1, side_state.slot_1), (2, side_state.slot_2)):
                if active is None:
                    continue
                targets.append(
                    Pokemon(species=active.species, side=side_name, slot=slot)  # type: ignore[arg-type]
                )

    turns = int(event.turns_remaining)
    for pokemon in targets:
        def update(active: ActivePokemon, *, _turns: int = turns) -> ActivePokemon:
            return active.model_copy(update={"perish_turns": _turns})

        state = _update_active(state, pokemon, update)
    return state


def _apply_switch_lock(state: GameState, event: SwitchLockStartedEvent) -> GameState:
    return state.model_copy(
        update={
            "field": state.field.model_copy(
                update={"fairy_lock_turns": _FAIRY_LOCK_TURNS}
            )
        }
    )


def _apply_side_condition(state: GameState, event: SideConditionEvent) -> GameState:
    side: Side = event.side
    side_state = state.player if side == "player" else state.opponent
    condition = event.condition
    ending = event.action == "end"
    patch: dict = {}

    if condition == "tailwind":
        patch["tailwind_turns"] = 0 if ending else _TAILWIND_TURNS
    elif condition == "reflect":
        patch["reflect_turns"] = 0 if ending else _SCREEN_TURNS
    elif condition == "light_screen":
        patch["light_screen_turns"] = 0 if ending else _SCREEN_TURNS
    elif condition == "aurora_veil":
        patch["aurora_veil_turns"] = 0 if ending else _SCREEN_TURNS
    elif condition == "safeguard":
        patch["safeguard_turns"] = 0 if ending else _SAFEGUARD_TURNS
    elif condition == "spikes":
        hazards = side_state.hazards
        if ending:
            patch["hazards"] = hazards.model_copy(update={"spikes": 0})
        else:
            layers = min(3, hazards.spikes + 1)
            patch["hazards"] = hazards.model_copy(update={"spikes": layers})
    elif condition == "toxic_spikes":
        hazards = side_state.hazards
        if ending:
            patch["hazards"] = hazards.model_copy(update={"toxic_spikes": 0})
        else:
            layers = min(2, hazards.toxic_spikes + 1)
            patch["hazards"] = hazards.model_copy(update={"toxic_spikes": layers})
    elif condition == "stealth_rocks":
        hazards = side_state.hazards
        patch["hazards"] = hazards.model_copy(
            update={"stealth_rocks": 0 if ending else 1}
        )
    elif condition == "sticky_web":
        hazards = side_state.hazards
        patch["hazards"] = hazards.model_copy(
            update={"sticky_web": 0 if ending else 1}
        )
    else:
        return state

    return state.model_copy(update={side: side_state.model_copy(update=patch)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _side_from_bring(
    *,
    leads: tuple[str, str],
    bench: tuple[str, ...],
    player_team: PlayerTeam | None,
    is_player: bool,
) -> SideState:
    team = player_team if is_player else None
    return SideState(
        slot_1=_new_active(leads[0], player_team=team),
        slot_2=_new_active(leads[1], player_team=team),
        benched=[_new_benched(s, player_team=team) for s in bench],
        hazards=Hazards(),
    )


def _new_active(species: str, *, player_team: PlayerTeam | None = None) -> ActivePokemon:
    known = _player_known(player_team, species)
    item = known.get("revealed_item")
    return ActivePokemon(
        species=species,
        hp_percentage=100,
        stat_stages=StatStages(),
        revealed_ability=known.get("revealed_ability"),
        revealed_item=item,
        item_state="held" if item else "unknown",
        revealed_moves=list(known.get("revealed_moves") or []),
    )


def _new_benched(species: str, *, player_team: PlayerTeam | None = None) -> BenchedPokemon:
    known = _player_known(player_team, species)
    item = known.get("revealed_item")
    return BenchedPokemon(
        species=species,
        hp_percentage=100,
        revealed_ability=known.get("revealed_ability"),
        revealed_item=item,
        item_state="held" if item else "unknown",
        revealed_moves=list(known.get("revealed_moves") or []),
    )


def _player_known(team: PlayerTeam | None, species: str) -> dict:
    if team is None:
        return {}
    for mon in team.pokemon:
        if _species_matches(mon.species, species):
            return {
                "revealed_ability": mon.ability,
                "revealed_item": mon.item,
                "revealed_moves": list(mon.moves),
            }
    return {}


def _to_benched(active: ActivePokemon) -> BenchedPokemon:
    return BenchedPokemon(
        species=active.species,
        hp_percentage=active.hp_percentage,
        status_condition=active.status_condition,
        item_state=active.item_state,
        revealed_ability=active.revealed_ability,
        revealed_item=active.revealed_item,
        revealed_moves=list(active.revealed_moves),
    )


def _benched_to_active(benched: BenchedPokemon) -> ActivePokemon:
    status = benched.status_condition
    if status not in ("none", "brn", "par", "slp", "psn", "tox", "frz"):
        status = "none"
    return ActivePokemon(
        species=benched.species,
        hp_percentage=benched.hp_percentage,
        status_condition=status,  # type: ignore[arg-type]
        stat_stages=StatStages(),
        item_state=benched.item_state,
        revealed_ability=benched.revealed_ability,
        revealed_item=benched.revealed_item,
        revealed_moves=list(benched.revealed_moves),
    )


def _take_from_bench(
    bench: list[BenchedPokemon],
    species: str,
) -> tuple[BenchedPokemon | None, list[BenchedPokemon]]:
    for index, mon in enumerate(bench):
        if _species_matches(mon.species, species):
            remaining = [*bench[:index], *bench[index + 1 :]]
            return mon, remaining
    return None, list(bench)


def _resolve_switch_in_slot(side: SideState, pokemon: Pokemon) -> Slot:
    # Prefer explicit empty slot matching event.slot when free or same species.
    preferred = pokemon.slot
    current = side.slot_1 if preferred == 1 else side.slot_2
    if current is None or _species_matches(current.species, pokemon.species):
        return preferred
    other: Slot = 2 if preferred == 1 else 1
    other_mon = side.slot_2 if preferred == 1 else side.slot_1
    if other_mon is None:
        return other
    return preferred


def _find_active(
    side: SideState,
    pokemon: Pokemon,
) -> tuple[Slot | None, ActivePokemon | None]:
    slots: tuple[tuple[Slot, ActivePokemon | None], ...] = (
        (1, side.slot_1),
        (2, side.slot_2),
    )
    # Exact slot + species
    by_slot = side.slot_1 if pokemon.slot == 1 else side.slot_2
    if by_slot is not None and _species_matches(by_slot.species, pokemon.species):
        return pokemon.slot, by_slot

    matches = [
        (slot, active)
        for slot, active in slots
        if active is not None and _species_matches(active.species, pokemon.species)
    ]
    if len(matches) == 1:
        return matches[0]
    if by_slot is not None and pokemon.species:
        # Slot known but species OCR drifted — still update that slot.
        return pokemon.slot, by_slot
    return None, None


def _update_active(
    state: GameState,
    pokemon: Pokemon,
    updater,
) -> GameState:
    side = pokemon.side
    side_state = state.player if side == "player" else state.opponent
    slot, active = _find_active(side_state, pokemon)
    if active is None or slot is None:
        return state
    updated = updater(active)
    patch = {"slot_1": updated} if slot == 1 else {"slot_2": updated}
    return state.model_copy(update={side: side_state.model_copy(update=patch)})


def _is_protected_this_turn(pokemon: ActivePokemon | None) -> bool:
    return pokemon is not None and pokemon.is_protected_this_turn


def _update_protect_flags(side: SideState) -> SideState:
    patch: dict = {}
    if side.slot_1 is not None:
        patch["slot_1"] = side.slot_1.model_copy(
            update={
                "is_protected_last_turn": _is_protected_this_turn(side.slot_1),
                "is_protected_this_turn": False,
            }
        )
    if side.slot_2 is not None:
        patch["slot_2"] = side.slot_2.model_copy(
            update={
                "is_protected_last_turn": _is_protected_this_turn(side.slot_2),
                "is_protected_this_turn": False,
            }
        )
    return side.model_copy(update=patch) if patch else side


def _tick_extendible_turns(turns: int) -> int:
    """Decrement an item-extendible duration (weather, terrain, screens).

    Expiry for these is driven by battle text, not by reaching 0 here. If a
    condition is still at 1 turn with no end event, assume an extender item and
    grant 3 more turns. Inactive (0) stays 0.
    """
    if turns <= 0:
        return 0
    if turns > 1:
        return turns - 1
    return 3


def _tick_perish(side: SideState) -> SideState:
    patch: dict = {}
    for slot_name, active in (("slot_1", side.slot_1), ("slot_2", side.slot_2)):
        if active is None or active.perish_turns <= 0:
            continue
        next_turns = active.perish_turns - 1
        update: dict = {"perish_turns": next_turns}
        if next_turns <= 0:
            update["hp_percentage"] = 0
        patch[slot_name] = active.model_copy(update=update)
    return side.model_copy(update=patch) if patch else side


def _tick_field_and_sides(state: GameState) -> GameState:
    field = state.field
    # Trick Room / rooms / Gravity / Fairy Lock are not item-extendible.
    new_field = field.model_copy(
        update={
            "weather_turns": _tick_extendible_turns(field.weather_turns),
            "terrain_turns": _tick_extendible_turns(field.terrain_turns),
            "trick_room_turns": max(0, field.trick_room_turns - 1),
            "gravity_turns": max(0, field.gravity_turns - 1),
            "magic_room_turns": max(0, field.magic_room_turns - 1),
            "wonder_room_turns": max(0, field.wonder_room_turns - 1),
            "fairy_lock_turns": max(0, field.fairy_lock_turns - 1),
        }
    )
    return state.model_copy(
        update={
            "field": new_field,
            "player": _tick_perish(_tick_side(state.player)),
            "opponent": _tick_perish(_tick_side(state.opponent)),
        }
    )


def _tick_side(side: SideState) -> SideState:
    # Tailwind has no extender item — it can tick to 0.
    # Screens/Aurora Veil/Safeguard are Light Clay–extendible; end via battle text.
    return side.model_copy(
        update={
            "tailwind_turns": max(0, side.tailwind_turns - 1),
            "reflect_turns": _tick_extendible_turns(side.reflect_turns),
            "light_screen_turns": _tick_extendible_turns(side.light_screen_turns),
            "aurora_veil_turns": _tick_extendible_turns(side.aurora_veil_turns),
            "safeguard_turns": _tick_extendible_turns(side.safeguard_turns),
        }
    )


def _species_matches(left: str, right: str) -> bool:
    a = left.strip().lower()
    b = right.strip().lower()
    if a == b:
        return True

    def base_species(name: str) -> str:
        if name.startswith("mega "):
            name = name[len("mega ") :]
        if name.endswith((" x", " y", " z")):
            name = name[:-2].rstrip()
        return name

    return base_species(a) == base_species(b)

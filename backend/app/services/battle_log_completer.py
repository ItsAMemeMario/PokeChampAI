"""Retroactively complete partial BattleLogEvents from later log evidence.

Frame-by-frame OCR often emits incomplete events (empty move targets, blank
ability effect text, default actor slots). After each append, scan the current
turn and patch fields in place when later events or known move properties
supply the missing data — before the reducer / Gemini see the log.
"""

from __future__ import annotations

from app.data.spread_moves import spread_kind
from app.schema.battle_log import (
    AbilityTriggeredEvent,
    BattleLogEvent,
    FaintEvent,
    HPChangeEvent,
    MoveUsedEvent,
    StatChangeEvent,
    StatusAppliedEvent,
    VolatileAppliedEvent,
)
from app.schema.common import Pokemon, Side, Slot
from app.schema.gamestate import ActivePokemon, GameState, SideState
from app.services.session import SessionStore

# Events that close the evidence window for a move / ability.
_MOVE_BOUNDARY_TYPES = frozenset({"move_used", "turn_start"})
_ABILITY_BOUNDARY_TYPES = frozenset(
    {"move_used", "ability_triggered", "turn_start", "item_used"}
)


def complete_battle_logs(store: SessionStore) -> list[tuple[int, int]]:
    """Patch incomplete events in ``store.battle_logs`` using later evidence.

    Returns ``(turn_number, event_index)`` pairs for events that were modified.
    """
    patched: list[tuple[int, int]] = []
    game_state = store.game_state

    for turn, logs in enumerate(store.battle_logs):
        if turn == 0 or not logs:
            continue
        for index, event in enumerate(logs):
            changed = False
            if isinstance(event, MoveUsedEvent):
                changed = _complete_move_used(logs, index, game_state)
            elif isinstance(event, AbilityTriggeredEvent):
                changed = _complete_ability_triggered(logs, index)

            if changed:
                patched.append((turn, index))

    return patched


def _complete_move_used(
    logs: list[BattleLogEvent],
    index: int,
    game_state: GameState | None,
) -> bool:
    event = logs[index]
    assert isinstance(event, MoveUsedEvent)

    changed = False
    actor = _resolve_actor_slot(event.actor, game_state)
    if actor != event.actor:
        event = event.model_copy(update={"actor": actor})
        logs[index] = event
        changed = True

    following = _following_until(logs, index, _MOVE_BOUNDARY_TYPES)
    evidence_targets = _targets_from_evidence(following, actor)
    if evidence_targets:
        if not _same_targets(event.targets, evidence_targets):
            logs[index] = event.model_copy(update={"targets": evidence_targets})
            return True
        return changed

    if event.targets:
        return changed

    kind = spread_kind(event.move)
    if kind is not None and game_state is not None:
        spread_targets = _targets_from_spread(actor, kind, game_state)
        if spread_targets:
            logs[index] = event.model_copy(update={"targets": spread_targets})
            return True

    return changed


def _same_targets(left: list[Pokemon], right: list[Pokemon]) -> bool:
    return {(p.species, p.side, p.slot) for p in left} == {
        (p.species, p.side, p.slot) for p in right
    }


def _complete_ability_triggered(
    logs: list[BattleLogEvent],
    index: int,
) -> bool:
    event = logs[index]
    assert isinstance(event, AbilityTriggeredEvent)

    following = _following_until(logs, index, _ABILITY_BOUNDARY_TYPES)
    snippets: list[str] = []
    seen: set[str] = set()
    for later in following:
        if not isinstance(
            later, (StatChangeEvent, StatusAppliedEvent, VolatileAppliedEvent)
        ):
            continue
        text = later.raw_text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        snippets.append(text)

    if not snippets:
        return False

    effect_text = "; ".join(snippets)
    if event.effect_text == effect_text:
        return False

    logs[index] = event.model_copy(update={"effect_text": effect_text})
    return True


def _following_until(
    logs: list[BattleLogEvent],
    index: int,
    boundary_types: frozenset[str],
) -> list[BattleLogEvent]:
    """Return events after ``index`` until a boundary type (exclusive)."""
    collected: list[BattleLogEvent] = []
    for later in logs[index + 1 :]:
        event_type = getattr(later, "type", None)
        if event_type in boundary_types:
            break
        collected.append(later)
    return collected


def _targets_from_evidence(
    following: list[BattleLogEvent],
    actor: Pokemon,
) -> list[Pokemon]:
    """Infer move targets from subsequent HP / faint events."""
    targets: list[Pokemon] = []
    seen: set[tuple[str, Side, Slot]] = set()

    for later in following:
        candidate: Pokemon | None = None
        if isinstance(later, FaintEvent):
            candidate = later.pokemon
        elif isinstance(later, HPChangeEvent):
            # Self HP changes (recoil / drain heal) are not move targets.
            if _same_combatant(later.pokemon, actor):
                continue
            if later.hp_pct_change == 0:
                continue
            candidate = later.pokemon

        if candidate is None:
            continue
        key = (candidate.species, candidate.side, candidate.slot)
        if key in seen:
            continue
        seen.add(key)
        targets.append(candidate)

    return targets


def _targets_from_spread(
    actor: Pokemon,
    kind: str,
    game_state: GameState,
) -> list[Pokemon]:
    """Geometric targets for known spread moves from current field state."""
    targets: list[Pokemon] = []

    if kind == "all_foes":
        foe_side: Side = "opponent" if actor.side == "player" else "player"
        targets.extend(_active_pokemon_on_side(game_state, foe_side))
    elif kind == "all_adjacent":
        sides: tuple[Side, ...] = ("player", "opponent")
        for side in sides:
            for mon in _active_pokemon_on_side(game_state, side):
                if _same_combatant(mon, actor):
                    continue
                targets.append(mon)

    return targets


def _active_pokemon_on_side(game_state: GameState, side: Side) -> list[Pokemon]:
    side_state: SideState = game_state.player if side == "player" else game_state.opponent
    result: list[Pokemon] = []
    slots: tuple[tuple[Slot, ActivePokemon | None], ...] = (
        (1, side_state.slot_1),
        (2, side_state.slot_2),
    )
    for slot, active in slots:
        if active is None:
            continue
        result.append(Pokemon(species=active.species, side=side, slot=slot))
    return result


def _resolve_actor_slot(
    actor: Pokemon,
    game_state: GameState | None,
) -> Pokemon:
    """Fill actor slot from GameState when species uniquely matches an active."""
    if game_state is None:
        return actor

    side_state = game_state.player if actor.side == "player" else game_state.opponent
    matches: list[tuple[Slot, str]] = []
    slots: tuple[tuple[Slot, ActivePokemon | None], ...] = (
        (1, side_state.slot_1),
        (2, side_state.slot_2),
    )
    for slot, active in slots:
        if active is None:
            continue
        if _species_matches(active.species, actor.species):
            matches.append((slot, active.species))

    if len(matches) != 1:
        return actor

    slot, species = matches[0]
    if actor.slot == slot and actor.species == species:
        return actor
    return Pokemon(species=species, side=actor.side, slot=slot)


def _species_matches(left: str, right: str) -> bool:
    a = left.strip().lower()
    b = right.strip().lower()
    if a == b:
        return True
    # Mega / form prefixes: "Mega Scizor" ↔ "Scizor"
    for x, y in ((a, b), (b, a)):
        if x.startswith("mega ") and x[len("mega ") :] == y:
            return True
    return False


def _same_combatant(left: Pokemon, right: Pokemon) -> bool:
    if left.side != right.side:
        return False
    if left.slot == right.slot and _species_matches(left.species, right.species):
        return True
    # Slot may still be unresolved (OCR default); fall back to species on side.
    return _species_matches(left.species, right.species)

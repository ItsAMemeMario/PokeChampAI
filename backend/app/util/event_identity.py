"""Shared semantic identity for battle-log dedupe and OCR re-emit suppression."""

from __future__ import annotations

from app.schema.battle_log import BattleLogEvent, LeadInEvent


def semantic_key(
    event: BattleLogEvent,
    *,
    include_raw_text: bool = False,
) -> tuple:
    """Return a stable identity for ``event``.

    Used by session turn-level dedupe and OCR region fingerprints. Parser-level
    within-line dedupe may set ``include_raw_text=True``.
    """
    key: list[object] = [event.type]
    if include_raw_text:
        key.append(event.raw_text)

    if isinstance(event, LeadInEvent):
        key.extend([event.side, event.slot_1.species, event.slot_2.species])
        return tuple(key)

    pokemon = getattr(event, "pokemon", None) or getattr(event, "actor", None)
    if pokemon is None:
        pokemon = getattr(event, "source", None)
    if pokemon is not None:
        # switch_in/out: omit slot — a bad single-lead OCR often defaults slot=1,
        # then a cleaner reading may assign the true slot.
        if event.type in {"switch_in", "switch_out"}:
            key.extend([pokemon.species, pokemon.side])
        else:
            key.extend([pokemon.species, pokemon.side, pokemon.slot])

    if event.type == "stat_change":
        key.extend(
            [getattr(event, "stat", None), getattr(event, "stages_delta", None)]
        )
    if event.type == "stat_stage_operation":
        key.extend(
            [
                getattr(event, "operation", None),
                getattr(getattr(event, "target", None), "species", None),
                tuple(getattr(event, "stats", ()) or ()),
            ]
        )
    if event.type in {"move_used", "move_failed"}:
        key.append(getattr(event, "move", None))
        if event.type == "move_failed":
            key.append(getattr(event, "reason", None))
    if event.type == "move_availability_changed":
        key.extend(
            [
                getattr(event, "restriction", None),
                getattr(event, "move", None),
                getattr(event, "source_item", None),
            ]
        )
    if event.type == "move_outcome":
        key.extend(
            [
                getattr(event, "outcome", None),
                getattr(getattr(event, "target", None), "species", None),
                getattr(event, "count", None),
            ]
        )
    if event.type == "item_used":
        key.append(getattr(event, "item", None))
    if event.type == "held_item_changed":
        key.extend(
            [
                getattr(event, "change", None),
                getattr(event, "item", None),
                getattr(getattr(event, "source", None), "species", None),
            ]
        )
    if event.type == "ability_triggered":
        key.append(getattr(event, "ability", None))
    if event.type in {"status_applied", "status_cured", "volatile_applied", "volatile_cured"}:
        key.append(getattr(event, "status", None) or getattr(event, "volatile", None))
    if event.type in {"weather_start", "weather_end"}:
        key.append(getattr(event, "weather", None))
    if event.type in {"terrain_start", "terrain_end"}:
        key.append(getattr(event, "terrain", None))
    if event.type == "side_condition":
        key.extend(
            [
                getattr(event, "side", None),
                getattr(event, "condition", None),
                getattr(event, "action", None),
            ]
        )
    if event.type == "field_effect_changed":
        key.extend(
            [
                getattr(event, "effect", None),
                getattr(event, "action", None),
            ]
        )
    if event.type == "perish_song_started":
        key.append(getattr(event, "turns_remaining", None))
    if event.type == "switch_lock_started":
        key.append(getattr(event, "scope", None))
    return tuple(key)

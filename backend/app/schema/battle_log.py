from __future__ import annotations

from datetime import datetime
from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from app.schema.common import Side, Pokemon

StatName = Literal["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

MoveFailedReason = Literal[
    "failed",
    "nothing_happened",
    "flinch",
    "paralysis",
    "freeze",
    "sleep",
    "recharge",
    "no_pp",
    "gravity",
    "insufficient_hp",
    "unusable",
    "cooldown",
    "cant",
]

FieldEffect = Literal[
    "gravity",
    "magic_room",
    "wonder_room",
    "weather_suppression",
]

StatStageOperation = Literal[
    "clear_all",
    "clear_one",
    "invert",
    "copy",
    "swap_all",
    "swap_offensive",
    "swap_defensive",
]

HeldItemChange = Literal[
    "revealed",
    "activated",
    "consumed",
    "lost",
    "obtained",
    "stolen",
]

MoveAvailabilityRestriction = Literal[
    "no_pp",
    "forced_move",
    "cooldown",
    "unusable",
]

MoveOutcome = Literal[
    "extremely_effective",
    "super_effective",
    "resisted",
    "mostly_ineffective",
    "critical",
    "immune",
    "miss",
    "ohko",
    "hit_count",
]

SideCondition = Literal[
    "tailwind",
    "reflect",
    "light_screen",
    "aurora_veil",
    "spikes",
    "toxic_spikes",
    "stealth_rocks",
    "safeguard",
    "sticky_web",
]


class BattleLogEventBase(BaseModel):
    raw_text: str
    timestamp: datetime = Field(default_factory=datetime.now)


class TurnStartEvent(BattleLogEventBase):
    type: Literal["turn_start"] = "turn_start"
    turn_number: int

class MegaEvolutionEvent(BattleLogEventBase):
    type: Literal["mega_evolution"] = "mega_evolution"
    pokemon: Pokemon
    variant: Literal["regular", "X", "Y", "Z"] = "regular"

class MoveUsedEvent(BattleLogEventBase):
    type: Literal["move_used"] = "move_used"
    actor: Pokemon
    move: str
    targets: List[Pokemon]

class MoveFailedEvent(BattleLogEventBase):
    type: Literal["move_failed"] = "move_failed"
    actor: Pokemon | None = None
    move: str = ""
    reason: MoveFailedReason = "failed"


class AbilityTriggeredEvent(BattleLogEventBase):
    type: Literal["ability_triggered"] = "ability_triggered"
    actor: Pokemon
    ability: str
    effect_text: str


class SwitchInEvent(BattleLogEventBase):
    type: Literal["switch_in"] = "switch_in"
    pokemon: Pokemon


class LeadInEvent(BattleLogEventBase):
    """Opening dual lead send-out as a single game message.

    Covers texts like ``Go! Sneasler and Grimmsnarl!`` or
    ``Blue sent out Musharna and Dragapult!``.
    """

    type: Literal["lead_in"] = "lead_in"
    side: Side
    slot_1: Pokemon
    slot_2: Pokemon


class SwitchOutEvent(BattleLogEventBase):
    type: Literal["switch_out"] = "switch_out"
    pokemon: Pokemon

class ItemUsedEvent(BattleLogEventBase):
    type: Literal["item_used"] = "item_used"
    pokemon: Pokemon
    item: str

class HPChangeEvent(BattleLogEventBase):
    type: Literal["hp_change"] = "hp_change"
    pokemon: Pokemon
    hp_pct_change: int = Field(ge=-100, le=100)

class StatChangeEvent(BattleLogEventBase):
    type: Literal["stat_change"] = "stat_change"
    pokemon: Pokemon
    stat: StatName
    stages_delta: int = Field(ge=-6, le=6)


class StatusAppliedEvent(BattleLogEventBase):
    type: Literal["status_applied"] = "status_applied"
    pokemon: Pokemon
    status: Literal["brn", "par", "slp", "psn", "tox", "frz"]

class StatusCuredEvent(BattleLogEventBase):
    type: Literal["status_cured"] = "status_cured"
    pokemon: Pokemon
    status: Literal["brn", "par", "slp", "psn", "tox", "frz"]


class VolatileAppliedEvent(BattleLogEventBase):
    type: Literal["volatile_applied"] = "volatile_applied"
    pokemon: Pokemon
    volatile: Literal["taunted", "encore", "confused"]

class VolatileCuredEvent(BattleLogEventBase):
    type: Literal["volatile_cured"] = "volatile_cured"
    pokemon: Pokemon
    volatile: Literal["taunted", "encore", "confused"]


class FaintEvent(BattleLogEventBase):
    type: Literal["faint"] = "faint"
    pokemon: Pokemon


Weather = Literal["sunny", "rain", "sandstorm", "snow"]
class WeatherStartEvent(BattleLogEventBase):
    type: Literal["weather_start"] = "weather_start"
    weather: Weather

class WeatherEndEvent(BattleLogEventBase):
    type: Literal["weather_end"] = "weather_end"
    weather: Weather

Terrain = Literal["electric_terrain", "grassy_terrain", "misty_terrain", "psychic_terrain"]
class TerrainStartEvent(BattleLogEventBase):
    type: Literal["terrain_start"] = "terrain_start"
    terrain: Terrain

class TerrainEndEvent(BattleLogEventBase):
    type: Literal["terrain_end"] = "terrain_end"
    terrain: Terrain


class TrickRoomStartEvent(BattleLogEventBase):
    type: Literal["trick_room_start"] = "trick_room_start"

class TrickRoomEndEvent(BattleLogEventBase):
    type: Literal["trick_room_end"] = "trick_room_end"

FieldEvent = Union[
    WeatherStartEvent,
    WeatherEndEvent,
    TerrainStartEvent,
    TerrainEndEvent,
    TrickRoomStartEvent,
    TrickRoomEndEvent,
]

class SideConditionEvent(BattleLogEventBase):
    type: Literal["side_condition"] = "side_condition"
    side: Side
    condition: SideCondition
    action: Literal["start", "end"] = "start"


class FieldEffectChangedEvent(BattleLogEventBase):
    type: Literal["field_effect_changed"] = "field_effect_changed"
    effect: FieldEffect
    action: Literal["start", "end"]
    source: Pokemon | None = None


class PerishSongStartedEvent(BattleLogEventBase):
    type: Literal["perish_song_started"] = "perish_song_started"
    turns_remaining: Literal[3] = 3
    source: Pokemon | None = None
    affected: List[Pokemon] = Field(default_factory=list)


class SwitchLockStartedEvent(BattleLogEventBase):
    type: Literal["switch_lock_started"] = "switch_lock_started"
    scope: Literal["all_active"] = "all_active"
    activates_next_turn: Literal[True] = True
    source: Pokemon | None = None


class StatStageOperationEvent(BattleLogEventBase):
    type: Literal["stat_stage_operation"] = "stat_stage_operation"
    operation: StatStageOperation
    pokemon: Pokemon | None = None
    target: Pokemon | None = None
    stats: List[StatName] = Field(default_factory=list)


class HeldItemChangedEvent(BattleLogEventBase):
    type: Literal["held_item_changed"] = "held_item_changed"
    change: HeldItemChange
    pokemon: Pokemon | None = None
    item: str | None = None
    source: Pokemon | None = None
    associated_move: str | None = None


class MoveAvailabilityChangedEvent(BattleLogEventBase):
    type: Literal["move_availability_changed"] = "move_availability_changed"
    restriction: MoveAvailabilityRestriction
    pokemon: Pokemon | None = None
    move: str | None = None
    source_item: str | None = None
    clears_on_switch: bool | None = None


class MoveOutcomeEvent(BattleLogEventBase):
    type: Literal["move_outcome"] = "move_outcome"
    outcome: MoveOutcome
    target: Pokemon | None = None
    count: int | None = Field(default=None, ge=1)


BattleLogEvent = Annotated[
    Union[
        TurnStartEvent,
        MegaEvolutionEvent,
        MoveUsedEvent,
        MoveFailedEvent,
        AbilityTriggeredEvent,
        LeadInEvent,
        SwitchInEvent,
        SwitchOutEvent,
        ItemUsedEvent,
        HPChangeEvent,
        StatChangeEvent,
        StatusAppliedEvent,
        StatusCuredEvent,
        VolatileAppliedEvent,
        VolatileCuredEvent,
        FaintEvent,
        FieldEvent,
        SideConditionEvent,
        FieldEffectChangedEvent,
        PerishSongStartedEvent,
        SwitchLockStartedEvent,
        StatStageOperationEvent,
        HeldItemChangedEvent,
        MoveAvailabilityChangedEvent,
        MoveOutcomeEvent,
    ],
    Field(discriminator="type"),
]

_battle_log_event_adapter = TypeAdapter(BattleLogEvent)


def parse_battle_log_event(data: dict) -> BattleLogEvent:
    """Deserialize a dict into the appropriate BattleLogEvent variant."""
    return _battle_log_event_adapter.validate_python(data)

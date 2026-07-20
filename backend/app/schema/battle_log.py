from __future__ import annotations

from datetime import datetime
from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from app.schema.common import Side, Pokemon

class BattleLogEventBase(BaseModel):
    raw_text: str
    timestamp: datetime = Field(default_factory=datetime.now)


class TurnStartEvent(BattleLogEventBase):
    type: Literal["turn_start"] = "turn_start"
    turn_number: int

class MegaEvolutionEvent(BattleLogEventBase):
    type: Literal["mega_evolution"] = "mega_evolution"
    pokemon: Pokemon

class MoveUsedEvent(BattleLogEventBase):
    type: Literal["move_used"] = "move_used"
    actor: Pokemon
    move: str
    targets: List[Pokemon]

class MoveFailedEvent(BattleLogEventBase):
    type: Literal["move_failed"] = "move_failed"
    actor: Pokemon | None = None
    move: str = ""


class AbilityTriggeredEvent(BattleLogEventBase):
    type: Literal["ability_triggered"] = "ability_triggered"
    actor: Pokemon
    ability: str
    effect_text: str


class SwitchInEvent(BattleLogEventBase):
    type: Literal["switch_in"] = "switch_in"
    pokemon: Pokemon


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
    stat: Literal["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    stages_delta: int = Field(ge=-6, le=6)


class StatusAppliedEvent(BattleLogEventBase):
    type: Literal["status_applied"] = "status_applied"
    pokemon: Pokemon
    status: Literal["brn", "par", "slp", "psn", "tox", "frz"]


class VolatileAppliedEvent(BattleLogEventBase):
    type: Literal["volatile_applied"] = "volatile_applied"
    pokemon: Pokemon
    volatile: Literal["taunted", "encore", "confused"]


class FaintEvent(BattleLogEventBase):
    type: Literal["faint"] = "faint"
    pokemon: Pokemon


Weather = Literal["sunny", "rain", "sandstorm", "snow", "none"]
class WeatherChangeEvent(BattleLogEventBase):
    type: Literal["weather_change"] = "weather_change"
    weather: Weather

Terrain = Literal["electric_terrain", "grassy_terrain", "misty_terrain", "psychic_terrain", "none"]
class TerrainChangeEvent(BattleLogEventBase):
    type: Literal["terrain_change"] = "terrain_change"
    terrain: Terrain


class TrickRoomChangeEvent(BattleLogEventBase):
    type: Literal["trick_room_change"] = "trick_room_change"
    active: bool

FieldEvent = Union[WeatherChangeEvent, TerrainChangeEvent, TrickRoomChangeEvent]

class SideConditionEvent(BattleLogEventBase):
    type: Literal["side_condition"] = "side_condition"
    side: Side
    condition: Literal[
        "tailwind",
        "reflect",
        "light_screen",
        "aurora_veil",
        "spikes",
        "toxic_spikes",
        "stealth_rocks",
    ]


BattleLogEvent = Annotated[
    Union[
        TurnStartEvent,
        MegaEvolutionEvent,
        MoveUsedEvent,
        MoveFailedEvent,
        AbilityTriggeredEvent,
        SwitchInEvent,
        SwitchOutEvent,
        ItemUsedEvent,
        HPChangeEvent,
        StatChangeEvent,
        StatusAppliedEvent,
        VolatileAppliedEvent,
        FaintEvent,
        FieldEvent,
        SideConditionEvent,
    ],
    Field(discriminator="type"),
]

_battle_log_event_adapter = TypeAdapter(BattleLogEvent)


def parse_battle_log_event(data: dict) -> BattleLogEvent:
    """Deserialize a dict into the appropriate BattleLogEvent variant."""
    return _battle_log_event_adapter.validate_python(data)

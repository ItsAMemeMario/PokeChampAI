from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from app.schema.battle_log import MoveAvailabilityRestriction


class StatStages(BaseModel):
    atk: int = Field(0, ge=-6, le=6)
    def_: int = Field(0, alias="def", ge=-6, le=6)
    spa: int = Field(0, ge=-6, le=6)
    spd: int = Field(0, ge=-6, le=6)
    spe: int = Field(0, ge=-6, le=6)
    evasion: int = Field(0, ge=-6, le=6)
    accuracy: int = Field(0, ge=-6, le=6)


class MoveRestrictionState(BaseModel):
    """Known restriction on what a Pokemon may legally choose next turn."""

    restriction: MoveAvailabilityRestriction
    move: Optional[str] = None
    source_item: Optional[str] = None
    clears_on_switch: bool = True


# Pokemon currently active on the field
class ActivePokemon(BaseModel):
    species: str
    hp_percentage: int = Field(ge=0, le=100)
    status_condition: Literal["none", "brn", "par", "slp", "psn", "tox", "frz"] = "none"
    stat_stages: StatStages
    volatile_statuses: List[str] = Field(default_factory=list) # e.g., ["taunted", "encore", "confused"]
    is_protected_this_turn: bool = False
    is_protected_last_turn: bool = False
    # 0 = not under Perish Song; 1-3 = turns remaining until faint.
    perish_turns: int = Field(0, ge=0, le=3)
    # unknown: never seen; held: known and still present; consumed/lost: gone.
    item_state: Literal["unknown", "held", "consumed", "lost"] = "unknown"
    move_restrictions: List[MoveRestrictionState] = Field(default_factory=list)

    # For the opponent, these start as None and populate as the vision model detects them
    revealed_ability: Optional[str] = None
    revealed_item: Optional[str] = None
    revealed_moves: List[str] = Field(default_factory=list)

# Pokemon currently on the bench
# For the opponent, these are added as active pokemon are benched
class BenchedPokemon(BaseModel):
    species: str
    hp_percentage: int = Field(ge=0, le=100)
    status_condition: Literal["none", "brn", "par", "slp", "psn", "tox", "frz"] = "none"
    item_state: Literal["unknown", "held", "consumed", "lost"] = "unknown"

    # For the opponent, these start as None and populate as the vision model detects them
    revealed_ability: Optional[str] = None
    revealed_item: Optional[str] = None
    revealed_moves: List[str] = Field(default_factory=list)

class Hazards(BaseModel):
    spikes: Literal[0, 1, 2, 3] = 0
    toxic_spikes: Literal[0, 1, 2] = 0
    stealth_rocks: Literal[0, 1] = 0
    sticky_web: Literal[0, 1] = 0

class SideState(BaseModel):
    # Explicit slots make targeting logic much easier for the LLM
    slot_1: Optional[ActivePokemon] = None
    slot_2: Optional[ActivePokemon] = None
    benched: List[BenchedPokemon]
    mega_used: bool = False
    tailwind_turns: int = 0
    reflect_turns: int = 0
    light_screen_turns: int = 0
    aurora_veil_turns: int = 0
    safeguard_turns: int = 0
    hazards: Hazards

class FieldState(BaseModel):
    weather: Literal["none", "sun", "rain", "sand", "snow"] = "none"
    weather_turns: int = 0
    # True when Cloud Nine / Air Lock suppresses weather without ending it.
    weather_suppressed: bool = False
    terrain: Literal["none", "electric", "grassy", "misty", "psychic"] = "none"
    terrain_turns: int = 0
    trick_room_turns: int = 0
    gravity_turns: int = 0
    magic_room_turns: int = 0
    wonder_room_turns: int = 0
    # 0 = inactive; 1 = Fairy Lock applies to the upcoming turn / current lock window.
    fairy_lock_turns: int = 0

class GameState(BaseModel):
    turn_number: int
    field: FieldState
    player: SideState
    opponent: SideState

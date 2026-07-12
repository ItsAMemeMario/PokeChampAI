from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class StatStages(BaseModel):
    atk: int = Field(0, ge=-6, le=6)
    def_: int = Field(0, alias="def", ge=-6, le=6)
    spa: int = Field(0, ge=-6, le=6)
    spd: int = Field(0, ge=-6, le=6)
    spe: int = Field(0, ge=-6, le=6)
    evasion: int = Field(0, ge=-6, le=6)
    accuracy: int = Field(0, ge=-6, le=6)

# Pokemon currently active on the field
class ActivePokemon(BaseModel):
    species: str
    hp_percentage: int = Field(ge=0, le=100)
    status_condition: Literal["none", "brn", "par", "slp", "psn", "tox", "frz"] = "none"
    stat_stages: StatStages
    volatile_statuses: List[str] = Field(default_factory=list) # e.g., ["taunted", "encored", "confused"]
    is_protected_last_turn: bool = False
    
    # For the opponent, these start as None and populate as the vision model detects them
    revealed_ability: Optional[str] = None
    revealed_item: Optional[str] = None
    revealed_moves: List[str] = Field(default_factory=list)

# Pokemon currently on the bench
# For the opponent, these are added as active pokemon are benched
class BenchedPokemon(BaseModel):
    species: str
    hp_percentage: int = Field(ge=0, le=100)
    status_condition: str = "none"

    # For the opponent, these start as None and populate as the vision model detects them
    revealed_ability: Optional[str] = None
    revealed_item: Optional[str] = None
    revealed_moves: List[str] = Field(default_factory=list)

class SideState(BaseModel):
    # Explicit slots make targeting logic much easier for the LLM
    slot_1: Optional[ActivePokemon] = None 
    slot_2: Optional[ActivePokemon] = None
    benched: List[BenchedPokemon]
    tailwind_turns: int = 0
    reflect_turns: int = 0
    light_screen_turns: int = 0

class FieldState(BaseModel):
    weather: Literal["none", "sun", "rain", "sand", "snow"] = "none"
    weather_turns: int = 0
    terrain: Literal["none", "electric", "grassy", "misty", "psychic"] = "none"
    terrain_turns: int = 0
    trick_room_turns: int = 0

class GameState(BaseModel):
    turn_number: int
    field: FieldState
    player: SideState
    opponent: SideState
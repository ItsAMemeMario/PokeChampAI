from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

Side = Literal["player", "opponent"]
Slot = Literal[1, 2]

class Pokemon(BaseModel):
    species: str
    side: Side
    slot: Slot
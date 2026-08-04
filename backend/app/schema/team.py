from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.data.items import REGULATION_MB_ITEMS
from app.util.pokepaste_parser import parse


class PlayerPokemon(BaseModel):
    species: str
    item: str
    ability: str
    evs: dict[str, int]
    nature: str
    moves: list[str]

    @field_validator("item")
    @classmethod
    def item_must_be_regulation_mb_legal(cls, value: str) -> str:
        item = value.strip()
        if item not in REGULATION_MB_ITEMS:
            raise ValueError(f"Item '{item}' is not legal in Regulation M-B")
        return item


class PlayerTeam(BaseModel):
    pokemon: list[PlayerPokemon]
    regulation: Literal["M-B"] = "M-B"

    @field_validator("pokemon")
    @classmethod
    def exactly_six_pokemon(cls, value: list[PlayerPokemon]) -> list[PlayerPokemon]:
        if len(value) != 6:
            raise ValueError(f"Team must contain exactly 6 Pokemon, got {len(value)}")
        return value


class OpponentTeamPreview(BaseModel):
    """Six opponent species identified from team-preview sprite crops."""

    species: list[str] = Field(description="Exactly 6 opponent species, top to bottom")

    @field_validator("species")
    @classmethod
    def exactly_six_species(cls, value: list[str]) -> list[str]:
        if len(value) != 6:
            raise ValueError(f"Expected exactly 6 opponent species, got {len(value)}")
        return value


def player_pokemon_from_dict(data: dict) -> PlayerPokemon:
    """Convert a pokepaste_parser.parse() entry into a PlayerPokemon."""
    return PlayerPokemon(
        species=data["species"],
        item=data["item"],
        ability=data["ability"],
        evs=data["evs"],
        nature=data["nature"],
        moves=data["moves"],
    )


def parse_team(pokepaste: str, *, regulation: Literal["M-B"] = "M-B") -> PlayerTeam:
    """Parse a Showdown pokepaste string into a validated PlayerTeam."""
    parsed = parse(pokepaste)
    if len(parsed) != 6:
        raise ValueError(f"Expected exactly 6 Pokemon in pokepaste, got {len(parsed)}")
    try:
        return PlayerTeam(
            pokemon=[player_pokemon_from_dict(entry) for entry in parsed],
            regulation=regulation,
        )
    except ValidationError as exc:
        messages: list[str] = []
        for error in exc.errors():
            msg = error.get("msg", "Invalid team data")
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, ") :]
            messages.append(msg)
        raise ValueError("; ".join(messages) if messages else str(exc)) from exc

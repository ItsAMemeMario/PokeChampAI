from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.util.pokepaste_parser import parse


class PlayerPokemon(BaseModel):
    species: str
    item: str
    ability: str
    evs: dict[str, int]
    nature: str
    moves: list[str]


class PlayerTeam(BaseModel):
    pokemon: list[PlayerPokemon]
    regulation: Literal["M-B"] = "M-B"

    @field_validator("pokemon")
    @classmethod
    def exactly_six_pokemon(cls, value: list[PlayerPokemon]) -> list[PlayerPokemon]:
        if len(value) != 6:
            raise ValueError(f"Team must contain exactly 6 Pokemon, got {len(value)}")
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
    return PlayerTeam(
        pokemon=[player_pokemon_from_dict(entry) for entry in parsed],
        regulation=regulation,
    )

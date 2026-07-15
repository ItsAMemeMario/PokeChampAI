"""Tests for PlayerTeam parsing and Regulation M-B item validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schema.team import PlayerPokemon, PlayerTeam, parse_team

_LEGAL_PASTE = """
Sinistcha @ Sitrus Berry
Ability: Hospitality
Level: 50
EVs: 252 HP / 4 Def / 252 SpD
Bold Nature
- Matcha Gotcha
- Strength Sap
- Rage Powder
- Protect

Staraptor @ Staraptorite
Ability: Intimidate
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Brave Bird
- Close Combat
- U-turn
- Protect

Garchomp @ Lum Berry
Ability: Rough Skin
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Earthquake
- Outrage
- Rock Slide
- Protect

Grimmsnarl @ Light Clay
Ability: Prankster
Level: 50
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Reflect
- Light Screen
- Thunder Wave
- Foul Play

Charizard @ Charizardite Y
Ability: Blaze
Level: 50
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Heat Wave
- Solar Beam
- Air Slash
- Protect

Sneasler @ Focus Sash
Ability: Poison Touch
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Dire Claw
- Close Combat
- Fake Out
- Protect
"""

_ILLEGAL_ITEM_PASTE = """
Sinistcha @ Sitrus Berry
Ability: Hospitality
Level: 50
EVs: 252 HP / 4 Def / 252 SpD
Bold Nature
- Matcha Gotcha
- Strength Sap
- Rage Powder
- Protect

Staraptor @ Choice Band
Ability: Intimidate
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Brave Bird
- Close Combat
- U-turn
- Protect

Garchomp @ Lum Berry
Ability: Rough Skin
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Earthquake
- Outrage
- Rock Slide
- Protect

Grimmsnarl @ Light Clay
Ability: Prankster
Level: 50
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Reflect
- Light Screen
- Thunder Wave
- Foul Play

Charizard @ Charizardite Y
Ability: Blaze
Level: 50
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Heat Wave
- Solar Beam
- Air Slash
- Protect

Sneasler @ Focus Sash
Ability: Poison Touch
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Dire Claw
- Close Combat
- Fake Out
- Protect
"""


def test_parse_team_accepts_legal_regulation_mb_items() -> None:
    team = parse_team(_LEGAL_PASTE)
    assert len(team.pokemon) == 6
    assert team.pokemon[0].item == "Sitrus Berry"
    assert team.pokemon[1].item == "Staraptorite"


def test_parse_team_rejects_illegal_item() -> None:
    with pytest.raises(ValueError, match="Choice Band.*not legal in Regulation M-B"):
        parse_team(_ILLEGAL_ITEM_PASTE)


def test_player_pokemon_rejects_illegal_item() -> None:
    with pytest.raises(ValidationError, match="not legal in Regulation M-B"):
        PlayerPokemon(
            species="Garchomp",
            item="Assault Vest",
            ability="Rough Skin",
            evs={"Atk": 252},
            nature="Jolly",
            moves=["Earthquake"],
        )


def test_player_team_rejects_illegal_item_on_construction() -> None:
    legal = PlayerPokemon(
        species="Sinistcha",
        item="Sitrus Berry",
        ability="Hospitality",
        evs={"HP": 252},
        nature="Bold",
        moves=["Matcha Gotcha"],
    )
    with pytest.raises(ValidationError, match="not legal in Regulation M-B"):
        PlayerTeam(
            pokemon=[
                legal,
                PlayerPokemon(
                    species="Staraptor",
                    item="Choice Band",
                    ability="Intimidate",
                    evs={"Atk": 252},
                    nature="Jolly",
                    moves=["Brave Bird"],
                ),
                *[
                    PlayerPokemon(
                        species=f"Mon{i}",
                        item="Leftovers",
                        ability="Ability",
                        evs={"HP": 0},
                        nature="Timid",
                        moves=["Protect"],
                    )
                    for i in range(4)
                ],
            ]
        )

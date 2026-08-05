"""Tests for the Showdown pokepaste parser."""

from __future__ import annotations

from app.util.pokepaste_parser import parse

_PASTE = """
Staraptor @ Staraptite  
Ability: Intimidate  
Level: 50  
EVs: 2 HP / 32 Atk / 32 Spe  
Jolly Nature  
- Close Combat  
- Brave Bird  
- Roost  
- Protect  

Grimmsnarl @ Wide Lens  
Ability: Prankster  
Level: 50  
EVs: 29 HP / 22 Def / 15 SpD  
Careful Nature  
- Spirit Break  
- Swagger  
- Scary Face  
- Parting Shot  

Charizard @ Charizardite Y  
Ability: Blaze  
Level: 50  
EVs: 8 HP / 17 Def / 20 SpA / 21 Spe  
Modest Nature  
- Heat Wave  
- Solar Beam  
- Weather Ball  
- Protect  

Garchomp @ Lum Berry  
Ability: Rough Skin  
Level: 50  
EVs: 2 HP / 32 Atk / 32 Spe  
Jolly Nature  
- Dragon Claw  
- Earthquake  
- Rock Slide  
- Protect  

Sneasler @ Persim Berry  
Ability: Unburden  
Level: 50  
EVs: 2 HP / 32 Atk / 32 Spe  
Adamant Nature  
- Close Combat  
- Dire Claw  
- Rock Tomb  
- Throat Chop  

Sinistcha @ Sitrus Berry  
Ability: Hospitality  
Level: 50  
EVs: 32 HP / 2 Def / 32 SpD  
Bold Nature  
- Matcha Gotcha  
- Strength Sap  
- Life Dew  
- Rage Powder
"""


def test_parse_vgc_team_paste() -> None:
    team = parse(_PASTE)

    assert len(team) == 6

    assert team[0] == {
        "species": "Staraptor",
        "item": "Staraptite",
        "ability": "Intimidate",
        "evs": {"HP": 2, "Atk": 32, "Spe": 32},
        "nature": "Jolly",
        "moves": ["Close Combat", "Brave Bird", "Roost", "Protect"],
    }
    assert team[1] == {
        "species": "Grimmsnarl",
        "item": "Wide Lens",
        "ability": "Prankster",
        "evs": {"HP": 29, "Def": 22, "SpD": 15},
        "nature": "Careful",
        "moves": ["Spirit Break", "Swagger", "Scary Face", "Parting Shot"],
    }
    assert team[2] == {
        "species": "Charizard",
        "item": "Charizardite Y",
        "ability": "Blaze",
        "evs": {"HP": 8, "Def": 17, "SpA": 20, "Spe": 21},
        "nature": "Modest",
        "moves": ["Heat Wave", "Solar Beam", "Weather Ball", "Protect"],
    }
    assert team[3] == {
        "species": "Garchomp",
        "item": "Lum Berry",
        "ability": "Rough Skin",
        "evs": {"HP": 2, "Atk": 32, "Spe": 32},
        "nature": "Jolly",
        "moves": ["Dragon Claw", "Earthquake", "Rock Slide", "Protect"],
    }
    assert team[4] == {
        "species": "Sneasler",
        "item": "Persim Berry",
        "ability": "Unburden",
        "evs": {"HP": 2, "Atk": 32, "Spe": 32},
        "nature": "Adamant",
        "moves": ["Close Combat", "Dire Claw", "Rock Tomb", "Throat Chop"],
    }
    assert team[5] == {
        "species": "Sinistcha",
        "item": "Sitrus Berry",
        "ability": "Hospitality",
        "evs": {"HP": 32, "Def": 2, "SpD": 32},
        "nature": "Bold",
        "moves": ["Matcha Gotcha", "Strength Sap", "Life Dew", "Rage Powder"],
    }

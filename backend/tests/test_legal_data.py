"""Tests for Regulation M-B static libraries and RapidFuzz snapping."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.cv.event_parser import parse_battle_text, parse_side_banner
from app.data.abilities import REGULATION_MB_ABILITIES
from app.data.moves import ALL_ADJACENT_MOVES, REGULATION_MB_MOVES, spread_kind
from app.data.species import REGULATION_MB_SPECIES
from app.schema.team import PlayerPokemon, parse_team
from app.util.legal_snap import snap_to_legal


def test_species_includes_alternate_forms_not_megas() -> None:
    assert "Goodra-Hisui" in REGULATION_MB_SPECIES
    assert "Arcanine-Hisui" in REGULATION_MB_SPECIES
    assert "Venusaur" in REGULATION_MB_SPECIES
    assert "Venusaur-Mega" not in REGULATION_MB_SPECIES
    assert "Charizard-Mega-X" not in REGULATION_MB_SPECIES


def test_abilities_include_mega_only_abilities() -> None:
    # Drought is on Charizard-Mega-Y; should be in the ability pool.
    assert "Drought" in REGULATION_MB_ABILITIES
    assert "Intimidate" in REGULATION_MB_ABILITIES


def test_moves_pool_and_spread_kind() -> None:
    assert "Earthquake" in REGULATION_MB_MOVES
    assert "Earthquake" in ALL_ADJACENT_MOVES
    assert spread_kind("Earthquake") == "all_adjacent"
    assert spread_kind("Rock Slide") == "all_foes"


def test_snap_to_legal_typo() -> None:
    assert snap_to_legal("Garchmp", REGULATION_MB_SPECIES) == "Garchomp"
    assert snap_to_legal("earthquak", REGULATION_MB_MOVES) == "Earthquake"


def test_snap_to_legal_known_list_resolves_forms() -> None:
    # Species clause: only one Arcanine* can be known; OCR never has form suffixes.
    known = ["Arcanine-Hisui", "Incineroar", "Sinistcha", "Staraptor"]
    assert snap_to_legal("Arcanine", known) == "Arcanine-Hisui"
    assert snap_to_legal("Incineroar", known) == "Incineroar"
    assert snap_to_legal("Garchmp", ["Garchomp", "Scizor", "Hatterene", "Milotic"]) == "Garchomp"


def test_parse_battle_text_snaps_species_and_move() -> None:
    events = parse_battle_text(
        "The opposing Garchmp used Earthquak!",
        opponent_species=["Garchomp", "Scizor", "Hatterene", "Milotic", "Blaziken", "Amoonguss"],
    )
    assert len(events) == 1
    event = events[0]
    assert event.type == "move_used"
    assert event.actor.species == "Garchomp"
    assert event.move == "Earthquake"


def test_parse_battle_text_uses_side_specific_species_lists() -> None:
    events = parse_battle_text(
        "Arcanine used Flare Blitz!",
        player_species=["Arcanine-Hisui", "Sinistcha", "Staraptor", "Garchomp"],
        opponent_species=["Arcanine", "Scizor", "Hatterene", "Milotic", "Blaziken", "Amoonguss"],
    )
    assert events[0].actor.species == "Arcanine-Hisui"

    events = parse_battle_text(
        "The opposing Arcanine used Flare Blitz!",
        player_species=["Arcanine-Hisui", "Sinistcha", "Staraptor", "Garchomp"],
        opponent_species=["Arcanine", "Scizor", "Hatterene", "Milotic", "Blaziken", "Amoonguss"],
    )
    assert events[0].actor.species == "Arcanine"



def test_parse_side_banner_snaps_item() -> None:
    event = parse_side_banner("Garchomp's Leftovers", "player", slot=1)
    assert event is not None
    assert event.type == "item_used"
    assert event.item == "Leftovers"


def test_player_pokemon_rejects_illegal_species() -> None:
    with pytest.raises(ValidationError, match="Species .* not legal"):
        PlayerPokemon(
            species="MissingNo",
            item="Leftovers",
            ability="Intimidate",
            evs={"Atk": 252},
            nature="Jolly",
            moves=["Earthquake"],
        )


def test_player_pokemon_rejects_illegal_move() -> None:
    with pytest.raises(ValidationError, match="Move .* not legal"):
        PlayerPokemon(
            species="Garchomp",
            item="Lum Berry",
            ability="Rough Skin",
            evs={"Atk": 252},
            nature="Jolly",
            moves=["V-create"],
        )


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

Staraptor @ Staraptite
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


def test_parse_team_still_accepts_legal_paste() -> None:
    team = parse_team(_LEGAL_PASTE)
    assert team.pokemon[0].species == "Sinistcha"
    assert team.pokemon[5].species == "Sneasler"

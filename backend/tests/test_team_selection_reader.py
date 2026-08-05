"""Tests for player team selection via selection-order badge OCR."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.cv.regions import default_assets_dir, load_regions
from app.cv.team_selection_reader import (
    read_player_selected_species,
    read_selection_orders,
    split_player_selection_slots,
)
from app.schema.team import PlayerPokemon, PlayerTeam


def _load_asset(name: str) -> np.ndarray:
    path = default_assets_dir() / name
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


@pytest.fixture
def region_config():
    return load_regions()


@pytest.fixture
def player_team() -> PlayerTeam:
    species = [
        "Staraptor",
        "Grimmsnarl",
        "Charizard",
        "Sneasler",
        "Garchomp",
        "Sinistcha",
    ]
    return PlayerTeam(
        pokemon=[
            PlayerPokemon(
                species="Staraptor",
                item="Staraptite",
                ability="Intimidate",
                evs={"hp": 2, "atk": 32, "spe": 32},
                nature="Jolly",
                moves=["Close Combat", "Brave Bird", "Roost", "Protect"],
            ),
            PlayerPokemon(
                species="Grimmsnarl",
                item="Wide Lens",
                ability="Prankster",
                evs={"hp": 29, "def": 22, "spd": 15},
                nature="Careful",
                moves=["Spirit Break", "Swagger", "Scary Face", "Parting Shot"],
            ),
            PlayerPokemon(
                species="Charizard",
                item="Charizardite Y",
                ability="Blaze",
                evs={"hp": 8, "def": 17, "spa": 20, "spe": 21},
                nature="Modest",
                moves=["Heat Wave", "Solar Beam", "Weather Ball", "Protect"],
            ),
            PlayerPokemon(
                species="Sneasler",
                item="Persim Berry",
                ability="Unburden",
                evs={"hp": 2, "atk": 32, "spe": 32},
                nature="Adamant",
                moves=["Close Combat", "Dire Claw", "Rock Tomb", "Throat Chop"],
            ),
            PlayerPokemon(
                species="Garchomp",
                item="Lum Berry",
                ability="Rough Skin",
                evs={"hp": 2, "atk": 32, "spe": 32},
                nature="Jolly",
                moves=["Dragon Claw", "Earthquake", "Rock Slide", "Protect"],
            ),
            PlayerPokemon(
                species="Sinistcha",
                item="Sitrus Berry",
                ability="Hospitality",
                evs={"hp": 32, "def": 2, "spd": 32},
                nature="Bold",
                moves=["Matcha Gotcha", "Strength Sap", "Life Dew", "Rage Powder"],
            )
        ]
    )


def test_split_player_selection_slots(region_config) -> None:
    image = _load_asset("team_selection.png")
    slots = split_player_selection_slots(image, region_config)
    assert len(slots) == 6
    for slot in slots:
        assert slot.ndim == 3
        assert slot.shape[0] > 0
        assert slot.shape[1] > 0


def test_read_selection_orders_on_reference(region_config) -> None:
    image = _load_asset("team_selection.png")
    orders = read_selection_orders(image, region_config)
    assert orders == [1, 2, None, 4, 3, None]


def test_read_player_selected_species_uses_selection_order(
    region_config, player_team: PlayerTeam
) -> None:
    image = _load_asset("team_selection.png")
    selected = read_player_selected_species(image, region_config, player_team)
    # Badge order on reference: Staraptor=1, Grimmsnarl=2, Garchomp=3, Sneasler=4
    assert selected == ["Staraptor", "Grimmsnarl", "Garchomp", "Sneasler"]

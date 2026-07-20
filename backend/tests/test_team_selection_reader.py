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
                species=name,
                item="Sitrus Berry",
                ability="Ability",
                evs={"hp": 0},
                nature="Adamant",
                moves=["Move1", "Move2", "Move3", "Move4"],
            )
            for name in species
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

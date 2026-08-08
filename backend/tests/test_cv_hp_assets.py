"""End-to-end slot-card HP OCR tests for backend/tests/assets/cv/hp screenshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.cv.hp_reader import SlotCardRead, read_slot_card
from app.cv.regions import load_regions


def _hp_assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets" / "cv" / "hp"


def _load_hp_asset(name: str) -> np.ndarray:
    path = _hp_assets_dir() / name
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


@pytest.fixture
def region_config():
    return load_regions()


def test_staraptor_58_garchomp_100_delphox_2_aerodactyl_38(region_config) -> None:
    """Filename lists species/HP; calibrated opponent slots are right=1, left=2."""
    image = _load_hp_asset("staraptor_58_garchomp_100_delphox_2_aerodactyl_38.png")
    expected = {
        "player_slot_1_card": SlotCardRead(
            species="Staraptor",
            hp_pct=58,
            raw_text="Staraptor 94 / 162",
        ),
        "player_slot_2_card": SlotCardRead(
            species="Garchomp",
            hp_pct=100,
            raw_text="Garchomp 185 / 185",
        ),
        # Config places opponent slot 1 on the right (far) card.
        "opponent_slot_1_card": SlotCardRead(
            species="Aerodactyl",
            hp_pct=38,
            raw_text="Aerodactyl 38%",
        ),
        "opponent_slot_2_card": SlotCardRead(
            species="Delphox",
            hp_pct=2,
            raw_text="Delphox 2%",
        ),
    }

    for region_name, expected_reading in expected.items():
        reading = read_slot_card(image, region_config, region_name)
        assert reading is not None, region_name
        assert reading.species == expected_reading.species, region_name
        assert reading.hp_pct == expected_reading.hp_pct, region_name


def test_charizard_100_grimmsnarl_100_staraptor_100_whimsicott_100(region_config) -> None:
    """Filename lists species/HP; calibrated opponent slots are right=1, left=2."""
    image = _load_hp_asset("charizard_100_grimmsnarl_100_staraptor_100_whimsicott_100.png")
    expected = {
        "player_slot_1_card": SlotCardRead(
            species="Charizard",
            hp_pct=100,
            raw_text="Charizard 161 / 161",
        ),
        "player_slot_2_card": SlotCardRead(
            species="Grimmsnarl",
            hp_pct=100,
            raw_text="Grimmsnarl 199 / 199",
        ),
        # Config places opponent slot 1 on the right (far) card.
        "opponent_slot_1_card": SlotCardRead(
            species="Staraptor",
            hp_pct=100,
            raw_text="Staraptor 100%",
        ),
        "opponent_slot_2_card": SlotCardRead(
            species="Whimsicott",
            hp_pct=100,
            raw_text="Whimsicott 100%",
        ),
    }

    for region_name, expected_reading in expected.items():
        reading = read_slot_card(image, region_config, region_name)
        assert reading is not None, region_name
        assert reading.species == expected_reading.species, region_name
        assert reading.hp_pct == expected_reading.hp_pct, region_name


def test_grimmsnarl_53_musharna_41_metagross_67(region_config) -> None:
    """Varying amounts of HP; one player slot card absent."""
    image = _load_hp_asset("grimmsnarl_53_musharna_41_metagross_67.png")
    expected = {
        "player_slot_1_card": None,
        "player_slot_2_card": SlotCardRead(
            species="Grimmsnarl",
            hp_pct=53,
            raw_text="Grimmsnarl 106/199",
        ),
        # Config places opponent slot 1 on the right (far) card.
        "opponent_slot_1_card": SlotCardRead(
            species="Musharna",
            hp_pct=41,
            raw_text="Musharna 41%",
        ),
        "opponent_slot_2_card": SlotCardRead(
            species="Metagross",
            hp_pct=67,
            raw_text="Metagross 67%",
        ),
    }

    for region_name, expected_reading in expected.items():
        reading = read_slot_card(image, region_config, region_name)
        if expected_reading is None:
            assert reading is None, region_name
            continue
        assert reading is not None, region_name
        assert reading.species == expected_reading.species, region_name
        assert reading.hp_pct == expected_reading.hp_pct, region_name


def test_all_hp_assets_are_covered() -> None:
    on_disk = {path.name for path in _hp_assets_dir().glob("*.png")}
    assert on_disk == {
        "charizard_100_grimmsnarl_100_staraptor_100_whimsicott_100.png",
        "grimmsnarl_53_musharna_41_metagross_67.png",
        "staraptor_58_garchomp_100_delphox_2_aerodactyl_38.png",
    }

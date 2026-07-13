"""Regression tests for phase detection on reference screenshots."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.cv.phase_detector import PhaseDetector, detect_phase, has_battle_ended
from app.cv.regions import config_for_image, default_assets_dir, load_regions
from app.services.session import BattlePhase


def _load_asset(name: str) -> np.ndarray:
    path = default_assets_dir() / name
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _advance_to_battle_animation(detector: PhaseDetector) -> None:
    detector.detect(_load_asset("team_preview.png"))
    detector.detect(_load_asset("battle_text.png"))


def _advance_to_action_selection(detector: PhaseDetector) -> None:
    _advance_to_battle_animation(detector)
    detector.detect(_load_asset("action_selection.png"))


@pytest.fixture
def region_config():
    return load_regions()


@pytest.mark.parametrize(
    ("asset", "expected"),
    [
        ("team_preview.png", BattlePhase.TEAM_PREVIEW),
        ("action_selection.png", BattlePhase.ACTION_SELECTION),
        ("battle_text.png", BattlePhase.BATTLE_ANIMATION),
        ("player_side_ability.png", BattlePhase.BATTLE_ANIMATION),
        ("opponent_side_ability.png", BattlePhase.BATTLE_ANIMATION),
        ("player_side_item.png", BattlePhase.BATTLE_ANIMATION),
        ("opponent_side_item.png", BattlePhase.BATTLE_ANIMATION),
        ("standby.png", BattlePhase.BATTLE_ANIMATION),
        ("battle_end.png", BattlePhase.BATTLE_ANIMATION),
    ],
)
def test_detect_phase_on_reference_screenshots(asset: str, expected: BattlePhase, region_config) -> None:
    """Stateless detect_phase helper (used for signal checks, not strict transitions)."""
    image = _load_asset(asset)
    display_config = config_for_image(region_config, image)
    in_match = expected != BattlePhase.TEAM_PREVIEW
    saw_team_preview = expected == BattlePhase.BATTLE_ANIMATION
    phase = detect_phase(
        image,
        display_config,
        in_match=in_match,
        saw_team_preview=saw_team_preview,
    )
    assert phase == expected


def test_phase_detector_strict_idle_to_team_preview_only(region_config) -> None:
    detector = PhaseDetector(region_config)

    transition = detector.detect_transition(_load_asset("action_selection.png"))
    assert transition.previous == BattlePhase.IDLE
    assert transition.current == BattlePhase.IDLE

    transition = detector.detect_transition(_load_asset("battle_text.png"))
    assert transition.current == BattlePhase.IDLE


def test_phase_detector_stateful_flow(region_config) -> None:
    detector = PhaseDetector(region_config)

    transition = detector.detect_transition(_load_asset("team_preview.png"))
    assert transition.previous == BattlePhase.IDLE
    assert transition.current == BattlePhase.TEAM_PREVIEW

    transition = detector.detect_transition(_load_asset("battle_text.png"))
    assert transition.previous == BattlePhase.TEAM_PREVIEW
    assert transition.current == BattlePhase.BATTLE_ANIMATION

    transition = detector.detect_transition(_load_asset("action_selection.png"))
    assert transition.previous == BattlePhase.BATTLE_ANIMATION
    assert transition.current == BattlePhase.ACTION_SELECTION
    assert transition.entered_action_selection is True

    transition = detector.detect_transition(_load_asset("standby.png"))
    assert transition.previous == BattlePhase.ACTION_SELECTION
    assert transition.current == BattlePhase.BATTLE_ANIMATION

    transition = detector.detect_transition(_load_asset("battle_text.png"))
    assert transition.previous == BattlePhase.BATTLE_ANIMATION
    assert transition.current == BattlePhase.BATTLE_ANIMATION
    assert transition.entered_action_selection is False


def test_phase_detector_latches_action_selection_until_standby(region_config) -> None:
    """Strict rules: ACTION_SELECTION only exits via standby, not missing FIGHT."""
    detector = PhaseDetector(region_config)
    _advance_to_action_selection(detector)
    assert detector.phase == BattlePhase.ACTION_SELECTION

    transition = detector.detect_transition(_load_asset("battle_text.png"))
    assert transition.previous == BattlePhase.ACTION_SELECTION
    assert transition.current == BattlePhase.ACTION_SELECTION


def test_phase_detector_battle_end_returns_to_idle(region_config) -> None:
    detector = PhaseDetector(region_config)
    _advance_to_battle_animation(detector)
    assert detector.phase == BattlePhase.BATTLE_ANIMATION

    transition = detector.detect_transition(_load_asset("battle_end.png"))
    assert transition.previous == BattlePhase.BATTLE_ANIMATION
    assert transition.current == BattlePhase.IDLE


def test_has_battle_ended_on_battle_end_screenshot(region_config) -> None:
    image = _load_asset("battle_end.png")
    display_config = config_for_image(region_config, image)
    assert has_battle_ended(image, display_config) is True


def test_phase_detector_reset(region_config) -> None:
    detector = PhaseDetector(region_config)
    _advance_to_action_selection(detector)
    assert detector.phase == BattlePhase.ACTION_SELECTION

    detector.reset()
    assert detector.phase == BattlePhase.IDLE

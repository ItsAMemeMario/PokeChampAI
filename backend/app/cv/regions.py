"""Load and apply calibrated screen regions from JSON config."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RegionRect = tuple[int, int, int, int]


def repo_root() -> Path:
    """Repository root (parent of backend/)."""
    return Path(__file__).resolve().parents[3]


def default_config_path() -> Path:
    return repo_root() / "config" / "bluestacks_1600x900.json"


def default_assets_dir() -> Path:
    return repo_root() / "assets" / "cv"


@dataclass(frozen=True)
class RegionConfig:
    resolution: tuple[int, int]
    regions: dict[str, RegionRect]

    def get(self, name: str) -> RegionRect:
        if name not in self.regions:
            known = ", ".join(sorted(self.regions))
            raise KeyError(f"Unknown region '{name}'. Known regions: {known}")
        return self.regions[name]

    def names(self) -> list[str]:
        return sorted(self.regions)


def _parse_rect(raw: Any, name: str) -> RegionRect:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"Region '{name}' must be [x, y, w, h], got {raw!r}")
    x, y, w, h = (int(v) for v in raw)
    if w <= 0 or h <= 0:
        raise ValueError(f"Region '{name}' width and height must be positive, got {raw!r}")
    return x, y, w, h


def load_regions(path: Path | str | None = None) -> RegionConfig:
    config_path = Path(path) if path is not None else default_config_path()
    data = json.loads(config_path.read_text(encoding="utf-8"))

    resolution_raw = data.get("resolution")
    if not isinstance(resolution_raw, list) or len(resolution_raw) != 2:
        raise ValueError("Config 'resolution' must be [width, height]")

    regions_raw = data.get("regions")
    if not isinstance(regions_raw, dict) or not regions_raw:
        raise ValueError("Config 'regions' must be a non-empty object")

    regions = {name: _parse_rect(rect, name) for name, rect in regions_raw.items()}
    resolution = (int(resolution_raw[0]), int(resolution_raw[1]))
    return RegionConfig(resolution=resolution, regions=regions)


def save_regions(config: RegionConfig, path: Path | str | None = None) -> Path:
    config_path = Path(path) if path is not None else default_config_path()
    payload = {
        "resolution": list(config.resolution),
        "regions": {name: list(config.regions[name]) for name in sorted(config.regions)},
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path


def crop_region(image: np.ndarray, rect: RegionRect) -> np.ndarray:
    """Crop an RGB or BGR image using a (x, y, w, h) rectangle."""
    x, y, w, h = rect
    height, width = image.shape[:2]
    x2 = min(x + w, width)
    y2 = min(y + h, height)
    x1 = max(x, 0)
    y1 = max(y, 0)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Region {rect} is outside image bounds ({width}x{height})")
    return image[y1:y2, x1:x2].copy()


def assert_resolution(image: np.ndarray, config: RegionConfig) -> None:
    height, width = image.shape[:2]
    expected_w, expected_h = config.resolution
    if width != expected_w or height != expected_h:
        raise ValueError(
            f"Image resolution {width}x{height} does not match config {expected_w}x{expected_h}. "
            "Set BlueStacks to 1600x900 before capture."
        )


def scale_rect(rect: RegionRect, scale_x: float, scale_y: float) -> RegionRect:
    x, y, w, h = rect
    return (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        max(1, int(round(w * scale_x))),
        max(1, int(round(h * scale_y))),
    )


def config_for_image(config: RegionConfig, image: np.ndarray) -> RegionConfig:
    """Return region rects scaled to match the given image when resolutions differ."""
    height, width = image.shape[:2]
    expected_w, expected_h = config.resolution
    if width == expected_w and height == expected_h:
        return config
    scale_x = width / expected_w
    scale_y = height / expected_h
    scaled = {
        name: scale_rect(rect, scale_x, scale_y) for name, rect in config.regions.items()
    }
    return RegionConfig(resolution=(width, height), regions=scaled)

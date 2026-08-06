"""ADB screen capture for BlueStacks (Pokemon Champions)."""

from __future__ import annotations

import logging
import os
import subprocess
from io import BytesIO

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "127.0.0.1:5555"
ADB_TIMEOUT_SEC = 15


def get_adb_device() -> str:
    return os.environ.get("ADB_DEVICE", DEFAULT_DEVICE).strip() or DEFAULT_DEVICE


def _adb_base_cmd(device: str) -> list[str]:
    return ["adb", "-s", device]


def is_adb_connected(device: str | None = None) -> bool:
    """Return True when the configured device appears in `adb devices` as ready."""
    device = device or get_adb_device()
    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=ADB_TIMEOUT_SEC,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("ADB devices check failed: %s", exc)
        return False

    if result.returncode != 0:
        return False

    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == device and parts[1] == "device":
            return True
    return False


def capture_screenshot(device: str | None = None) -> np.ndarray:
    """
    Capture the emulator screen via `adb exec-out screencap -p`.

    Returns an RGB uint8 numpy array with shape (height, width, 3).
    Raises RuntimeError when ADB is unavailable or capture fails.
    """
    logger.info("Beginning ADB screencap")
    device = device or get_adb_device()
    cmd = [*_adb_base_cmd(device), "exec-out", "screencap", "-p"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=ADB_TIMEOUT_SEC,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "adb executable not found. Install Android platform-tools and ensure adb is on PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ADB screencap timed out after {ADB_TIMEOUT_SEC}s") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ADB screencap failed (exit {result.returncode}): {stderr}")

    if not result.stdout:
        raise RuntimeError("ADB screencap returned empty output")

    try:
        image = Image.open(BytesIO(result.stdout))
        rgb = image.convert("RGB")
    except Exception as exc:
        raise RuntimeError("Failed to decode ADB screencap PNG") from exc

    logger.info("Captured screenshot")
    return np.asarray(rgb, dtype=np.uint8)

"""ADB screen capture for BlueStacks (Pokemon Champions)."""

from __future__ import annotations

import logging
import os
import struct
import subprocess

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "127.0.0.1:5555"
ADB_TIMEOUT_SEC = 15

# Android PixelFormat values commonly emitted by `screencap` (no -p).
_PIXEL_FORMAT_RGBA_8888 = 1
_PIXEL_FORMAT_RGBX_8888 = 2
_PIXEL_FORMAT_BGRA_8888 = 5
_SUPPORTED_4BPP_FORMATS = {
    _PIXEL_FORMAT_RGBA_8888,
    _PIXEL_FORMAT_RGBX_8888,
    _PIXEL_FORMAT_BGRA_8888,
}


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


def parse_raw_screencap(data: bytes) -> np.ndarray:
    """
    Parse Android raw `screencap` output into an RGB uint8 array.

    Header is little-endian ``width``, ``height``, ``format`` (12 bytes).
    Newer Android builds append a 4-byte color-space field (16-byte header).
    Pixel payload is RGBA/RGBX/BGRA (4 bytes per pixel); alpha/X is dropped.
    When row stride exceeds width, only the active width columns are kept.
    """
    if len(data) < 12:
        raise RuntimeError(
            f"ADB screencap output too short for header ({len(data)} bytes)"
        )

    width, height, pixel_format = struct.unpack_from("<III", data, 0)
    if width == 0 or height == 0:
        raise RuntimeError(f"Invalid screencap dimensions: {width}x{height}")

    if pixel_format not in _SUPPORTED_4BPP_FORMATS:
        raise RuntimeError(
            f"Unsupported screencap pixel format {pixel_format} "
            f"(expected RGBA/RGBX/BGRA 8888)"
        )

    bytes_per_pixel = 4
    expected_tight = width * height * bytes_per_pixel

    header_size: int | None = None
    payload: bytes | None = None
    for candidate in (12, 16):
        if len(data) < candidate:
            continue
        body = data[candidate:]
        if len(body) < expected_tight:
            continue
        if len(body) == expected_tight:
            header_size = candidate
            payload = body
            break
        # Strided buffer: row bytes * height == payload length, stride >= width.
        if height > 0 and len(body) % height == 0:
            stride_bytes = len(body) // height
            if stride_bytes >= width * bytes_per_pixel and stride_bytes % bytes_per_pixel == 0:
                header_size = candidate
                payload = body
                break

    if header_size is None or payload is None:
        raise RuntimeError(
            f"ADB screencap payload size mismatch: got {len(data)} bytes for "
            f"{width}x{height} format={pixel_format} (expected {expected_tight} "
            f"pixels after 12- or 16-byte header)"
        )

    if len(payload) == expected_tight:
        rgba = np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 4))
    else:
        stride_bytes = len(payload) // height
        rows = np.frombuffer(payload, dtype=np.uint8).reshape((height, stride_bytes))
        rgba = rows[:, : width * bytes_per_pixel].reshape((height, width, 4))

    if pixel_format == _PIXEL_FORMAT_BGRA_8888:
        rgb = rgba[:, :, [2, 1, 0]]
    else:
        rgb = rgba[:, :, :3]

    # frombuffer views are read-only; callers may mutate the frame.
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def capture_screenshot(device: str | None = None) -> np.ndarray:
    """
    Capture the emulator screen via `adb exec-out screencap` (raw RGBA dump).

    Returns an RGB uint8 numpy array with shape (height, width, 3).
    Raises RuntimeError when ADB is unavailable or capture fails.
    """
    logger.info("Beginning ADB screencap")
    device = device or get_adb_device()
    cmd = [*_adb_base_cmd(device), "exec-out", "screencap"]
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
        rgb = parse_raw_screencap(result.stdout)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Failed to decode ADB raw screencap") from exc

    logger.info("Captured screenshot")
    return rgb

"""Unit tests for raw ADB screencap parsing."""

from __future__ import annotations

import struct
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.cv.adb_capture import capture_screenshot, parse_raw_screencap


def _rgba_frame(width: int, height: int) -> np.ndarray:
    """Deterministic RGBA test pattern (R=x%256, G=y%256, B=3, A=255)."""
    ys, xs = np.mgrid[0:height, 0:width]
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = (xs % 256).astype(np.uint8)
    rgba[:, :, 1] = (ys % 256).astype(np.uint8)
    rgba[:, :, 2] = 3
    rgba[:, :, 3] = 255
    return rgba


def _pack_raw(
    rgba: np.ndarray,
    *,
    pixel_format: int = 1,
    header_size: int = 12,
    stride_width: int | None = None,
) -> bytes:
    height, width = rgba.shape[:2]
    header = struct.pack("<III", width, height, pixel_format)
    if header_size == 16:
        header += struct.pack("<I", 0)  # color space (ignored)
    elif header_size != 12:
        raise ValueError(f"unsupported header_size {header_size}")

    if stride_width is None or stride_width == width:
        return header + rgba.tobytes()

    if stride_width < width:
        raise ValueError("stride_width must be >= width")
    row_pad = (stride_width - width) * 4
    rows = []
    for y in range(height):
        rows.append(rgba[y].tobytes())
        rows.append(b"\x00" * row_pad)
    return header + b"".join(rows)


def test_parse_raw_screencap_12_byte_header() -> None:
    rgba = _rgba_frame(4, 3)
    raw = _pack_raw(rgba, header_size=12)
    rgb = parse_raw_screencap(raw)
    assert rgb.shape == (3, 4, 3)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb, rgba[:, :, :3])


def test_parse_raw_screencap_16_byte_header() -> None:
    rgba = _rgba_frame(5, 2)
    raw = _pack_raw(rgba, header_size=16)
    rgb = parse_raw_screencap(raw)
    assert rgb.shape == (2, 5, 3)
    np.testing.assert_array_equal(rgb, rgba[:, :, :3])


def test_parse_raw_screencap_with_row_stride() -> None:
    rgba = _rgba_frame(3, 2)
    raw = _pack_raw(rgba, header_size=12, stride_width=4)
    rgb = parse_raw_screencap(raw)
    assert rgb.shape == (2, 3, 3)
    np.testing.assert_array_equal(rgb, rgba[:, :, :3])


def test_parse_raw_screencap_bgra_swaps_channels() -> None:
    height, width = 2, 2
    bgra = np.zeros((height, width, 4), dtype=np.uint8)
    bgra[:, :] = (10, 20, 30, 255)  # B, G, R, A
    header = struct.pack("<III", width, height, 5)  # BGRA_8888
    rgb = parse_raw_screencap(header + bgra.tobytes())
    expected = np.full((height, width, 3), (30, 20, 10), dtype=np.uint8)
    np.testing.assert_array_equal(rgb, expected)


def test_parse_raw_screencap_rejects_short_buffer() -> None:
    with pytest.raises(RuntimeError, match="too short"):
        parse_raw_screencap(b"\x00" * 8)


def test_parse_raw_screencap_rejects_unsupported_format() -> None:
    header = struct.pack("<III", 2, 2, 99)
    payload = b"\x00" * (2 * 2 * 4)
    with pytest.raises(RuntimeError, match="Unsupported screencap pixel format"):
        parse_raw_screencap(header + payload)


def test_parse_raw_screencap_rejects_size_mismatch() -> None:
    header = struct.pack("<III", 10, 10, 1)
    with pytest.raises(RuntimeError, match="payload size mismatch"):
        parse_raw_screencap(header + b"\x00" * 16)


def test_capture_screenshot_uses_raw_screencap(monkeypatch: pytest.MonkeyPatch) -> None:
    rgba = _rgba_frame(2, 2)
    raw = _pack_raw(rgba, header_size=12)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = raw
    mock_result.stderr = b""

    def fake_run(cmd, **_kwargs):
        assert cmd[-2:] == ["exec-out", "screencap"]
        assert "-p" not in cmd
        return mock_result

    monkeypatch.setattr("app.cv.adb_capture.subprocess.run", fake_run)
    rgb = capture_screenshot(device="127.0.0.1:5555")
    np.testing.assert_array_equal(rgb, rgba[:, :, :3])


def test_capture_screenshot_propagates_parse_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = b"\x00" * 8
    mock_result.stderr = b""
    monkeypatch.setattr(
        "app.cv.adb_capture.subprocess.run",
        lambda *_args, **_kwargs: mock_result,
    )
    with pytest.raises(RuntimeError, match="too short"):
        capture_screenshot(device="127.0.0.1:5555")

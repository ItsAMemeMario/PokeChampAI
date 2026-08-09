"""Unit tests for shared CUDA EasyOCR reader."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.cv import ocr_reader


@pytest.fixture(autouse=True)
def _reset_reader() -> None:
    ocr_reader.reset_ocr_reader()
    yield
    ocr_reader.reset_ocr_reader()


def test_get_ocr_reader_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_reader.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        ocr_reader.get_ocr_reader()


def test_get_ocr_reader_caches_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_reader = MagicMock()
    constructed: list[object] = []

    def fake_reader(*_args, **_kwargs):
        constructed.append(object())
        return mock_reader

    monkeypatch.setattr(ocr_reader.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ocr_reader.torch.cuda, "get_device_name", lambda _idx: "Fake GPU")
    monkeypatch.setattr(ocr_reader.easyocr, "Reader", fake_reader)

    assert ocr_reader.get_ocr_reader() is mock_reader
    assert ocr_reader.get_ocr_reader() is mock_reader
    assert len(constructed) == 1


def test_read_text_uses_lock_and_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = ["Hello", "World"]
    monkeypatch.setattr(ocr_reader, "get_ocr_reader", lambda: mock_reader)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    lines = ocr_reader.read_text(image, detail=0, paragraph=True)
    assert lines == ["Hello", "World"]
    mock_reader.readtext.assert_called_once()
    assert mock_reader.readtext.call_args.kwargs["detail"] == 0
    assert mock_reader.readtext.call_args.kwargs["paragraph"] is True


def test_map_parallel_preserves_order() -> None:
    assert ocr_reader.map_parallel(lambda x: x * 2, []) == []
    assert ocr_reader.map_parallel(lambda x: x * 2, [3]) == [6]
    assert ocr_reader.map_parallel(lambda x: x * 2, [1, 2, 3, 4]) == [2, 4, 6, 8]

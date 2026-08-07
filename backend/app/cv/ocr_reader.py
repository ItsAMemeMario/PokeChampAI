"""Shared EasyOCR reader (CUDA required)."""

from __future__ import annotations

import logging
import threading
from typing import Any

import easyocr
import numpy as np
import torch

logger = logging.getLogger(__name__)

_reader: easyocr.Reader | None = None
_reader_lock = threading.Lock()
_infer_lock = threading.Lock()


def get_ocr_reader() -> easyocr.Reader:
    """
    Return the process-wide EasyOCR reader (GPU only).

    Raises RuntimeError if CUDA is unavailable. There is no CPU fallback.
    """
    global _reader
    if _reader is not None:
        return _reader

    with _reader_lock:
        if _reader is not None:
            return _reader
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for EasyOCR. Install a CUDA-enabled PyTorch build "
                "matching your NVIDIA driver (stock `pip install torch` is often CPU-only)."
            )
        logger.info(
            "Initializing EasyOCR on CUDA device %s",
            torch.cuda.get_device_name(0),
        )
        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        return _reader


def read_text(
    image: np.ndarray,
    *,
    detail: int = 0,
    paragraph: bool = True,
    **kwargs: Any,
) -> list[str]:
    """Run EasyOCR ``readtext`` under a lock (single shared GPU model)."""
    reader = get_ocr_reader()
    with _infer_lock:
        lines = reader.readtext(image, detail=detail, paragraph=paragraph, **kwargs)
    if not lines:
        return []
    if detail == 0:
        return [str(line) for line in lines]
    return lines  # type: ignore[return-value]


def reset_ocr_reader() -> None:
    """Drop the cached reader (tests / forced re-init)."""
    global _reader
    with _reader_lock:
        with _infer_lock:
            _reader = None

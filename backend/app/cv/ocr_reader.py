"""Shared EasyOCR reader (CUDA required)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import easyocr
import numpy as np
import torch

logger = logging.getLogger(__name__)

_reader: easyocr.Reader | None = None
_reader_lock = threading.Lock()
_infer_lock = threading.Lock()
_pool_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None

_T = TypeVar("_T")
_R = TypeVar("_R")


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


def map_parallel(
    fn: Callable[[_T], _R],
    items: Iterable[_T],
    *,
    max_workers: int = 4,
) -> list[_R]:
    """
    Run ``fn`` over ``items`` in a shared thread pool.

    Single-item lists run inline (no pool hop). OCR inference still serializes
    on the shared GPU lock; preprocess work overlaps across workers.
    """
    materialised = list(items)
    if not materialised:
        return []
    if len(materialised) == 1:
        return [fn(materialised[0])]
    pool = _ocr_pool(max_workers=max_workers)
    return list(pool.map(fn, materialised))


def _ocr_pool(*, max_workers: int) -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="ocr",
            )
        return _pool


def reset_ocr_reader() -> None:
    """Drop the cached reader (tests / forced re-init)."""
    global _reader
    with _reader_lock:
        with _infer_lock:
            _reader = None

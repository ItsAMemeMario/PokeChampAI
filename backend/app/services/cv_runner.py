"""Background CV task lifecycle. Replaced by cv.pipeline in a later milestone."""

from __future__ import annotations

import asyncio
import logging

from app.services.session import SessionStore

logger = logging.getLogger(__name__)

_cv_task: asyncio.Task[None] | None = None


async def _cv_loop(store: SessionStore) -> None:
    logger.info("CV loop started")
    try:
        while store.cv_running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("CV loop stopped")
        raise


def start_cv(store: SessionStore) -> None:
    global _cv_task
    if _cv_task is not None and not _cv_task.done():
        return
    _cv_task = asyncio.create_task(_cv_loop(store))


async def stop_cv() -> None:
    global _cv_task
    if _cv_task is None:
        return
    _cv_task.cancel()
    try:
        await _cv_task
    except asyncio.CancelledError:
        pass
    _cv_task = None


async def shutdown_cv() -> None:
    """Cancel the CV task during application shutdown."""
    await stop_cv()

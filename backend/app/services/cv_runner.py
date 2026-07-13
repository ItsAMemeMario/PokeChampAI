"""Background CV task lifecycle. Replaced by cv.pipeline in a later milestone."""

from __future__ import annotations

import asyncio
import logging

from app.cv.adb_capture import is_adb_connected
from app.services.session import SessionStore

logger = logging.getLogger(__name__)

_cv_task: asyncio.Task[None] | None = None
_ADB_PROBE_INTERVAL_SEC = 5.0


async def _cv_loop(store: SessionStore) -> None:
    logger.info("CV loop started")
    try:
        while store.cv_running:
            store.adb_connected = await asyncio.to_thread(is_adb_connected)
            if not store.adb_connected:
                logger.debug("ADB not connected; retrying in %.0fs", _ADB_PROBE_INTERVAL_SEC)
            await asyncio.sleep(_ADB_PROBE_INTERVAL_SEC)
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

"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logger = logging.getLogger(__name__)

_cv_task: asyncio.Task | None = None


async def _cv_loop() -> None:
    """Background CV polling loop. Replaced by cv.pipeline in a later milestone."""
    logger.info("CV loop started")
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.info("CV loop stopped")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _cv_task
    logger.info("Starting application")
    _cv_task = asyncio.create_task(_cv_loop())
    yield
    if _cv_task is not None:
        _cv_task.cancel()
        try:
            await _cv_task
        except asyncio.CancelledError:
            pass
        _cv_task = None
    logger.info("Application shutdown complete")


app = FastAPI(
    title="PokeChamp AI",
    description="Pokemon Champions doubles battle assistant",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

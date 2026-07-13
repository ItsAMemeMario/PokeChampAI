"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import session_router, suggestions_router, team_router
from app.services.cv_runner import shutdown_cv

load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting application")
    yield
    await shutdown_cv()
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

app.include_router(team_router)
app.include_router(session_router)
app.include_router(suggestions_router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

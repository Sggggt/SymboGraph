from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from app.core.config import get_settings


_MODEL_SEMAPHORE: asyncio.Semaphore | None = None
_MODEL_SEMAPHORE_LIMIT: int | None = None
_MODEL_SEMAPHORE_LOOP: asyncio.AbstractEventLoop | None = None


def _model_request_limit() -> int:
    return max(1, int(get_settings().model_request_concurrency or 1))


def model_request_semaphore() -> asyncio.Semaphore:
    global _MODEL_SEMAPHORE, _MODEL_SEMAPHORE_LIMIT, _MODEL_SEMAPHORE_LOOP
    loop = asyncio.get_running_loop()
    limit = _model_request_limit()
    if _MODEL_SEMAPHORE is None or _MODEL_SEMAPHORE_LIMIT != limit or _MODEL_SEMAPHORE_LOOP is not loop:
        _MODEL_SEMAPHORE = asyncio.Semaphore(limit)
        _MODEL_SEMAPHORE_LIMIT = limit
        _MODEL_SEMAPHORE_LOOP = loop
    return _MODEL_SEMAPHORE


@asynccontextmanager
async def model_request_slot() -> AsyncIterator[None]:
    semaphore = model_request_semaphore()
    async with semaphore:
        yield

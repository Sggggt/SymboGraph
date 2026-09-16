"""One future source preparation; durable file mutations remain sequential."""
from __future__ import annotations

import asyncio
import os
import threading
import time
from contextvars import ContextVar

_CURRENT = ContextVar("source_parse_pipeline", default=None)
_STOP = ContextVar("source_parse_stop", default=None)
_PARSE_LOCK = threading.Lock()


def _after_fork():
    global _PARSE_LOCK
    _PARSE_LOCK = threading.Lock()


if hasattr(os,"register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def check_parse_cancellation():
    stop = _STOP.get()
    if stop is not None and stop.is_set():
        from app.services.cancellation import IngestionCancelled
        raise IngestionCancelled("Source preparation stopped")


def serialized_parse(parser, snapshot):
    # Native PDF parsing cannot run concurrently in the same process. The
    # thread lock remains held even if its awaiting coroutine is cancelled.
    with _PARSE_LOCK:
        check_parse_cancellation()
        result = parser(snapshot)
        check_parse_cancellation()
        return result


def preparation_started(args, kwargs):
    pipeline = _CURRENT.get()
    path = args[1] if len(args)>1 else kwargs.get("source_path")
    if pipeline and pipeline.pending and pipeline.pending[0] == path:
        return pipeline.pending[2]
    return None


async def prepared_source(path, factory):
    pipeline = _CURRENT.get()
    return await pipeline.take(path, factory) if pipeline else await factory()


class SourcePreparationPipeline:
    def __init__(self, paths, prepare):
        self.next_path = {a:b for a,b in zip(paths,paths[1:])}
        self.prepare = prepare
        self.pending = None

    async def __aenter__(self):
        self.token = _CURRENT.set(self)
        return self

    async def _prepare(self,path,stop):
        token = _STOP.set(stop)
        try:
            check_parse_cancellation()
            return await self.prepare(path)
        finally:
            _STOP.reset(token)

    async def take(self,path,factory):
        if self.pending:
            if self.pending[0] != path:
                raise RuntimeError("Source preparation order changed")
            pending = self.pending
            self.pending = None
            # Shield keeps ownership of the parsing thread until cleanup.
            try:
                result = await asyncio.shield(pending[1])
            except asyncio.CancelledError:
                self.pending = pending
                raise
        else:
            result = await factory()
        next_path = self.next_path.get(path)
        if next_path is not None:
            stop = threading.Event()
            self.pending = (next_path,asyncio.create_task(self._prepare(next_path,stop)),time.perf_counter(),stop)
        return result

    async def discard(self,path):
        if self.pending and self.pending[0] == path:
            pending = self.pending
            self.pending = None
            pending[3].set()
            try:
                await asyncio.shield(pending[1])
            except Exception:
                pass  # The failed current file already owns its error audit.

    async def __aexit__(self,exc_type,exc,tb):
        try:
            if self.pending:
                pending = self.pending
                self.pending = None
                pending[3].set()
                try:
                    await asyncio.shield(pending[1])
                except Exception:
                    # A prefetched error belongs to its file when consumed;
                    # after an earlier batch failure it cannot replace that
                    # original error or create a document/version mutation.
                    if exc_type is None:
                        raise
        finally:
            _CURRENT.reset(self.token)

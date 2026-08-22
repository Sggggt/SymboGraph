from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Iterator, Protocol, TypeVar

from app.core.config import get_settings


logger = logging.getLogger(__name__)
T = TypeVar("T")
ADMISSION_KEY_PREFIX = "symbograph:agent_admission:v1"


@dataclass(frozen=True)
class AgentAdmissionConfig:
    active_limit: int
    queue_limit: int
    queue_timeout_seconds: int
    lease_ttl_seconds: int

    @classmethod
    def from_settings(cls) -> "AgentAdmissionConfig":
        settings = get_settings()
        return cls(
            active_limit=settings.agent_request_concurrency,
            queue_limit=settings.agent_request_queue_limit,
            queue_timeout_seconds=settings.agent_request_queue_timeout_seconds,
            lease_ttl_seconds=settings.agent_request_lease_ttl_seconds,
        )


@dataclass(frozen=True)
class AdmissionResult:
    state: str
    active_count: int = 0
    queue_count: int = 0
    position: int | None = None


class AgentAdmissionError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        status_code: int,
        message: str,
        config: AgentAdmissionConfig,
        request_kind: str,
        active_count: int | None = None,
        queue_count: int | None = None,
        position: int | None = None,
        retry_after_seconds: int = 1,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message
        self.config = config
        self.request_kind = request_kind
        self.active_count = active_count
        self.queue_count = queue_count
        self.position = position
        self.retry_after_seconds = retry_after_seconds

    def payload(self) -> dict:
        fix_commands = [
            "Verify the API and worker containers can reach REDIS_URL.",
            "Retry after in-flight Agent requests finish.",
            "Adjust AGENT_REQUEST_CONCURRENCY or AGENT_REQUEST_QUEUE_LIMIT only after checking database and model capacity.",
        ]
        return {
            "code": self.code,
            "title": "Agent request admission failed",
            "message": self.message,
            "issues": [
                {
                    "code": self.code,
                    "title": "Agent request admission failed",
                    "message": self.message,
                    "fix_commands": fix_commands,
                }
            ],
            "retryable": True,
            "retry_after_seconds": self.retry_after_seconds,
            "diagnostics": {
                "backend": "redis",
                "request_kind": self.request_kind,
                "active_limit": self.config.active_limit,
                "queue_limit": self.config.queue_limit,
                "queue_timeout_seconds": self.config.queue_timeout_seconds,
                "lease_ttl_seconds": self.config.lease_ttl_seconds,
                "active_count": self.active_count,
                "queue_count": self.queue_count,
                "queue_position": self.position,
            },
            "fix_commands": fix_commands,
        }


class AgentAdmissionAdapter(Protocol):
    backend_name: str

    async def register(self, token: str, config: AgentAdmissionConfig, *, now_ms: int) -> AdmissionResult: ...

    async def poll(self, token: str, config: AgentAdmissionConfig, *, now_ms: int) -> AdmissionResult: ...

    async def heartbeat(self, token: str, config: AgentAdmissionConfig, *, now_ms: int) -> AdmissionResult: ...

    async def release(self, token: str) -> None: ...

    async def close(self) -> None: ...


class RedisAgentAdmissionAdapter:
    backend_name = "redis"

    _REGISTER_SCRIPT = """
local server_time = redis.call('TIME')
local now = server_time[1] * 1000 + math.floor(server_time[2] / 1000)
local expired = redis.call('ZRANGEBYSCORE', KEYS[3], '-inf', now)
for _, token in ipairs(expired) do
  redis.call('ZREM', KEYS[2], token)
  redis.call('ZREM', KEYS[3], token)
end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
local active_score = redis.call('ZSCORE', KEYS[1], ARGV[1])
if active_score then
  redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[1])
  return {'active', redis.call('ZCARD', KEYS[1]), redis.call('ZCARD', KEYS[2]), 0}
end
local queued_score = redis.call('ZSCORE', KEYS[2], ARGV[1])
if queued_score then
  local rank = redis.call('ZRANK', KEYS[2], ARGV[1])
  return {'queued', redis.call('ZCARD', KEYS[1]), redis.call('ZCARD', KEYS[2]), rank + 1}
end
local active_count = redis.call('ZCARD', KEYS[1])
local queue_count = redis.call('ZCARD', KEYS[2])
if active_count < tonumber(ARGV[4]) and queue_count == 0 then
  redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[1])
  return {'active', active_count + 1, queue_count, 0}
end
if tonumber(ARGV[5]) <= 0 or queue_count >= tonumber(ARGV[5]) then
  return {'rejected', active_count, queue_count, -1}
end
local sequence = redis.call('INCR', KEYS[4])
redis.call('ZADD', KEYS[2], sequence, ARGV[1])
redis.call('ZADD', KEYS[3], now + tonumber(ARGV[3]), ARGV[1])
return {'queued', active_count, queue_count + 1, queue_count + 1}
"""

    _POLL_SCRIPT = """
local server_time = redis.call('TIME')
local now = server_time[1] * 1000 + math.floor(server_time[2] / 1000)
local expired = redis.call('ZRANGEBYSCORE', KEYS[3], '-inf', now)
for _, token in ipairs(expired) do
  redis.call('ZREM', KEYS[2], token)
  redis.call('ZREM', KEYS[3], token)
end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZSCORE', KEYS[1], ARGV[1]) then
  redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[1])
  return {'active', redis.call('ZCARD', KEYS[1]), redis.call('ZCARD', KEYS[2]), 0}
end
local rank = redis.call('ZRANK', KEYS[2], ARGV[1])
if not rank then
  return {'timed_out', redis.call('ZCARD', KEYS[1]), redis.call('ZCARD', KEYS[2]), -1}
end
local active_count = redis.call('ZCARD', KEYS[1])
if rank == 0 and active_count < tonumber(ARGV[3]) then
  redis.call('ZREM', KEYS[2], ARGV[1])
  redis.call('ZREM', KEYS[3], ARGV[1])
  redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[1])
  return {'active', active_count + 1, redis.call('ZCARD', KEYS[2]), 0}
end
return {'queued', active_count, redis.call('ZCARD', KEYS[2]), rank + 1}
"""

    _HEARTBEAT_SCRIPT = """
local server_time = redis.call('TIME')
local now = server_time[1] * 1000 + math.floor(server_time[2] / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if not redis.call('ZSCORE', KEYS[1], ARGV[1]) then
  return {'lost', redis.call('ZCARD', KEYS[1]), redis.call('ZCARD', KEYS[2]), -1}
end
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[1])
return {'active', redis.call('ZCARD', KEYS[1]), redis.call('ZCARD', KEYS[2]), 0}
"""

    _RELEASE_SCRIPT = """
redis.call('ZREM', KEYS[1], ARGV[1])
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('ZREM', KEYS[3], ARGV[1])
return 1
"""

    def __init__(self, redis_url: str, *, key_prefix: str = ADMISSION_KEY_PREFIX, client=None) -> None:
        if client is None:
            import redis.asyncio as redis

            client = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        self._client = client
        self._keys = [
            f"{key_prefix}:active",
            f"{key_prefix}:queue",
            f"{key_prefix}:deadlines",
            f"{key_prefix}:sequence",
        ]

    @staticmethod
    def _parse(raw) -> AdmissionResult:
        state = str(raw[0])
        if state not in {"active", "queued", "rejected", "timed_out", "lost"}:
            raise RuntimeError("agent_admission_invalid_redis_response")
        return AdmissionResult(
            state=state,
            active_count=int(raw[1]),
            queue_count=int(raw[2]),
            position=None if int(raw[3]) < 0 else int(raw[3]),
        )

    async def _eval(self, script: str, keys: list[str], *args) -> AdmissionResult:
        try:
            raw = await self._client.eval(script, len(keys), *keys, *args)
        except Exception as exc:
            logger.warning("Agent admission Redis operation failed (%s)", type(exc).__name__)
            raise RuntimeError("agent_admission_redis_unavailable") from exc
        return self._parse(raw)

    async def register(self, token: str, config: AgentAdmissionConfig, *, now_ms: int) -> AdmissionResult:
        return await self._eval(
            self._REGISTER_SCRIPT,
            self._keys,
            token,
            config.lease_ttl_seconds * 1000,
            config.queue_timeout_seconds * 1000,
            config.active_limit,
            config.queue_limit,
        )

    async def poll(self, token: str, config: AgentAdmissionConfig, *, now_ms: int) -> AdmissionResult:
        return await self._eval(
            self._POLL_SCRIPT,
            self._keys[:3],
            token,
            config.lease_ttl_seconds * 1000,
            config.active_limit,
        )

    async def heartbeat(self, token: str, config: AgentAdmissionConfig, *, now_ms: int) -> AdmissionResult:
        return await self._eval(
            self._HEARTBEAT_SCRIPT,
            self._keys[:2],
            token,
            config.lease_ttl_seconds * 1000,
        )

    async def release(self, token: str) -> None:
        try:
            await self._client.eval(self._RELEASE_SCRIPT, 3, *self._keys[:3], token)
        except Exception as exc:
            logger.warning("Agent admission Redis release failed (%s)", type(exc).__name__)
            raise RuntimeError("agent_admission_redis_unavailable") from exc

    async def close(self) -> None:
        close = getattr(self._client, "aclose", None)
        if close is not None:
            await close()


class LocalAgentAdmissionAdapter:
    """Explicit unit-test adapter; production never selects this automatically."""

    backend_name = "local_test"

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active: dict[str, int] = {}
        self._queue: list[tuple[int, str]] = []
        self._deadlines: dict[str, int] = {}
        self._sequence = 0

    def _cleanup(self, now_ms: int) -> None:
        self._active = {token: expiry for token, expiry in self._active.items() if expiry > now_ms}
        expired = {token for token, expiry in self._deadlines.items() if expiry <= now_ms}
        if expired:
            self._queue = [(sequence, token) for sequence, token in self._queue if token not in expired]
            for token in expired:
                self._deadlines.pop(token, None)

    def _result(self, state: str, token: str, position: int | None = None) -> AdmissionResult:
        if position is None and state == "queued":
            position = next((index + 1 for index, (_, queued) in enumerate(self._queue) if queued == token), None)
        return AdmissionResult(state, len(self._active), len(self._queue), position)

    async def register(self, token: str, config: AgentAdmissionConfig, *, now_ms: int) -> AdmissionResult:
        async with self._lock:
            self._cleanup(now_ms)
            if token in self._active:
                self._active[token] = now_ms + config.lease_ttl_seconds * 1000
                return self._result("active", token, 0)
            if any(queued == token for _, queued in self._queue):
                return self._result("queued", token)
            if len(self._active) < config.active_limit and not self._queue:
                self._active[token] = now_ms + config.lease_ttl_seconds * 1000
                return self._result("active", token, 0)
            if config.queue_limit <= 0 or len(self._queue) >= config.queue_limit:
                return self._result("rejected", token)
            self._sequence += 1
            self._queue.append((self._sequence, token))
            self._deadlines[token] = now_ms + config.queue_timeout_seconds * 1000
            return self._result("queued", token)

    async def poll(self, token: str, config: AgentAdmissionConfig, *, now_ms: int) -> AdmissionResult:
        async with self._lock:
            self._cleanup(now_ms)
            if token in self._active:
                self._active[token] = now_ms + config.lease_ttl_seconds * 1000
                return self._result("active", token, 0)
            position = next((index for index, (_, queued) in enumerate(self._queue) if queued == token), None)
            if position is None:
                return self._result("timed_out", token)
            if position == 0 and len(self._active) < config.active_limit:
                self._queue.pop(0)
                self._deadlines.pop(token, None)
                self._active[token] = now_ms + config.lease_ttl_seconds * 1000
                return self._result("active", token, 0)
            return self._result("queued", token, position + 1)

    async def heartbeat(self, token: str, config: AgentAdmissionConfig, *, now_ms: int) -> AdmissionResult:
        async with self._lock:
            self._cleanup(now_ms)
            if token not in self._active:
                return self._result("lost", token)
            self._active[token] = now_ms + config.lease_ttl_seconds * 1000
            return self._result("active", token, 0)

    async def release(self, token: str) -> None:
        async with self._lock:
            self._active.pop(token, None)
            self._queue = [(sequence, queued) for sequence, queued in self._queue if queued != token]
            self._deadlines.pop(token, None)

    async def close(self) -> None:
        return None

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            self._cleanup(int(time.time() * 1000))
            return {"active": len(self._active), "queued": len(self._queue)}


_ADAPTER_OVERRIDE: ContextVar[AgentAdmissionAdapter | None] = ContextVar("agent_admission_adapter", default=None)


@contextmanager
def use_agent_admission_adapter(adapter: AgentAdmissionAdapter) -> Iterator[AgentAdmissionAdapter]:
    token = _ADAPTER_OVERRIDE.set(adapter)
    try:
        yield adapter
    finally:
        _ADAPTER_OVERRIDE.reset(token)


def _admission_error(
    code: str,
    config: AgentAdmissionConfig,
    request_kind: str,
    result: AdmissionResult | None = None,
) -> AgentAdmissionError:
    if code == "agent_admission_queue_full":
        status_code = 429
        message = "The bounded Agent request queue is full; retry after an in-flight request finishes."
    elif code == "agent_admission_queue_timeout":
        status_code = 429
        message = "The Agent request exceeded its bounded queue wait timeout and was removed from the queue."
    elif code == "agent_admission_lease_lost":
        status_code = 503
        message = "The global Agent admission lease was lost, so the request was stopped to preserve the resource budget."
    else:
        status_code = 503
        message = "Redis admission coordination is unavailable; the Agent request was not run and no local fallback was used."
    return AgentAdmissionError(
        code=code,
        status_code=status_code,
        message=message,
        config=config,
        request_kind=request_kind,
        active_count=result.active_count if result else None,
        queue_count=result.queue_count if result else None,
        position=result.position if result else None,
        retry_after_seconds=max(1, min(30, config.queue_timeout_seconds)),
    )


class AgentAdmissionLease:
    def __init__(self, token: str, adapter: AgentAdmissionAdapter, config: AgentAdmissionConfig, request_kind: str) -> None:
        self.token = token
        self.adapter = adapter
        self.config = config
        self.request_kind = request_kind
        self._released = False
        self._release_lock = asyncio.Lock()
        self._lost_event = asyncio.Event()
        self._lost_error: AgentAdmissionError | None = None
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        interval = max(0.25, min(2.0, self.config.lease_ttl_seconds / 3))
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    result = await self.adapter.heartbeat(self.token, self.config, now_ms=int(time.time() * 1000))
                except Exception:
                    self._lost_error = _admission_error("agent_admission_unavailable", self.config, self.request_kind)
                    self._lost_event.set()
                    return
                if result.state != "active":
                    self._lost_error = _admission_error("agent_admission_lease_lost", self.config, self.request_kind, result)
                    self._lost_event.set()
                    return
        except asyncio.CancelledError:
            raise

    def raise_if_lost(self) -> None:
        if self._lost_event.is_set():
            raise self._lost_error or _admission_error("agent_admission_lease_lost", self.config, self.request_kind)

    async def run(self, awaitable: Awaitable[T]) -> T:
        try:
            self.raise_if_lost()
        except BaseException:
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise
        work = asyncio.create_task(awaitable)
        lost = asyncio.create_task(self._lost_event.wait())
        try:
            done, _ = await asyncio.wait({work, lost}, return_when=asyncio.FIRST_COMPLETED)
            if lost in done and self._lost_event.is_set():
                if not work.done():
                    work.cancel()
                    with suppress(asyncio.CancelledError):
                        await work
                self.raise_if_lost()
            lost.cancel()
            with suppress(asyncio.CancelledError):
                await lost
            return await work
        except asyncio.CancelledError:
            if not work.done():
                work.cancel()
                with suppress(asyncio.CancelledError):
                    await work
            raise
        finally:
            if not lost.done():
                lost.cancel()

    async def release(self) -> None:
        async with self._release_lock:
            if self._released:
                return
            self._released = True
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
            try:
                try:
                    await self.adapter.release(self.token)
                finally:
                    await self.adapter.close()
            except Exception as exc:
                raise _admission_error("agent_admission_unavailable", self.config, self.request_kind) from exc


async def acquire_agent_request_slot(
    request_kind: str,
    *,
    adapter: AgentAdmissionAdapter | None = None,
    config: AgentAdmissionConfig | None = None,
) -> AgentAdmissionLease:
    config = config or AgentAdmissionConfig.from_settings()
    adapter = adapter or _ADAPTER_OVERRIDE.get()
    if adapter is None:
        try:
            adapter = RedisAgentAdmissionAdapter(get_settings().redis_url)
        except Exception as exc:
            raise _admission_error("agent_admission_unavailable", config, request_kind) from exc
    token = uuid.uuid4().hex
    start = time.monotonic()
    last_result: AdmissionResult | None = None
    try:
        try:
            last_result = await adapter.register(token, config, now_ms=int(time.time() * 1000))
        except Exception as exc:
            raise _admission_error("agent_admission_unavailable", config, request_kind) from exc
        if last_result.state == "rejected":
            raise _admission_error("agent_admission_queue_full", config, request_kind, last_result)
        while last_result.state != "active":
            remaining = config.queue_timeout_seconds - (time.monotonic() - start)
            if remaining <= 0:
                raise _admission_error("agent_admission_queue_timeout", config, request_kind, last_result)
            await asyncio.sleep(min(0.1, remaining))
            try:
                last_result = await adapter.poll(token, config, now_ms=int(time.time() * 1000))
            except Exception as exc:
                raise _admission_error("agent_admission_unavailable", config, request_kind, last_result) from exc
            if last_result.state == "timed_out":
                raise _admission_error("agent_admission_queue_timeout", config, request_kind, last_result)
        return AgentAdmissionLease(token, adapter, config, request_kind)
    except BaseException:
        with suppress(asyncio.CancelledError, Exception):
            await asyncio.shield(adapter.release(token))
        with suppress(asyncio.CancelledError, Exception):
            await adapter.close()
        raise


@asynccontextmanager
async def admitted_agent_request(request_kind: str) -> AsyncIterator[AgentAdmissionLease]:
    lease = await acquire_agent_request_slot(request_kind)
    try:
        yield lease
    finally:
        await lease.release()

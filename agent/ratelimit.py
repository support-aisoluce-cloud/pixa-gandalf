"""Token-bucket rate limiter for outbound requests to target endpoints.

One bucket per (target_id, host). Configurable per engagement; defaults are
conservative to avoid getting the client's IP banned by the target.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from urllib.parse import urlparse

DEFAULT_RPS = float(os.getenv("PIXA_TARGET_RPS", "1.0"))      # 1 req/s per target
DEFAULT_BURST = int(os.getenv("PIXA_TARGET_BURST", "3"))      # short bursts allowed


@dataclass
class Bucket:
    rps: float
    capacity: int
    tokens: float
    last: float

    def take(self) -> float:
        """Returns seconds to wait before the request can fire."""
        now = time.monotonic()
        elapsed = now - self.last
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rps)
        self.last = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0
        # wait until we have a token
        need = 1.0 - self.tokens
        wait = need / self.rps
        # consume the token we're about to produce
        self.tokens = 0.0
        self.last = now + wait
        return wait


class RateLimiter:
    def __init__(self, rps: float = DEFAULT_RPS, burst: int = DEFAULT_BURST) -> None:
        self.rps = rps
        self.burst = burst
        self._buckets: dict[str, Bucket] = {}
        self._lock = asyncio.Lock()

    def _key(self, target_id: str, url: str) -> str:
        host = urlparse(url if url.startswith("http") else "https://" + url).netloc.lower()
        return f"{target_id}::{host}"

    async def throttle(self, target_id: str, url: str) -> float:
        """Block until a token is available. Returns wait seconds for telemetry."""
        key = self._key(target_id, url)
        async with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = Bucket(rps=self.rps, capacity=self.burst,
                           tokens=float(self.burst), last=time.monotonic())
                self._buckets[key] = b
            wait = b.take()
        if wait > 0:
            await asyncio.sleep(wait)
        return wait


# Singleton
limiter = RateLimiter()

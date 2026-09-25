"""Minimal stdlib client for a FOR-Engine / OpenJev `POST /v1/systemone` server.

Kept dependency-free on purpose: the inference server runs in its own env
(FOR-Engine, Python 3.12 + mlx), Utama only needs HTTP. Identical requests are
answered from an in-process cache — the readout is deterministic for a given
prompt, and bucketed states repeat often from tick to tick.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass
class Decision:
    answers: dict[str, Any]
    latency_ms: float
    cached: bool


class OpenJevClient:
    def __init__(
        self,
        url: str = "http://127.0.0.1:3000",
        timeout_s: float = 120.0,
        cache_size: int = 4096,
    ):
        u = urlparse(url)
        self.url = url
        self._host, self._port = u.hostname or "127.0.0.1", u.port or 80
        self._timeout = timeout_s
        self._conn: http.client.HTTPConnection | None = None
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_size = cache_size

    def _request(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        for attempt in (0, 1):  # one reconnect on a dropped keep-alive connection
            if self._conn is None:
                self._conn = http.client.HTTPConnection(
                    self._host, self._port, timeout=self._timeout
                )
            try:
                self._conn.request(method, path, body=data, headers=headers)
                resp = self._conn.getresponse()
                out = json.loads(resp.read() or b"{}")
            except (http.client.HTTPException, ConnectionError, OSError):
                self._conn.close()
                self._conn = None
                if attempt:
                    raise
                continue
            if resp.status != 200:
                raise RuntimeError(f"{method} {path} -> HTTP {resp.status}: {out}")
            return out
        raise AssertionError("unreachable")

    def version(self) -> dict[str, Any]:
        return self._request("GET", "/v1/version")

    def prewarm(self, state: str) -> dict[str, Any]:
        return self._request("POST", "/v1/prewarm", {"state": state})

    def decide(self, state: str, questions: dict[str, Any]) -> Decision:
        key = hashlib.sha256(
            (state + "\x00" + json.dumps(questions, sort_keys=True)).encode()
        ).hexdigest()
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return Decision(answers=hit, latency_ms=0.0, cached=True)
        t0 = time.perf_counter()
        out = self._request(
            "POST", "/v1/systemone", {"state": state, "questions": questions}
        )
        latency = (time.perf_counter() - t0) * 1000
        self._cache[key] = out["answers"]
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return Decision(answers=out["answers"], latency_ms=latency, cached=False)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

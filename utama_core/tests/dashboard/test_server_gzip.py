"""Large JSON responses (a replay's frames) are gzipped when the browser accepts it; small ones are not."""

from __future__ import annotations

import gzip
import socket
import time
import urllib.request

from utama_core.dashboard.server import _GZIP_MIN_BYTES, DashboardServer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _get(port: int, path: str, accept: str):
    request = urllib.request.Request(f"http://localhost:{port}{path}", headers={"Accept-Encoding": accept})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.headers.get("Content-Encoding"), response.read()


def test_only_responses_over_the_threshold_are_gzipped_and_only_when_accepted():
    big = b"[" + b"1," * _GZIP_MIN_BYTES + b"1]"
    small = b"[1]"
    port = _free_port()
    server = DashboardServer(port)
    server.add_route("/big", lambda: big)
    server.add_route("/small", lambda: small)
    server.start()
    for _ in range(50):
        try:
            encoding, body = _get(port, "/big", "gzip, deflate")
            break
        except OSError:  # not listening yet
            time.sleep(0.05)
    assert encoding == "gzip" and gzip.decompress(body) == big
    assert _get(port, "/big", "identity") == (None, big)
    assert _get(port, "/small", "gzip") == (None, small)

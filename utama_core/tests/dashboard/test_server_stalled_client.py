"""A dashboard client that stops reading (a phone locked with the page open, a laptop asleep,
WiFi gone) must not stall `notify`: it runs on the control loop (`CustomReferee.step`), so a
blocked write there stopped radio commands and HALT. Found 2026-10-21, before the first field
test with a phone."""

from __future__ import annotations

import socket
import threading
import time

from utama_core.dashboard.server import DashboardServer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_a_client_that_never_reads_is_dropped_instead_of_blocking_notify():
    port = _free_port()
    server = DashboardServer(port)
    server.start()
    stalled = socket.socket()
    stalled.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    for _ in range(50):
        try:
            stalled.connect(("localhost", port))
            break
        except OSError:  # not listening yet
            time.sleep(0.05)
    stalled.sendall(b"GET /events/live HTTP/1.1\r\nHost: x\r\n\r\n")  # and never reads again
    for _ in range(50):
        if server._sse_clients.get("live"):
            break
        time.sleep(0.02)

    state = {"pad": "x" * 100_000}  # a few of these fill every buffer on the way
    done = threading.Event()

    def push():
        for _ in range(100):
            server.notify("live", state)
        done.set()

    threading.Thread(target=push, daemon=True).start()
    assert done.wait(5), "notify blocked on a client that stopped reading"
    assert not server._sse_clients["live"]
    stalled.close()


def test_a_client_that_reads_gets_every_state():
    port = _free_port()
    server = DashboardServer(port)
    server.start()
    client = socket.socket()
    for _ in range(50):
        try:
            client.connect(("localhost", port))
            break
        except OSError:
            time.sleep(0.05)
    client.sendall(b"GET /events/live HTTP/1.1\r\nHost: x\r\n\r\n")
    for _ in range(50):
        if server._sse_clients.get("live"):
            break
        time.sleep(0.02)
    for i in range(200):
        server.notify("live", {"i": i})
    client.settimeout(2)
    received = b""
    while b'"i": 199' not in received:
        received += client.recv(65536)
    assert all(f'"i": {i}}}'.encode() in received for i in range(200))
    client.close()

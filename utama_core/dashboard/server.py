"""`DashboardServer` — one HTTP+SSE server shared by every dashboard view.

Domain-agnostic on purpose: this module knows nothing about referees,
tactics, or replays. It serves the shell page, broadcasts whatever JSON
state a view pushes via `notify(channel, state)` to that channel's SSE
subscribers, and dispatches POSTed commands to whichever view registered a
handler for the path. Each `views/*.py` module owns its own state shape and
registers itself against a running server — this module just moves bytes.

Extracted from `custom_referee/gui.py`'s `_RefereeGUIServer`, which used to
run one server per referee instance with a single hardcoded SSE channel and
a single hardcoded `/command` handler. Multiple views sharing one server
means one browser tab, one port, one process — not a GUI per concern.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List
from urllib.parse import parse_qs, urlsplit

CommandHandler = Callable[[dict], None]


def attach_dashboard(port: int = 8080) -> "DashboardServer":
    """Start the dashboard's HTTP server in a background daemon thread.

    Tournament and Replay views are attached automatically — both are
    stateless readers of files on disk (`replays/*/summary.json`, `*.pkl`),
    so every dashboard instance gets them working with no per-script wiring.
    The Referee and Live views depend on a specific running match/referee
    instance, so callers attach those explicitly (see
    `dashboard.views.referee.attach`).
    """
    from utama_core.dashboard.page import build_page
    from utama_core.dashboard.views import replay as replay_view
    from utama_core.dashboard.views import tournament as tournament_view

    server = DashboardServer(port)
    server.set_page(build_page())
    tournament_view.attach(server)
    replay_view.attach(server)
    server.start()
    print(f"Dashboard  →  http://localhost:{port}")
    return server


class DashboardServer(threading.Thread):
    """HTTP server + SSE broadcast, running in a daemon thread.

    Views push state with `notify(channel, state)`; every SSE subscriber on
    that channel receives it as one `data:` frame. Views register POST
    handlers with `add_command_handler(path, handler)` and static payload
    getters with `add_route(path, getter)` for simple `GET -> JSON` endpoints
    (e.g. tournament summaries, replay frame lists).
    """

    def __init__(self, port: int) -> None:
        super().__init__(daemon=True, name="DashboardServer")
        self._port = port

        self._state_lock = threading.Lock()
        self._latest_state: Dict[str, str] = {}  # channel -> last JSON payload

        self._sse_lock = threading.Lock()
        self._sse_clients: Dict[str, List] = {}  # channel -> [wfile, ...]

        self._command_handlers: Dict[str, CommandHandler] = {}
        self._routes: Dict[str, Callable[[], bytes]] = {}
        self._page_bytes = b""

    # ---- registration (called by views before/while the server runs) ----

    def set_page(self, html: str) -> None:
        """Set the shell page served at `/`."""
        self._page_bytes = html.encode()

    def add_route(self, path: str, getter: Callable[..., bytes]) -> None:
        """Register a `GET path` endpoint returning raw JSON bytes.

        `getter` is called with no arguments, unless the request has a query
        string, in which case it is called with one positional arg: a
        `dict[str, str]` of the first value per query key (e.g.
        `?path=foo` -> `{"path": "foo"}`). A getter that never needs query
        params can keep taking zero args — it just won't be called with the
        dict unless a request actually includes one.
        """
        self._routes[path] = getter

    def add_command_handler(self, path: str, handler: CommandHandler) -> None:
        """Register a `POST path` endpoint; handler receives the parsed JSON body."""
        self._command_handlers[path] = handler

    # ---- push state from a view ----

    def notify(self, channel: str, state: dict) -> None:
        """Push a new state snapshot to `channel`'s SSE subscribers."""
        payload = json.dumps(state)
        with self._state_lock:
            self._latest_state[channel] = payload

        frame = (f"event: {channel}\ndata: {payload}\n\n").encode()
        with self._sse_lock:
            clients = list(self._sse_clients.get(channel, ()))

        dead = []
        for wfile in clients:
            try:
                wfile.write(frame)
                wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                dead.append(wfile)

        if dead:
            with self._sse_lock:
                for w in dead:
                    bucket = self._sse_clients.get(channel)
                    if bucket and w in bucket:
                        bucket.remove(w)

    # ---- thread entry point ----

    def run(self) -> None:
        handler_factory = self._make_handler_class()
        http_server = ThreadingHTTPServer(("", self._port), handler_factory)
        http_server.serve_forever()

    def _make_handler_class(self):
        server = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # suppress default access log

            def _send_json_bytes(self, body: bytes, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_index(self) -> None:
                body = server._page_bytes
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_sse(self, channel: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                self.wfile.flush()

                with server._sse_lock:
                    server._sse_clients.setdefault(channel, []).append(self.wfile)

                # Replay the last known state immediately so a newly-opened
                # tab isn't blank until the next tick.
                with server._state_lock:
                    last = server._latest_state.get(channel)
                if last is not None:
                    try:
                        self.wfile.write(f"event: {channel}\ndata: {last}\n\n".encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass

                try:
                    while True:
                        threading.Event().wait(0.5)
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with server._sse_lock:
                        bucket = server._sse_clients.get(channel)
                        if bucket and self.wfile in bucket:
                            bucket.remove(self.wfile)

            def _handle_command(self, path: str) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                handler = server._command_handlers.get(path)
                if handler is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    payload = json.loads(body) if body else {}
                    handler(payload)
                    self.send_response(204)
                    self.end_headers()
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    msg = f"Bad request: {exc}".encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)

            def do_GET(self):
                split = urlsplit(self.path)
                path = split.path
                if path == "/":
                    self._serve_index()
                elif path.startswith("/events/"):
                    self._serve_sse(path[len("/events/") :])
                elif path in server._routes:
                    getter = server._routes[path]
                    if split.query:
                        query = {k: v[0] for k, v in parse_qs(split.query).items()}
                        self._send_json_bytes(getter(query))
                    else:
                        self._send_json_bytes(getter())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                self._handle_command(self.path)

        return _Handler

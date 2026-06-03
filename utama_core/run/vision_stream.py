"""Browser stream for StrategyRunner frames."""

from __future__ import annotations

import json
import logging
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class GameFrameRenderer:
    """Render refined StrategyRunner game frames using the RSIM visual style."""

    def __init__(self, field_dims, scale: float = 100.0) -> None:
        import pygame

        from utama_core.rsoccer_simulator.src.Render import SSLRenderField

        self._pygame = pygame
        self._field_renderer = SSLRenderField(
            length=2 * field_dims.full_field_half_length,
            width=2 * field_dims.full_field_half_width,
            penalty_length=2 * field_dims.half_defense_area_depth,
            penalty_width=2 * field_dims.half_defense_area_width,
            goal_width=2 * field_dims.half_goal_width,
            goal_depth=field_dims.goal_depth,
            center_circle_r=field_dims.center_circle_radius,
            scale=scale,
        )
        self._surface = pygame.Surface(self._field_renderer.window_size)

    def render(self, game_frame) -> np.ndarray:
        """Render a `GameFrame` into an RGB numpy array."""
        from utama_core.rsoccer_simulator.src.Render import (
            COLORS,
            RenderBall,
            RenderSSLRobot,
        )

        self._field_renderer.draw(self._surface)

        for robot in game_frame.friendly_robots.values():
            color = COLORS["YELLOW"] if game_frame.my_team_is_yellow else COLORS["BLUE"]
            self._draw_robot(RenderSSLRobot, robot, color)

        for robot in game_frame.enemy_robots.values():
            color = COLORS["BLUE"] if game_frame.my_team_is_yellow else COLORS["YELLOW"]
            self._draw_robot(RenderSSLRobot, robot, color)

        if game_frame.ball is not None:
            ball = RenderBall(
                *self._pos_transform(game_frame.ball.p.x, -game_frame.ball.p.y),
                self._field_renderer.scale,
            )
            ball.draw(self._surface)

        return np.transpose(
            np.array(self._pygame.surfarray.pixels3d(self._surface)),
            axes=(1, 0, 2),
        ).copy()

    def _draw_robot(self, render_robot_cls, robot, color) -> None:
        x, y = self._pos_transform(robot.p.x, -robot.p.y)
        render_robot_cls(
            x,
            y,
            np.rad2deg(robot.orientation),
            self._field_renderer.scale,
            robot.id,
            color,
        ).draw(self._surface)

    def _pos_transform(self, pos_x: float, pos_y: float) -> tuple[int, int]:
        return (
            int(pos_x * self._field_renderer.scale + self._field_renderer.center_x),
            int(pos_y * self._field_renderer.scale + self._field_renderer.center_y),
        )


class RSimVisionStreamServer:
    """Serve rendered StrategyRunner frames to a browser.

    The page is served with the stdlib HTTP server. Metadata is pushed through
    Server-Sent Events, while frame pixels are fetched from ``/frame`` as raw
    RGB bytes with an 8-byte big-endian ``uint32 width`` + ``uint32 height``
    header. This keeps the stream dependency-free in the pixi environment.
    """

    def __init__(
        self,
        http_host: str = "127.0.0.1",
        http_port: int = 8765,
        websocket_host: str = "127.0.0.1",
        websocket_port: int = 8766,
        max_fps: float = 30.0,
    ) -> None:
        self.http_host = http_host
        self.http_port = http_port
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        self.max_fps = max_fps

        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame_condition = threading.Condition(self._lock)
        self._latest_frame: Optional[bytes] = None
        self._latest_status: Optional[str] = None
        self._sse_clients: List = []
        self._sse_lock = threading.Lock()
        self._last_publish_time = 0.0

    @property
    def url(self) -> str:
        return f"http://{self.http_host}:{self.http_port}/"

    def start(self) -> None:
        """Start the HTTP server in a daemon thread."""
        if self._thread is not None:
            return

        handler = self._make_handler_class()
        self._server = ThreadingHTTPServer((self.http_host, self.http_port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="rsim-vision-stream",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the stream server."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None
        with self._sse_lock:
            self._sse_clients.clear()

    def publish_rgb_frame(self, frame: np.ndarray) -> None:
        """Publish an RGB frame rendered from the latest game state.

        Args:
            frame: A numpy array with shape ``(height, width, 3)`` and RGB data.
        """
        if self._server is None:
            return

        now = time.monotonic()
        min_interval = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
        if now - self._last_publish_time < min_interval:
            return
        self._last_publish_time = now

        height, width = frame.shape[:2]
        payload = struct.pack("!II", width, height) + np.ascontiguousarray(frame[:, :, :3]).tobytes()
        with self._frame_condition:
            self._latest_frame = payload
            self._frame_condition.notify_all()

    def publish_status(self, status: dict[str, object]) -> None:
        """Publish status metadata to the stream client."""
        if self._server is None:
            return

        payload = json.dumps({"type": "status", **status})
        with self._lock:
            self._latest_status = payload
        self._broadcast_status(payload)

    def _broadcast_status(self, payload: str) -> None:
        message = f"data: {payload}\n\n".encode("utf-8")
        dead = []
        with self._sse_lock:
            clients = list(self._sse_clients)

        for wfile in clients:
            try:
                wfile.write(message)
                wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                dead.append(wfile)

        if dead:
            with self._sse_lock:
                for wfile in dead:
                    if wfile in self._sse_clients:
                        self._sse_clients.remove(wfile)

    def _make_handler_class(self):
        server_instance = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _send_body(self, body: bytes, content_type: str, *, cache: bool = True) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                if not cache:
                    self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _serve_index(self) -> None:
                self._send_body(server_instance._index_html().encode("utf-8"), "text/html; charset=utf-8")

            def _serve_status(self) -> None:
                with server_instance._lock:
                    payload = server_instance._latest_status or '{"type":"status"}'
                self._send_body(payload.encode("utf-8"), "application/json; charset=utf-8", cache=False)

            def _serve_frame(self) -> None:
                with server_instance._lock:
                    payload = server_instance._latest_frame
                if payload is None:
                    self.send_response(204)
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    return
                self._send_body(payload, "application/octet-stream", cache=False)

            def _serve_events(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                self.wfile.flush()

                with server_instance._sse_lock:
                    server_instance._sse_clients.append(self.wfile)

                with server_instance._lock:
                    latest_status = server_instance._latest_status
                if latest_status is not None:
                    try:
                        self.wfile.write(f"data: {latest_status}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        latest_status = None

                try:
                    while True:
                        time.sleep(0.5)
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with server_instance._sse_lock:
                        if self.wfile in server_instance._sse_clients:
                            server_instance._sse_clients.remove(self.wfile)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/":
                    self._serve_index()
                elif path == "/frame":
                    self._serve_frame()
                elif path == "/status":
                    self._serve_status()
                elif path == "/events":
                    self._serve_events()
                else:
                    self.send_response(404)
                    self.end_headers()

        return _Handler

    def _index_html(self) -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vision Stream</title>
  <style>
    :root {
      box-sizing: border-box;
      color-scheme: dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #15171a;
      color: #f1f4f8;
    }
    *, *::before, *::after {
      box-sizing: inherit;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 12px;
      padding: 16px;
    }
    #stream-info {
      display: grid;
      gap: 12px;
      grid-template-columns: minmax(160px, 1fr) minmax(180px, 240px) minmax(160px, 1fr);
      align-items: start;
      width: min(100%, 980px);
      margin: 0 auto;
      min-width: 0;
    }
    .team-panel,
    #time-box {
      display: grid;
      gap: 8px;
      min-width: 0;
    }
    #time-box {
      align-items: center;
      justify-items: center;
    }
    .info-box {
      background: #1e242c;
      border: 1px solid #343b45;
      border-radius: 10px;
      padding: 10px 14px;
      min-width: 0;
      width: 100%;
    }
    #time-box .info-box {
      text-align: center;
    }
    .info-title {
      color: #8fa1b3;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }
    main {
      display: grid;
      place-items: center;
      padding: 16px;
    }
    canvas {
      width: min(100%, 1200px);
      height: auto;
      aspect-ratio: 4 / 3;
      background: #0b0d0f;
      border: 1px solid #343b45;
    }
  </style>
</head>
<body>
  <div id="stream-info">
    <div class="team-panel">
      <div class="info-box">
        <div class="info-title">Blue Strategy</div>
        <div id="blue-strategy">N/A</div>
      </div>
      <div class="info-box">
        <div class="info-title">Blue Score</div>
        <div id="blue-score">0</div>
      </div>
    </div>
    <div id="time-box">
      <div class="info-box">
        <div class="info-title">Time</div>
        <div id="time-left">--:--</div>
      </div>
    </div>
    <div class="team-panel">
      <div class="info-box">
        <div class="info-title">Yellow Score</div>
        <div id="yellow-score">0</div>
      </div>
      <div class="info-box">
        <div class="info-title">Yellow Strategy</div>
        <div id="yellow-strategy">N/A</div>
      </div>
    </div>
  </div>
  <main>
    <canvas id="stream"></canvas>
  </main>
  <script>
    const canvas = document.getElementById("stream");
    const ctx = canvas.getContext("2d");
    const timeLeft = document.getElementById("time-left");
    const blueScore = document.getElementById("blue-score");
    const yellowScore = document.getElementById("yellow-score");
    const blueStrategy = document.getElementById("blue-strategy");
    const yellowStrategy = document.getElementById("yellow-strategy");

    function updateStatus(info) {
      timeLeft.textContent = info.time_left ?? "--:--";
      blueScore.textContent = info.score_blue ?? 0;
      yellowScore.textContent = info.score_yellow ?? 0;
      blueStrategy.textContent = info.strategy_blue ?? "N/A";
      yellowStrategy.textContent = info.strategy_yellow ?? "N/A";
    }

    function drawFrame(buffer) {
      const data = new Uint8Array(buffer);
      if (data.length < 8) {
        return;
      }
      const view = new DataView(buffer);
      const width = view.getUint32(0);
      const height = view.getUint32(4);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const rgb = data.subarray(8);
      const rgba = new Uint8ClampedArray(width * height * 4);
      for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
        rgba[j] = rgb[i];
        rgba[j + 1] = rgb[i + 1];
        rgba[j + 2] = rgb[i + 2];
        rgba[j + 3] = 255;
      }
      ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
    }

    async function fetchInitialStatus() {
      try {
        const response = await fetch("/status", { cache: "no-store" });
        if (response.ok) {
          updateStatus(await response.json());
        }
      } catch (err) {
        console.warn("Unable to fetch stream status", err);
      }
    }

    async function pollFrame() {
      try {
        const response = await fetch("/frame", { cache: "no-store" });
        if (response.status === 204) {
          return;
        }
        if (response.ok) {
          drawFrame(await response.arrayBuffer());
        }
      } catch (err) {
        console.warn("Unable to fetch stream frame", err);
      }
    }

    const events = new EventSource("/events");
    events.onmessage = (event) => {
      try {
        updateStatus(JSON.parse(event.data));
      } catch (err) {
        console.warn("Unable to parse stream status", err);
      }
    };
    events.onerror = () => console.warn("Vision stream status connection interrupted");

    fetchInitialStatus();
    setInterval(pollFrame, 33);
  </script>
</body>
</html>
"""

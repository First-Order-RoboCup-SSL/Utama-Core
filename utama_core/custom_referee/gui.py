"""Browser-based operator panel for the CustomReferee.

Library mode (attach to a referee already driven by your own loop):
    from utama_core.custom_referee.gui import attach_gui
    referee = CustomReferee.from_profile_name("simulation")
    attach_gui(referee, profile, port=8080)   # starts server in background
    # your loop: referee.step(frame, time.time()) as normal

Or use the convenience flag on CustomReferee:
    referee = CustomReferee(profile, enable_gui=True, gui_port=8080)

Serves a single-page HTML GUI over a stdlib HTTP server.
State is pushed via SSE (~30 Hz); commands come back via POST /command.
The active profile's configuration (geometry + rules + game settings) is
available at GET /config and displayed in a read-only panel on the page.

No external dependencies beyond what the project already installs.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from utama_core.custom_referee import CustomReferee
    from utama_core.custom_referee.profiles.profile_loader import RefereeProfile

from utama_core.entities.data.vector import Vector3D
from utama_core.entities.game.ball import Ball
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attach_gui(
    referee: "CustomReferee",
    profile: "RefereeProfile",
    port: int = 8080,
    *,
    run_tick_loop: bool = False,
) -> None:
    """Attach the web GUI to an existing CustomReferee instance.

    Starts the HTTP server in a background daemon thread.  The GUI will show
    live state as long as the caller keeps calling ``referee.step()`` from its
    own loop.

    Args:
        referee:       The CustomReferee instance to observe / control.
        profile:       The RefereeProfile used to create the referee (used to
                       populate the /config endpoint and the config panel).
        port:          HTTP port to listen on (default 8080).
        run_tick_loop: If True, also start an internal tick loop that calls
                       ``referee.step()`` with a static frame at ~30 Hz.  Use
                       this when you have *no* external game loop (standalone
                       operator-panel mode).  Leave False when your own loop
                       drives ``referee.step()``.
    """
    server = _RefereeGUIServer(referee, profile, port, run_tick_loop=run_tick_loop)
    server.start()
    print(f"Referee GUI  →  http://localhost:{port}")
    print(f"Profile:        {profile.profile_name}")


# ---------------------------------------------------------------------------
# Internal server (one instance per attach_gui call)
# ---------------------------------------------------------------------------


class _RefereeGUIServer(threading.Thread):
    """HTTP server + optional tick loop, all in daemon threads."""

    def __init__(
        self,
        referee: "CustomReferee",
        profile: "RefereeProfile",
        port: int,
        *,
        run_tick_loop: bool,
    ) -> None:
        super().__init__(daemon=True, name="RefereeGUIServer")
        self._referee = referee
        self._port = port
        self._run_tick_loop = run_tick_loop
        self._static_config = _build_static_config(profile)

        self._lock = threading.Lock()
        self._ref_data = None
        self._game_frame = None  # Optional[GameFrame]
        self._bt_data: dict = {}
        self._robot_feedback_data: list[dict] = []
        self._sse_clients: List = []
        self._sse_lock = threading.Lock()

    # ---- threading.Thread entry point ----

    def run(self) -> None:
        if self._run_tick_loop:
            threading.Thread(target=self._tick_loop, daemon=True, name="RefereeGUITick").start()

        handler_factory = self._make_handler_class()
        server = ThreadingHTTPServer(("", self._port), handler_factory)
        server.serve_forever()

    def start(self) -> None:
        super().start()

    # ---- tick loop (standalone mode only) ----

    def _tick_loop(self) -> None:
        frame = _make_static_frame()
        while True:
            result = self._referee.step(frame, time.time())
            with self._lock:
                self._ref_data = result
                self._game_frame = frame
            self._broadcast()
            time.sleep(1 / 30)

    def _build_config_json(self) -> str:
        g = self._referee.geometry
        config = dict(self._static_config)
        config["geometry"] = {
            "half_length": g.half_length,
            "half_width": g.half_width,
            "half_goal_width": g.half_goal_width,
            "half_defense_depth": g.half_defense_depth,
            "half_defense_width": g.half_defense_width,
            "center_circle_radius": g.center_circle_radius,
            "goal_depth": g.goal_depth,
        }
        return json.dumps(config)

    # ---- called by external loops to push a new state snapshot ----

    def notify(self, ref_data, game_frame=None, bt_data=None, robot_feedback_data=None) -> None:
        """Push a RefereeData snapshot from an external game loop."""
        with self._lock:
            self._ref_data = ref_data
            self._game_frame = game_frame
            if bt_data is not None:
                self._bt_data = bt_data
            if robot_feedback_data is not None:
                self._robot_feedback_data = [dict(row) for row in robot_feedback_data]
        self._broadcast()

    # ---- SSE broadcast ----

    def _broadcast(self) -> None:
        with self._lock:
            data = self._ref_data
            frame = self._game_frame
            bt = self._bt_data
            robot_feedback = self._robot_feedback_data
        if data is None:
            return

        payload = ("data: " + _serialise_state(data, frame, bt, robot_feedback) + "\n\n").encode()
        dead: List = []

        with self._sse_lock:
            clients = list(self._sse_clients)

        for wfile in clients:
            try:
                wfile.write(payload)
                wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                dead.append(wfile)

        if dead:
            with self._sse_lock:
                for w in dead:
                    if w in self._sse_clients:
                        self._sse_clients.remove(w)

    # ---- handler class factory (captures self) ----

    def _make_handler_class(self):
        server_instance = self

        class _Handler(BaseHTTPRequestHandler):

            def log_message(self, fmt, *args):
                pass  # suppress default access log

            def _serve_index(self):
                body = _HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_config(self):
                body = server_instance._build_config_json().encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_sse(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                self.wfile.flush()

                with server_instance._sse_lock:
                    server_instance._sse_clients.append(self.wfile)

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

            def _handle_command(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body)
                    cmd = RefereeCommand[payload["command"]]
                    designated = payload.get("designated")
                    with server_instance._lock:
                        ref = server_instance._referee
                        if designated is not None and hasattr(ref, "force_command"):
                            target = (float(designated[0]), float(designated[1]))
                            ref.force_command(cmd, time.time(), ball_placement_target=target)
                        else:
                            ref.set_command(cmd, time.time())
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
                if self.path == "/":
                    self._serve_index()
                elif self.path == "/config":
                    self._serve_config()
                elif self.path == "/events":
                    self._serve_sse()
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == "/command":
                    self._handle_command()
                else:
                    self.send_response(404)
                    self.end_headers()

        return _Handler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_static_frame() -> GameFrame:
    """Minimal GameFrame with no robots and ball at centre."""
    ball = Ball(
        p=Vector3D(0.0, 0.0, 0.0),
        v=Vector3D(0.0, 0.0, 0.0),
        a=Vector3D(0.0, 0.0, 0.0),
    )
    return GameFrame(
        ts=time.time(),
        my_team_is_yellow=True,
        my_team_is_right=False,
        friendly_robots={},
        enemy_robots={},
        ball=ball,
        referee=None,
    )


def _serialise_robots(game_frame) -> dict:
    if game_frame is None:
        return {"friendly": [], "enemy": []}

    def _robot_list(robots_dict):
        return [
            {
                "id": r.id,
                "x": r.p.x,
                "y": r.p.y,
                "vx": r.v.x,
                "vy": r.v.y,
                "orientation": r.orientation,
                "has_ball": r.has_ball,
            }
            for r in robots_dict.values()
        ]

    return {
        "friendly": _robot_list(game_frame.friendly_robots),
        "enemy": _robot_list(game_frame.enemy_robots),
    }


def _serialise_ball(game_frame):
    if game_frame is None or game_frame.ball is None:
        return None
    return {"x": game_frame.ball.p.x, "y": game_frame.ball.p.y, "vx": game_frame.ball.v.x, "vy": game_frame.ball.v.y}


def _serialise_vector(vector):
    if vector is None:
        return None
    data = {"x": vector.x, "y": vector.y}
    if hasattr(vector, "z"):
        data["z"] = vector.z
    return data


def _serialise_game_frame_robot(robot) -> dict:
    return {
        "id": robot.id,
        "is_friendly": robot.is_friendly,
        "has_ball": robot.has_ball,
        "p": _serialise_vector(robot.p),
        "v": _serialise_vector(robot.v),
        "a": _serialise_vector(robot.a),
        "orientation": robot.orientation,
    }


def _serialise_game_frame_ball(ball) -> Optional[dict]:
    if ball is None:
        return None
    return {
        "p": _serialise_vector(ball.p),
        "v": _serialise_vector(ball.v),
        "a": _serialise_vector(ball.a),
    }


def _serialise_game_frame(game_frame) -> Optional[dict]:
    if game_frame is None:
        return None
    return {
        "type": "GameFrame",
        "ts": game_frame.ts,
        "my_team_is_yellow": game_frame.my_team_is_yellow,
        "my_team_is_right": game_frame.my_team_is_right,
        "friendly_robots": {
            robot_id: _serialise_game_frame_robot(robot) for robot_id, robot in game_frame.friendly_robots.items()
        },
        "enemy_robots": {
            robot_id: _serialise_game_frame_robot(robot) for robot_id, robot in game_frame.enemy_robots.items()
        },
        "ball": _serialise_game_frame_ball(game_frame.ball),
        "referee": repr(game_frame.referee) if game_frame.referee is not None else None,
    }


def _serialise_state(ref_data, game_frame=None, bt_data=None, robot_feedback_data=None) -> str:
    designated = None
    if ref_data.designated_position is not None:
        try:
            designated = list(ref_data.designated_position)
        except TypeError:
            designated = [
                ref_data.designated_position.x,
                ref_data.designated_position.y,
            ]

    return json.dumps(
        {
            "command": ref_data.referee_command.name,
            "next_command": (ref_data.next_command.name if ref_data.next_command else None),
            "stage": ref_data.stage.name,
            "stage_time_left": ref_data.stage_time_left or 0.0,
            "yellow_score": ref_data.yellow_team.score,
            "blue_score": ref_data.blue_team.score,
            "designated": designated,
            "status_message": ref_data.status_message,
            "my_team_is_right": getattr(game_frame, "my_team_is_right", False),
            "my_team_is_yellow": getattr(game_frame, "my_team_is_yellow", True),
            "robots": _serialise_robots(game_frame),
            "ball": _serialise_ball(game_frame),
            "game_frame": _serialise_game_frame(game_frame),
            "bt_nodes": bt_data or {},
            "robot_feedback": robot_feedback_data or [],
        }
    )


def _build_static_config(profile: "RefereeProfile") -> dict:
    r = profile.rules
    gm = profile.game
    return {
        "profile_name": profile.profile_name,
        "rules": {
            "goal_detection": {
                "enabled": r.goal_detection.enabled,
                "cooldown_seconds": r.goal_detection.cooldown_seconds,
            },
            "out_of_bounds": {
                "enabled": r.out_of_bounds.enabled,
                "free_kick_assigner": r.out_of_bounds.free_kick_assigner,
            },
            "defense_area": {
                "enabled": r.defense_area.enabled,
                "max_defenders": r.defense_area.max_defenders,
                "attacker_infringement": r.defense_area.attacker_infringement,
            },
            "keep_out": {
                "enabled": r.keep_out.enabled,
                "radius_meters": r.keep_out.radius_meters,
                "violation_persistence_frames": r.keep_out.violation_persistence_frames,
            },
        },
        "game": {
            "half_duration_seconds": gm.half_duration_seconds,
            "kickoff_team": gm.kickoff_team,
            "force_start_after_goal": gm.force_start_after_goal,
            "stop_duration_seconds": gm.stop_duration_seconds,
            "auto_advance": {
                "stop_to_next_command": gm.auto_advance.stop_to_next_command,
                "prepare_kickoff_to_normal": gm.auto_advance.prepare_kickoff_to_normal,
                "prepare_penalty_to_normal": gm.auto_advance.prepare_penalty_to_normal,
                "direct_free_to_normal": gm.auto_advance.direct_free_to_normal,
                "ball_placement_to_next": gm.auto_advance.ball_placement_to_next,
                "normal_start_to_force": gm.auto_advance.normal_start_to_force,
            },
        },
    }


# ---------------------------------------------------------------------------
# Inline HTML page
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Custom Referee</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #1a1a2e;
    --surface:   #16213e;
    --border:    #0f3460;
    --text:      #e0e0e0;
    --muted:     #888;
    --yellow:    #f4c542;
    --blue:      #4da6ff;
    --red:       #e74c3c;
    --orange:    #e67e22;
    --green:     #2ecc71;
    --radius:    6px;
  }

  html, body {
    height: 100%;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: ui-monospace, "Cascadia Code", "Fira Code", monospace;
    height: 100vh;
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    grid-template-areas:
      "field  controls"
      "log    config";
    gap: 0;
    overflow: hidden;
  }

  /* ── Quadrant base ── */
  .quad {
    background: var(--surface);
    border: 1px solid var(--border);
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .quad-title {
    padding: 8px 16px;
    font-size: .65rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  /* ── Top-left: field ── */
  #quad-field {
    grid-area: field;
  }
  #field-canvas {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
  .field-wrap {
    flex: 1;
    min-height: 0;
    position: relative;
  }
  .field-wrap canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }

  /* ── Top-right: controls ── */
  #quad-controls {
    grid-area: controls;
    overflow-y: auto;
  }
  .scoreboard {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .score-cell {
    padding: 12px 16px;
    text-align: center;
  }
  .score-cell:first-child { border-right: 1px solid var(--border); }
  .team-name {
    font-size: .7rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .team-name.yellow { color: var(--yellow); }
  .team-name.blue   { color: var(--blue); }
  .score-value {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
  }
  .score-value.yellow { color: var(--yellow); }
  .score-value.blue   { color: var(--blue); }

  .info-block {
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .info-row {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: .85rem;
  }
  .label {
    color: var(--muted);
    min-width: 84px;
    font-size: .7rem;
    text-transform: uppercase;
    letter-spacing: .05em;
    flex-shrink: 0;
  }

  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: .8rem;
    font-weight: 600;
    letter-spacing: .05em;
    text-transform: uppercase;
    transition: background .2s, color .2s;
  }
  .badge.HALT              { background: var(--red);    color: #fff; }
  .badge.STOP              { background: var(--orange); color: #fff; }
  .badge.NORMAL_START,
  .badge.FORCE_START       { background: var(--green);  color: #111; }
  .badge.PREPARE_KICKOFF_YELLOW,
  .badge.DIRECT_FREE_YELLOW,
  .badge.INDIRECT_FREE_YELLOW,
  .badge.PREPARE_PENALTY_YELLOW,
  .badge.BALL_PLACEMENT_YELLOW,
  .badge.TIMEOUT_YELLOW    { background: var(--yellow); color: #111; }
  .badge.PREPARE_KICKOFF_BLUE,
  .badge.DIRECT_FREE_BLUE,
  .badge.INDIRECT_FREE_BLUE,
  .badge.PREPARE_PENALTY_BLUE,
  .badge.BALL_PLACEMENT_BLUE,
  .badge.TIMEOUT_BLUE      { background: var(--blue);   color: #111; }
  .badge.unknown           { background: #444; color: #ccc; }

  .buttons {
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex-shrink: 0;
  }
  .btn-row { display: flex; flex-wrap: wrap; gap: 6px; }
  button {
    padding: 7px 13px;
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-family: inherit;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .05em;
    text-transform: uppercase;
    cursor: pointer;
    transition: filter .15s, transform .1s;
  }
  button:active { transform: scale(.96); }
  button:hover  { filter: brightness(1.15); }
  .btn-halt          { background: var(--red);    color: #fff; }
  .btn-stop          { background: var(--orange); color: #fff; }
  .btn-normal-start,
  .btn-force-start   { background: var(--green);  color: #111; }
  .btn-yellow        { background: var(--yellow); color: #111; }
  .btn-blue          { background: var(--blue);   color: #111; }

  .conn {
    padding: 8px 16px;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: .7rem;
    color: var(--muted);
    border-top: 1px solid var(--border);
    flex-shrink: 0;
    margin-top: auto;
  }
  .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #555;
    transition: background .3s;
  }
  .dot.live { background: var(--green); }

  /* ── Bottom-left: robot status ── */
  #quad-status {
    grid-area: log;
  }
  .status-entries {
    overflow-y: auto;
    flex: 0 0 auto;
    max-height: 45%;
    padding: 6px 10px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .status-section-title {
    font-size: .6rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 4px 0 2px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2px;
  }
  .status-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: .68rem;
    line-height: 1.4;
    padding: 1px 0;
  }
  .status-bot-id {
    font-weight: 700;
    min-width: 16px;
    text-align: right;
  }
  .status-bot-id.friendly { color: var(--yellow); }
  .status-bot-id.enemy    { color: var(--blue); }
  .status-pos { color: var(--text); opacity: .85; }
  .status-vel { color: var(--muted); }
  .status-feedback-id {
    font-weight: 700;
    color: var(--text);
    min-width: 54px;
  }
  .status-feedback-meta {
    color: var(--muted);
    font-size: .62rem;
  }
  .status-feedback-ball {
    min-width: 54px;
    color: var(--muted);
    font-weight: 600;
  }
  .status-feedback-ball.yes { color: var(--orange); }
  .status-pill {
    display: inline-block;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: .58rem;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
  }
  .status-pill.connected { background: #17361f; color: var(--green); border: 1px solid #245c33; }
  .status-pill.disconnected { background: #3a1a1a; color: #c0605a; border: 1px solid #5a2a2a; }
  .status-pill.ball { background: #3a2a13; color: var(--orange); border: 1px solid #73501b; }
  .status-ball-indicator {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    flex-shrink: 0;
  }
  .status-bt {
    font-size: .6rem;
    color: var(--orange);
    padding-left: 22px;
    line-height: 1.3;
    opacity: .85;
  }
  .game-frame-log {
    border-top: 1px solid var(--border);
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .game-frame-log-title {
    font-size: .6rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 7px 10px 5px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .game-frame-log-entries {
    overflow-y: auto;
    flex: 1;
    min-height: 0;
    padding: 6px 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .game-frame-entry {
    border: 1px solid var(--border);
    border-radius: 4px;
    background: rgba(0, 0, 0, .12);
    padding: 6px 7px;
  }
  .game-frame-entry-head {
    color: var(--muted);
    font-size: .58rem;
    letter-spacing: .05em;
    margin-bottom: 4px;
  }
  .game-frame-entry pre {
    color: var(--text);
    font-family: inherit;
    font-size: .62rem;
    line-height: 1.35;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* ── Bottom-right: config ── */
  #quad-config {
    grid-area: config;
    overflow-y: auto;
  }
  .config-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    flex: 1;
    min-height: 0;
  }
  .config-section {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
  }
  .config-section:nth-child(odd)       { border-right: 1px solid var(--border); }
  .config-section:nth-last-child(-n+2) { border-bottom: none; }
  .config-section-title {
    font-size: .63rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .cfg-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: .75rem;
    padding: 2px 0;
    gap: 8px;
  }
  .cfg-key { color: var(--muted); flex-shrink: 0; }
  .cfg-val { text-align: right; word-break: break-word; }
  .pill {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: .7rem;
    font-weight: 600;
  }
  .pill.on  { background: #1a3a1a; color: var(--green); border: 1px solid #2a5a2a; }
  .pill.off { background: #3a1a1a; color: #c0605a;      border: 1px solid #5a2a2a; }

  /* ── God mode ── */
  .btn-god { background: #4a0d6e; color: #d49eff; border: 1px solid #7a3dae; }
  .btn-god.active { background: #7a3dae; color: #fff; border-color: #d49eff; }

  #ctx-menu {
    display: none;
    position: fixed;
    z-index: 999;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    min-width: 180px;
    box-shadow: 0 4px 16px rgba(0,0,0,.5);
    overflow: hidden;
  }
  #ctx-menu button {
    display: block;
    width: 100%;
    text-align: left;
    border-radius: 0;
    border: none;
    background: none;
    color: var(--text);
    padding: 9px 14px;
    font-size: .78rem;
    text-transform: none;
    letter-spacing: 0;
    font-weight: 400;
  }
  #ctx-menu button:hover { background: var(--border); }
  #ctx-menu .ctx-sep { height: 1px; background: var(--border); margin: 2px 0; }
</style>
</head>
<body>

<!-- Top-left: field render -->
<div class="quad" id="quad-field">
  <div class="quad-title" id="page-title">Field</div>
  <div class="field-wrap">
    <canvas id="field-canvas"></canvas>
  </div>
</div>

<!-- Top-right: scoreboard + game state + buttons -->
<div class="quad" id="quad-controls">
  <div class="quad-title">Controls</div>
  <div class="scoreboard">
    <div class="score-cell">
      <div class="team-name yellow">Yellow</div>
      <div class="score-value yellow" id="yellow-score">—</div>
    </div>
    <div class="score-cell">
      <div class="team-name blue">Blue</div>
      <div class="score-value blue" id="blue-score">—</div>
    </div>
  </div>
  <div class="info-block">
    <div class="info-row">
      <span class="label">Command</span>
      <span class="badge unknown" id="cmd-badge">—</span>
    </div>
    <div class="info-row">
      <span class="label">Next</span>
      <span id="next-cmd">—</span>
    </div>
    <div class="info-row">
      <span class="label">Stage</span>
      <span id="stage">—</span>&nbsp;
      <span id="stage-time" style="color:var(--muted)"></span>
    </div>
    <div class="info-row">
      <span class="label">Designated</span>
      <span id="designated">—</span>
    </div>
    <div class="info-row" id="status-row" style="display:none">
      <span class="label">Status</span>
      <span id="status-msg" style="color:var(--muted);font-size:.75rem;"></span>
    </div>
  </div>
  <div class="buttons">
    <div class="btn-row">
      <button class="btn-halt"         onclick="send('HALT')"
        title="Emergency stop — all robots immediately cease movement.">Halt</button>
      <button class="btn-stop"         onclick="send('STOP')"
        title="Pause play. Robots slow to ≤1.5 m/s, stay ≥0.5 m from ball.">Stop</button>
      <button class="btn-normal-start" onclick="send('NORMAL_START')"
        title="Begin/resume play after kickoff or free kick positioning is complete.">Normal Start</button>
      <button class="btn-force-start"  onclick="send('FORCE_START')"
        title="Resume immediately without set-piece positioning (double-touch, stalled play).">Force Start</button>
    </div>
    <div class="btn-row">
      <button class="btn-yellow" onclick="send('PREPARE_KICKOFF_YELLOW')"
        title="Award Yellow a kickoff. If game is running, STOP is issued first so robots can clear the ball — then click Normal Start to begin kickoff.">Kickoff Yellow</button>
      <button class="btn-blue"   onclick="send('PREPARE_KICKOFF_BLUE')"
        title="Award Blue a kickoff. If game is running, STOP is issued first so robots can clear the ball — then click Normal Start to begin kickoff.">Kickoff Blue</button>
    </div>
    <div class="btn-row">
      <button class="btn-yellow" onclick="send('DIRECT_FREE_YELLOW')"
        title="Award Yellow a direct free kick. If game is running, STOP is issued first so robots can clear — then click Normal Start.">Free Kick Yellow</button>
      <button class="btn-blue"   onclick="send('DIRECT_FREE_BLUE')"
        title="Award Blue a direct free kick. If game is running, STOP is issued first so robots can clear — then click Normal Start.">Free Kick Blue</button>
    </div>
    <div class="btn-row">
      <button class="btn-god" id="god-btn" onclick="toggleGod()"
        title="God mode: right-click anywhere on the field to move the ball or issue ball placement.">God Mode</button>
    </div>
  </div>
  <div class="conn">
    <div class="dot" id="conn-dot"></div>
    <span id="conn-label">connecting…</span>
    <span style="margin-left:auto;color:var(--muted);font-size:.6rem;">Space = Halt / Resume</span>
  </div>
</div>

<!-- Bottom-left: robot status -->
<div class="quad" id="quad-status">
  <div class="quad-title">Robot Status</div>
  <div class="status-entries" id="status-entries"></div>
  <div class="game-frame-log">
    <div class="game-frame-log-title">Game Frame Log</div>
    <div class="game-frame-log-entries" id="game-frame-log"></div>
  </div>
</div>

<!-- God mode context menu -->
<div id="ctx-menu">
  <button id="ctx-place-left" onclick="ctxPlace('left')">Ball placement here</button>
  <button id="ctx-place-right" onclick="ctxPlace('right')">Ball placement here</button>
</div>

<!-- Bottom-right: profile config -->
<div class="quad" id="quad-config">
  <div class="quad-title" id="cfg-title">Profile — loading…</div>
  <div class="config-grid" id="cfg-grid"></div>
</div>

<script>
function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// --- Robot status rendering ---
function renderStatus(d) {
  const c = document.getElementById('status-entries');
  if (!c) return;
  let html = '';

  // Controller feedback from the robot port
  const feedback = d.robot_feedback || [];
  html += '<div class="status-section-title">Controller Feedback (' + feedback.length + ')</div>';
  if (feedback.length === 0) {
    html += '<div class="status-row"><span class="status-pos">none</span></div>';
  }
  for (const row of feedback) {
    const age = Number(row.age_seconds);
    const ageLabel = Number.isFinite(age) ? age.toFixed(2) + 's' : '--';
    const robotId = row.vision_id !== undefined && row.vision_id !== null ? row.vision_id : row.port_id;
    const robotLabel = row.team_color
      ? row.team_color.charAt(0).toUpperCase() + row.team_color.slice(1) + ' Robot ' + robotId
      : 'Robot ' + robotId;
    const ballLabel = row.connected ? (row.has_ball ? 'ball: YES' : 'ball: no') : 'ball: --';
    let mapped = 'unmapped';
    if (row.vision_id !== undefined && row.vision_id !== null) {
      const bits = [];
      if (row.team) bits.push(row.team);
      if (row.team_color) bits.push(row.team_color);
      mapped = bits.join(' / ');
    }
    html += '<div class="status-row">'
      + '<span class="status-feedback-id">' + escapeHtml(robotLabel) + '</span>'
      + '<span class="status-feedback-ball ' + (row.connected && row.has_ball ? 'yes' : '') + '">'
      + escapeHtml(ballLabel) + '</span>'
      + '<span class="status-pill ' + (row.connected ? 'connected' : 'disconnected') + '">'
      + (row.connected ? 'connected' : 'no data') + '</span>'
      + '<span class="status-feedback-meta">port ' + escapeHtml(row.port_id) + ' · '
      + escapeHtml(mapped) + ' · age ' + escapeHtml(ageLabel) + '</span>'
      + '</div>';
  }

  // Ball
  html += '<div class="status-section-title">Ball</div>';
  if (d.ball) {
    html += '<div class="status-row"><span class="status-pos">('
      + d.ball.x.toFixed(3) + ', ' + d.ball.y.toFixed(3)
      + ')</span><span class="status-vel">v=('
      + d.ball.vx.toFixed(2) + ', ' + d.ball.vy.toFixed(2)
      + ')</span></div>';
  } else {
    html += '<div class="status-row"><span class="status-pos">—</span></div>';
  }

  // Friendly robots
  const friendly = (d.robots && d.robots.friendly) || [];
  const btNodes = d.bt_nodes || {};
  html += '<div class="status-section-title">Friendly (' + friendly.length + ')</div>';
  if (friendly.length === 0) {
    html += '<div class="status-row"><span class="status-pos">none</span></div>';
  }
  for (const bot of friendly) {
    html += '<div class="status-row">'
      + '<span class="status-bot-id friendly">' + bot.id + '</span>'
      + '<span class="status-pos">(' + bot.x.toFixed(3) + ', ' + bot.y.toFixed(3) + ')</span>'
      + '<span class="status-vel">v=(' + bot.vx.toFixed(2) + ', ' + bot.vy.toFixed(2) + ')</span>'
      + (bot.has_ball ? '<span class="status-ball-indicator" title="has ball"></span>' : '')
      + '</div>';
    const nodes = btNodes[bot.id];
    if (nodes && nodes.length > 0) {
      html += '<div class="status-bt">' + nodes.join(' › ') + '</div>';
    }
  }

  // Enemy robots
  const enemy = (d.robots && d.robots.enemy) || [];
  html += '<div class="status-section-title">Enemy (' + enemy.length + ')</div>';
  if (enemy.length === 0) {
    html += '<div class="status-row"><span class="status-pos">none</span></div>';
  }
  for (const bot of enemy) {
    html += '<div class="status-row">'
      + '<span class="status-bot-id enemy">' + bot.id + '</span>'
      + '<span class="status-pos">(' + bot.x.toFixed(3) + ', ' + bot.y.toFixed(3) + ')</span>'
      + '<span class="status-vel">v=(' + bot.vx.toFixed(2) + ', ' + bot.vy.toFixed(2) + ')</span>'
      + '</div>';
  }

  c.innerHTML = html;
}

const GAME_FRAME_LOG_INTERVAL_MS = 2000;
const GAME_FRAME_LOG_LIMIT = 8;
let _lastGameFrameLoggedAt = 0;

function _gameFrameLogLabel(frame) {
  const ts = Number(frame && frame.ts);
  if (!Number.isFinite(ts)) return 'ts=—';

  const date = new Date(ts * 1000);
  const looksLikeEpochSeconds =
    !Number.isNaN(date.getTime()) && date.getFullYear() >= 2020 && date.getFullYear() <= 2100;
  const tsLabel = 'ts=' + ts.toFixed(3);
  return looksLikeEpochSeconds
    ? date.toLocaleTimeString() + ' · ' + tsLabel
    : tsLabel;
}

function logGameFrame(d) {
  const frame = d.game_frame;
  const c = document.getElementById('game-frame-log');
  if (!c || !frame) return;

  const now = Date.now();
  if (_lastGameFrameLoggedAt && now - _lastGameFrameLoggedAt < GAME_FRAME_LOG_INTERVAL_MS) return;
  _lastGameFrameLoggedAt = now;

  const entry = document.createElement('div');
  entry.className = 'game-frame-entry';

  const head = document.createElement('div');
  head.className = 'game-frame-entry-head';
  head.textContent = _gameFrameLogLabel(frame);

  const body = document.createElement('pre');
  body.textContent = JSON.stringify(frame, null, 2);

  entry.appendChild(head);
  entry.appendChild(body);
  c.prepend(entry);

  while (c.children.length > GAME_FRAME_LOG_LIMIT) {
    c.removeChild(c.lastElementChild);
  }
}

// --- Canvas globals ---
let _cfg = null, _lastFrame = {};
let _currentCmd = null;
let _myTeamIsRight = false;
let _myTeamIsYellow = true;

function _fieldView(g) {
  const goalDepth = Math.max(0, Number(g.goal_depth) || 0);
  return {
    minX: -g.half_length - goalDepth,
    maxX:  g.half_length + goalDepth,
    minY: -g.half_width,
    maxY:  g.half_width,
    width: 2 * (g.half_length + goalDepth),
    height: 2 * g.half_width,
    goalDepth,
  };
}

function _fieldTransform(canvas) {
  if (!canvas || !_cfg) return null;
  const view = _fieldView(_cfg);
  const M = 12;
  const usableW = Math.max(1, canvas.width - 2 * M);
  const usableH = Math.max(1, canvas.height - 2 * M);
  const scale = Math.min(usableW / view.width, usableH / view.height);
  const originX = (canvas.width - view.width * scale) / 2;
  const originY = (canvas.height - view.height * scale) / 2;

  return {
    scale,
    toX(fx) {
      const displayX = _myTeamIsRight ? fx : -fx;
      return originX + (displayX - view.minX) * scale;
    },
    toY(fy) {
      return originY + (view.maxY - fy) * scale;
    },
    toField(px, py) {
      let fx = view.minX + (px - originX) / scale;
      if (!_myTeamIsRight) fx = -fx;
      const fy = view.maxY - (py - originY) / scale;
      return { x: fx, y: fy };
    },
  };
}

function _yellowIsRight() {
  return _myTeamIsRight === _myTeamIsYellow;
}

function resizeCanvas() {
  const canvas = document.getElementById('field-canvas');
  if (!canvas || !_cfg) return;
  const wrap = canvas.parentElement;
  const cw = wrap.clientWidth;
  const ch = wrap.clientHeight;
  const view = _fieldView(_cfg);
  const fieldAspect = view.width / view.height;
  const wrapAspect  = cw / ch;
  let pw, ph;
  if (wrapAspect > fieldAspect) {
    // container is wider than the field — constrain by height
    ph = ch;
    pw = Math.round(ch * fieldAspect);
  } else {
    // container is taller — constrain by width
    pw = cw;
    ph = Math.round(cw / fieldAspect);
  }
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width  = pw;
    canvas.height = ph;
  }
  // Centre the canvas inside the wrap
  canvas.style.left = Math.round((cw - pw) / 2) + 'px';
  canvas.style.top  = Math.round((ch - ph) / 2) + 'px';
  canvas.style.width  = pw + 'px';
  canvas.style.height = ph + 'px';
  drawField(_lastFrame);
}

function initCanvas(g) {
  _cfg = g;
  resizeCanvas();
  new ResizeObserver(resizeCanvas).observe(
    document.getElementById('field-canvas').parentElement
  );
}

function drawField(d) {
  const canvas = document.getElementById('field-canvas');
  if (!canvas || !_cfg) return;
  const ctx = canvas.getContext('2d');
  const g = _cfg;
  const CW = canvas.width, CH = canvas.height;
  const tx = _fieldTransform(canvas);
  if (!tx) return;
  const scale = tx.scale;
  const toX = tx.toX;
  const toY = tx.toY;

  function rectBounds(x1, y1, x2, y2) {
    const px1 = toX(x1), px2 = toX(x2);
    const py1 = toY(y1), py2 = toY(y2);
    return {
      x: Math.min(px1, px2),
      y: Math.min(py1, py2),
      w: Math.abs(px2 - px1),
      h: Math.abs(py2 - py1),
    };
  }
  function strokeWorldRect(x1, y1, x2, y2) {
    const r = rectBounds(x1, y1, x2, y2);
    ctx.strokeRect(r.x, r.y, r.w, r.h);
  }
  function fillWorldRect(x1, y1, x2, y2) {
    const r = rectBounds(x1, y1, x2, y2);
    ctx.fillRect(r.x, r.y, r.w, r.h);
  }

  // Green background
  ctx.fillStyle = '#2d7a2d';
  ctx.fillRect(0, 0, CW, CH);

  // Field boundary
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1.5;
  strokeWorldRect(-g.half_length, -g.half_width, g.half_length, g.half_width);

  // Centre line
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(-g.half_width));
  ctx.lineTo(toX(0), toY(g.half_width));
  ctx.stroke();

  // Centre circle
  ctx.beginPath();
  ctx.arc(toX(0), toY(0), g.center_circle_radius * scale, 0, 2 * Math.PI);
  ctx.stroke();

  // Centre dot
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(toX(0), toY(0), 2, 0, 2 * Math.PI);
  ctx.fill();

  // Defence areas — full depth = 2 * half_defense_depth (matches physical model)
  const dl = 2 * g.half_defense_depth, dw = g.half_defense_width;
  // Left (negative x)
  strokeWorldRect(-g.half_length, -dw, -g.half_length + dl, dw);
  // Right (positive x)
  strokeWorldRect(g.half_length - dl, -dw, g.half_length, dw);

  // Goal bars (depth from geometry, outside field boundary)
  const goalDepth = Math.max(0, Number(g.goal_depth) || 0);
  const yellowGoal = 'rgba(244,197,66,0.6)';
  const blueGoal = 'rgba(77,166,255,0.6)';
  ctx.fillStyle = _yellowIsRight() ? blueGoal : yellowGoal;
  fillWorldRect(-g.half_length - goalDepth, -g.half_goal_width,
                -g.half_length, g.half_goal_width);
  ctx.fillStyle = _yellowIsRight() ? yellowGoal : blueGoal;
  fillWorldRect(g.half_length, -g.half_goal_width,
                g.half_length + goalDepth, g.half_goal_width);

  // Designated position marker
  if (d.designated) {
    const dx = toX(d.designated[0]), dy = toY(d.designated[1]);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    const s = 5;
    ctx.beginPath();
    ctx.moveTo(dx - s, dy - s); ctx.lineTo(dx + s, dy + s);
    ctx.moveTo(dx + s, dy - s); ctx.lineTo(dx - s, dy + s);
    ctx.stroke();
  }

  // Robots
  const robots = d.robots;
  if (robots) {
    const r = 5; // robot radius px
    // Color depends on which team is yours
    const friendlyColor = _myTeamIsYellow ? '#f4c542' : '#4da6ff';
    const friendlyStroke = _myTeamIsYellow ? '#111' : '#fff';
    const enemyColor = _myTeamIsYellow ? '#4da6ff' : '#f4c542';
    const enemyStroke = _myTeamIsYellow ? '#fff' : '#111';
    // Enemy
    for (const bot of (robots.enemy || [])) {
      const cx = toX(bot.x), cy = toY(bot.y);
      ctx.fillStyle = enemyColor;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2*Math.PI); ctx.fill();
      ctx.strokeStyle = enemyStroke; ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + r * Math.cos(bot.orientation), cy - r * Math.sin(bot.orientation));
      ctx.stroke();
      ctx.fillStyle = enemyStroke;
      ctx.font = '7px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(bot.id, cx, cy - r - 1);
    }
    // Friendly
    for (const bot of (robots.friendly || [])) {
      const cx = toX(bot.x), cy = toY(bot.y);
      ctx.fillStyle = friendlyColor;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2*Math.PI); ctx.fill();
      ctx.strokeStyle = friendlyStroke; ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + r * Math.cos(bot.orientation), cy - r * Math.sin(bot.orientation));
      ctx.stroke();
      ctx.fillStyle = friendlyStroke;
      ctx.font = '7px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(bot.id, cx, cy - r - 1);
    }
  }

  // Ball
  if (d.ball) {
    const bx = toX(d.ball.x), by = toY(d.ball.y);
    const br = Math.max(4, 3);
    ctx.fillStyle = '#e67e22';
    ctx.beginPath(); ctx.arc(bx, by, br, 0, 2*Math.PI); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 0.8;
    ctx.stroke();
  }
}

// --- SSE ---
const es = new EventSource('/events');

es.onopen = () => {
  document.getElementById('conn-dot').classList.add('live');
  document.getElementById('conn-label').textContent = 'live';
};
es.onerror = () => {
  document.getElementById('conn-dot').classList.remove('live');
  document.getElementById('conn-label').textContent = 'disconnected — retrying…';
};
es.onmessage = (ev) => {
  const d = JSON.parse(ev.data);

  document.getElementById('yellow-score').textContent = d.yellow_score ?? '—';
  document.getElementById('blue-score').textContent   = d.blue_score   ?? '—';

  const badge = document.getElementById('cmd-badge');
  badge.textContent = (d.command ?? '—').replace(/_/g, ' ');
  badge.className = 'badge ' + (d.command ?? 'unknown');
  _currentCmd = d.command;

  document.getElementById('next-cmd').textContent =
    d.next_command ? d.next_command.replace(/_/g, ' ') : '—';
  document.getElementById('stage').textContent =
    d.stage ? d.stage.replace(/_/g, ' ') : '—';

  const secs = d.stage_time_left;
  if (secs != null && secs > 0) {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    document.getElementById('stage-time').textContent =
      m + ':' + String(s).padStart(2, '0');
  } else {
    document.getElementById('stage-time').textContent = '';
  }

  document.getElementById('designated').textContent = d.designated
    ? '(' + d.designated[0].toFixed(2) + ', ' + d.designated[1].toFixed(2) + ')'
    : '—';

  // Status message
  if (d.status_message) {
    document.getElementById('status-row').style.display = '';
    document.getElementById('status-msg').textContent = d.status_message;
  } else {
    document.getElementById('status-row').style.display = 'none';
  }

  // Update team orientation and context menu labels
  let teamLayoutChanged = false;
  if (d.my_team_is_right !== undefined && d.my_team_is_right !== _myTeamIsRight) {
    _myTeamIsRight = d.my_team_is_right;
    teamLayoutChanged = true;
  }
  if (d.my_team_is_yellow !== undefined && d.my_team_is_yellow !== _myTeamIsYellow) {
    _myTeamIsYellow = d.my_team_is_yellow;
    teamLayoutChanged = true;
  }
  if (teamLayoutChanged) _updateCtxMenuLabels();

  // Robot status panel
  renderStatus(d);
  logGameFrame(d);

  // Canvas update
  _lastFrame = d;
  if (_cfg) drawField(d);
};

function send(command) {
  fetch('/command', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command }),
  }).catch(err => console.error('command error:', err));
}


function pill(val) {
  if (val === true)  return '<span class="pill on">ON</span>';
  if (val === false) return '<span class="pill off">OFF</span>';
  return val;
}
function cfgRow(key, val) {
  return `<div class="cfg-row"><span class="cfg-key">${key}</span><span class="cfg-val">${pill(val)}</span></div>`;
}
function cfgSection(title, rows) {
  return `<div class="config-section"><div class="config-section-title">${title}</div>${rows}</div>`;
}

// --- God mode ---
let _godMode = false;
let _ctxFieldPos = null; // {x, y} in field coords at last right-click

function toggleGod() {
  _godMode = !_godMode;
  const btn = document.getElementById('god-btn');
  btn.classList.toggle('active', _godMode);
  btn.title = _godMode
    ? 'God mode ON — right-click on the field canvas to move the ball.'
    : 'God mode: right-click anywhere on the field to move the ball or issue ball placement.';
}

function _canvasToField(canvas, clientX, clientY) {
  if (!_cfg) return null;
  const rect = canvas.getBoundingClientRect();
  const cw = canvas.width, ch = canvas.height;
  const tx = _fieldTransform(canvas);
  if (!tx) return null;
  const px = (clientX - rect.left) * (cw / rect.width);
  const py = (clientY - rect.top)  * (ch / rect.height);
  return tx.toField(px, py);
}

function _hideCtxMenu() {
  document.getElementById('ctx-menu').style.display = 'none';
}

document.addEventListener('click', _hideCtxMenu);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') _hideCtxMenu();
  // Space → HALT, or FORCE_START if already in HALT
  if (e.key === ' ') {
    e.preventDefault();
    if (_currentCmd === 'HALT') {
      send('FORCE_START');
      const btn = document.querySelector('.btn-force-start');
      if (btn) { btn.style.boxShadow = '0 0 12px 4px rgba(46,204,113,.8)'; setTimeout(() => btn.style.boxShadow = '', 300); }
    } else {
      send('HALT');
      const btn = document.querySelector('.btn-halt');
      if (btn) { btn.style.boxShadow = '0 0 12px 4px rgba(231,76,60,.8)'; setTimeout(() => btn.style.boxShadow = '', 300); }
    }
  }
});

document.getElementById('field-canvas').addEventListener('contextmenu', function(e) {
  if (!_godMode) return;
  e.preventDefault();
  _ctxFieldPos = _canvasToField(this, e.clientX, e.clientY);
  if (!_ctxFieldPos) return;
  const menu = document.getElementById('ctx-menu');
  menu.style.display = 'block';
  const mx = Math.min(e.clientX, window.innerWidth  - menu.offsetWidth  - 4);
  const my = Math.min(e.clientY, window.innerHeight - menu.offsetHeight - 4);
  menu.style.left = mx + 'px';
  menu.style.top  = my + 'px';
});

function _updateCtxMenuLabels() {
  const yellowIsRight = _yellowIsRight();
  const leftTeam  = yellowIsRight ? 'Blue'   : 'Yellow';
  const rightTeam = yellowIsRight ? 'Yellow' : 'Blue';
  document.getElementById('ctx-place-left').textContent  = 'Ball placement ' + leftTeam  + ' here';
  document.getElementById('ctx-place-right').textContent = 'Ball placement ' + rightTeam + ' here';
}

function ctxPlace(side) {
  _hideCtxMenu();
  if (!_ctxFieldPos) return;
  const yellowIsRight = _yellowIsRight();
  const leftCmd  = yellowIsRight ? 'BALL_PLACEMENT_BLUE'   : 'BALL_PLACEMENT_YELLOW';
  const rightCmd = yellowIsRight ? 'BALL_PLACEMENT_YELLOW' : 'BALL_PLACEMENT_BLUE';
  const command = (side === 'left') ? leftCmd : rightCmd;
  fetch('/command', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command, designated: [_ctxFieldPos.x, _ctxFieldPos.y] }),
  }).catch(err => console.error('command error:', err));
}

_updateCtxMenuLabels();

fetch('/config').then(r => r.json()).then(c => {
  document.getElementById('page-title').textContent = 'Custom Referee — ' + c.profile_name;
  document.getElementById('cfg-title').textContent  = 'Profile: ' + c.profile_name;

  const g = c.geometry;
  const geomRows =
    cfgRow('half length',           g.half_length          + ' m') +
    cfgRow('half width',            g.half_width           + ' m') +
    cfgRow('half goal width',       g.half_goal_width      + ' m') +
    cfgRow('defense length (half)', g.half_defense_depth  + ' m') +
    cfgRow('defense width (half)',  g.half_defense_width   + ' m') +
    cfgRow('centre circle r',       g.center_circle_radius + ' m');

  const gm = c.game;
  const gameRows =
    cfgRow('half duration',          Math.round(gm.half_duration_seconds / 60) + ' min') +
    cfgRow('kickoff team',           gm.kickoff_team) +
    cfgRow('force start after goal', gm.force_start_after_goal) +
    cfgRow('stop duration',          gm.stop_duration_seconds + ' s');

  const aa = gm.auto_advance || {};
  const aaRows =
    cfgRow('stop → next command',      aa.stop_to_next_command) +
    cfgRow('prepare kickoff → normal', aa.prepare_kickoff_to_normal) +
    cfgRow('prepare penalty → normal', aa.prepare_penalty_to_normal) +
    cfgRow('direct free → normal',     aa.direct_free_to_normal) +
    cfgRow('ball placement → next',    aa.ball_placement_to_next) +
    cfgRow('normal start → force',     aa.normal_start_to_force);

  const r = c.rules;
  const goalRows =
    cfgRow('enabled',  r.goal_detection.enabled) +
    cfgRow('cooldown', r.goal_detection.cooldown_seconds + ' s');
  const oobRows =
    cfgRow('enabled',  r.out_of_bounds.enabled) +
    cfgRow('assigner', r.out_of_bounds.free_kick_assigner);
  const daRows =
    cfgRow('enabled',        r.defense_area.enabled) +
    cfgRow('max defenders',  r.defense_area.max_defenders) +
    cfgRow('attacker foul',  r.defense_area.attacker_infringement);
  const koRows =
    cfgRow('enabled',     r.keep_out.enabled) +
    cfgRow('radius',      r.keep_out.radius_meters + ' m') +
    cfgRow('persistence', r.keep_out.violation_persistence_frames + ' frames');

  document.getElementById('cfg-grid').innerHTML =
    cfgSection('Field geometry', geomRows) +
    cfgSection('Game settings',  gameRows) +
    cfgSection('Auto-advance',   aaRows)   +
    cfgSection('Goal detection', goalRows) +
    cfgSection('Out of bounds',  oobRows)  +
    cfgSection('Defense area',   daRows)   +
    cfgSection('Keep-out zone',  koRows);

  initCanvas(c.geometry);
}).catch(err => {
  document.getElementById('cfg-title').textContent = 'Profile config unavailable';
  console.error('config fetch error:', err);
});
</script>
</body>
</html>
"""

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
        self._config_json = _build_config_json(profile)

        self._lock = threading.Lock()
        self._ref_data = None
        self._game_frame = None  # Optional[GameFrame]
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

    # ---- called by external loops to push a new state snapshot ----

    def notify(self, ref_data, game_frame=None) -> None:
        """Push a RefereeData snapshot from an external game loop."""
        with self._lock:
            self._ref_data = ref_data
            self._game_frame = game_frame
        self._broadcast()

    # ---- SSE broadcast ----

    def _broadcast(self) -> None:
        with self._lock:
            data = self._ref_data
            frame = self._game_frame
        if data is None:
            return

        payload = ("data: " + _serialise_state(data, frame) + "\n\n").encode()
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
                body = server_instance._config_json.encode()
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
                    with server_instance._lock:
                        server_instance._referee.set_command(cmd, time.time())
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
        return [{"id": r.id, "x": r.p.x, "y": r.p.y, "orientation": r.orientation} for r in robots_dict.values()]

    return {"friendly": _robot_list(game_frame.friendly_robots), "enemy": _robot_list(game_frame.enemy_robots)}


def _serialise_ball(game_frame):
    if game_frame is None or game_frame.ball is None:
        return None
    return {"x": game_frame.ball.p.x, "y": game_frame.ball.p.y}


def _serialise_state(ref_data, game_frame=None) -> str:
    designated = None
    if ref_data.designated_position is not None:
        try:
            designated = list(ref_data.designated_position)
        except TypeError:
            designated = [ref_data.designated_position.x, ref_data.designated_position.y]

    return json.dumps(
        {
            "command": ref_data.referee_command.name,
            "next_command": ref_data.next_command.name if ref_data.next_command else None,
            "stage": ref_data.stage.name,
            "stage_time_left": ref_data.stage_time_left or 0.0,
            "yellow_score": ref_data.yellow_team.score,
            "blue_score": ref_data.blue_team.score,
            "designated": designated,
            "status_message": ref_data.status_message,
            "robots": _serialise_robots(game_frame),
            "ball": _serialise_ball(game_frame),
        }
    )


def _build_config_json(profile: "RefereeProfile") -> str:
    g = profile.geometry
    r = profile.rules
    gm = profile.game
    return json.dumps(
        {
            "profile_name": profile.profile_name,
            "geometry": {
                "half_length": g.half_length,
                "half_width": g.half_width,
                "half_goal_width": g.half_goal_width,
                "half_defense_length": g.half_defense_length,
                "half_defense_width": g.half_defense_width,
                "center_circle_radius": g.center_circle_radius,
            },
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
    )


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

  body {
    background: var(--bg);
    color: var(--text);
    font-family: ui-monospace, "Cascadia Code", "Fira Code", monospace;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px 16px;
    gap: 20px;
  }

  h1 {
    font-size: 1.4rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--text);
    opacity: .9;
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    width: 100%;
    max-width: 640px;
    overflow: hidden;
  }

  .panel-title {
    padding: 10px 20px;
    font-size: .7rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
  }

  .scoreboard {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-bottom: 1px solid var(--border);
  }

  .score-cell {
    padding: 18px 24px;
    text-align: center;
  }
  .score-cell:first-child { border-right: 1px solid var(--border); }

  .team-name {
    font-size: .75rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .team-name.yellow { color: var(--yellow); }
  .team-name.blue   { color: var(--blue); }

  .score-value {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
  }
  .score-value.yellow { color: var(--yellow); }
  .score-value.blue   { color: var(--blue); }

  .info-block {
    padding: 16px 24px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .info-row {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: .9rem;
  }

  .label {
    color: var(--muted);
    min-width: 90px;
    font-size: .75rem;
    text-transform: uppercase;
    letter-spacing: .05em;
    flex-shrink: 0;
  }

  .badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: .85rem;
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

  .config-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
  }

  .config-section {
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
  }
  .config-section:nth-child(odd)      { border-right: 1px solid var(--border); }
  .config-section:nth-last-child(-n+2) { border-bottom: none; }

  .config-section-title {
    font-size: .68rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
  }

  .cfg-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: .8rem;
    padding: 2px 0;
    gap: 8px;
  }

  .cfg-key { color: var(--muted); flex-shrink: 0; }
  .cfg-val { text-align: right; word-break: break-word; }

  .pill {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 3px;
    font-size: .75rem;
    font-weight: 600;
  }
  .pill.on  { background: #1a3a1a; color: var(--green); border: 1px solid #2a5a2a; }
  .pill.off { background: #3a1a1a; color: #c0605a;      border: 1px solid #5a2a2a; }

  .buttons {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    width: 100%;
    max-width: 640px;
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .btn-row { display: flex; flex-wrap: wrap; gap: 8px; }

  button {
    padding: 8px 16px;
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-family: inherit;
    font-size: .8rem;
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
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: .75rem;
    color: var(--muted);
  }
  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #555;
    transition: background .3s;
  }
  .dot.live { background: var(--green); }

  /* Phase 2 additions */
  .viz-row { display:flex; gap:16px; width:100%; max-width:640px; align-items:flex-start; }
  .field-panel { flex:1 1 0; min-width:0; overflow:hidden; }
  #field-canvas { display:block; width:100%; height:auto; }
  .log-panel { flex:1 1 0; min-width:0; display:flex; flex-direction:column; max-height:260px; }
  .log-entries { overflow-y:auto; flex:1; padding:8px 12px; display:flex; flex-direction:column; gap:4px; }
  .log-entry { font-size:.72rem; line-height:1.4; border-bottom:1px solid var(--border); padding-bottom:3px; }
  .log-time { color:var(--muted); margin-right:6px; }
  .log-cmd  { color:var(--green); }
  .log-score{ color:var(--yellow); }
  .log-msg  { color:var(--text); opacity:.8; }
  .adv-toggle { display:flex; align-items:center; gap:8px; font-size:.75rem; color:var(--muted); padding:4px 0 0 0; cursor:pointer; user-select:none; }
  .adv-toggle input { cursor:pointer; }
  .adv-row { display:none; }
  .adv-row.visible { display:flex; }
</style>
</head>
<body>

<h1 id="page-title">Custom Referee</h1>

<div class="panel">
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
      <span id="status-msg" style="color:var(--muted);font-size:.8rem;"></span>
    </div>
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
  <label class="adv-toggle">
    <input type="checkbox" id="adv-toggle" onchange="toggleAdvanced(this.checked)">
    Advanced controls (penalty &amp; ball placement)
  </label>
  <div class="btn-row adv-row">
    <button class="btn-yellow" onclick="send('PREPARE_PENALTY_YELLOW')"
      title="(Advanced) Award Yellow a penalty. Usually auto-detected. Follow with Normal Start.">Penalty Yellow</button>
    <button class="btn-blue"   onclick="send('PREPARE_PENALTY_BLUE')"
      title="(Advanced) Award Blue a penalty. Usually auto-detected. Follow with Normal Start.">Penalty Blue</button>
  </div>
  <div class="btn-row adv-row">
    <button class="btn-yellow" onclick="send('BALL_PLACEMENT_YELLOW')"
      title="(Advanced) Manually command Yellow to place ball at designated position.">Ball Placement Yellow</button>
    <button class="btn-blue"   onclick="send('BALL_PLACEMENT_BLUE')"
      title="(Advanced) Manually command Blue to place ball at designated position.">Ball Placement Blue</button>
  </div>
</div>

<div class="viz-row">
  <div class="panel field-panel">
    <div class="panel-title">Field</div>
    <canvas id="field-canvas"></canvas>
  </div>
  <div class="panel log-panel">
    <div class="panel-title">Event Log</div>
    <div class="log-entries" id="log-entries"></div>
  </div>
</div>

<div class="panel" id="cfg-panel">
  <div class="panel-title" id="cfg-title">Profile — loading…</div>
  <div class="config-grid" id="cfg-grid"></div>
</div>

<div class="conn">
  <div class="dot" id="conn-dot"></div>
  <span id="conn-label">connecting…</span>
</div>

<script>
// --- Event log state ---
const MAX_LOG = 20;
let _prevCmd = null, _prevYS = null, _prevBS = null, _prevStatusMsg = null;

function _now() {
  return new Date().toLocaleTimeString('en-GB', {hour12:false});
}
function addLog(cssClass, text) {
  const c = document.getElementById('log-entries');
  const e = document.createElement('div');
  e.className = 'log-entry';
  e.innerHTML = `<span class="log-time">${_now()}</span><span class="${cssClass}">${text}</span>`;
  c.insertBefore(e, c.firstChild);
  while (c.children.length > MAX_LOG) c.removeChild(c.lastChild);
}

// --- Canvas globals ---
let _cfg = null, _lastFrame = {};
function initCanvas(g) {
  _cfg = g;
  const canvas = document.getElementById('field-canvas');
  const CW = 400;
  const fieldW = 2 * g.half_length;
  const fieldH = 2 * g.half_width;
  canvas.width = CW;
  canvas.height = Math.round(CW * fieldH / fieldW);
  drawField(_lastFrame);
}

function drawField(d) {
  const canvas = document.getElementById('field-canvas');
  if (!canvas || !_cfg) return;
  const ctx = canvas.getContext('2d');
  const g = _cfg;
  const CW = canvas.width, CH = canvas.height;
  const M = 12;
  const scale = (CW - 2 * M) / (2 * g.half_length);

  function toX(fx) { return M + (fx + g.half_length) * scale; }
  function toY(fy) { return CH - M - (fy + g.half_width) * scale; }

  // Green background
  ctx.fillStyle = '#2d7a2d';
  ctx.fillRect(0, 0, CW, CH);

  // Field boundary
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(toX(-g.half_length), toY(g.half_width),
                 2 * g.half_length * scale, 2 * g.half_width * scale);

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

  // Defence areas
  const dl = g.half_defense_length, dw = g.half_defense_width;
  // Left (negative x)
  ctx.strokeRect(toX(-g.half_length), toY(dw), dl * scale, 2 * dw * scale);
  // Right (positive x)
  ctx.strokeRect(toX(g.half_length - dl), toY(dw), dl * scale, 2 * dw * scale);

  // Goal bars (10 cm deep, outside field boundary)
  const goalDepth = 0.1;
  // Left goal (yellow)
  ctx.fillStyle = 'rgba(244,197,66,0.6)';
  ctx.fillRect(toX(-g.half_length - goalDepth), toY(g.half_goal_width),
               goalDepth * scale, 2 * g.half_goal_width * scale);
  // Right goal (blue)
  ctx.fillStyle = 'rgba(77,166,255,0.6)';
  ctx.fillRect(toX(g.half_length), toY(g.half_goal_width),
               goalDepth * scale, 2 * g.half_goal_width * scale);

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
    // Enemy (blue)
    for (const bot of (robots.enemy || [])) {
      const cx = toX(bot.x), cy = toY(bot.y);
      ctx.fillStyle = '#4da6ff';
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2*Math.PI); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + r * Math.cos(bot.orientation), cy - r * Math.sin(bot.orientation));
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.font = '7px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(bot.id, cx, cy - r - 1);
    }
    // Friendly (yellow)
    for (const bot of (robots.friendly || [])) {
      const cx = toX(bot.x), cy = toY(bot.y);
      ctx.fillStyle = '#f4c542';
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2*Math.PI); ctx.fill();
      ctx.strokeStyle = '#111'; ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + r * Math.cos(bot.orientation), cy - r * Math.sin(bot.orientation));
      ctx.stroke();
      ctx.fillStyle = '#111';
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
    if (d.status_message !== _prevStatusMsg) { addLog('log-msg', d.status_message); _prevStatusMsg = d.status_message; }
  } else {
    document.getElementById('status-row').style.display = 'none';
  }

  // Event log tracking
  if (_prevCmd !== null && d.command !== _prevCmd) addLog('log-cmd', d.command.replace(/_/g,' '));
  _prevCmd = d.command;
  if (_prevYS !== null && d.yellow_score !== _prevYS) addLog('log-score','Yellow '+d.yellow_score);
  _prevYS = d.yellow_score;
  if (_prevBS !== null && d.blue_score !== _prevBS) addLog('log-score','Blue '+d.blue_score);
  _prevBS = d.blue_score;

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

function toggleAdvanced(on) {
  document.querySelectorAll('.adv-row').forEach(el =>
    el.classList.toggle('visible', on));
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

fetch('/config').then(r => r.json()).then(c => {
  document.getElementById('page-title').textContent = 'Custom Referee — ' + c.profile_name;
  document.getElementById('cfg-title').textContent  = 'Profile: ' + c.profile_name;

  const g = c.geometry;
  const geomRows =
    cfgRow('half length',           g.half_length          + ' m') +
    cfgRow('half width',            g.half_width           + ' m') +
    cfgRow('half goal width',       g.half_goal_width      + ' m') +
    cfgRow('defense length (half)', g.half_defense_length  + ' m') +
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

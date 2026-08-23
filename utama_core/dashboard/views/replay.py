"""Replay view — lists and serves recorded `.pkl` match replays.

Fully standalone: reads files under `REPLAY_BASE_PATH` only, no coupling to
a running match. `/replay/list` enumerates available files; `/replay/frames`
loads one file fully (a full-length match replay is small enough — tens of
thousands of small dataclasses — that full-load-then-serialize is simpler
than a seek-aware streaming reader, same tradeoff `replay_player.py`'s own
`load_frames_in_range` already makes) and returns every frame pre-serialized
into the same JSON shape `dashboard.views.referee` produces, so the shared
`FieldCanvas.draw()` on the frontend needs no replay-specific branch.

If a sibling `<match_tag>.intentions.jsonl` sits next to the `.pkl` (written
by `full_match_tournament.py` via `StrategyRunner(match_log_path=...)`), its
sparse `IntentionEvent`/`RefereeEvent` rows are shipped to the browser
as-is — `tactic_events`/`referee_events` — rather than forward-filled onto
every frame here. Frame counts run into the tens of thousands while event
counts stay in the low hundreds (one row per assignment/state *change*, not
per tick), so pre-expanding server-side would mean copying the same handful
of dicts onto thousands of frames just to ship them over the wire. The
frontend (`replay.js`) does the same "last event whose sim_time <= current
ts wins" forward-fill this module used to do, just at scrub/playback time
instead of at load time — cheap given the event counts involved, and it
means a backward scrub is just a cursor reset + cheap replay rather than a
second algorithm.

`ReplayMetadata` (see `replay/entities.py`) does not carry field geometry, so
there is no ground truth for it in the replay file itself. `/replay/frames`
always reports `STANDARD_FIELD_DIMS` — correct for tournament/normal-match
replays, wrong for anything recorded on a non-standard field (e.g. the
Exhibition Road demo's 4x3m field). Fixing that properly needs geometry
written into `ReplayMetadata` at record time, which is out of scope here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.dashboard.server import DashboardServer
from utama_core.dashboard.views.referee import _serialise_ball, _serialise_robots
from utama_core.engine.match_log import IntentionEvent, RefereeEvent, load_jsonl
from utama_core.entities.game.game_frame import GameFrame
from utama_core.replay.replay_player import _load_replay

_DEFAULT_GEOMETRY = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)


def attach(server: DashboardServer) -> None:
    server.add_route("/replay/list", _list_bytes)
    server.add_route("/replay/frames", _frames_bytes)


def _list_bytes() -> bytes:
    return json.dumps(_list_replays()).encode()


def _list_replays() -> List[str]:
    if not REPLAY_BASE_PATH.exists():
        return []
    return sorted(str(p.relative_to(REPLAY_BASE_PATH)) for p in REPLAY_BASE_PATH.rglob("*.pkl"))


def _frames_bytes(query: Optional[dict] = None) -> bytes:
    query = query or {}
    rel_path = query.get("path")
    if not rel_path:
        return json.dumps({"error": "missing 'path' query param"}).encode()

    replay_path = (REPLAY_BASE_PATH / rel_path).resolve()
    if REPLAY_BASE_PATH.resolve() not in replay_path.parents or not replay_path.exists():
        return json.dumps({"error": "replay not found"}).encode()

    intention_events, referee_events = _load_match_log_events(replay_path)

    frames = []
    my_team_is_right = None
    my_team_is_yellow = True
    for obj in _load_replay(replay_path):
        if isinstance(obj, GameFrame):
            if my_team_is_right is None:
                my_team_is_right = obj.my_team_is_right
                my_team_is_yellow = obj.my_team_is_yellow

            frames.append(
                {
                    "ts": obj.ts,
                    "robots": _serialise_robots(obj),
                    "ball": _serialise_ball(obj),
                }
            )

    payload = {
        "my_team_is_right": bool(my_team_is_right),
        "my_team_is_yellow": my_team_is_yellow,
        "geometry": {
            "half_length": _DEFAULT_GEOMETRY.half_length,
            "half_width": _DEFAULT_GEOMETRY.half_width,
            "half_goal_width": _DEFAULT_GEOMETRY.half_goal_width,
            "half_defense_depth": _DEFAULT_GEOMETRY.half_defense_depth,
            "half_defense_width": _DEFAULT_GEOMETRY.half_defense_width,
            "center_circle_radius": _DEFAULT_GEOMETRY.center_circle_radius,
            "goal_depth": _DEFAULT_GEOMETRY.goal_depth,
        },
        "frames": frames,
        # Sparse — one row per assignment/state *change*, not per frame. The
        # frontend forward-fills these against the currently-viewed frame's
        # `ts`, mirroring what this module used to do server-side.
        "tactic_events": [
            {"sim_time": e.sim_time, "tactic_id": e.tactic_id, "robot_ids": list(e.robot_ids)} for e in intention_events
        ],
        "referee_events": [
            {
                "sim_time": e.sim_time,
                "command": e.command,
                "stage": e.stage,
                "yellow_score": e.yellow_score,
                "blue_score": e.blue_score,
                "designated": list(e.designated) if e.designated is not None else None,
            }
            for e in referee_events
        ],
        "has_tactic_data": len(intention_events) > 0,
        "has_referee_data": len(referee_events) > 0,
    }
    return json.dumps(payload).encode()


def _load_match_log_events(replay_path: Path) -> tuple[list, list]:
    """Best-effort: load a sibling `.intentions.jsonl`'s events, split by kind, each sorted by sim_time.

    Returns ([], []) if no such file exists or it fails to parse — this
    overlay is optional, absence must never break loading the replay itself.
    """
    intentions_path = replay_path.with_suffix("").with_suffix(".intentions.jsonl")
    if not intentions_path.exists():
        return [], []

    try:
        events = load_jsonl(intentions_path)
    except (OSError, json.JSONDecodeError):
        return [], []

    intention_events = sorted((e for e in events if isinstance(e, IntentionEvent)), key=lambda e: e.sim_time)
    referee_events = sorted((e for e in events if isinstance(e, RefereeEvent)), key=lambda e: e.sim_time)
    return intention_events, referee_events

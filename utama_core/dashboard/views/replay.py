"""Replay view — lists and serves recorded match replays (`.pkl` or the
faster columnar `.npz` — see `utama_core.replay.columnar_writer`).

Fully standalone: reads files under `REPLAY_BASE_PATH` only, no coupling to
a running match. `/replay/list` enumerates available files; `/replay/frames`
loads one file fully (a full-length match replay is small enough — tens of
thousands of small dataclasses — that full-load-then-serialize is simpler
than a seek-aware streaming reader, same tradeoff `replay_player.py`'s own
`load_frames_in_range` already makes) and returns every frame pre-serialized
into the same JSON shape `dashboard.views.referee` produces, so the shared
`FieldCanvas.draw()` on the frontend needs no replay-specific branch.

If a sibling `<match_tag>.intentions.jsonl` sits next to the `.pkl` (written
by a round-robin run via `StrategyRunner(match_log_path=...)`), its
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

`markers` are the moments worth jumping to, in time order: goals (a score
change in the referee events), fouls and stalls (the sibling `.stats.json`),
and the recorded team's real ball losses. Ball losses and `shots` come from
`turnover_breakdown.analyse_match` (columnar replays only), which re-reads the
replay frame by frame, several seconds; its result is kept beside the replay in
`<match>.timeline.json`, redone when the replay or `ANALYSIS_VERSION` changes.

`play` is the match told as its situations (`goal_ledger`: a restart going live
or a side winning the ball), one row each with how it ended: a goal, a stoppage
and its reason, or the other side winning the ball (with how we lost it).

`ReplayMetadata` (see `replay/entities.py`) does not carry field geometry, so
there is no ground truth for it in the replay file itself. `/replay/frames`
always reports `STANDARD_FIELD_DIMS` — correct for tournament/normal-match
replays, wrong for anything recorded on a non-standard field (e.g. the
Exhibition Road demo's 4x3m field). Fixing that properly needs geometry
written into `ReplayMetadata` at record time, which is out of scope here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from utama_core.analysis import goal_ledger
from utama_core.analysis.turnover_breakdown import analyse_match, is_real
from utama_core.config.field_params import STANDARD_FIELD_DIMS
from utama_core.config.settings import REPLAY_BASE_PATH
from utama_core.custom_referee.geometry import RefereeGeometry
from utama_core.dashboard.server import DashboardServer
from utama_core.dashboard.views.referee import _serialise_ball, _serialise_robots
from utama_core.engine.match_log import (
    IntentionEvent,
    RefereeEvent,
    TraceEvent,
    load_jsonl,
)
from utama_core.entities.game.game_frame import GameFrame
from utama_core.replay.columnar_reader import load_columnar_replay
from utama_core.replay.replay_player import _load_replay

_DEFAULT_GEOMETRY = RefereeGeometry.from_field_dims(STANDARD_FIELD_DIMS)
ANALYSIS_VERSION = 1  # bump when what `_analysis` keeps changes, to redo every sidecar
_LIVE_COMMANDS = ("NORMAL_START", "FORCE_START")


def attach(server: DashboardServer) -> None:
    server.add_route("/replay/list", _list_bytes)
    server.add_route("/replay/frames", _frames_bytes)


def _list_bytes() -> bytes:
    return json.dumps(_list_replays()).encode()


def _list_replays() -> List[str]:
    if not REPLAY_BASE_PATH.exists():
        return []
    # `*.sparse_referee.pkl` is a columnar replay's rare-field sidecar (see
    # `columnar_writer.py`), not a browsable replay on its own — it also
    # ends in ".pkl" so `rglob("*.pkl")` would otherwise list it twice
    # (once under its own name, once implicitly via its paired ".npz").
    pkl_paths = [p for p in REPLAY_BASE_PATH.rglob("*.pkl") if not p.name.endswith(".sparse_referee.pkl")]
    paths = pkl_paths + list(REPLAY_BASE_PATH.rglob("*.npz"))
    return sorted(str(p.relative_to(REPLAY_BASE_PATH)) for p in paths)


def _iter_game_frames(replay_path: Path):
    """Yield `GameFrame`s from either replay format, dispatching on
    extension — same idea as `replay_player.load_frames_in_range`."""
    if replay_path.suffix == ".npz":
        yield from load_columnar_replay(replay_path).iter_frames()
        return
    for obj in _load_replay(replay_path):
        if isinstance(obj, GameFrame):
            yield obj


def columnar_frames(replay) -> list[dict]:
    """Every tick of a columnar replay in `_serialise_robots`/`_serialise_ball`'s shape, straight
    from its arrays: building a `GameFrame` per tick first took most of a replay's load time."""

    def side(ids, p, v, orientation, has_ball):
        p, v, o = np.round(p, 3).tolist(), np.round(v, 2).tolist(), np.round(orientation, 3).tolist()
        has_ball = has_ball.tolist()
        ids = [int(i) for i in ids]
        return [
            [
                {"id": rid, "x": p[t][i][0], "y": p[t][i][1], "vx": v[t][i][0], "vy": v[t][i][1]}
                | {"orientation": o[t][i], "has_ball": bool(has_ball[t][i])}
                for i, rid in enumerate(ids)
                if not math.isnan(p[t][i][0])
            ]
            for t in range(len(p))
        ]

    friendly = side(
        replay.friendly_ids,
        replay.friendly_p,
        replay.friendly_v,
        replay.friendly_orientation,
        replay.friendly_has_ball,
    )
    enemy = side(replay.enemy_ids, replay.enemy_p, replay.enemy_v, replay.enemy_orientation, replay.enemy_has_ball)
    ball_p, ball_v = np.round(replay.ball_p, 3).tolist(), np.round(replay.ball_v, 2).tolist()
    right = replay.side_is_right.tolist() if replay.side_is_right is not None else None
    return [
        {
            "ts": round(float(ts), 4),
            "my_team_is_right": bool(right[t]) if right is not None else bool(replay.my_team_is_right),
            "robots": {"friendly": friendly[t], "enemy": enemy[t]},
            "ball": (
                None
                if math.isnan(ball_p[t][0])
                else {"x": ball_p[t][0], "y": ball_p[t][1], "vx": ball_v[t][0], "vy": ball_v[t][1]}
            ),
        }
        for t, ts in enumerate(replay.ts.tolist())
    ]


def _frames_bytes(query: Optional[dict] = None) -> bytes:
    query = query or {}
    rel_path = query.get("path")
    if not rel_path:
        return json.dumps({"error": "missing 'path' query param"}).encode()

    replay_path = (REPLAY_BASE_PATH / rel_path).resolve()
    if REPLAY_BASE_PATH.resolve() not in replay_path.parents or not replay_path.exists():
        return json.dumps({"error": "replay not found"}).encode()

    intention_events, referee_events, trace_events = _load_match_log_events(replay_path)
    analysis = _analysis(replay_path)

    frames = []
    my_team_is_right = None
    my_team_is_yellow = True
    if replay_path.suffix == ".npz":
        replay = load_columnar_replay(replay_path)
        frames = columnar_frames(replay)
        my_team_is_right, my_team_is_yellow = replay.my_team_is_right, replay.my_team_is_yellow
    else:
        for obj in _iter_game_frames(replay_path):
            if my_team_is_right is None:
                my_team_is_right = obj.my_team_is_right
                my_team_is_yellow = obj.my_team_is_yellow

            frames.append(
                {
                    "ts": round(obj.ts, 4),
                    "my_team_is_right": bool(obj.my_team_is_right),  # the teams change ends at half-time
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
            {
                "sim_time": e.sim_time,
                "tactic_id": e.tactic_id,
                "robot_ids": list(e.robot_ids),
                "tag": e.tag.value,
                "note": e.note,
            }
            for e in intention_events
        ],
        "referee_events": [
            {
                "sim_time": e.sim_time,
                "command": e.command,
                "stage": e.stage,
                "yellow_score": e.yellow_score,
                "blue_score": e.blue_score,
                "designated": list(e.designated) if e.designated is not None else None,
                "note": e.note,
            }
            for e in referee_events
        ],
        # Sparse, same forward-fill contract as tactic_events — arbitrary
        # per-tactic geometry (e.g. shadow_and_mark.marks: {marker_id:
        # opponent_id}) keyed by whatever string a tactic chose when calling
        # `ctx.match_log.trace_if_changed(...)`. The frontend doesn't
        # interpret `key`/`value` here, it just forward-fills and hands them
        # to whichever overlay renderer knows that key.
        "trace_events": [{"sim_time": e.sim_time, "key": e.key, "value": e.value} for e in trace_events],
        "markers": _markers(replay_path, referee_events, my_team_is_yellow, analysis["losses"]),
        "shots": analysis["shots"],
        "play": _play(replay_path, referee_events, analysis["losses"], frames[-1]["ts"] if frames else 0.0),
        "has_tactic_data": len(intention_events) > 0,
        "has_referee_data": len(referee_events) > 0,
    }
    return json.dumps(payload).encode()


def _load_match_log_events(replay_path: Path) -> tuple[list, list, list]:
    """Best-effort: load a sibling `.intentions.jsonl`'s events, split by kind, each sorted by sim_time.

    Returns ([], [], []) if no such file exists or it fails to parse — this
    overlay is optional, absence must never break loading the replay itself.
    """
    intentions_path = replay_path.with_suffix("").with_suffix(".intentions.jsonl")
    if not intentions_path.exists():
        return [], [], []

    try:
        events = load_jsonl(intentions_path)
    except (OSError, json.JSONDecodeError):
        return [], [], []

    intention_events = sorted((e for e in events if isinstance(e, IntentionEvent)), key=lambda e: e.sim_time)
    referee_events = sorted((e for e in events if isinstance(e, RefereeEvent)), key=lambda e: e.sim_time)
    trace_events = sorted((e for e in events if isinstance(e, TraceEvent)), key=lambda e: e.sim_time)
    return intention_events, referee_events, trace_events


def _analysis(replay_path: Path) -> dict:
    """The recorded team's real ball losses (`turnover_breakdown.is_real`) and both sides' shots
    ({t, side: friendly|enemy}), from the sidecar when it is for this replay and version."""
    empty = {"losses": [], "shots": []}
    if replay_path.suffix != ".npz":
        return empty
    sidecar = replay_path.with_name(replay_path.name[: -len(".npz")] + ".timeline.json")
    stat = replay_path.stat()
    key = {"version": ANALYSIS_VERSION, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    try:
        cached = json.loads(sidecar.read_text())
        if cached.get("key") == key:
            return cached["analysis"]
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    try:
        result = analyse_match(str(replay_path))
    except Exception:  # noqa: BLE001 - markers are optional, never break loading the replay
        return empty
    analysis = {
        "losses": [t for t in result["turnovers"] if is_real(t)],
        "shots": [{"t": s["t"], "side": s["side"]} for s in result["chances"]["shots"]],
    }
    try:
        sidecar.write_text(json.dumps({"key": key, "analysis": analysis}))
    except OSError:
        pass  # a read-only replay directory: recompute next time
    return analysis


def _play(replay_path: Path, referee_events: list, losses: list[dict], end_t: float) -> list[dict]:
    """The match's situations (columnar replays only) as `play_rows`."""
    if replay_path.suffix != ".npz":
        return []
    try:
        match = goal_ledger.match_ledger(replay_path)
    except Exception:  # noqa: BLE001 - optional, like the markers
        return []
    return play_rows(goal_ledger.situations(match), referee_events, losses, end_t)


def play_rows(starts: list[dict], referee_events: list, losses: list[dict], end_t: float) -> list[dict]:
    """One row per situation start ({t, kind, side: friendly|enemy|None, third}): `t`, `t_end`
    (the next start, or play stopping first), `side`, `kind`, `third`, and `ended`: the
    stoppage's reason (the referee's note, else its command), or the next side winning the ball,
    with how a ball we lost was lost."""
    stops = [e for e in referee_events if e.command not in _LIVE_COMMANDS]
    rows = []
    for i, s in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else None
        t_end = nxt["t"] if nxt else end_t
        stop = next((e for e in stops if s["t"] < e.sim_time <= t_end), None)
        row = {"t": s["t"], "side": s["side"], "kind": s["kind"], "third": s["third"]}
        if stop is not None:
            row["t_end"] = stop.sim_time
            row["ended"] = stop.note or stop.command.replace("_", " ").lower()
        elif nxt is not None and nxt["kind"] == "ball won":
            row["t_end"] = t_end
            row["ended"] = ("won back by " if nxt["side"] == s["side"] else "won by ") + nxt["side"]
            loss = next((x for x in losses if s["t"] < x["t"] <= t_end + 0.2), None)
            if s["side"] == "friendly" and loss is not None:
                row["ended"] += " (" + loss["kind"].replace("_", " ") + ")"
        else:
            row["t_end"] = t_end
            row["ended"] = None
        rows.append(row)
    return rows


def _markers(replay_path: Path, referee_events: list, my_team_is_yellow: bool, losses: list[dict]) -> list[dict]:
    markers = []
    yellow, blue = 0, 0
    for e in referee_events:
        if e.yellow_score > yellow or e.blue_score > blue:
            scorer = "yellow" if e.yellow_score > yellow else "blue"
            markers.append(
                {"sim_time": e.sim_time, "kind": "goal", "label": f"Goal {scorer}, {e.yellow_score}-{e.blue_score}"}
            )
        yellow, blue = e.yellow_score, e.blue_score

    stats_path = replay_path.with_suffix("").with_suffix(".stats.json")
    try:
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        stats = {}
    colour = "yellow" if my_team_is_yellow else "blue"
    other = "blue" if my_team_is_yellow else "yellow"
    fouls: dict[tuple, list[dict]] = {}  # a crash is logged once per side at one instant
    for foul in stats.get("fouls", []):
        fouls.setdefault((foul["sim_time"], foul["rule"]), []).append(foul)
    for (sim_time, rule), same in fouls.items():
        side = {"friendly": colour, "enemy": other}
        who = ", ".join(
            f"{side.get(f.get('side'), f.get('side'))} {f.get('robot_id')} {f.get('tactic') or ''}".strip()
            for f in same
        )
        markers.append({"sim_time": sim_time, "kind": "foul", "label": f"Foul: {rule.replace('_', ' ')} ({who})"})
    for stall in stats.get("stall_events", []):
        detail = f", {stall['diagnosis']}" if stall.get("diagnosis") else ""
        label = f"Stall: {stall['kind']} for {stall['duration_s']:.0f}s{detail}"
        markers.append({"sim_time": stall["sim_time"], "kind": "stall", "label": label})

    for loss in losses:
        label = f"{colour.capitalize()} lost the ball: {loss['kind'].replace('_', ' ')} ({loss['tactic']})"
        markers.append({"sim_time": loss["t"], "kind": "loss", "label": label})
    return sorted(markers, key=lambda m: m["sim_time"])

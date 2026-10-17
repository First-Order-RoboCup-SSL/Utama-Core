"""repro_from_replay.py — reload a replay's field state at a given timestamp
into a fresh headless rsim match, and tick forward with match_log tracing on.

Why this exists: finding a stall today means running a 231-match round-robin
(`round_robin.py`), spotting a suspicious match, then tracing its replay
(`render_window`/`load_frames_in_range`) to find the stall window. Once the
window is known (e.g. "ball frozen from t=264s"), reproducing it previously
meant replaying the *whole* match from t=0 in a fresh run just to get back
to that one tick with tracing enabled. This script instead builds the exact
same match `tournament.run_match` would (same profile, same robot counts,
same two strategies — read from the replay's sidecar/summary.json, or given
explicitly), teleports the field into the replay's state at `--t` seconds
(`utama_core.replay.scenario.apply_scenario`), and ticks forward for
`--duration` sim seconds with `match_log` enabled so
`ctx.match_log.trace(...)` calls already in the tactics start recording from
right before the stall — no need to re-run the first N minutes of the match
just to observe the last ten seconds of it.

All the actual scenario-loading/coordinate-frame logic lives in
`utama_core.replay.scenario` — this script is a thin CLI wrapper: parse args,
build the runner the same way `tournament.run_match` does, apply the
scenario, tick, print a summary.

Sim noise: by default it plays with the replay's own `SimNoise` (its seed and settings, stored in
the .npz), so latency, kick offsets and motion limits redraw the match's values; `--sim-noise PATH`
uses another settings file, `--sim-noise none` plays noiseless. Vision and kick-speed noise are not
replayed (their streams start afresh at `--t`).

Known limitations (see `utama_core/replay/scenario.py`'s module docstring
for the full rationale):
- Tactic `mem` (each `Tactic`'s internal per-slot state — phase, committed
  targets, timers) is not in the replay, so every repro starts with FRESH
  tactic state: the kernel re-partitions robots from scratch on the first
  tick. This reproduces stalls caused by field geometry/referee state, not
  stalls that depend on accumulated tactic history to reach.
- Robot velocities are recorded in the replay but cannot be applied — the
  sim controller's `teleport_robot` has no velocity parameter — so every
  robot resumes from rest. Ball velocity IS applied.
- The referee command is read from the frame itself when present, else from
  the `<name>.intentions.jsonl` sidecar; if neither has one, the repro
  starts with whatever the freshly-seeded `CustomReferee` defaults to.

Usage:
    pixi run python tools/repro_from_replay.py replays/tournament_.../match.npz \\
        --t 260 --duration 15 --control-scheme trajsample \\
        --trace-out /tmp/repro_trace.jsonl

`--strategy` / `--opponent` take the short name of a `build_*_kernel_strategy` factory in
`utama_core/strategy/` (e.g. `clear_press_plus`, `counter_flow`), the same names as the
replay's `summary.json` config names. The trace shows the tactics those strategies use, not
the strategy names (e.g. `decoy_and_overload.*` keys come from a strategy's DecoyOverloadTactic).

`--trace-out` (default /tmp/repro_from_replay.intentions.jsonl; write under a scratch dir instead)
is a JSON Lines file, one object per line, written when the run closes. Field `event` names the
kind; every row has `tick` (the strategy's tick count) and `sim_time` (seconds, the replay's clock):
  "intention": `tactic_id` (the kernel slot, e.g. "overload", "block", "attack"), `robot_ids` (the
               robots the slot now holds; [] = released), `tag`, `note`: on an assignment the tactic
               class name (e.g. "DecoyOverloadTactic"), on a release why, else null. Written when a
               slot's robot set changes, not every tick. No `key`.
  "trace":     `key` and `value`. Keys are `highlights.<tactic_id>` (value: {robot_id: label}),
               `committed_robot_ids`, and tactic points such as `give_and_go.pass_target`
               ({receiver_id, x, y}), `switch_of_play.runner_target` ({runner_id, x, y}),
               `clear_ball.clear_target` ({from, to}), `dribble.segment_target` ({x, y}),
               `go_to_ball[<id>].approach`. Written only when the value changes. No `tactic_id`.
  "referee":  `command`, `stage`, scores, `designated`, `note`. Written on referee-state changes.
               No `key` and no `robot_ids`.
Read it by checking `event` first; `key` exists only on "trace" rows.

`--summary ROBOT_IDS` (our side, e.g. `3` or `1,3`) prints, every 0.25 s of the run and at most
40 lines, per listed robot: `t`, `robot`, `tactic` (the tactic class holding it, from the intention
rows up to that time; the slot id if the class is unknown; `-` if none), `pos` (its position), `dist_ball` (to the ball), `ref` (the referee
command at that tick) and `target`: the latest traced point (a "trace" row with a `*_id` field
equal to the robot's id, e.g. give_and_go.pass_target's receiver_id), else `-`. Tactics that trace
points without a robot id (clear_ball, dribble, decoy_and_overload) show `-`.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # before any import: hides pygame's banner

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.match import N_OUTFIELD, OUTFIELD_ROBOT_IDS, TICKS_PER_SECOND
from utama_core.custom_referee import CustomReferee
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.match_log import load_jsonl
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.columnar_reader import load_sim_noise
from utama_core.replay.scenario import Scenario, apply_scenario, scenario_from_replay
from utama_core.rsoccer_simulator.src.Utils.sim_noise import SimNoise
from utama_core.run import StrategyRunner
from utama_core.strategy import kernel_strategy

_STILL_STALLED_WINDOW_S = 10.0
_STILL_STALLED_THRESHOLD_M = 0.05  # 5 cm


def _resolve_config_name(explicit: Optional[str], from_scenario: Optional[str], role: str) -> str:
    name = explicit or from_scenario
    if name is None:
        raise SystemExit(
            f"Could not determine the {role} strategy config: no --{'strategy' if role == 'friendly' else 'opponent'} "
            f"given and the replay's directory has no summary.json (or no matching entry) to derive it from. "
            f"Pass --strategy/--opponent explicitly."
        )
    if not name.startswith("build_"):
        name = f"build_{name}"
    if not name.endswith("_kernel_strategy"):
        name = f"{name}_kernel_strategy"
    if not hasattr(kernel_strategy, name):
        raise SystemExit(
            f"Unknown strategy config {name!r} (from {'CLI override' if explicit else 'replay metadata'})."
        )
    return name


def resolve_sim_noise(override: Optional[str], replay_noise: Optional[dict]) -> tuple[Optional[SimNoise], str]:
    """The `SimNoise` a repro plays with, and the line that says so.

    `override` is `--sim-noise`: "none" for noiseless, a path to a `SimNoise` JSON file, or None for
    the replay's own. `replay_noise` is what `load_sim_noise` returned for the replay (`{"settings":
    ..., "draw": ...}`), or None when it had none. The replay's settings carry its seed, so
    `SimNoise.draw` gives the same per-robot values the match drew.
    """
    if override == "none":
        return None, "sim noise: none (--sim-noise none)"
    if override is not None:
        noise = SimNoise.from_dict(json.loads(Path(override).read_text()))
        return noise, f"sim noise: {override} (--sim-noise, seed {noise.seed})"
    if replay_noise is None:
        return None, "sim noise: none (the replay stores none)"
    noise = SimNoise.from_dict(replay_noise["settings"])
    return noise, f"sim noise: the replay's own (seed {noise.seed})"


SUMMARY_EVERY_S = 0.25
SUMMARY_MAX_LINES = 40


def _parse_ids(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def _snapshot(runner, robot_ids: list[int]) -> dict:
    """Our robots in `robot_ids` and the ball at the runner's current frame."""
    frame = runner.my.current_game_frame
    ref_data = runner.my.game.referee
    ball = (frame.ball.p.x, frame.ball.p.y) if frame.ball is not None else None
    robots = {
        rid: (frame.friendly_robots[rid].p.x, frame.friendly_robots[rid].p.y)
        for rid in robot_ids
        if rid in frame.friendly_robots
    }
    return {
        "t": frame.ts,
        "referee": ref_data.referee_command.name if ref_data is not None else None,
        "ball": ball,
        "robots": robots,
    }


def summary_rows(samples: list[dict], robot_ids: list[int], events: list, every: float = SUMMARY_EVERY_S) -> list[dict]:
    """One row per listed robot every `every` seconds. `samples`: dicts with t, referee, ball (x, y) or
    None, and robots {id: (x, y)}, in time order. `events`: IntentionEvent-like (sim_time, tactic_id,
    robot_ids) and TraceEvent-like (sim_time, key, value) objects, any order; applied up to each row's t."""
    intentions = sorted((e for e in events if hasattr(e, "tactic_id")), key=lambda e: e.sim_time)
    traces = sorted((e for e in events if hasattr(e, "key")), key=lambda e: e.sim_time)
    holder: dict[int, str] = {}  # robot -> slot id
    classes: dict[str, str] = {}  # slot id -> tactic class, from assignment notes
    target: dict[int, str] = {}
    ii = ti = 0
    last_t: Optional[float] = None
    rows: list[dict] = []
    for s in samples:
        t = s["t"]
        if last_t is not None and t - last_t < every - 1e-9:
            continue
        last_t = t
        while ii < len(intentions) and intentions[ii].sim_time <= t:
            e = intentions[ii]
            ii += 1
            if e.robot_ids and e.note and not e.note.startswith(("released", "deadline")):
                classes[e.tactic_id] = e.note
            # The event says which robots the tactic now holds: drop the ones it lost, add the new ones.
            for rid in [r for r, tid in holder.items() if tid == e.tactic_id and r not in e.robot_ids]:
                del holder[rid]
            for rid in e.robot_ids:
                holder[rid] = e.tactic_id
        while ti < len(traces) and traces[ti].sim_time <= t:
            e = traces[ti]
            ti += 1
            value = e.value
            if e.key.endswith("target") and isinstance(value, dict) and "x" in value and "y" in value:
                for field, rid in value.items():
                    if field.endswith("_id"):
                        target[rid] = f"{e.key} ({value['x']:.2f},{value['y']:.2f})"
        for rid in robot_ids:
            pos = s["robots"].get(rid)
            ball = s["ball"]
            rows.append(
                {
                    "t": t,
                    "robot": rid,
                    "tactic": classes.get(holder[rid], holder[rid]) if rid in holder else "-",
                    "pos": pos,
                    "dist_ball": math.hypot(pos[0] - ball[0], pos[1] - ball[1]) if pos and ball else None,
                    "ref": s["referee"],
                    "target": target.get(rid, "-"),
                }
            )
    return rows


def format_summary(rows: list[dict], max_lines: int = SUMMARY_MAX_LINES) -> list[str]:
    """`summary_rows` as text lines, at most `max_lines` and then "... N more"."""
    lines = []
    for r in rows[:max_lines]:
        pos = "(  -  ,  -  )" if r["pos"] is None else f"({r['pos'][0]:+.2f},{r['pos'][1]:+.2f})"
        dist = "   -" if r["dist_ball"] is None else f"{r['dist_ball']:.2f}"
        lines.append(
            f"{r['t']:8.2f}s robot {r['robot']} {r['tactic']:<24} at {pos} ball {dist} m  "
            f"ref {r['ref'] or '-':<22} target {r['target']}"
        )
    if len(rows) > max_lines:
        lines.append(f"... {len(rows) - max_lines} more")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay_path", type=Path, help="path to a .pkl or .npz replay")
    parser.add_argument("--t", type=float, required=True, help="sim_time (seconds) to load the scenario from")
    parser.add_argument("--duration", type=float, default=15.0, help="sim seconds to tick forward (default 15)")
    parser.add_argument(
        "--control-scheme",
        default="fpp",
        help="motion control scheme for both sides, e.g. fpp, dwa, trajsample (default fpp)",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="override the friendly (config_a) strategy: the short name (e.g. clear_press_plus) or the factory "
        "name (e.g. build_clear_press_plus_kernel_strategy); default: the replay's summary.json",
    )
    parser.add_argument(
        "--opponent",
        default=None,
        help="override the enemy (config_b) strategy, same two name forms (e.g. counter_flow); default: the "
        "replay's summary.json",
    )
    parser.add_argument(
        "--trace-out", default="/tmp/repro_from_replay.intentions.jsonl", help="where to write the match_log JSONL"
    )
    parser.add_argument("--print-trace", action="store_true", help="print every TraceEvent as it's written back")
    parser.add_argument(
        "--sim-noise",
        default=None,
        metavar="PATH|none",
        help="sim noise to play with: none (noiseless), or a SimNoise JSON file; default: the replay's own "
        "(see the module docstring)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        metavar="ROBOT_IDS",
        help="print tactic, position, distance to ball, referee command and traced target every 0.25 s for "
        "these of our robots, e.g. 3 or 1,3 (see the module docstring)",
    )
    args = parser.parse_args()

    scenario: Scenario = scenario_from_replay(args.replay_path, args.t)
    replay_noise = load_sim_noise(args.replay_path) if args.replay_path.suffix == ".npz" else None
    sim_noise, noise_line = resolve_sim_noise(args.sim_noise, replay_noise)

    strategy_a_name = _resolve_config_name(args.strategy, scenario.config_a_name, "friendly")
    strategy_b_name = _resolve_config_name(args.opponent, scenario.config_b_name, "enemy")

    build_a = getattr(kernel_strategy, strategy_a_name)
    build_b = getattr(kernel_strategy, strategy_b_name)
    strategy_a = AbstractStrategy(build_kernel_strategy=build_a(OUTFIELD_ROBOT_IDS))
    strategy_b = AbstractStrategy(build_kernel_strategy=build_b(OUTFIELD_ROBOT_IDS))

    referee = CustomReferee.from_profile_name(
        "simulation", n_robots_yellow=N_OUTFIELD + 1, n_robots_blue=N_OUTFIELD + 1
    )

    runner = StrategyRunner(
        strategy=strategy_a,
        opp_strategy=strategy_b,
        my_team_is_yellow=True,
        my_team_is_right=True,
        mode="rsim",
        exp_friendly=N_OUTFIELD + 1,
        exp_enemy=N_OUTFIELD + 1,
        exp_ball=True,
        referee=referee,
        enable_vision_stream=False,
        referee_initial_command=RefereeCommand.PREPARE_KICKOFF_YELLOW,
        control_scheme=args.control_scheme,
        match_log_path=args.trace_out,
        sim_noise=sim_noise,
    )

    print(noise_line)
    if sim_noise is not None and replay_noise is not None and args.sim_noise is None:
        drawn = runner.rsim_env.sim_draw.to_dict()
        same = json.loads(json.dumps(drawn)) == replay_noise["draw"]
        print(f"  per-robot draws {'match' if same else 'DO NOT match'} the replay's stored draw")
    print(f"Loaded scenario from {args.replay_path} at t={args.t}s (nearest frame ts={scenario.frame_ts:.3f}s)")
    print(f"  friendly={strategy_a_name}  enemy={strategy_b_name}")
    print(f"  referee_command at load: {scenario.referee_command}")

    try:
        apply_scenario(runner, scenario, verify=True)
        print("apply_scenario: positions verified within tolerance.")

        n_ticks = int(args.duration * TICKS_PER_SECOND)
        start_frame = runner.my.current_game_frame
        start_ts = start_frame.ts
        ball_positions = [(start_frame.ball.p.x, start_frame.ball.p.y)] if start_frame.ball is not None else []
        start_robot_pos = {
            rid: (r.p.x, r.p.y) for rid, r in {**start_frame.friendly_robots, **start_frame.enemy_robots}.items()
        }
        referee_commands_seen: list[str] = []
        last_command = None
        summary_ids = _parse_ids(args.summary) if args.summary else []
        samples: list[dict] = []
        if summary_ids:
            samples.append(_snapshot(runner, summary_ids))

        for _ in range(n_ticks):
            runner.step_once()
            frame = runner.my.current_game_frame
            if frame.ball is not None:
                ball_positions.append((frame.ball.p.x, frame.ball.p.y))
            ref_data = runner.my.game.referee
            command = ref_data.referee_command.name if ref_data is not None else None
            if summary_ids:
                samples.append(_snapshot(runner, summary_ids))
            if command != last_command:
                referee_commands_seen.append(command)
                last_command = command

        end_frame = runner.my.current_game_frame
        end_ts = end_frame.ts

        ball_travel_m = sum(
            math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(ball_positions, ball_positions[1:])
        )

        end_robot_pos = {
            rid: (r.p.x, r.p.y) for rid, r in {**end_frame.friendly_robots, **end_frame.enemy_robots}.items()
        }
        displacements = {
            rid: math.hypot(x2 - x1, y2 - y1)
            for rid, (x1, y1) in start_robot_pos.items()
            if rid in end_robot_pos
            for (x2, y2) in [end_robot_pos[rid]]
        }

        # "still stalled" verdict: ball moved < 5cm total over the final 10s
        # (or the whole run, if shorter than that window).
        window_s = min(_STILL_STALLED_WINDOW_S, args.duration)
        window_ticks = int(window_s * TICKS_PER_SECOND)
        tail = ball_positions[-(window_ticks + 1) :] if window_ticks > 0 else ball_positions
        tail_travel_m = sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(tail, tail[1:]))
        still_stalled = tail_travel_m < _STILL_STALLED_THRESHOLD_M

        print(f"\nSim time range: {start_ts:.3f}s -> {end_ts:.3f}s ({end_ts - start_ts:.3f}s ticked)")
        print(f"Ball travel over the run: {ball_travel_m:.3f}m")
        print(f"Ball travel over final {window_s:.0f}s: {tail_travel_m:.3f}m")
        print("Per-robot displacement (m):")
        for rid in sorted(displacements):
            print(f"  robot {rid}: {displacements[rid]:.3f}")
        print(f"Referee commands seen: {referee_commands_seen}")
        print(
            f"Verdict: {'STILL STALLED' if still_stalled else 'not stalled'} "
            f"(ball moved {tail_travel_m:.3f}m in the final {window_s:.0f}s, threshold {_STILL_STALLED_THRESHOLD_M}m)"
        )

    finally:
        runner.close()

    if summary_ids:
        lines = format_summary(summary_rows(samples, summary_ids, load_jsonl(args.trace_out)))
        print(f"\nSummary for robot(s) {summary_ids} (every {SUMMARY_EVERY_S}s, at most {SUMMARY_MAX_LINES} lines):")
        print("\n".join(lines))

    if args.print_trace and runner.match_log is not None:
        last_value: dict[str, object] = {}
        for event in load_jsonl(args.trace_out):
            if type(event).__name__ == "TraceEvent":
                if last_value.get(event.key) == event.value:
                    continue
                last_value[event.key] = event.value
            print(event)


if __name__ == "__main__":
    main()

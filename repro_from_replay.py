"""repro_from_replay.py — reload a replay's field state at a given timestamp
into a fresh headless rsim match, and tick forward with match_log tracing on.

Why this exists: finding a stall today means running a 231-match round-robin
(`smoke_tournament.py`), spotting a suspicious match, then tracing its replay
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
    pixi run python repro_from_replay.py replays/tournament_.../match.pkl \\
        --t 260 --duration 15 --control-scheme trajsample \\
        --trace-out /tmp/repro_trace.jsonl
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

from tournament_lib import N_OUTFIELD, OUTFIELD_ROBOT_IDS, TICKS_PER_SECOND
from utama_core.custom_referee import CustomReferee
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.match_log import load_jsonl
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.scenario import Scenario, apply_scenario, scenario_from_replay
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
        help="override the friendly (config_a) strategy, e.g. build_tiki_taka_kernel_strategy",
    )
    parser.add_argument(
        "--opponent", default=None, help="override the enemy (config_b) strategy, e.g. build_low_block_kernel_strategy"
    )
    parser.add_argument(
        "--trace-out", default="/tmp/repro_from_replay.intentions.jsonl", help="where to write the match_log JSONL"
    )
    parser.add_argument("--print-trace", action="store_true", help="print every TraceEvent as it's written back")
    args = parser.parse_args()

    scenario: Scenario = scenario_from_replay(args.replay_path, args.t)

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
    )

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

        for _ in range(n_ticks):
            runner.step_once()
            frame = runner.my.current_game_frame
            if frame.ball is not None:
                ball_positions.append((frame.ball.p.x, frame.ball.p.y))
            ref_data = runner.my.game.referee
            command = ref_data.referee_command.name if ref_data is not None else None
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

"""debug_match.py — one-off ad hoc match runner for tactic debugging.

Run:
    pixi run python tools/debug_match.py --strategy build_counter_flow_kernel_strategy \\
        --opponent build_high_press_kernel_strategy --duration 90 --headless

Replaces the pattern of hand-writing a fresh `StrategyRunner(...)` block per
bug (`trace_relay_stall.py`, `trace_finish_detail.py`, `check_keeper.py`,
etc. from the 2026-08-23 `counter_press` investigation were all this same
~25 lines of setup, copy-pasted and tweaked). Reuses `match.py`'s own
match-setup constants/helpers (`N_OUTFIELD`, `OUTFIELD_ROBOT_IDS`,
`TICKS_PER_SECOND`) rather than redefining them.

This only *runs* the match and writes `match_log_path`/`stats_path` (see
`utama_core.engine.match_log.MatchLog`) — it does not itself decide what a
tactic records. To debug a specific tactic, add a few
`ctx.match_log.trace(tick, sim_time, "key", value)` calls at the decision
points you care about (phase transitions, gate checks, computed targets)
inside the tactic file itself; those calls are always present and cheap
(append to an in-memory list) but only ever produce a file when
`match_log_path` is set here, so normal runs/tests/tournaments are
unaffected. Read the result back with `utama_core.engine.match_log.load_jsonl`.

`--stop-on-goal` ends the match early — most debugging doesn't need the tail
of a full match once the phase of interest has been seen once or twice.

`--dump-ticks PATH` writes one JSON row per tick: sim time, referee command,
score, ball, every robot's pose, and each robot's commanded position target
(what a tactic asked for, as opposed to where physics put the robot; keys
`y<id>` / `b<id>` because both teams number robots from 0). Use it when a
position alone cannot tell a bad target from a transient collision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.match import (
    _CONFIG_NAMES,
    N_OUTFIELD,
    OUTFIELD_ROBOT_IDS,
    TICKS_PER_SECOND,
)
from utama_core.custom_referee import CustomReferee
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.match_log import load_jsonl
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.strategy import kernel_strategy


class _TargetRecorder:
    """MotionController wrapper that records the target of every `calculate()` call.

    Keys are ``y<id>`` for the strategy side and ``b<id>`` for the opponent.
    """

    def __init__(self, inner, side: str, targets: dict[str, tuple[float, float]]):
        self._inner = inner
        self._side = side
        self._targets = targets

    def __getattr__(self, name):
        # Tactics read controller attributes (`mode`, `rsim_env`, ...): pass them through.
        return getattr(self._inner, name)

    def calculate(self, game, robot_id, target_pos, target_oren, **kwargs):
        # go_to_point accepts Vector2D or plain (x, y) tuples.
        x, y = (target_pos.x, target_pos.y) if hasattr(target_pos, "x") else (target_pos[0], target_pos[1])
        self._targets[f"{self._side}{robot_id}"] = (round(float(x), 3), round(float(y), 3))
        return self._inner.calculate(
            game=game, robot_id=robot_id, target_pos=target_pos, target_oren=target_oren, **kwargs
        )


def _record_targets(runner: StrategyRunner) -> dict[str, tuple[float, float]]:
    """Wrap both kernels' motion controllers; the returned dict holds the latest target per robot."""
    targets: dict[str, tuple[float, float]] = {}
    for side, prefix in ((runner.my, "y"), (runner.opp, "b")):
        kernel = getattr(side.strategy, "_kernel_strategy", None)
        if kernel is not None:
            kernel._ctx.motion_controller = _TargetRecorder(kernel._ctx.motion_controller, prefix, targets)
    return targets


def _tick_row(runner: StrategyRunner, tick: int, targets: dict[str, tuple[float, float]]) -> dict:
    frame = runner.my.current_game_frame
    referee = runner.my.game.referee
    ball = frame.ball

    def poses(robots):
        return {
            str(rid): [round(r.p.x, 2), round(r.p.y, 2), round(r.orientation, 3)] for rid, r in sorted(robots.items())
        }

    return {
        "tick": tick,
        "t": round(tick / TICKS_PER_SECOND, 2),
        "cmd": str(referee.referee_command),
        "score": {"yellow": referee.yellow_team.score, "blue": referee.blue_team.score},
        "ball": [round(ball.p.x, 3), round(ball.p.y, 3), round(ball.v.x, 3), round(ball.v.y, 3)] if ball else None,
        "yellow": poses(frame.friendly_robots or {}),
        "blue": poses(frame.enemy_robots or {}),
        "targets": dict(targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", required=True, choices=sorted(_CONFIG_NAMES))
    parser.add_argument("--opponent", required=True, choices=sorted(_CONFIG_NAMES))
    parser.add_argument("--duration", type=float, default=90.0, help="sim seconds (default 90)")
    parser.add_argument("--stop-on-goal", action="store_true", help="end the match as soon as either side scores")
    parser.add_argument(
        "--match-log", default="/tmp/debug_match.intentions.jsonl", help="where to write the match_log JSONL"
    )
    parser.add_argument("--dump-ticks", default=None, metavar="PATH", help="write one JSON row per tick to PATH")
    parser.add_argument("--print-trace", action="store_true", help="print every TraceEvent as it's written back")
    parser.add_argument("--headless", action="store_true", help="accepted for CLI-convention compatibility; unused")
    parser.add_argument(
        "--control-scheme",
        default="fpp",
        help="motion control scheme for both sides, e.g. fpp, dwa, trajsample (default fpp)",
    )
    parser.add_argument(
        "--stats-path",
        default=None,
        help=(
            "If set, also accumulate and write the same possession/shots/ball-travel "
            "summary round_robin.py records (utama_core.engine.match_stats.MatchStats) to "
            "this path. Off by default since most debugging sessions only care about the "
            "match_log/trace output, not aggregate stats."
        ),
    )
    args = parser.parse_args()

    build_a = getattr(kernel_strategy, args.strategy)
    build_b = getattr(kernel_strategy, args.opponent)
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
        match_log_path=args.match_log,
        stats_path=args.stats_path,
        control_scheme=args.control_scheme,
    )

    targets = _record_targets(runner) if args.dump_ticks else {}
    dump = open(args.dump_ticks, "w") if args.dump_ticks else None
    try:
        prev_score = (0, 0)
        for tick in range(int(args.duration * TICKS_PER_SECOND)):
            runner.step_once()
            if dump is not None:
                dump.write(json.dumps(_tick_row(runner, tick, targets)) + "\n")
            if args.stop_on_goal:
                ref = runner.my.game.referee
                score = (ref.yellow_team.score, ref.blue_team.score)
                if score != prev_score:
                    print(f"Goal! score={score}, stopping early.")
                    break
                prev_score = score
        ref_data = runner.my.game.referee
        print(f"Final score: yellow={ref_data.yellow_team.score} blue={ref_data.blue_team.score}")
    finally:
        if dump is not None:
            dump.close()
            print(f"tick dump written: {args.dump_ticks}")
        # match_stats.finalize()/.to_json() (if --stats-path was given) happens
        # inside close() itself — see StrategyRunner.close().
        runner.close()

    if args.stats_path and runner.match_stats is not None:
        stats = runner.match_stats.finalize()
        print(f"Stats written: {args.stats_path}")
        poss = stats.possession_pct
        shots = stats.shots
        print(
            f"possession {poss['friendly']:.0%}/{poss['enemy']:.0%}  "
            f"shots {shots['friendly']}-{shots['enemy']}  ball_travel {stats.ball_travel_m:.1f}m"
        )

    if runner.match_log is not None:
        runner.match_log.to_jsonl(args.match_log)
        print(f"match_log written: {args.match_log}")
        if args.print_trace:
            # Print IntentionEvents (already one-per-change, never per-tick)
            # and only the *changed* TraceEvents per key -- a naive dump of
            # every event is 1000+ near-identical lines for a 20s match (one
            # TraceEvent per tick per trace() call site), which is exactly
            # the terminal-noise failure mode this tool exists to avoid.
            last_value: dict[str, object] = {}
            for event in load_jsonl(args.match_log):
                if type(event).__name__ == "TraceEvent":
                    if last_value.get(event.key) == event.value:
                        continue
                    last_value[event.key] = event.value
                print(event)


if __name__ == "__main__":
    main()

"""debug_match.py — one-off ad hoc match runner for tactic debugging.

Run:
    pixi run python debug_match.py --strategy build_counter_press_kernel_strategy \\
        --opponent build_low_block_kernel_strategy --duration 90 --headless

Replaces the pattern of hand-writing a fresh `StrategyRunner(...)` block per
bug (`trace_relay_stall.py`, `trace_finish_detail.py`, `check_keeper.py`,
etc. from the 2026-08-23 `counter_press` investigation were all this same
~25 lines of setup, copy-pasted and tweaked). Reuses `tournament.py`'s own
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

`--stop-on-goal` and `--stop-after` end the match early — most debugging
doesn't need the tail of a full match once the phase of interest has been
seen once or twice.
"""

from __future__ import annotations

import argparse

from tournament import _CONFIG_NAMES, N_OUTFIELD, OUTFIELD_ROBOT_IDS, TICKS_PER_SECOND
from utama_core.custom_referee import CustomReferee
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.match_log import load_jsonl
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.run import StrategyRunner
from utama_core.strategy import kernel_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", required=True, choices=sorted(_CONFIG_NAMES))
    parser.add_argument("--opponent", required=True, choices=sorted(_CONFIG_NAMES))
    parser.add_argument("--duration", type=float, default=90.0, help="sim seconds (default 90)")
    parser.add_argument("--stop-on-goal", action="store_true", help="end the match as soon as either side scores")
    parser.add_argument(
        "--match-log", default="/tmp/debug_match.intentions.jsonl", help="where to write the match_log JSONL"
    )
    parser.add_argument("--print-trace", action="store_true", help="print every TraceEvent as it's written back")
    parser.add_argument("--headless", action="store_true", help="accepted for CLI-convention compatibility; unused")
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
    )

    try:
        prev_score = (0, 0)
        for _ in range(int(args.duration * TICKS_PER_SECOND)):
            runner.step_once()
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
        runner.close()

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

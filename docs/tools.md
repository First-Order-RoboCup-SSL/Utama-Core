# Tools and scripts

Every script in the repository, by what it is for. Run them from the repository root with
`pixi run python <path>`; each script's docstring has its full usage and flags.

## pixi tasks

`pixi run <task>` (defined in `pixi.toml`):

| Task | What it does |
|---|---|
| `test` | pytest over `utama_core/tests/` (pass `--headless` to pytest directly in agent loops) |
| `lint` | the whole pre-commit stack (black, ruff, isort) on every file |
| `precommit-install` / `precommit-uninstall` | install or remove the pre-commit hook |
| `main` | `main.py`: watch one match between two strategies in rsim (below) |
| `build-planner` | build the C++ fast-path planner (`utama_core/motion_planning/src/fastpathplanning/cpp`) into `build/fpp` with cmake |
| `replay [-n NAME] [-p]` | play a legacy pickle replay `replays/NAME.pkl` in the rSoccer viewer (newest if no name; `-p` step by step). Today's `.npz` replays open in the dashboard instead |
| `runs` | list the tournament runs in `replays/`: start time, commit, matches, stalls, arguments |
| `debug-robots` | keyboard/click teleoperation GUI for real robots over the serial radio |

## Playing matches

| Script | Purpose |
|---|---|
| `tools/evaluation/round_robin.py` | Every strategy config against every other, one full match per pair (two halves of 300 s of playing time); writes `replays/tournament_<id>/` with `summary.json`. `--pair A B` plays one match; `--reuse` replays only matches whose code changed, plus a 5% spot-check of reused ones; `summary.json`'s `reuse.mismatches` lists any that came out different, which means a match no longer replays exactly. `--sim-noise JSON` (inline or a file) plays every match with the same seeded latency, kick speed and vision noise (`SimNoise`, `docs/STRATEGY_DEVELOPMENT.md`). The ground truth for which strategy is better |
| `tools/evaluation/ladder.py` | One candidate against a reference pool (default: the top 5 of the newest complete round-robin): mirrored pairs over side and kickoff, up to 8 matches per opponent, stopping once a pairing is clear (`decided`). Reuses and fills the round-robin's match cache; writes `replays/ladder_<id>/ladder.json`. A cheaper check of one strategy than a round-robin, not a ranking of all |
| `tools/evaluation/match.py` | Match construction shared by the two above (not run directly) |
| `dashboard_server.py` | Standalone browser dashboard at http://localhost:8080: replays and tournament results |

## Evaluating strategies

What these measure and how far to trust each: [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md),
[signals.md](signals.md).

| Script | Purpose |
|---|---|
| `tools/scenario_bench.py` | Paired A/B of a candidate against a baseline on a bank of situation starts harvested from a round-robin (10 s each, against the opponent each came from): change in P(goal) per situation type, weighted by real or sim situation rates (`--weights`) into goals per match; also builds new banks (`--harvest-from`), and plays one start with a timeline and picture (`--play`) |
| `tools/signal_report.py` | Figures of a round-robin's strategy signals for [signal_report.md](signal_report.md) |
| `utama_core/analysis/goal_ledger.py` | `python -m utama_core.analysis.goal_ledger <run_dir or replays>`: every goal attributed to the situation that started its play (a restart, or the ball won in open play, by third); per strategy, situations per match × P(goal) = goals per match, scored and conceded. Reads only the replay and its referee file, so it runs on converted official logs too. `--rates` writes the situation rates the scenario bench weights by. See [signals.md](signals.md#where-do-goals-come-from) |
| `utama_core/analysis/physics_ranges.py` | `python -m utama_core.analysis.physics_ranges --real <official logs> --sim <run_dir>`: physics parameter ranges for the sim-realism plan ([sim_realism.md](sim_realism.md)): ball deceleration by speed and time since a kick, rolling-segment fits, robot speed, acceleration, braking and turn-rate percentiles per robot, and dropped detections, with spreads between matches, robots and teams |
| `tools/bench_vs_standings.py` | How well the scenario bench's effect on goal difference per match predicts a round-robin's, for several candidates against one baseline (Pearson, Spearman) |
| `tools/metric_correlation.py` | Offline study of which cheap per-match metrics predict results; computes the *offline* signals in [signals.md](signals.md) |
| `tools/check_strategy_branch.py` | CI check on `strategy/*` pull requests: the branch must not change the evaluation or its opponents |

## Real matches

| Script | Purpose |
|---|---|
| `tools/convert_ssl_log.py` | Official RoboCup SSL game logs to replays in our format, so the analyses run on real matches: given logs, or `--corpus 2022-2026` to download and convert every published Division B match, deleting each raw log once converted. Also writes sidecars for measuring physics: the tracker's kicks and visibility, and with `--raw-vision` every raw camera detection; `--backfill-extras` adds them to matches already converted. What a converted match holds and its limits: [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md#real-matches) |

## Debugging a match

| Script | Purpose |
|---|---|
| `tools/replay_trace.py` | Text trace of a replay window: referee commands, ball, nearest robot per side. The first thing to run on a stall or a voided restart |
| `tools/repro_from_replay.py` | Load a replay's field state at a timestamp into a fresh headless rsim match and tick forward with tracing on |
| `tools/debug_match.py` | One ad hoc match between two strategies for tactic debugging; `--dump-ticks` writes per-tick poses and commanded targets |

Library helpers for the same job (`render_window`, `render_clip`, `MatchLog.trace()`) are
described in [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md).

## Benchmarks

| Script | Purpose |
|---|---|
| `tools/motion_planning_benchmark.py` | The motion planners (`fpp`, `dwa`, `trajsample`) on fixed headless rsim scenarios; see [motion_planning_comparison.md](motion_planning_comparison.md) |

## Demos and external environments

| Script | Purpose |
|---|---|
| `main.py` | Watch one match in rsim with the referee and dashboard: `pixi run main [yellow] [blue]`, default `split_shape` vs `high_press` |
| `examples/demo_custom_referee.py` | Every `CustomReferee` rule in a scripted scenario, in a pygame window; no simulator needed |
| `examples/demo_referee_gui_rsim.py` | `CustomReferee` with the dashboard's referee tab, over rsim |
| `examples/demo_referee_feedback_gui.py` | The referee UI's controller-feedback panel with fake feedback, no hardware |
| `examples/demo_exhibition_road.py` | The Exhibition Road festival demo on the 4 m x 3 m field, over rsim |
| `examples/demo_dribbler_test.py` | One robot dribbling round a rectangle on the Exhibition Road field |
| `examples/demo_split_shape_match.py` | Visible grSim 6v6 of `split_shape` against itself |

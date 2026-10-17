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

## Worktrees

| Script | Purpose |
|---|---|
| `tools/setup_worktree.sh` | Run inside a git worktree: links `.pixi` to the main checkout's and copies the gitignored compiled `.so` extensions (planner, rsim). Idempotent; one line per action |

## Playing matches

| Script | Purpose |
|---|---|
| `tools/evaluation/round_robin.py` | The whole-catalog health and stall check: every strategy config against every other, one full match per pair (two halves of 300 s of playing time); writes `replays/tournament_<id>/` with `summary.json`, and adds its matches to the evaluation ledger. `--pair A B` plays one match; `--reuse` replays only matches whose code changed, plus a 5% spot-check of reused ones; `summary.json`'s `reuse.mismatches` lists any that came out different, which means a match no longer replays exactly. `--sim-noise JSON` (inline or a file) plays every match with the same seeded latency, kick speed and direction noise, ball friction, motion limits and vision noise (`SimNoise`, `docs/STRATEGY_DEVELOPMENT.md`). A single strategy is judged with `evaluate.py`, not by its place here |
| `tools/evaluation/evaluate.py` | One or more candidates against the champion of a fixed reference pool (`tools/evaluation/pools/pool_v1.json`), in rounds under seeded sim noise, until its paired results against the champion say better (P >= 0.99), worse (P <= 0.2) or undecided after 6 rounds; `--held-out` plays the pool's held-out opponents before promoting. Results go to the evaluation ledger `replays/eval/ledger.jsonl` (round-robins add theirs too), reports to `replays/eval/reports/`; no replays kept. The way to judge a strategy (`docs/STRATEGY_DEVELOPMENT.md`, "Evaluating a strategy") |
| `tools/evaluation/match.py` | Match construction shared by the two above (not run directly) |
| `dashboard_server.py` | Standalone browser dashboard at http://localhost:8080: replays and tournament results |

## Evaluating strategies

What these measure and how far to trust each: [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md),
[signals.md](signals.md).

| Script | Purpose |
|---|---|
| `tools/signal_report.py` | Figures of a round-robin's strategy signals for [signal_report.md](signal_report.md) |
| `utama_core/analysis/goal_ledger.py` | `python -m utama_core.analysis.goal_ledger <run_dir or replays>`: every goal attributed to the situation that started its play (a restart, or the ball won in open play, by third); per strategy, situations per match × P(goal) = goals per match, scored and conceded. Reads only the replay and its referee file, so it runs on converted official logs too. `--rates` writes the situation rates the scenario bench weights by. See [signals.md](signals.md#where-do-goals-come-from) |
| `utama_core/analysis/physics_ranges.py` | `python -m utama_core.analysis.physics_ranges --real <official logs> --sim <run_dir>`: physics parameter ranges for the sim-realism plan ([sim_realism.md](sim_realism.md)): ball deceleration by speed and time since a kick, rolling-segment fits, robot speed, acceleration, braking and turn-rate percentiles per robot, and dropped detections, with spreads between matches, robots and teams |
| `tools/check_strategy_branch.py` | CI check on `strategy/*` pull requests: the branch must not change the evaluation or its opponents |

### Validation studies

These check how far cheap metrics agree with round-robin results. They do not judge a strategy: use `evaluate.py`.

| Script | Purpose |
|---|---|
| `tools/metric_correlation.py` | Which cheap per-match metrics predict results; computes the *offline* signals in [signals.md](signals.md) |

## Real matches

| Script | Purpose |
|---|---|
| `tools/convert_ssl_log.py` | Official RoboCup SSL game logs to replays in our format, so the analyses run on real matches: given logs, or `--corpus 2022-2026` to download and convert every published Division B match, deleting each raw log once converted (`--keep-raw` keeps it; downloads cut short are resumed; each conversion's memory is capped by `--max-memory-gb`). Also writes sidecars for measuring physics: the tracker's kicks and visibility, and with `--raw-vision` every raw camera detection; `--backfill-extras` adds them to matches already converted. What a converted match holds and its limits: [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md#real-matches) |
| `tools/realism_scorecard.py` | Our sim against real Division B matches, component by component: attempts per live minute (open play and set pieces), attempt distance, real xG per attempt, finishing, goal rate and keeper saves per xG band, shot outcomes (keeper, defender, wide, post, crossed with no goal), aim outside the posts and miss distance, ball slow-down, kick speed, keeper depth, keeper and nearest-defender reaction time (`utama_core/analysis/realism/reaction.py`), pass line offset and completion (`passes.py`). Each row: the real value and 95% interval, the sim value, and whether it is inside. `--real` rebuilds the committed reference (`utama_core/analysis/realism/real_2025_2026.json`: the real xG model and rows) from converted logs; `--sim RUN_DIR` scores a round-robin against it. Attempts exclude passes and kicks from beyond 6 m. See [sim_realism.md](sim_realism.md) |

## Debugging a match

| Script | Purpose |
|---|---|
| `tools/replay_trace.py` | Text trace of a replay window: referee commands, ball, nearest robot per side. `--events` lists the window's kicks (with the nearest robot as kicker), possession changes, referee changes and fouls, one line each; `--json` writes rows or events as JSON. The first thing to run on a stall, a voided restart or a foul |
| `tools/repro_from_replay.py` | Load a replay's field state at a timestamp into a fresh headless rsim match and tick forward with tracing on. Plays with the replay's own sim noise by default (`--sim-noise none` for noiseless, `--sim-noise PATH` for another). `--summary ROBOT_IDS` prints each listed robot's tactic, position, distance to the ball, referee command and traced target every 0.25 s; `--trace-out` JSONL row kinds are in `--help` |
| `tools/moments.py` | Regression moments (`utama_core/tests/moments/`): `fouls RUN_DIR` lists our side's fouls in a round-robin with the restart side (`--skip-own-restart`, `--by strategy,rule` for counts, `--include-stoppages` for out_of_bounds, `--json` for either as JSON; each foul names its match's `replay` .npz relative to the repo root), `capture REPLAY --t S [--out PATH]` writes a moment from a replay, `new NAME --command CMD` writes a hand-made template, `play JSON [--seed N \| --seeds 0-4] [--against REF] [--events] [--strategy FILE.py:FACTORY]` plays one and prints what broke its checks (a seed table, the same moment at another commit, its events, a scratch strategy). See [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md), "Debugging a foul or failure" |
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

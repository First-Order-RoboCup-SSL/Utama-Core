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
| `main` | `main.py`, the exhibition demo (below) |
| `replay [-n NAME] [-p]` | play a legacy pickle replay `replays/NAME.pkl` in the rSoccer viewer (newest if no name; `-p` step by step). Today's `.npz` replays open in the dashboard instead |
| `runs` | list the tournament runs in `replays/`: start time, commit, matches, stalls, arguments |
| `debug-robots` | keyboard/click teleoperation GUI for real robots over the serial radio |

## Playing matches

| Script | Purpose |
|---|---|
| `tools/tournament/round_robin.py` | Every strategy config against every other, one full match per pair (two halves of 300 s of playing time); writes `replays/tournament_<id>/` with `summary.json`. `--pair A B` plays one match; `--reuse` replays only matches whose code changed. The ground truth for which strategy is better |
| `tools/tournament/full_match_tournament.py` | Full-match round-robin among the `competitive`-tier strategies only, each pair played 4 times: both sides x both kickoffs, so a result can be attributed to side or kickoff |
| `tools/tournament/tournament_lib.py` | Match construction shared by the two above (not run directly) |
| `dashboard_server.py` | Standalone browser dashboard at http://localhost:8080: replays and tournament results |

## Evaluating strategies

What these measure and how far to trust each: [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md),
[signals.md](signals.md).

| Script | Purpose |
|---|---|
| `tools/scenario_bench.py` | Paired A/B of a candidate against a baseline on a bank of 20 s starts harvested from a round-robin; also builds new banks (`--harvest-from`), and plays one start with a timeline and picture (`--play`) |
| `tools/signal_report.py` | Figures of a round-robin's strategy signals for [signal_report.md](signal_report.md) |
| `tools/elo.py` / `tools/plot_elo.py` | Elo ratings from round-robin `summary.json` files, and plots of them (rating history, W/D/L matrix, goal difference) |
| `tools/bench_vs_standings.py` | How far scenario-bench scores rank strategies the way round-robin standings do (Spearman) |
| `tools/metric_correlation.py` | Offline study of which cheap per-match metrics predict results; computes the *offline* signals in [signals.md](signals.md) |
| `tools/check_strategy_branch.py` | CI check on `strategy/*` pull requests: the branch must not change the evaluation or its opponents |

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
| `main.py` | Exhibition demo: one `GiveAndGoTactic` robot plus keeper over grSim, with the dashboard (grSim must be running) |
| `examples/demo_custom_referee.py` | Every `CustomReferee` rule in a scripted scenario, in a pygame window; no simulator needed |
| `examples/demo_referee_gui_rsim.py` | `CustomReferee` with the dashboard's referee tab, over rsim |
| `examples/demo_referee_feedback_gui.py` | The referee UI's controller-feedback panel with fake feedback, no hardware |
| `examples/demo_exhibition_road.py` | The Exhibition Road festival demo on the 4 m x 3 m field, over rsim |
| `examples/demo_dribbler_test.py` | One robot dribbling round a rectangle on the Exhibition Road field |
| `examples/demo_split_shape_match.py` | Visible grSim 6v6 of `split_shape` against itself |
| `start_test_env.sh` | Starts grSim, the GameController and AutoReferee together; see [setup_external.md](setup_external.md) |

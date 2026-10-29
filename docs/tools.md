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
| `debug-robots` | keyboard/click teleoperation GUI for real robots over the serial radio (robots chosen by radio ID; [real_robot_setup.md](real_robot_setup.md)) |
| `calibrate` | measure one real robot from the terminal (`examples/calibrate_robot.py --robot yellow:0 --radio 0`): fixed driving moves, then the robot fetches the ball itself for a carry ladder (the fastest it keeps the ball each way) and 10 kicks; asks only to roll the ball back; then DONE, the ladder, and its loop delay, speed gain, acceleration, vision noise, kick speed and kick bend; `--ball-only` repeats the ball part ([real_robot_setup.md](real_robot_setup.md#5-measuring-a-robot)) |
| `calibrate-tag` | where one real robot's vision tag sits on it (`examples/calibrate_tag.py --robot yellow:0 --radio 0`): its offset from the centre and its heading offset, printed as the `RobotProfile` fields that correct vision ([real_robot_setup.md](real_robot_setup.md#5-measuring-a-robot)) |

## Worktrees

| Script | Purpose |
|---|---|
| `tools/setup_worktree.sh` | Run inside a git worktree: links `.pixi` to the main checkout's and copies the gitignored compiled `.so` extensions (planner, rsim). Idempotent; one line per action |

## Playing matches

| Script | Purpose |
|---|---|
| `tools/evaluation/round_robin.py` | The whole-catalog health and stall check: every strategy config against every other, one full match per pair (two halves of 300 s of playing time); writes `replays/tournament_<id>/` with `summary.json`, and adds its matches to the evaluation ledger. `--pair A B` plays one match; `--reuse` replays only matches whose code changed, plus a 5% spot-check of reused ones; `summary.json`'s `reuse.mismatches` lists any that came out different, which means a match no longer replays exactly. `--sim-noise JSON` (inline or a file) plays every match with the same seeded latency, kick speed and direction noise, ball friction, motion limits and vision noise (`SimNoise`, `docs/STRATEGY_DEVELOPMENT.md`). A single strategy is judged with `evaluate.py`, not by its place here |
| `tools/evaluation/evaluate.py` | One or more candidates against the champion of a fixed reference pool (`tools/evaluation/pools/pool_v1.json`), in rounds under seeded sim noise, until its paired results against the champion say better (P >= 0.99), worse (P <= 0.2) or undecided after 6 rounds; `--held-out` plays the pool's held-out opponents before promoting. Results go to the evaluation ledger `replays/eval/ledger.jsonl` (round-robins add theirs too), reports to `replays/eval/reports/`; no replays kept. The way to judge a strategy (`docs/STRATEGY_DEVELOPMENT.md`, "Evaluating a strategy"). `--explain` (off by default) also writes pictures and a `summary.json` of each candidate against the champion to `replays/eval/explain/` (`tools/evaluation/explain.py`); they explain the verdict, never change it |
| `tools/evaluation/explain.py` | `explain.py RUN_DIR CANDIDATE CHAMPION --out DIR`: positions (with and without the ball), shots for and against, and fouls by rule, for two strategies from the matches in a run dir (a round-robin's or an `evaluate.py` run's), plus `summary.json` with the numbers behind them; goals stay the verdict. Used by `evaluate.py --explain` |
| `tools/evaluation/match.py` | Match construction shared by the two above (not run directly) |
| `dashboard_server.py` | Standalone browser dashboard at http://localhost:8080: replays and tournament results |

## Evaluating strategies

What these measure and how far to trust each: [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md),
[signals.md](signals.md).

| Script | Purpose |
|---|---|
| `tools/signal_report.py` | Figures of a round-robin's strategy signals for [signal_report.md](signal_report.md) |
| `tools/pitch_maps.py` | Where each strategy's robots stand, shoots and fouls, drawn on the pitch, for [signal_report.md](signal_report.md#on-the-pitch). `--findings PATH`: six descriptive `zone_share` facts per strategy (its in- and out-of-possession samples in each third; no threshold) |
| `utama_core/analysis/goal_ledger.py` | `python -m utama_core.analysis.goal_ledger <run_dir or replays>`: every goal attributed to the situation that started its play (a restart, or the ball won in open play, by third); per strategy, situations per match × P(goal) = goals per match, scored and conceded. Reads only the replay and its referee file, so it runs on converted official logs too. `--rates` writes the situation rates per team per live minute. See [signals.md](signals.md#where-do-goals-come-from) |
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

The four picture tools (`match_timeline.py`, `restart_picture.py`, `plan_trace.py`, `pitch_maps.py`) each take
`--findings PATH` (`-` for stdout) and write the problems they flag as JSON lines, one `Finding` per line
(`utama_core/analysis/findings.py`):

```json
{"tool": "match_timeline", "kind": "tactic_thrash", "replay": "replays/.../a_vs_b.npz", "side": "friendly", "t0": 0.0, "t1": 480.0, "robots": [4], "value": 5.5, "threshold": 2.0, "unit": "flip-backs per live min", "message": "R4 took a tactic for under 1 s and reverted 11 times in 2.0 min of live play (5.50 per min, threshold 2); first at t=12.3 s.", "image": null}
```

Fields: `tool`, `kind`, `replay` (path or null), `side` (`friendly`, `enemy` or null), `t0` and `t1` (seconds or null),
`robots`, `value`, `threshold` (null for a descriptive fact), `unit`, `message` (one sentence with the numbers) and
`image` (the PNG drawn, or null). Filter on the JSON; open the picture to understand a flag, never conclude from it alone.

| Script | Purpose |
|---|---|
| `tools/triage.py` | **Start here.** `RUN_DIR [--strategy NAME] [--matches 3] [--restarts] [--out DIR] [--max-pictures 8] [--workers 2]`: each strategy's N worst matches by goal difference, among those it played as config_a (the side with an intentions log: about half its round-robin matches, every other round in an evaluation), every flagged problem with its numbers (`tactic_thrash`, `stall`, and with `--restarts` the `target_in_keep_out` / `keep_out_short` / `target_in_defense_area` restart flags), the top K drawn with the picture tool for that kind. Writes `OUT/index.md` (read first), `OUT/findings.jsonl` and `OUT/pictures/`; default OUT is `RUN_DIR/triage/`. `--restarts` is slow (about 45 s a match). Read-only. See [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md), "Debugging a foul or failure" |
| `tools/replay_trace.py` | Text trace of a replay window: referee commands, ball, nearest robot per side. `--events` lists the window's kicks (with the nearest robot as kicker), possession changes, referee changes and fouls, one line each; `--json` writes rows or events as JSON. The first thing to run on a stall, a voided restart or a foul |
| `tools/match_timeline.py` | One match window as swimlanes: `REPLAY.npz T0 T1 [--side friendly\|enemy]`. Referee command band, one lane per robot coloured by its tactic (ticks at switches, strip where it had the ball), distance of each robot to the ball, and goals, fouls and shots on top; prints switches per minute and the longest tactic span per robot. Use it for thrashing and everyone-chasing-the-ball; example in [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md). `--findings PATH`: one `tactic_thrash` per robot at or above `FLIP_BACKS_PER_MIN` (2) flip-backs per minute of live play (a tactic held under 1 s, then reverted; raw switch rate is not the signal, most strategies switch 10-25 times a minute by design); `timeline_findings(replay, t0, t1, side)` computes it without drawing |
| `tools/repro_from_replay.py` | Load a replay's field state at a timestamp into a fresh headless rsim match and tick forward with tracing on. Plays with the replay's own sim noise by default (`--sim-noise none` for noiseless, `--sim-noise PATH` for another). `--summary ROBOT_IDS` prints each listed robot's tactic, position, distance to the ball, referee command and traced target every 0.25 s; `--trace-out` JSONL row kinds are in `--help` |
| `tools/plan_snapshot.py` | Picture of what the fast-path planner decided for one robot on one tick: `REPLAY.npz --t S --robot ID [--target X Y] [--side enemy]` runs the planner's `_path_to` on the replay's field state and draws the pitch around the robot: every robot with its id and radius, the ball and its keep-out circle, each clearance ring, the target, the carrot returned, and the referee command and distances. `--full-pitch` shows the whole pitch; with no `--target` the robot's position 0.5 s later stands in. Read-only. See [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md), "Debugging a foul or failure" |
| `tools/restart_picture.py` | Picture of where the referee override sends every friendly robot for one restart: `REPLAY.npz --t S [--side enemy]`, or `--all-restarts` for one small panel per restart command in the replay's referee sidecar (at most 8). Runs `RefereeOverride.tick` for the replay's command and draws each robot's commanded target with its role (kicker, placer, keeper), the ball and its 0.8 m keep-out, the placement spot and the other team faded. A robot whose target is inside the keep-out or a defence area it must avoid is flagged in red, with the text box. Read-only. `--findings PATH`: `target_in_keep_out`, `keep_out_short` (the 1 cm stop deadband) and `target_in_defense_area` for every restart in the match, each occurrence, without drawing (`restart_findings(replay, side)`). See [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md), "Referee handling" |
| `tools/plan_trace.py` | The fast-path planner over a time window: `REPLAY.npz --t0 S --t1 S --robot ID [--side enemy] [--target X Y] [--step 0.1] [--lookahead 0.5] [--json]` runs the planner's `_path_to` on every sampled tick and draws one PNG: the robot's path coloured by time with each tick's carrot joined to it, the ball path, the keep-out circle and clearance rings at the end, and time panels for robot-to-ball distance, robot-to-carrot length, speed and the referee command. Without `--target` each tick's target is the robot's position `--lookahead` s later. Robot positions are the replay's, not re-simulated: it shows what the current planner would do from each recorded state. Read-only; `--json` prints the per-tick rows. `--findings PATH` writes one `parked` or `pushed_out` finding per run of at least 1 s (`PARKED_MIN_S`) of short carrot with the target far. See [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md), "Debugging a foul or failure" |
| `tools/moments.py` | Regression moments (`utama_core/tests/moments/`): `fouls RUN_DIR` lists our side's fouls in a round-robin with the restart side (`--skip-own-restart`, `--by strategy,rule` for counts, `--include-stoppages` for out_of_bounds, `--json` for either as JSON; each foul names its match's `replay` .npz relative to the repo root), `capture REPLAY --t S [--out PATH]` writes a moment from a replay, `new NAME --command CMD` writes a hand-made template, `play JSON [--seed N \| --seeds 0-4] [--against REF] [--events] [--strategy FILE.py:FACTORY]` plays one and prints what broke its checks (a seed table, the same moment at another commit, its events, a scratch strategy). See [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md), "Debugging a foul or failure" |
| `tools/debug_match.py` | One ad hoc match between two strategies for tactic debugging; `--dump-ticks` writes per-tick poses and commanded targets |

Library helpers for the same job (`render_window`, `render_clip`, `MatchLog.trace()`) are
described in [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md).

## Architecture

| Script | Purpose |
|---|---|
| `tools/package_graph.py` | The import graph between `utama_core/` packages, as a Mermaid flowchart with import counts, from `ast` (nothing is imported), plus any violation of the stated import rules (`replay/` never imports `analysis/`). Its output is kept in [architecture.md](architecture.md) |

## Benchmarks

| Script | Purpose |
|---|---|
| `tools/motion_planning_benchmark.py` | The motion planners (`fpp`, `dwa`, `trajsample`) on fixed headless rsim scenarios; `--figures DIR` also writes trajectory and summary PNGs; see [motion_planning_comparison.md](motion_planning_comparison.md) |

## Demos and external environments

| Script | Purpose |
|---|---|
| `main.py` | Watch one match in rsim with the referee and dashboard: `pixi run main [yellow] [blue]`, default `split_shape` vs `high_press` |
| `examples/demo_custom_referee.py` | Every `CustomReferee` rule in a scripted scenario, in a pygame window; no simulator needed |
| `examples/demo_referee_gui_rsim.py` | `CustomReferee` with the dashboard's referee tab, over rsim |
| `examples/demo_referee_feedback_gui.py` | The referee UI's controller-feedback panel with fake feedback, no hardware |
| `examples/demo_exhibition_road.py` | The Exhibition Road festival demo on the 4 m x 3 m field, over rsim |
| `examples/calibrate_robot.py` | `pixi run calibrate`: one real robot's profile from the terminal, no dashboard; `--mode rsim --yes` rehearses it |
| `examples/ball_trials.py` | `pixi run calibrate`'s ball part: fetches the ball itself, carries it at rising speeds each way until it loses it, then kicks |
| `examples/shooting_drill.py` | One robot shoots at a goal the other defends (the keeper feeds the ball back, so it loops) on real robots (or `--mode rsim`/`grsim`): field, camera half, IDs and each robot's working hardware in one config block; the Live tab's field check gates the start. Records every camera run as an SSL game log in `replays/field/`; `--calibrate attacker\|defender` plays fixed moves for measuring that robot |
| `examples/save_and_counter.py` | A 1v1 game where a keeper that saves takes the ball and counterattacks, on real robots (or `--mode rsim`/`grsim`), a goal each: each robot keeps its goal and attacks the other's, reading every tick from the same world whether it attacks or keeps; a keeper's save turns it into the attacker. Field, rules and hardware type from `shooting_drill.py` |
| `examples/pass_receive.py` | Two robots (one per team, so two colours) pass to each other on real robots (or `--mode rsim`/`grsim`): the robot nearer the ball passes once its partner is still and facing it, the other gets open and receives. The Live tab shows each robot's role |
| `examples/passing_rally.py` | `pass_receive.py`'s pair passing their way along the field in two lanes, turning round `TURN_M` from each end line, for ever (no shots) |
| `tools/vision_check.py` | Does ssl-vision's multicast reach this machine: sender, packets per second per camera, detections and the field size, in 5 s with no radio or runner. The first check of a test day ([real_robot_setup.md](real_robot_setup.md)). `--guided ball` (or `"blue 2"`): each camera's error on 9 known spots, saved as a baseline |
| `tools/ssl-vision-lab-field.patch` | Not a script: adds a **Lab Field** preset (our 3 m x 2.25 m field, two cameras) to an ssl-vision checkout; `git apply` it there and rebuild |
| `tools/robot_calibration.py` | A robot's numbers from a recorded field test (`replays/field/*.log`): loop delay, speed gain and acceleration per axis (the sim's move model fitted to the camera's positions), vision noise at rest and missed detections, kick speed. Writes `<log>.calibration.json`. The same log converts to a replay with `tools/convert_ssl_log.py` |
| `utama_core/analysis/close_calls.py` | Where two robots came within 0.25 m of each other in field logs (`replays/field/*.log`): how near, how fast they were closing, and the referee command in force; only the robots the run drove, `--tag` per robot. `python -m utama_core.analysis.close_calls replays/field/*real*.log` |
| `examples/planner_ab.py` | Two robots swap ends head on and cross, again and again, in blocks alternating the planner (fpp, trajsample): contacts, close calls, nearest approach and time per planner; every block a game log. The field A/B of the motion planners |
| `examples/demo_dribbler_test.py` | One robot dribbling round a rectangle on the Exhibition Road field |
| `examples/demo_split_shape_match.py` | Visible grSim 6v6 of `split_shape` against itself |

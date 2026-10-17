# AGENTS.md

Durable context for any coding agent (not just Claude) working in this repo.

## What this repo is

Utama-Core is the RoboCup SSL (Small Size League) team's robot-control stack: vision
ingestion, motion planning, robot control, a simulator (`rsoccer_simulator`, "rsim"), and
the strategy layer that decides what each robot does. `Utama-Strategy` is a sibling repo
and is stale — all active strategy work happens here, on top of the tactic-kernel model.

## Work areas

Pick the row for what you are changing and read its doc first. "Doc" is `TODO` where none
exists yet: say so in your change rather than guessing.

| Area | Paths | Doc | Notes |
|---|---|---|---|
| Strategy / tactics | `utama_core/engine/`, `strategy/`, `tactics/`, `skills/` | `docs/STRATEGY_DEVELOPMENT.md` | Engine changes are rare: a new primitive, not a new strategy |
| Strategy evaluation | `tools/evaluation/`, `tools/scenario_bench.py`, `utama_core/scenario_bench/`, `utama_core/replay/`, `utama_core/analysis/` | `docs/STRATEGY_DEVELOPMENT.md`, `docs/signals.md` | Change evaluation and strategy in separate commits, so results stay comparable |
| Referee | `utama_core/custom_referee/` | `docs/custom_referee.md` | |
| Motion planning | `utama_core/motion_planning/`, `tools/motion_planning_benchmark.py` | `docs/motion_planning_comparison.md` | |
| Simulation | `utama_core/rsoccer_simulator/`, `vendor/rSim/` | `vendor/rSim/FORK_NOTES.md` | `vendor/rSim` is the C++ physics fork; `rsoccer_simulator` is the Python env on top of it |
| Real robots (radio, vision, controllers) | `utama_core/team_controller/`, `utama_core/data_processing/` | `utama_core/team_controller/README.md` (brief) | |

## Glossary

One term per thing; reuse these instead of coining new ones.

- **Round-robin / tournament run** — every strategy config plays every other once
  (`tools/evaluation/round_robin.py`); writes `replays/tournament_<id>/summary.json`. The ground truth for
  "which strategy is better". Slow in full; with `--reuse` only matches whose code changed rerun.
- **Standings** — a round-robin's ranking. `round_robin.py` prints wins and draws;
  `metric_correlation.py` uses points per match (3 a win, 1 a draw) and goal difference.
- **Start** — one situation (see below) the scenario bench replays for 10 s, tagged with its
  type from the candidate's view. In code: `BenchScenario` (`scenario_bench/start.py`), and
  "scenario" in flags and enums.
- **Bank** — a versioned list of starts (`utama_core/scenario_bench/banks/bank_vN.json`),
  harvested from one round-robin's replays, up to 40 of each situation type.
- **Scenario bench** — `tools/scenario_bench.py`: a fast paired A/B screen of a candidate
  against a baseline on a bank, each start against the opponent it came from; it estimates the
  change in goals per match as Σ situations per match (real or sim weights) × the change in
  P(goal within 10 s). A diagnostic, not a ranking: it cannot see a candidate change how often
  situations happen or score from build-up longer than 10 s, so rank with matches.
- **Bench validators** — `tools/bench_vs_standings.py` and `tools/metric_correlation.py` check
  how far the bench and cheap proxy metrics agree with round-robin results.
- **Motion planning benchmark** — `tools/motion_planning_benchmark.py`: the planner alone on
  fixed scenarios. Unrelated to the scenario bench or any strategy.
- **Pitch zones** — defensive / middle / attacking third, left wing / centre / right wing, the
  box, danger: defined once, with a diagram, in `docs/pitch_zones.md`. Use those names.
- **Catch rate** — a diagnostic reported from the bank, never a gate (±4–8 pts between runs).
- **Signal** — an interpretable measure of how a strategy plays (shots, danger conceded, real
  ball losses, stalls ...), for explaining results, never a target. All listed in `docs/signals.md`.
- **Outcome estimator / expected standings** — a match stat validated to predict other matches'
  results better than goals do, used to rank and screen (today only shots on target; the round-robin's
  EXPECTED STANDINGS). Kept only while it keeps predicting; decide with goals. `docs/signals.md`, "Ranking".
- **Situation / goal ledger** — a situation starts when a restart goes live or a side wins the
  ball in open play; the goal ledger (`utama_core/analysis/goal_ledger.py`) gives each goal to
  the latest situation less than 10 s before it, so goals per match = Σ situations per match ×
  P(goal). Situation types: own / opponent ball won or free kick in each third, own / opponent
  kickoff or penalty, force start. See `docs/signals.md`.
- **Situation rates** — how often each situation type happens per team per live minute, from
  real Division B matches or our round-robin (`banks/situation_rates_{real,sim}.json`,
  `goal_ledger --rates`): the scenario bench's weights.

## Repo map

- `utama_core/engine/` — the scheduler/protocol infra: `Strategy`, `Tactic`, `TickContext`,
  `MatchLog`, `AbstractStrategy`, referee-override plumbing. Strategy-dev work touches this
  rarely, mostly to add a new primitive, not a new strategy. Named `engine/`, not `kernel/`,
  specifically to avoid colliding with "kernel strategy" — the model's own established
  vocabulary (every factory is `build_*_kernel_strategy`, e.g. `build_tiki_taka_kernel_strategy`)
  — so "where do I find the kernel strategies" unambiguously means `strategy/` below, not here.
- `utama_core/strategy/` — the actual strategies people write, run, and compare
  (`build_*_kernel_strategy` factories — `tiki_taka`, `counter_flow`, etc.), one module per
  strategy (`strategy/<name>.py`, with the pickers only it uses) plus `pickers.py` for helpers
  shared by several; `kernel_strategy.py` re-exports every factory, so a new strategy must be
  imported there to be discovered. This is where day-to-day strategy-dev edits land.
- `utama_core/tactics/` — reusable `Tactic` implementations (`GiveAndGoTactic`,
  `PressAndContainTactic`, ...) that strategies compose.
- `utama_core/skills/` — lower-level per-robot primitives (`go_to_ball`, `block_attacker`,
  ...) tactics call directly.
- `utama_core/custom_referee/` — the in-process referee (rules, restart positioning).
- `utama_core/replay/` — the replay file format (writer, player, columnar reader/writer), scenario
  rebuilding from a replay, and the match-result cache/fingerprint.
- `utama_core/analysis/` — offline analyses of tournament runs and replays (`turnover_breakdown`,
  `restart_outcomes`, `stuck_detector`, `render_window`, `render_clip`, `list_runs`). Imports
  `replay/`; `replay/` never imports it.
- Everything else (`motion_planning/`, `rsoccer_simulator/`, `team_controller/`,
  `data_processing/`, `entities/`, `global_utils/`) is infrastructure the strategy layer
  sits on top of. Writing a strategy rarely needs to change it, but it is the main work area
  for planner, simulator, vision and hardware work: see the Work areas table above.

**Before touching `utama_core/engine/`, `utama_core/tactics/`, `utama_core/strategy/`,
`utama_core/skills/`, or `tools/evaluation/`/`docs/strategies.md`, read
[`docs/STRATEGY_DEVELOPMENT.md`](docs/STRATEGY_DEVELOPMENT.md)** — the tactic-kernel model,
referee-restart handling, lessons from past tactic bugs, and the observability tooling
(`MatchLog.trace()`, `render_window()`, the strategy catalog) all live there, scoped to
that half of the repo rather than duplicated here for every task.

## Minimalism discipline

Add a new concept (a new base class, a new scheduling mechanism, a new config knob) only
after a concrete case forces it — not in anticipation of one. If a change starts
accumulating multiple new named concepts, stop and check whether that was actually asked
for. This is a deliberate, repeatedly-stated project preference, not an oversight to fix:
prefer the smallest mechanism that solves the problem actually in front of you. `docs/
tactic_model_design_decisions.md` documents several concepts explicitly deferred for
exactly this reason (concurrent-slot priority tuning, `Tactic` min/max robot-count
declarations) — check there before reintroducing one of them.

## Testing

- **Always pass `--headless`** when running any simulator/integration test:
  `pixi run pytest utama_core/tests/ --headless`. Much faster; there is no reason to run
  with graphics in an agent loop.
- `pixi run test` runs the default suite; `pixi run lint` runs the full pre-commit stack
  (black, ruff, isort) — run both before considering a change done.
- `--level quick|full` (defined in the root `conftest.py`, not `utama_core/tests/
  conftest.py`) scales the `robot_id`/`my_team_is_right` parametrizations; no test takes
  those parameters today, so both levels run the same tests. CI runs `--level quick` on
  PRs, `--level full` on push to a branch, both with `--ignore-glob "**/*grsim*"` (grsim
  tests need an external grsim process CI can't provide).
- The root `conftest.py` turns off `StrategyRunner`'s browser vision stream for every test.
  In rsim, `run_test`'s `episode_timeout` is game time (rsim runs several times faster than
  real time), so a test's budget doesn't depend on machine load.
- A known defect is a `strict=True` xfail on exactly the cases that hit it (see
  `bang_bang_edge_cases_test.py`'s `_seeds_with_known_defect`), so a fix shows as XPASS and
  a regression elsewhere fails. Don't chase rsim-only dribble test failures as if they were
  kernel bugs; verify on grsim if genuinely unsure.
- Before trusting any test result (yours or another agent's), prefer independently
  re-running it and reading real output over trusting a self-report — this codebase has
  been touched by both humans and agents, and a claimed "all tests pass" is only as
  trustworthy as the last time someone actually ran it.
- **A bug-fix commit is not done without a regression test that fails before the fix and
  passes after.** "Full suite unchanged" is necessary but not sufficient — it proves the
  fix broke nothing, not that the fix itself is protected. The test should pin the exact
  boundary condition (a tolerance, a tie, a zero, a timeout) the fix introduced, not just
  exercise the surrounding function. Prefer a pure unit test over an rsim fixture whenever
  the logic allows it — faster, and easier to isolate the one boundary that changed.

## Where things live

`docs/README.md` indexes every doc by kind and `docs/tools.md` every script; the ones agents
need most:

- `docs/STRATEGY_DEVELOPMENT.md` — tactic-kernel model, referee handling, writing a
  `Tactic`, observability tooling. Read before any strategy-layer change.
- `docs/tactic_model_design_decisions.md` — kernel/Tactic/Partitioner design rationale.
- `docs/custom_referee.md` — `CustomReferee` architecture/usage; its "Known gaps" section
  tracks genuinely open items (don't assume something is missing without checking there
  first — it may already be resolved and the surrounding doc just stale).
- `docs/custom_referee_design_decisions.md` — referee rule-by-rule design decisions.
- `docs/roadmap.md` — running list of larger, not-yet-scheduled workstreams; check before
  assuming a doc's claim about "not yet built" is still accurate — these drift.
- `docs/signals.md` — every signal we record about how a strategy plays (stalls, fouls, chances
  created and conceded, ball losses, pass receptions, restarts), grouped by the question it
  answers, with where it lives in `summary.json`, a reference range and what an off value means.
  Read it when designing or debugging a strategy, and before adding a metric; they are
  diagnostics, not objectives, except the outcome estimators in its "Ranking" section.
- `docs/strategies.md` — strategy catalog: status, description, and real round-robin
  results per `build_*_kernel_strategy` factory.
- `utama_core/tests/engine/` and `utama_core/tests/strategy_runner/` — the real
  tactic-kernel test surface; everywhere else is largely infrastructure (motion planning,
  vision, controllers) that predates and sits below the kernel model.

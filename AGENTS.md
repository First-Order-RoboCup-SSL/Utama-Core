# AGENTS.md

Durable context for any coding agent (not just Claude) working in this repo.

## What this repo is

Utama-Core is the RoboCup SSL (Small Size League) team's robot-control stack: vision
ingestion, motion planning, robot control, a simulator (`rsoccer_simulator`, "rsim"), and
the strategy layer that decides what each robot does. `Utama-Strategy` is a sibling repo
and is stale — all active strategy work happens here, on top of the tactic-kernel model.

## Repo map

- `utama_core/kernel/` — the scheduler/protocol infra: `Strategy`, `Tactic`, `KernelContext`,
  `MatchLog`, `AbstractStrategy`, referee-override plumbing. Strategy-dev work touches this
  rarely, mostly to add a new primitive, not a new strategy.
- `utama_core/strategy/` — the actual strategies people write, run, and compare
  (`kernel_strategy.py`'s `build_*_kernel_strategy` factories — `tiki_taka`, `counter_flow`,
  etc.). This is where day-to-day strategy-dev edits land.
- `utama_core/tactics/` — reusable `Tactic` implementations (`GiveAndGoTactic`,
  `PressAndContainTactic`, ...) that strategies compose.
- `utama_core/skills/` — lower-level per-robot primitives (`go_to_ball`, `block_attacker`,
  ...) tactics call directly.
- `utama_core/custom_referee/` — the in-process referee (rules, restart positioning).
- Everything else (`motion_planning/`, `rsoccer_simulator/`, `team_controller/`,
  `data_processing/`, `entities/`, `global_utils/`) is infrastructure the strategy layer
  sits on top of and mostly doesn't need to change to write a new strategy.

**Before touching `utama_core/kernel/`, `utama_core/tactics/`, `utama_core/strategy/`,
`utama_core/skills/`, or `tournament.py`/`docs/strategies.md`, read
`utama_core/kernel/AGENTS.md`** — the tactic-kernel model, referee-restart handling,
lessons from past tactic bugs, and the observability tooling (`MatchLog.trace()`,
`render_window()`, the strategy catalog) all live there, scoped to that half of the repo
rather than duplicated here for every task.

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
  conftest.py`) scales certain test parametrizations. CI runs `--level quick` on PRs,
  `--level full` on push to a branch, both with `--ignore-glob "**/*grsim*"` (grsim tests
  need an external grsim process CI can't provide).
- rsim has known, pre-existing dribble-physics/timing flakiness unrelated to strategy
  logic — see the `xfail(strict=False, ...)` markers already in the suite for the accepted
  pattern when a test is genuinely rsim-flaky, not a real bug. Don't chase rsim-only
  dribble test failures as if they were kernel bugs; verify on grsim if genuinely unsure.
- Before trusting any test result (yours or another agent's), prefer independently
  re-running it and reading real output over trusting a self-report — this codebase has
  been touched by both humans and agents, and a claimed "all tests pass" is only as
  trustworthy as the last time someone actually ran it.

## Where things live

- `utama_core/kernel/AGENTS.md` — tactic-kernel model, referee handling, writing a
  `Tactic`, observability tooling. Read before any strategy-layer change.
- `docs/tactic_model_design_decisions.md` — kernel/Tactic/Partitioner design rationale.
- `docs/custom_referee.md` — `CustomReferee` architecture/usage; its "Known gaps" section
  tracks genuinely open items (don't assume something is missing without checking there
  first — it may already be resolved and the surrounding doc just stale).
- `docs/custom_referee_design_decisions.md` — referee rule-by-rule design decisions.
- `docs/roadmap.md` — running list of larger, not-yet-scheduled workstreams; check before
  assuming a doc's claim about "not yet built" is still accurate — these drift.
- `docs/strategies.md` — strategy catalog: status, description, and real round-robin
  results per `build_*_kernel_strategy` factory.
- `utama_core/tests/kernel/` and `utama_core/tests/strategy_runner/` — the real
  tactic-kernel test surface; everywhere else is largely infrastructure (motion planning,
  vision, controllers) that predates and sits below the kernel model.

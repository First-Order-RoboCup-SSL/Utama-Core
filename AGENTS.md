# AGENTS.md

Durable context for any coding agent (not just Claude) working in this repo, before
touching `utama_core/kernel/`, `utama_core/tactics/`, or anything downstream of them.

## What this repo is

Utama-Core is the RoboCup SSL (Small Size League) team's robot-control stack: vision
ingestion, motion planning, robot control, a simulator (`rsoccer_simulator`, "rsim"), and
the strategy layer that decides what each robot does. `Utama-Strategy` is a sibling repo
and is stale — all active strategy work happens here, on top of the tactic-kernel model
described below.

## The tactic-kernel model

The strategy layer is not a behaviour tree. A team's play is a `kernel.Strategy`: a
scheduler that partitions outfield robots across concurrently-running `Tactic`s every
tick, re-deciding that partition fresh each tick via a `Partitioner` function. Robot 0
(goalkeeper) is pinned outside the scheduler and never scheduled.

- **`Tactic`** (`utama_core/kernel/tactic.py`) — a plain object: `tick(game, ctx,
  robot_ids, mem) -> (commands, mem)`, plus optional `applicable()`/`is_committed()`/
  `suggest_next()` hooks with sane defaults. No py_trees, no blackboard, no state-machine
  base class — `mem` is a plain dataclass the kernel only ever replaces wholesale or
  threads through unchanged.
- **`Strategy`** (`utama_core/kernel/strategy.py`) — runs N≥1 `Tactic`s concurrently, each
  owning a disjoint slice of the outfield pool.
- **`Partitioner`** — a plain function deciding how to split the *free* robot pool (robots
  no committed `Tactic` currently holds) across tactic slots this tick. No bid/fitness
  scoring system.
- **`AbstractStrategy`** (`utama_core/strategy/common/abstract_strategy.py`) — the base
  class `StrategyRunner` actually drives; wraps a `kernel.Strategy` built via a
  `build_kernel_strategy(motion_controller) -> kernel.Strategy` factory (see
  `utama_core/kernel/kernel_strategy.py` for the existing factory functions).

**Single-writer partition invariant:** exactly one place (the scheduler) decides the
*entire* partition once per tick, before any `Tactic` runs. By the time a `Tactic.tick()`
is called, its robot set for that tick is final — there is no window where two `Tactic`s
could contend for the same robot. This is what makes concurrent `Tactic`s safe without
locks or explicit synchronization; do not reintroduce a code path that lets a `Tactic`
claim or release robots outside the `Partitioner`'s decision.

Full rationale, rejected alternatives, and the "why" behind every one of these choices
lives in `docs/tactic_model_design_decisions.md` — read it before proposing a change to
the kernel's shape, not just this summary.

## Referee handling

`CustomReferee` (`utama_core/custom_referee/`) is an in-process, mode-agnostic referee —
works identically across rsim/grsim/real, no network dependency. During a restart
(kickoff/ball-placement/free-kick/penalty), `kernel.RefereeOverride`
(`utama_core/kernel/referee_override.py`) takes over every outfield robot's command
directly — this happens *before* any `Tactic` ticks, not as a `Tactic` itself. Design
rationale and the full rule-by-rule audit against the SSL rulebook: `docs/custom_referee.md`
and `docs/custom_referee_design_decisions.md`.

## Writing a Tactic

Lessons from building `SwitchOfPlayTactic` (`utama_core/tactics/switch_of_play.py`), a
multi-phase relay tactic that took two full debugging rounds to get right. The bugs below
weren't one-offs — the same *shape* of bug recurred twice in the same file, so they're
worth checking for deliberately rather than trusting "it worked once."

- **`go_to_point()` always faces the ball — it has no orientation parameter.** If a robot
  needs to hold a specific orientation while stationary (e.g. facing a pass target, not
  the ball), use `move()` directly with an explicit `target_oren`. This bit the same
  tactic twice: once in an early phase (carrier holding the ball before passing) and again,
  independently, one phase later (a different robot's own hold-and-wait branch) — a fix in
  one call site does not imply the pattern is fixed everywhere it appears. Grep every
  `go_to_point(` call in a new tactic and check whether the robot's orientation while
  stationary actually matters there.
- **`intercept_point()` (`shared/pass_and_score_geometry.py`) projects the receive point
  along the *passer's current orientation*,** not toward anything about the receiver's
  actual position. A passer facing the wrong way (see above) silently sends the receiver
  toward a nonsense point — this fails quietly (the receiver just never arrives) rather
  than erroring, so it reads as a vague "stall" until traced.
- **`is_committed()` returning `True` is a promise, not a suggestion.** It blocks the
  kernel scheduler from ever reassigning that tactic slot's robots — by design, not a bug
  (see "Single-writer partition invariant" above). Every code path that sets it `True` for
  a phase transition needs a matching path back to `False`, covering both the success case
  *and* every failure/timeout case. A timeout-driven phase reset is easy to write in a way
  that gets silently undone within the same tick, if the reset-target phase's own logic
  immediately re-advances past it — check that a timeout reset actually sticks for at least
  one full tick before the tactic can re-advance.
- **Verify by tracing a real match, not by reading the phase-transition logic.** Bugs in
  this tactic were invisible from the code alone — `intercept_point()` computing a
  plausible-looking point that happened to be wrong, or a phase timeout resetting state
  that then got immediately overwritten — and only showed up as `src_oren`/`intercept_pos`
  oscillating tick-to-tick in an actual traced match. Add a phase-transition + key-variable
  trace (see prior debugging sessions' harnesses for the pattern) before concluding a phase
  is stuck for some subtler reason than it looks.
- **A tactic can trip referee rules that have nothing to do with its own logic.** Positions
  computed correctly by one tactic can still walk a robot through another tactic's
  keep-out zone (e.g. `SwitchOfPlayTactic`'s relay play putting an attacker inside its own
  defense area, tripping the same `DefenseAreaRule` foul that a broken `DefenseTactic` also
  trips). When a match stalls on a referee foul, check *which* robot and *which* rule
  before assuming the fix belongs in the tactic that seems most related.

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

- `docs/tactic_model_design_decisions.md` — kernel/Tactic/Partitioner design rationale.
- `docs/custom_referee.md` — `CustomReferee` architecture/usage; its "Known gaps" section
  tracks genuinely open items (don't assume something is missing without checking there
  first — it may already be resolved and the surrounding doc just stale).
- `docs/custom_referee_design_decisions.md` — referee rule-by-rule design decisions.
- `docs/roadmap.md` — running list of larger, not-yet-scheduled workstreams; check before
  assuming a doc's claim about "not yet built" is still accurate — these drift.
- `utama_core/tests/kernel/` and `utama_core/tests/strategy_runner/` — the real
  tactic-kernel test surface; everywhere else is largely infrastructure (motion planning,
  vision, controllers) that predates and sits below the kernel model.

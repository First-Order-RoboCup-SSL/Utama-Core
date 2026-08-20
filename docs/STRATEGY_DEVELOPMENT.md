# Strategy development

Durable context for building/debugging anything under `utama_core/engine/`,
`utama_core/tactics/`, `utama_core/skills/`, `utama_core/strategy/`, or
`tournament.py`/`docs/strategies.md` — the strategy-layer half of the repo. See the root
`AGENTS.md` for repo-wide facts (what the repo is, testing commands, minimalism
discipline). That file links here; this file assumes you've read it first.

`engine/` (infra: `Strategy`, `Tactic`, `KernelContext`, `MatchLog`, `AbstractStrategy`,
referee-override plumbing) and `strategy/` (the actual `build_*_kernel_strategy` factories
— `tiki_taka`, `counter_flow`, etc.) used to both be named `kernel/`, which was confusing:
"kernel strategy" is the model's own vocabulary (baked into every factory/class name below),
so a directory also named `kernel` collided with it. `engine/` removes that collision — the
vocabulary "kernel strategy" is unchanged, only the infra directory's name is.

## The tactic-kernel model

The strategy layer is not a behaviour tree. A team's play is a `kernel.Strategy`: a
scheduler that partitions outfield robots across concurrently-running `Tactic`s every
tick, re-deciding that partition fresh each tick via a `Partitioner` function. Robot 0
(goalkeeper) is pinned outside the scheduler and never scheduled.

- **`Tactic`** (`utama_core/engine/tactic.py`) — a plain object: `tick(game, ctx,
  robot_ids, mem) -> (commands, mem)`, plus optional `applicable()`/`is_committed()`/
  `suggest_next()` hooks with sane defaults. No py_trees, no blackboard, no state-machine
  base class — `mem` is a plain dataclass the kernel only ever replaces wholesale or
  threads through unchanged.
- **`Strategy`** (`utama_core/engine/strategy.py`) — runs N≥1 `Tactic`s concurrently, each
  owning a disjoint slice of the outfield pool.
- **`Partitioner`** — a plain function deciding how to split the *free* robot pool (robots
  no committed `Tactic` currently holds) across tactic slots this tick. No bid/fitness
  scoring system.
- **`AbstractStrategy`** (`utama_core/engine/abstract_strategy.py`) — the base class
  `StrategyRunner` actually drives; wraps a `kernel.Strategy` built via a
  `build_kernel_strategy(motion_controller) -> kernel.Strategy` factory (see
  `utama_core/strategy/kernel_strategy.py` for the existing factory functions).

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
(`utama_core/engine/referee_override.py`) takes over every outfield robot's command
directly — this happens *before* any `Tactic` ticks, not as a `Tactic` itself. It reuses
the restart-positioning `*Step` classes in `utama_core/custom_referee/actions.py` rather
than reimplementing keep-out-distance geometry. Design rationale and the full rule-by-rule
audit against the SSL rulebook: `docs/custom_referee.md` and
`docs/custom_referee_design_decisions.md`.

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
  oscillating tick-to-tick in an actual traced match. Use `ctx.match_log.trace(...)` (see
  Observability below) to record the key variable per tick and read it back — not a
  hand-added `print()` you have to remember to revert — before concluding a phase is stuck
  for some subtler reason than it looks.
- **A tactic can trip referee rules that have nothing to do with its own logic.** Positions
  computed correctly by one tactic can still walk a robot through another tactic's
  keep-out zone (e.g. `SwitchOfPlayTactic`'s relay play putting an attacker inside its own
  defense area, tripping the same `DefenseAreaRule` foul that a broken `DefenseTactic` also
  trips). When a match stalls on a referee foul, check *which* robot and *which* rule
  before assuming the fix belongs in the tactic that seems most related.

## Observability — don't hand-roll a debug print, these already exist

Answering "why did this match go the way it did" has dedicated tooling; reach for it
before adding an `os.environ`-gated `print()` you'll have to remember to add and revert.

- **`MatchLog`** (`utama_core/engine/match_log.py`) — one JSONL trace per match, two event
  kinds sharing the same file/reader:
  - `intention(...)` — auto-recorded by `Strategy` itself, one row per tactic-slot
    assignment *change* (not per tick), so a robot holding the same tactic for seconds is
    one line, not thousands. You don't call this directly.
  - `trace(tick, sim_time, key, value)` — call this yourself, from inside any `Tactic.tick()`
    or any skill that receives `ctx`, via `ctx.match_log.trace(...)`. Record any
    JSON-serializable scalar/string per tick (a branch taken, `has_ball`, a computed angle).
    Guard with `if ctx.match_log is not None:` — it's `None` (a no-op) on every
    tournament/CI run, so leaving trace calls in permanently costs nothing. See
    `go_to_ball()` (`utama_core/skills/src/go_to_ball.py`) and `GiveAndGoTactic.tick()`
    (`utama_core/tactics/give_and_go.py`) for the pattern already in place.
  - Enable per-match by passing `match_log_path=...` to `StrategyRunner`/`AbstractStrategy`,
    or via `tournament.py run_match(..., run_dir=...)` which wires it automatically.
  - Read back with `utama_core.engine.match_log.load_jsonl(path)` — returns a list of
    `IntentionEvent`/`TraceEvent` in tick order; filter by `isinstance`.
- **`render_window()`** (`utama_core/replay/render_window.py`) — renders a PNG of robot/ball
  trails over a time window from a replay `.pkl`, for the one thing text traces are bad at
  (spatial motion). `render_around_event()` anchors the window on a `MatchLog` event index
  directly, instead of guessing a raw timestamp.
- **`docs/strategies.md`** — the strategy catalog: every `build_*_kernel_strategy` factory,
  its status (`baseline`/`competitive`/`parked`/`experimental`), and real round-robin
  results. Check here before treating an old strategy's win/loss record as current, and
  before assuming a strategy is worth using as a comparison target — `baseline`-status
  strategies (e.g. `default`, `low_block`) are not meant to be competitive; don't spend
  effort making them "win." Its own "Updating this file" section explains when to add/edit
  a row.
- **`tournament.py`** — round-robin match runner, `--max-workers N` for concurrency;
  `run_match(config_a_name, config_b_name, run_dir=None)` is directly importable for a
  one-off match with full observability recorded, not just the CLI's exclusion-filtered
  round-robin (e.g. `default` is excluded from the CLI sweep but reachable via `run_match`
  directly). Config names passed to `run_match` are the full factory name
  (`build_tiki_taka_kernel_strategy`), not the short catalog name (`tiki_taka`).

## Where things live

- `utama_core/engine/` — scheduler/protocol infra (`Strategy`, `Tactic`, `KernelContext`,
  `MatchLog`, `AbstractStrategy`, referee-override plumbing). Rarely touched to add a new
  strategy; touched to add a new kernel-level primitive.
- `utama_core/strategy/kernel_strategy.py` — every `build_*_kernel_strategy` factory. This
  is where day-to-day strategy-dev edits land.
- `docs/tactic_model_design_decisions.md` — kernel/Tactic/Partitioner design rationale.
- `docs/custom_referee.md` — `CustomReferee` architecture/usage; its "Known gaps" section
  tracks genuinely open items (don't assume something is missing without checking there
  first — it may already be resolved and the surrounding doc just stale).
- `docs/custom_referee_design_decisions.md` — referee rule-by-rule design decisions.
- `docs/strategies.md` — strategy catalog: status, description, and real round-robin
  results per `build_*_kernel_strategy` factory. See Observability above.
- `utama_core/tests/engine/` and `utama_core/tests/strategy_runner/` — the real
  tactic-kernel test surface.

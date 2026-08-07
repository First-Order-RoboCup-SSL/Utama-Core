# Tactic Model — Design Decisions

This document captures the design rationale for the multi-tactic scheduling architecture
(the "Tactic model") being implemented in `utama_core/kernel/` and `utama_core/tactics/`
on `spike/tactic-kernel`. It replaces a single fixed behaviour tree per team colour with a
scheduler that can run different tactics on different robots and change that assignment
as the game situation changes.

It followed a comparative study of Sumatra (TIGERs Mannheim's open-source SSL AI) and an
explicit OS-process-scheduling analogy, both used to borrow vocabulary and stress-test the
design, not as a mandate to copy either one's implementation weight.

Items marked **Settled** are decisions the team has committed to. Items marked **Deferred**
are recognised open questions being deliberately left unbuilt until a concrete case forces
them.

---

## 1. ✅ Settled — Naming: `Strategy` (scheduler) and `Tactic` (per-tactic unit)

**Decision:** the top-level, per-team-colour object is called `Strategy` — it matches the
existing `StrategyRunner` in `utama_core/run/strategy_runner.py`, which already expects to
be handed a "Strategy"-shaped thing, so no new top-level noun was needed. Each individual
behaviour (goalkeeper, two-robot-attack, dribble, …) is called a `Tactic`.

**Rejected alternatives and why:**

| Name considered | Rejected because |
|---|---|
| `Runner` (for the scheduler) | Collides with the existing `StrategyRunner` class name. |
| `Assigner` / `Dispatcher` / `Formation` / `Squad` | Either read as robot-roster/substitution management (not what this does), or carried unwanted sports-formation connotations. No separate class was needed once `Strategy` already denotes the scheduling layer. |
| `Play` (for the per-tactic unit) | Matched Sumatra's own `APlay`/`EPlay` vocabulary, but read as too close to an arbitrary BT/py_trees command term rather than a concrete concept. |
| `Tactic` (final choice) | Plain domain vocabulary — names what a goalkeeper/attack/dribble routine actually is, independent of any OS or scheduling framing, without requiring a reader to already know the analogy. |

---

## 2. ✅ Settled — `Tactic` interface

```python
def tick(game, ctx, robot_ids, mem) -> tuple[dict[RobotId, RobotCommand], mem]:
    ...

def committed(game, mem) -> bool:
    return False  # default: freely reassignable

def make_initial_mem() -> mem:
    ...
```

`robot_ids` is passed into `tick()` rather than fixed at construction time, since a Tactic
can no longer assume ownership of a fixed robot roster — the scheduler tells it who it owns
this tick. `mem` is a plain dataclass holding whatever state the Tactic needs; there is no
named-state-machine base class (see §6).

An optional `suggest_next` value may be returned alongside commands, naming another Tactic
the scheduler might want to switch to. It is purely advisory — the scheduler is free to
ignore it. This was called `yield_to` earlier; renamed because a plain "I'm done, no
preference" yield is expected to be the common case, and a name built around always naming
a target overweights the rarer case.

---

## 3. ✅ Settled — `committed()`: an absolute, self-declared veto

**Decision:** a Tactic may return `committed(game, mem) -> True` to prevent the scheduler
from reassigning its robots away mid-action (e.g. mid-pass). This veto is currently
**absolute** — no timeout, no forced eviction, no scheduler override — except via the
barrier reset described in §4.

**Naming history:** called `locked()` for a period, then renamed. "Lock" is mutex
vocabulary — mutual exclusion between concurrent actors contending for one shared resource.
That is not what this is: one Tactic, unilaterally refusing a request from the scheduler.
`committed()` names a one-sided declaration, not a concurrency primitive, and matches the
word already used naturally to describe the concept ("once the pass is committed, it can't
be interrupted").

**Why not add a timeout / forced-eviction path?** Considered and rejected — not because the
underlying concern (a buggy Tactic could get stuck committed forever) is wrong, but because
a bounded veto requires inventing a threshold with no principled way to pick one yet. The
realistic failure mode (a Tactic's phase logic has a bug and never returns to an
uncommitted phase) is not solved by the barrier reset either, since that only bounds the
damage to "until the next qualifying referee transition." It is deliberately left unsolved
at the architecture level. The mitigation is a debugging aid, not a runtime safety net: the
kernel logs every tick a commitment blocks a reassignment, so a Tactic committed for
hundreds of consecutive ticks is a visible signal of a bug to go fix in that Tactic's own
phase logic — not a case for scheduler-side eviction machinery guarding against bugs that
don't exist yet.

---

## 4. ✅ Settled — Three severities of reset, not two

Referee commands don't form one category for the scheduler's purposes. They split into
three distinct depths:

| Tier | Trigger | Effect |
|---|---|---|
| Reboot | New game / new half | A fresh `Strategy` entirely; everything below this restarts from nothing. |
| **Barrier reset** | Referee restart signal — kickoff prep, penalty prep, ball placement, a free-kick command, goal — anything meaning "a new phase of play is about to start" | **All** Tactics' `mem` resets and all `committed()` vetoes clear, uniformly, for every Tactic at once. Not a targeted eviction of one stuck Tactic — the previous tick's commitments stop being meaningful for everyone simultaneously. |
| SIGSTOP/CONT-equivalent | Referee `STOP`/`HALT` → resume | Game freezes and later continues from the **same** state. `mem` and commitments should survive this, not reset. |

**Correction made mid-design:** the barrier-reset case was initially described as
equivalent to `SIGKILL`. This was rejected — `SIGKILL` is targeted, ending one specific
process uncatchably, and the OS does not restart it. The barrier reset has no single
target: it is an all-Tactics reset, not a kill aimed at whichever Tactic happens to be
stuck. Keeping the `SIGKILL` label would have smuggled back in exactly the per-Tactic
forced-eviction mechanism deliberately left out in §3.

**Open judgment call:** the exact mapping from `RefereeCommand` enum values to these tiers
is a real decision, not something with an obviously correct answer derivable from the enum
alone — see the implementation for the concrete mapping chosen and the reasoning recorded
alongside it in code.

---

## 5. ✅ Settled — Single-writer partition; one Tactic claims the whole outfield pool

**Decision:** every outfield robot maps to exactly one Tactic per tick (robot 0 /
goalkeeper is pinned separately and never scheduled). For this pass, exactly **one** Tactic
is active at a time, claiming the entire outfield pool — there is no concurrent
multi-Tactic partitioning (e.g. 3 robots on attack while 2 hold defence, simultaneously).

**Why this avoids real shared-resource concurrency, not just simplifies it:** the hard
version of concurrency (locks, races, deadlock) arises when multiple independent actors can
act on a shared resource without knowing about each other, and correctness depends on
interleaving. Making assignment **single-writer** — exactly one place (the scheduler)
partitions all outfield robots into disjoint groups once per tick, before any Tactic runs —
eliminates that class of problem by construction rather than solving an instance of it. By
the time a Tactic's `tick()` is called, its robot set for that tick is already final; there
is no window in which two Tactics could contend for the same robot.

**Deferred, not rejected:** wanting simultaneous Tactics later does not require real
multi-core-style scheduling technique (time-slicing, preemption within a "core"). It is
still a partition — just with more than one non-empty part
(`{attack: [1,2,3], defence: [4,5]}` instead of `{attack: [1,2,3,4,5]}`). The single-writer
invariant survives unchanged; only the shape of the scheduler's output would change. This
is not built now because a fixed priority order for splitting robots across concurrent
Tactic kinds is itself a tuning decision (should defence outrank attack, always?) with no
forcing case yet. When two Tactic kinds genuinely can't be merged into one
robot-count-agnostic Tactic and must run concurrently, that is the concrete case that
justifies building it.

---

## 6. ✅ Settled — No state-machine framework for Tactics

**Decision:** `mem` is a plain dataclass; `committed()` is a plain boolean method. There is
no `IState`/`AState`-style state-machine base class, no named-transition framework.

**Context:** Sumatra's `ARole`/`ASkill` each carry a full `StateMachine`/`IState` with a
`MAX_STATE_CHANGES_PER_UPDATE` loop guard. This is a reasonable design at Sumatra's scale
(49 modules, live-tunable constants via `@Configurable`, a dedicated tactical-analysis layer
feeding role assignment) but was judged more machinery than justified for a handful of known
Tactics. The team's standing rule: add a name or concept only after a concrete case breaks
the plain version, not in anticipation of one.

---

## 7. Deferred — Splitting policy across concurrent Tactics

See §5. Not building a scheduler-level priority cascade for splitting the robot pool across
multiple simultaneously active Tactics. The chosen interim approach is to write
**robot-count-agnostic** Tactics that claim whatever pool they're given and behave sensibly
regardless of size, keeping any such complexity local to one Tactic's own logic rather than
centralising it in the scheduler where it would entangle with every other Tactic kind's
demands.

## 8. Deferred — Zombie/orphan robot cleanup

A robot going physically unreachable (disconnected) while its Tactic is `committed()` has no
reclaim path in the current design — the commitment would latch until the next barrier
reset. Not addressed now; mitigated only by the same commit-blocking log described in §3.

## 9. Deferred — Deadlock / starvation detection

No runtime mechanism detects a Tactic that never yields or never becomes uncommitted.
Mitigated by logging only, per §3. Revisit if this proves to be a real, observed problem
rather than a hypothetical one.

---

## 10. ✅ Settled — `StrategyRunner` integration via an `AbstractStrategy` adapter

**Decision:** `utama_core/kernel/kernel_strategy.py` adds `KernelStrategy(AbstractStrategy)`,
which satisfies `AbstractStrategy`'s contract (`create_behaviour_tree`, `assert_exp_robots`,
`assert_exp_goals`, `get_min_bounding_req`, `load_game`, `step`) but overrides `step()` to
tick a `kernel.Strategy` (for the outfield pool) plus the pinned goalkeeper tactic directly,
bypassing py_trees entirely for command computation. `StrategyRunner` drives it exactly like
any other `AbstractStrategy` — no changes to `StrategyRunner` itself were needed.

**Why an adapter and not a second runner:** `StrategyRunner` already owns vision/referee
ingestion, robot controllers, replay writing, and the real/rsim/grsim mode split — none of
that is specific to the behaviour-tree path. Reimplementing a parallel runner for the Tactic
model would duplicate all of that infrastructure to replace only the one part
(`AbstractStrategy.step()`) that actually differs. `create_behaviour_tree()` still returns an
empty `Selector`, since `AbstractStrategy.__init__` unconditionally builds a tree — it is
simply never ticked.

**Motion controller timing constraint:** `KernelContext` needs a `MotionController` instance,
but `KernelStrategy` doesn't have one at construction time — `StrategyRunner` only injects it
via `load_motion_controller()` onto the blackboard. Confirmed from `StrategyRunner.__init__`'s
call order that `_load_robot_controllers()` (which calls `load_motion_controller`) always runs
before `_load_game()`. `KernelStrategy` therefore defers building its `kernel.Strategy` until
`load_game()` is called, reading `self.blackboard.motion_controller` at that point rather than
requiring it be passed in some other way.

**Referee-command mapping validated against the SSL referee state machine:** the barrier/pause
tiers in §4 were cross-checked against the official referee-command transition diagram
(`Halted` → `Stopped` → `{PrepareKickoff, BallPlacement, PreparePenalty}` → `Running`, with
`Stop`/`Halt` reachable from any state). All of the diagram's restart-preparation states
(`PrepareKickoff`, `PreparePenalty`, `BallPlacement`, `FreeKick`) are entered from `Stop`, and
`Stop` itself is a distinct state — that already matches `_BARRIER_ENTRY_COMMANDS` treating
those as barrier-tier regardless of the preceding command, and `_PAUSE_COMMANDS` treating
`STOP`/`HALT` as a plain pause. The diagram also confirms `ForceStart` can fire directly out of
`Stop` (not only out of a barrier-tier command) — already handled correctly, since that path
resolves to `ResetTier.NONE` (résumé from a pause, not a new phase) rather than
`ResetTier.BARRIER`.

**Only one outfield tactic exists so far:** `build_default_kernel_strategy()` wires a
single-tactic pool (`two_robot_attack`) with a picker that has nothing to actually choose
between. This is not a stand-in allocation policy — it is the direct consequence of §7's
deferral: no second concrete outfield tactic exists yet to force a real splitting/allocation
decision, so none was invented. Callers needing more than one outfield tactic should construct
their own `kernel.Strategy` with a real `Picker` rather than use this helper.

---

## Explicitly rejected (do not reintroduce without a concrete forcing case)

- **Bid / fitness-scoring + arbiter** for choosing the active Tactic — recognised as
  equivalent to OS priority-preemption scheduling; more machinery than the current Tactic
  count justifies.
- **Timeout / forced eviction on `committed()`** — see §3.
- **Concurrent multi-Tactic partitioning** — see §5, §7.
- **`@Configurable`-style live-tuning/config system** — Sumatra-specific infrastructure
  judged out of scope at current scale.
- **Scheduling quantum / reduced check frequency** — solves a context-switch cost this
  system doesn't have; checking assignment every tick is already free here.

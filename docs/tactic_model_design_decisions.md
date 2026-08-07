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

## 5. ✅ Settled, later generalized (§11) — Single-writer partition

**Decision:** every outfield robot maps to exactly one Tactic per tick (robot 0 /
goalkeeper is pinned separately and never scheduled). For this pass, exactly **one** Tactic
is active at a time, claiming the entire outfield pool — there is no concurrent
multi-Tactic partitioning (e.g. 3 robots on attack while 2 hold defence, simultaneously).
**Superseded by §11**: `Strategy` was later generalized to run N≥1 Tactics concurrently:
what follows in this section is the original single-Tactic reasoning, which still holds as
the N=1 case of that generalization, not as a separate mechanism.

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

## 7. Deferred, then resolved — Splitting policy across concurrent Tactics

See §5. Originally deferred: not building a scheduler-level priority cascade for splitting
the robot pool across multiple simultaneously active Tactics, on the grounds that no two
concrete Tactic kinds existed yet to force the question, and robot-count-agnostic Tactics
were a sufficient interim answer. **Superseded by §11** once a genuine 5-robot
attack/defense split became a real, concrete requirement — see §11 for the resolution.
Robot-count-agnostic Tactics remain the right approach *within* a group (this section's
original point stands unchanged for that); what changed is that the pool can now be split
into more than one group at all.

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

## Ported tactics inventory

Beyond the three tactics ported alongside the kernel itself (`GoalkeeperTactic`,
`TwoRobotAttackTactic`, `DribbleTactic`), a survey of both repos' `plays/`, `strategies/`,
`examples/`, and `behaviours/` directories (across `Utama-Strategy`'s `spike/functional-strategy`
branch and several feature branches: `feat/dribbling`, `feat/pass`, `feat/pose`, `test/goalkeep`,
plus Utama-Core's own branch history) turned up:

- **`DefenseTactic`** (`utama_core/tactics/defense.py`) — ported from
  `utama_strategy.examples.defense_strategy.DefenceStrategy`. The original was almost entirely
  role-assignment plumbing around one already-self-contained skill function,
  `utama_core.skills.src.defend_parameter.defend_parameter`, which already lived in Core and did
  not itself need porting. Fills a real gap: no defensive positioning existed anywhere in the
  kernel before this. Carries over a pre-existing quirk unchanged (see the class docstring): the
  2-defender dynamic side-selection triggers on the whole team's robot count being exactly 2, not
  on how many robots this tactic was handed — correct for the original dedicated-defense
  strategy, not necessarily once a defense tactic runs alongside other concurrent tactics.
- **Not ported, candidates for later:** an off-ball "go-to-space" positioning tactic
  (`feat/pass` branch — scores open space by passing-lane openness, opponent race-to-point, goal
  threat, teammate spacing) and a 2v1 attack variant with a fast/slow-kicker recharge-cooldown
  role split (`plays/attack2v1.py`). Both are genuinely distinct from what's ported but were
  left for a follow-up pass — the former needs verification against the current `Game` API since
  it lives on a stale branch, the latter needs its cooldown timer reworked to live in `mem`.
- **Not ported, rejected as redundant:** `plays/pose.py`/`plays/receive.py` (subsumed by
  `two_robot_attack`'s existing setup-phase movement), the older random-target dribble picker in
  `utils/dribble_utils.py` (superseded by the already-ported `DribbleTactic`'s fixed
  rectangle-corner pattern), and all of Utama-Core's own non-`spike/tactic-kernel` branches
  (vision/UI/referee infra only, no tactical logic).

---

## 11. ✅ Settled — `Strategy` generalized to run N≥1 Tactics concurrently

**Decision:** `Strategy` (§5) itself was generalized to run more than one `Tactic` at once
by splitting the outfield pool into named tactic slots each tick, rather than building this
as a second, separate scheduler class. This is the concrete resolution of §7's deferral,
forced by a genuine requirement: a 5-outfield-robot (6-total, goalkeeper pinned) strategy
where some robots attack (`LeadAndSupportTactic`) while others defend
(`ShadowAndMarkTactic`) concurrently, not one full-pool Tactic switching wholesale between
the two.

**One class, not two.** An earlier pass at this built a second class, `PartitionedStrategy`,
kept separate from `Strategy` on the reasoning that `Strategy`'s `Picker` (one `TacticId`
out) and a group-partition picker (a full partition out) were different-shaped decisions.
That reasoning didn't hold up: a single-tactic pick is just the degenerate case of a
partition with one non-empty part (`{"only": all_robots}`), and carrying two schedulers
with near-identical tick loops (referee handling, barrier reset, per-slot committed-veto
logic) was duplicated machinery for one capability at two different N, which cuts against
the standing minimalism preference (see [[feedback_minimal_architecture]]) more than the
"different-shaped decision" argument justified keeping them apart. `Strategy` now always
operates on a partition (`GroupPicker`); the single-active-tactic case is reproduced exactly
via `Strategy.single_tactic_picker(picker)`, which adapts the original
`(Game, Optional[TacticId]) -> TacticId` shape into a trivial one-slot `GroupPicker` — so
existing single-tactic callers change their construction call, not their picker logic.

**The single-writer invariant is preserved, not weakened.** A `GroupPicker` still decides
the *entire* outfield pool's assignment in one place, once per tick, before any Tactic in
any slot runs — structurally identical to the original single-Tactic design, just producing
N≥1 slots instead of always exactly 1. There is still no tick in which two Tactics could
contend for the same robot.

**Per-slot `committed()`, not pool-wide.** One slot's Tactic returning `committed()` only
pins *that slot's* robot set; it has no bearing on any other slot's assignment. The
`GroupPicker` contract reflects this directly: it is only ever handed the *free* robot
pool — outfield robots not currently pinned by a committed slot — and must return a
partition that exactly covers that free pool, nothing more. This was chosen over an
earlier draft where the picker proposed a partition of the *entire* pool and the kernel
tried to reconcile it against pinned slots after the fact — that required guessing at
ambiguous cases (what if the picker's proposal disagreed with a pin?) instead of making
disagreement structurally impossible. The final contract cannot express a bad proposal for
a committed slot: the picker never sees committed slots' robots as available, so it cannot
suggest reassigning them, only strictly define what happens to the rest.

**`single_tactic_picker`'s "currently active" state is read from `Strategy`, not tracked
separately.** A first version of the adapter kept its own `active_id` inside the closure,
updated each time it was called. This desynced from `Strategy`'s actual state the moment a
caller replaced the picker mid-run (a fresh closure starts with no memory of what was
previously active) — caught by a test that swapped pickers after a tactic released its
commitment, which should have handed control to the newly-preferred tactic immediately but
didn't. Fixed by reading the previously active id out of `prev_partition` (which `Strategy`
always passes to the picker) instead of closure-local state, since `Strategy` is the single
source of truth for what was active.

**A slot dropped from the partition must be cleared, not left stale.** A second, related
bug: when a tactic id is simply absent from a tick's partition (its robots went to another
slot, or the picker stopped naming it), nothing in the main tick loop touches that slot at
all — it isn't "assigned an empty set," it's just not mentioned. A naive port kept the
slot's previous `assigned_robots` untouched in that case, which broke `active_tactic_id` for
single-tactic callers (the old, no-longer-active tactic still looked "assigned"). Fixed by
explicitly zeroing any slot's `assigned_robots`/`mem`/`committed_ticks` when its id is absent
from the current tick's partition.

**A slot assigned zero robots is not ticked.** Discovered while testing: comparing an empty
`frozenset()` assignment against a slot's *initial* default (also `frozenset()`) looks like
"no change," so a naive reset-on-change check would call `tactic.tick()` with an empty robot
tuple and `mem=None` on a slot's first zero-robot tick — meaningless, and a guaranteed crash
for any Tactic that reads `robot_ids[0]`. Fixed by skipping `tick()` entirely for any slot
with an empty assignment, and only calling `make_initial_mem()` when a slot's new assignment
is non-empty.

**Barrier reset still clears everything, unconditionally.** A referee restart clears every
slot's `mem` and every slot's `committed()` pin at once — a partial reset that left one
slot's commitment standing would reintroduce a targeted-eviction-shaped mechanism through
the back door, which §4 already rejected under the `SIGKILL` framing.

**Validation is loud, not best-effort.** `Strategy.tick()` raises if a `GroupPicker`'s
combined output (free-pool partition, merged with pinned committed slots) is not an exact,
disjoint cover of the outfield pool, or if it names a tactic id that was never registered,
or if it tries to assign robots to a currently-committed slot.

**Not built:** any priority/scoring system for *how* the split is decided — the picker used
for the 5-robot case (`_possession_split_picker` in `kernel_strategy.py`) is a single-signal
rule (whichever side is closer to the ball) with two fixed splits, not a tunable allocator.
Consistent with the standing rejection of bid/fitness-scoring machinery below; this section
is the mechanism for running a split, not a policy for choosing one.

---

## 12. Ported tactics — 5-robot pool: `LeadAndSupportTactic` and `ShadowAndMarkTactic`

Two new tactics, written from scratch rather than ported/renamed from Utama-Strategy's
`plays/`/`strategies/` (the team explicitly did not want a "2v1"/"go-to-space"-style
carryover) — reusing existing motion/geometry primitives (`go_to_ball`, `go_to_point`,
`defend_parameter`, `game.proximity_lookup`, `shared/pass_and_score_geometry`'s shot-finding
functions) since those are fundamental building blocks, not tactical decisions.

- **`LeadAndSupportTactic`** (`utama_core/tactics/lead_and_support.py`) — one ball-carrying
  leader (closest-to-ball, re-evaluated except while committed) dribbles/shoots toward
  goal; every other assigned robot continuously re-picks the best open support point,
  scored on lane-openness, shot-openness from that point, and forward progress. Named for
  what it does (one robot leads, the rest support), not a robot count or sports-formation
  term. Robot-count-agnostic (1 robot: leader only, no supports, up through however many
  assigned) per §7's original, still-standing local-agnosticism stance.
- **`ShadowAndMarkTactic`** (`utama_core/tactics/shadow_and_mark.py`) — the first two
  assigned robots shadow the shot line via the existing `defend_parameter` (unchanged,
  same as `DefenseTactic`); any further robots each mark the nearest not-yet-marked
  opponent, greedily, per tick. Built because `defend_parameter` itself has no concept of
  a 3rd+ defender — calling it per-robot for a larger defensive group would send every
  robot to the same post rather than spreading coverage. Marking assignment is not sticky
  (a marker can switch targets tick-to-tick); acceptable since marking carries no
  phase/commitment state to disrupt, revisit only if mark-flapping proves to be a real
  observed problem.
- **`build_split_shape_kernel_strategy`** (`utama_core/kernel/kernel_strategy.py`) wires
  both into a `Strategy` (§11) with `_possession_split_picker`, usable as a
  `KernelStrategy`'s `build_kernel_strategy` argument exactly like the existing
  `build_default_kernel_strategy` single-tactic factory.

---

## Explicitly rejected (do not reintroduce without a concrete forcing case)

- **Bid / fitness-scoring + arbiter** for choosing the active Tactic — recognised as
  equivalent to OS priority-preemption scheduling; more machinery than the current Tactic
  count justifies. Applies equally to `Strategy`'s multi-slot `GroupPicker` (§11) — the
  5-robot split picker is a one-signal rule, not a scored allocator.
- **Timeout / forced eviction on `committed()`** — see §3. Applies per-slot in the
  multi-Tactic case too (§11): a stuck committed slot is a bug to fix in that slot's
  Tactic, not a case for scheduler-side eviction.
- **`@Configurable`-style live-tuning/config system** — Sumatra-specific infrastructure
  judged out of scope at current scale.
- **Scheduling quantum / reduced check frequency** — solves a context-switch cost this
  system doesn't have; checking assignment every tick is already free here.

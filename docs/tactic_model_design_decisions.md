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

**Gap this table alone doesn't close:** a barrier reset clears scheduling state (`mem`,
commitments) but says nothing about what a robot should *physically do* while a restart
command is active — a Tactic ticking normally during e.g. the opponent's ball placement
would still drive straight at the ball, an SSL rule violation, not just a scheduling
wrinkle. See §13 for the override that actually closes this.

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

## 13. ✅ Settled — Referee-restart legality override, reusing the BT path's `actions.py`

**Problem:** §4's barrier reset only ever cleared scheduling state. Nothing stopped a
Tactic from ticking its ordinary logic during `PREPARE_KICKOFF_*`/`BALL_PLACEMENT_*`/
`DIRECT_FREE_*`/`PREPARE_PENALTY_*` — e.g. `LeadAndSupportTactic`'s leader driving
straight at the ball during the opponent's ball placement. Noticed only once the kernel
model was actually run against a real referee sequence, not caught by any scheduling-level
test, because scheduling was working exactly as designed — it was simply never asked to
cover this.

**Decision: reuse, don't reinvent.** The BT path (`utama_core/strategy/referee/
{actions,tree}.py`) already solves this correctly and is already tested
(`test_referee_unit.py`, `test_referee_rsim.py`, `test_ball_placement_rsim.py`) — a
priority Selector matches the live `RefereeCommand` and, when matched, takes over every
friendly robot's command for that tick via `*Step` classes (`_clear_to_legal_positions`,
`_project_outside_circle`, formation-position helpers), bypassing the strategy tree
entirely. Rebuilding equivalent keep-out-distance geometry inside the kernel model purely
to avoid a py_trees dependency would be duplicated logic with its own, separate bug
surface — worse than the coupling it avoids.

**Mechanism (`utama_core/kernel/referee_override.py`):** the `*Step` classes are
`AbstractBehaviour` (py_trees) subclasses, but every one of them touches exactly three
`blackboard` attributes — `game`, `motion_controller`, `cmd_map` — nothing else (verified
by grep, not assumed). `RefereeOverride` is a plain class holding one long-lived instance
of each relevant Step (long-lived because `BallPlacementOursStep` carries cross-tick state
— `_release_started_at`/`_placer_id` — that a fresh instance every tick would silently
discard, re-triggering its release-delay logic every single tick). Its `tick()` builds a
minimal duck-typed shim exposing just those three attributes, assigns it as the Step's
`.blackboard`, calls `.update()`, and returns the resulting `cmd_map`.

**Wiring:** `Strategy.tick()` checks `is_override_command(current_command)` right after
the existing pause check (`is_paused`) and before `_choose_partition` — if it matches, the
override computes every outfield robot's command for that tick and no Tactic ticks at all
that tick. Slots already had `mem`/commitments cleared by the barrier-reset transition on
the way in (§4); once the restart ends (`NORMAL_START`/`FORCE_START` arriving from a
non-barrier-tier command — see `classify_transition`), ticking simply resumes normally
with fresh `mem`, so there is nothing further to reconcile. `HALT`/`STOP`/`TIMEOUT_*` are
deliberately *not* routed through the override — `is_paused` already satisfies "stop
issuing motion" more directly by returning `{}`, matching the BT path's own `StopStep`
being just "stop cold, which happens to also satisfy the keep-distance rule."

**Goalkeeper:** `KernelStrategy.step()` ticks `GoalkeeperTactic` for robot 0 separately
from `Strategy` (§10's design). During an override, `RefereeOverride`'s Step classes
already compute a command for every `game.friendly_robots` id including the goalkeeper
(mirroring the BT path, which does the same) — so `KernelStrategy.step()` skips its own
goalkeeper tick whenever the override produced a command for that id, rather than
overwriting the override's placement with normal ball-tracking logic mid-restart.

**Follow-up fix:** `GoalkeeperTactic` was initially still ticked unconditionally during
`HALT`/`STOP` (unlike the outfield pool, which `is_paused` already froze) — a pre-existing
gap, not introduced by this change, but inherited silently at first. `KernelStrategy.step()`
now also checks `is_paused(current_command)` before ticking the goalkeeper, skipping to the
same `execute_default_action` → `empty_command(False)` fallback the outfield pool already
uses during a pause.

**A restart arriving mid-commitment is not a distinct case to test.** Considered whether a
Tactic that is `committed()` when a restart begins needs special handling (e.g. does the
override correctly override a stubborn committed slot). It cannot come up: `Strategy.tick()`
runs the barrier reset (unconditionally clearing every slot's `mem`/commitment) strictly
before checking `is_override_command` (see the wiring above), both on the very same tick the
restart command first arrives. By the time the override branch can possibly run, no slot has
been ticked since the reset, so there is no committed state left to override — this is a
structural guarantee from ordering, not a case needing its own test.

**Back-to-back restarts (no intervening `NORMAL_START`)** — e.g. a kickoff foul immediately
becoming a ball placement — are handled correctly: each new `RefereeCommand` re-runs
`classify_transition`/the barrier reset and `RefereeOverride._step_for` re-dispatches to
that command's own Step on the very next tick, with no memory of the previous restart's
target. Verified at the `Strategy.tick()`/command-dict level
(`test_back_to_back_restarts_dispatch_a_fresh_step_each_time`) rather than by watching rsim
robot positions settle — the ball itself can drift for several seconds after a restart (a
known rsim convergence quirk, see `dribble.py`'s KNOWN ISSUE note), which makes "did the
robot's position stabilize" an unreliable proxy for "did the override actually switch,"
even though the switch itself is instant and correct.

---

## 14. ✅ Settled, with a known trade-off — Opponent defense-area avoidance at the planner level

**Found via a live grsim run** of the split-shape strategy (`demo_split_shape_match.py`),
watched through the referee GUI: an outfield attacker chasing the ball drove straight into
the opponent's defense area — an SSL rule violation the referee flagged directly ("Yellow
attacker in blue defense area"). Neither §13's referee-restart override nor any Tactic had
anything to do with this — it's a live-play rule (applies during `NORMAL_START`/`FORCE_START`,
not a restart), and nothing anywhere in the codebase enforced it: not the BT path (only
`StopStep` clears the opponent's defense area, and only during `STOP`), not any tactic, not
the motion planner (`FastPathPlanner._get_obstacles` treated only robots and the field
boundary as obstacles).

**Decision:** fix it once, at the planner level (`utama_core/motion_planning/src/
fastpathplanning/planner.py`), not per-tactic — the rule applies to every non-goalkeeper
robot unconditionally, so it belongs where all tactics' motion already funnels through,
not as something each tactic author has to remember. The opponent's defense area becomes a
rectangular obstacle; the friendly defense area is deliberately NOT added (the goalkeeper,
ticked outside `Strategy` per §10, needs to enter it, and this planner has no notion of
"except the goalkeeper").

**Three distinct planner gaps, found in sequence, not one:**

1. **No obstacle at all** for the opponent's defense area — the base bug. Fixed by adding it
   to `_get_obstacles`.
2. **`sanitize_target` only reacts to a target near an obstacle *line***, not one *inside* an
   *enclosed* rectangle — a target placed at the defense area's exact center is farther than
   `OBSTACLE_CLEARANCE` from all four edges simultaneously, so the segment-only check never
   fires. Fixed by adding `_enemy_defense_rect`/`_project_outside_rect`, an explicit
   inside-the-rectangle check applied to the raw target before the rest of the pipeline runs.
3. **`smooth_path`'s projected "carrot"** (`PROJECTION_DISTANCE` = 1m along the direction to
   the first waypoint) is derived fresh from the robot's live position every tick, entirely
   unchecked against obstacles — confirmed by tracing a real crossing where the final,
   already-"smoothed" waypoint landed several centimetres inside the rule boundary even
   though the underlying `check_segment` trajectory correctly routed around it. This is a
   pre-existing gap in shared smoothing logic (affects any close obstacle, not just this
   one), only now exposed because a defense area sits still and close for long enough to
   matter — most obstacles (moving robots) rarely do. Fixed with a second, final
   `_project_outside_rect` safety net after smoothing, rather than chasing every individual
   unguarded point inside `check_segment`/`smooth_path`'s recursive subgoal search.

**Known trade-off, accepted:** the margin needed to actually stop the violation
(`OPPONENT_DEFENSE_AREA_KEEP_DISTANCE`, reused from `actions.py`'s restart-time standoff) is
large enough that it also redirects a target placed deliberately at/near the exact boundary
line — `utama_core/tests/motion_planning/multiple_robots_test.py::test_mirror_swap`'s
formation targets at `(3.5, ±0.75)` sit precisely on a standard-field defense area's edge,
and now fail. A zero-margin ("bare rule boundary") version of the same checks was tried and
measured to still let a fast, head-on approach cross several centimetres into the real
defense area — preventing the actual SSL violation took priority over that synthetic test's
exact-boundary targets. `_enemy_defense_rect(game, margin)` keeps `margin` as an explicit
parameter (rather than hardcoding the standoff into the geometry) specifically so a future,
better-tuned fix — e.g. a velocity-aware margin, or fixing the overshoot at its true source
inside `smooth_path` rather than papering over it with a final clamp — can revisit each call
site independently without re-deriving the rectangle logic.

**A second, distinct violation surfaced by the same grsim run:** "Too many yellow defenders in
own area" — a different SSL rule (max 1 non-goalkeeper robot in *your own* defense area, vs.
§14's "opponent's area, zero robots ever") that §14's planner-level fix has no bearing on at
all, since it only concerns which robot is allowed to be somewhere, not a universal
geometric constraint the shared planner can arbitrate.

**Root cause:** `ShadowAndMarkTactic`'s fallback for a marker with no opponent left to mark
called `defend_parameter` again — the same shot-shadowing call the two real shadow-defenders
use. Once a team has more than 2 robots, `defend_parameter`'s side-selection degrades to a
fixed `post_limit if robot_id == 1 else -post_limit` (see its source) — keyed purely off the
raw `robot_id`, with no notion of "which slot is calling me." A fallback marker therefore
lands on whichever post its numeric ID happens to map to, which very likely already belongs
to one of the two real shadow-defenders, converging multiple robots right at the edge of our
own defense area at once.

**Fix:** `_fallback_hold_target` — an open-space holding point roughly a third of the way
from the defense area's front edge to the centre line (clearly outside it, unlike
`defend_parameter`'s shadow post which sits only `ROBOT_RADIUS` outside on purpose), stacked
above/below the ball's `y` by index so multiple unmatched markers don't collide with each
other either. Does not touch `defend_parameter` itself — that function's behaviour for the
1-2 shadow-defender case is correct and unchanged; only `ShadowAndMarkTactic`'s own fallback
path was wrong.

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

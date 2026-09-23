# Tactic Model — Design Decisions

Why the strategy layer (`utama_core/engine/`, `utama_core/tactics/`,
`utama_core/strategy/kernel_strategy.py`) is shaped the way it is. The code is the source of
truth for *how*; this file records *why* and what was deliberately not built. The design
borrowed vocabulary from Sumatra (TIGERs Mannheim) and OS process scheduling to stress-test
ideas, not as a mandate to copy either's weight. For the general allocation framework see
[`scheduling_math_model.md`](scheduling_math_model.md).

Section numbers are stable because code comments cite them. Gaps are sections merged or
removed: §7 (splitting policy) was resolved by §11; §8/§9 (orphan robots, deadlock detection)
are under Deferred; §12/§17 (tactic and config inventories) live in `docs/strategies.md` and
the code.

## Settled

**§1. Names: `Strategy` (scheduler) and `Tactic` (unit of behaviour).** `Strategy` matches the
existing `StrategyRunner`. Rejected: `Runner` (collides), `Assigner`/`Dispatcher`/`Squad`
(read as roster management), `Play` (Sumatra's term, too close to BT vocabulary).

**§2. `Tactic` protocol** (`engine/tactic.py`): `tick(game, ctx, robot_ids, mem) ->
(commands, mem)`, mandatory `tag`, optional `applicable()`, `is_committed()`, `suggest_next()`
with safe defaults, plus `make_initial_mem()`. `robot_ids` is passed per tick because a Tactic
doesn't own a fixed roster. `mem` is a plain dataclass. `suggest_next` is advisory only.

**§3. `is_committed()` is a self-declared veto on reassignment** (e.g. mid-pass). Named for a
one-sided declaration, not a mutex (`locked()` was rejected). The veto is absolute *within*
the commitment deadline: `Strategy` releases a slot that has stayed committed past
`commitment_deadline_s` (default `DEFAULT_COMMITMENT_DEADLINE_S` = 15s) *and* whose ball has
moved less than ~5cm since the commitment began — a stall breaker added after a match froze
for 566s, not a play-length cap. A deadline release is logged to `MatchLog` and means the
tactic is missing a release path; fix the tactic, don't raise the deadline. The kernel also
warns every 100 consecutive committed ticks.

**§4. Three reset severities** (`engine/referee_reset.py` holds the exact command mapping):
- *Reboot* — new game/half: a fresh `Strategy`.
- *Barrier reset* — entering a new phase of play (kickoff/penalty prep, free kick, ball
  placement, goal; or resuming from one; or `FORCE_START` straight out of a pause, since the
  ball may have been moved during the stop — `46e962b`): every slot's `mem` resets and every
  commitment clears, uniformly. Not a targeted eviction (the "SIGKILL" framing was rejected
  for that reason).
- *Pause* — `HALT`/`STOP`: play freezes; `mem` and commitments survive, and a pause resumed
  by `NORMAL_START` picks up from the same state.

**§5. Single-writer partition.** One place (the `Partitioner`, via `Strategy`) decides the whole
outfield assignment once per tick, before any Tactic runs. Two Tactics can never contend for a
robot, so no locks are needed. The goalkeeper is pinned outside the scheduler
(`Strategy.set_goalkeeper`).

**§6. No state-machine framework.** No Sumatra-style `IState`/`AState` base class; phase logic
lives in `mem`. Add a concept only after a concrete case breaks the plain version.

**§11. `Strategy` runs N≥1 Tactics concurrently, in one class.** A single-tactic pick is the
degenerate one-slot partition (`Strategy.single_tactic_picker` adapts a
`(Game, Optional[TacticId]) -> TacticId` picker). A separate `PartitionedStrategy` was built
and removed as duplicated machinery. Consequences, each learned from a bug:
- Commitment is per slot: a committed slot pins only its own robots; the `Partitioner` only
  ever sees the free pool, so it cannot propose reassigning a pinned robot.
- `mem` resets exactly when a slot's robot set changes (compared as sets).
- A slot absent from the partition is cleared, not left stale; a slot with zero robots is
  not ticked.
- A free robot need not be assigned; an uncovered robot simply isn't ticked.
- `_validate_partition` raises on overlap, unknown ids, or claims outside the free pool —
  loud, not best-effort.
- The "currently active" id for `single_tactic_picker` is read from `prev_partition`, not
  closure state, so swapping pickers mid-run can't desync.

**§10. `AbstractStrategy` is kernel-native** (BT/py_trees removed: `087ee4b`, `960662c`,
`48affd6`). `StrategyRunner` drives it unchanged; it builds the `Strategy` in
`load_motion_controller()` because `build_kernel_strategy(motion_controller)` needs the
controller but not `game`.

**§13. Referee restarts are handled before any Tactic ticks** (`engine/referee_override.py`).
For `STOP`, `TIMEOUT_*` and every restart command, `RefereeOverride` computes every outfield
robot's command from the long-lived `*Step` classes in `custom_referee/actions.py` (reused,
not reimplemented — keep-out geometry has one home). Steps are driven through a three-attribute
shim (`game`, `motion_controller`, `cmd_map`). The goalkeeper's pinned tactic ticks only if the
override produced no command for it (kickoff steps exempt the keeper). A restart arriving
mid-commitment needs no special case: the barrier reset runs before the override on the same
tick. Strategies can replace any restart formation via `Strategy(referee_overrides=...)`
(`7cd1f61`). Tactics never contain referee logic.

**§14. Opponent defense-area avoidance lives in the planner**, not in tactics: the rule applies
to every non-goalkeeper robot, so it belongs where all motion funnels. The friendly area is not
an obstacle (the keeper must enter it). Known trade-off: the keep-out margin also redirects
targets placed exactly on the boundary (e.g. `test_mirror_swap`'s formation targets). The
separate "too many defenders in own area" violation was a `ShadowAndMarkTactic` fallback bug,
fixed with `_fallback_hold_target`.

**§15. Tactic selection at scale: boolean `applicable()` + a closed tag set, not scoring.**
- Eligibility filtering is `O(T)`: `Strategy` asks every uncommitted Tactic's `applicable(game)`
  and passes only applicable ids to the `Partitioner`. This keeps adding Tactic #200 from
  requiring a new `Partitioner` per subset (`O(2^T)`).
- `is_committed()` is the exit gate and is checked first; `applicable()` is the entry gate
  and only gates uncommitted slots. A Tactic that stops being applicable while committed is
  dropped the first tick its commitment ends, with no extra logic.
- Why boolean: a self-reported float or ordinal score is structurally unfalsifiable — an
  (agent-authored) Tactic has every incentive to over-report. A precondition is a local,
  unit-testable claim.
- `tag` is a closed set (`attack`/`defense`/`mixed`); an open vocabulary had no consumer.

**§16. One small hand-written `Partitioner` per `Strategy` config**, not a general-purpose
allocator. Allocation is disposable, config-local logic. Consequently `tag` is declared on
every Tactic but not read by any allocation code (it is logged to `MatchLog`); build a
tag-based helper only when one config must allocate across two same-tagged Tactics. `tag` is
used for dashboard colouring only.

**Testing convention:** generic Tactic/config properties are table-driven
(`tests/engine/test_all_tactics.py`, `test_all_strategy_configs.py`); behaviour specific to one
Tactic or config gets its own test.

## Explicitly rejected (don't reintroduce without a concrete forcing case)

- Bid/fitness-scoring arbiter for choosing tactics or splits — reaffirmed at large `T` (§15).
- Externally tuned per-tactic timeout/forced eviction — the commitment deadline (§3) is a
  kernel-wide stall breaker gated on no ball progress, not a tuned per-tactic timeout.
- `@Configurable`-style live tuning; scheduling quanta / reduced check frequency (checking
  every tick is already free).

## Deferred

- A scheduler-level priority cascade (or tunable allocator) for splitting robots across
  concurrent slots — each config's `Partitioner` decides its own split (§11, §16).
- Per-Tactic `min_robots`/`max_robots` declarations. Real gap:
  `PassAndShootTactic` reads `robot_ids[1]`, worked around by `_fixed_ratio_picker`'s
  `min_attack` (guarded by
  `test_fixed_ratio_picker_low_block_split_respects_min_attack_floor`).
- A pluggable/generic selection mechanism over `applicable()` results — revisit around
  `T`≈15-20 same-config tactics.
- A self-declared commitment horizon (tactic states its max commitment length at commit time).
- Scheduler-level agentic design (an agent tuning the `Partitioner` itself) — a bad allocation
  policy is a global failure, so sequence it after tactic-level scale and real match data.
- Reclaiming a committed robot that physically disconnects (currently latched until the
  deadline or next barrier).
- Unported candidates from `Utama-Strategy`: a "go-to-space" off-ball positioning tactic
  (`feat/pass` branch) and a 2v1 fast/slow-kicker variant (`plays/attack2v1.py`).

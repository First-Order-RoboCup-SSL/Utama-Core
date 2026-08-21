# Strategy catalog

Every `build_*_kernel_strategy` factory in `utama_core/strategy/kernel_strategy.py`,
what it does, and whether it's worth spending further effort on. Run any pair with
`pixi run python tournament.py <name1> <name2>` (name = the factory name with the
`build_`/`_kernel_strategy` trimmed, e.g. `tiki_taka`).

**Status legend**

- `baseline` — exists to exercise kernel machinery or serve as a minimal-risk
  reference point, not meant to be competitive. Don't spend effort making these
  "win"; that's not their job.
- `competitive` — a real playable team: reads live game state, has both an attack
  and a defense answer, allocation reacts to possession/zone. This is where
  strategy-quality effort should go.
- `parked` — a competitive attempt that was tried against a strong opponent,
  didn't win, and effort was deliberately redirected elsewhere rather than
  continuing to iterate. Root cause is documented; not scheduled for further work
  unless someone picks it back up.
- `experimental` — isolated/benchmark config, not meant to represent a realistic
  match posture at all (e.g. no defense slot).

## Backfill round-robin (2026-08-20, re-run after the block_attacker/go_to_ball fixes)

Every non-`default` config played every other once (91 matches — 14 configs,
up from 13 with `counter_flow` added — 60s sim time, 6v6 headless rsim, 8
concurrent workers), re-run specifically to see how the catalog looks after
the two shared-skill fixes below. Supersedes the original 2026-08-20 backfill
(`replays/tournament_20260820_104203/`), which predated both fixes and is no
longer representative. Current run: `replays/tournament_20260820_113853/summary.json`.

| Strategy | W-D-L | GF-GA |
|---|---|---|
| `tiki_taka` | 4-8-1 | 8-4 |
| `counter_flow` | 3-10-0 | 6-2 |
| `high_press` | 3-9-1 | 4-2 |
| `press_and_pass` | 3-9-1 | 5-1 |
| `overload_press` | 2-11-0 | 3-1 |
| `give_and_go_solo` | 2-9-2 | 3-3 |
| `split_shape` | 2-9-2 | 2-2 |
| `three_slot` | 1-9-3 | 1-3 |
| `zone_fluid` | 1-7-5 | 3-9 |
| `high_line_zone` | 0-13-0 | 0-0 |
| `low_block` | 0-12-1 | 0-2 |
| `switch_of_play` | 0-12-1 | 0-1 |
| `counter_press` | 0-11-2 | 0-3 |
| `decoy_and_overload` | 0-11-2 | 0-2 |

`tiki_taka` still has the best record (4W), but it is **no longer
undefeated** — its one loss is to `counter_flow` (2-1, see the Competitive
section). `counter_flow` itself is the only strategy with zero losses across
all 91 matches (3W-10D-0L) — on a losses-first read it is arguably the more
robust strategy now, even though `tiki_taka` still has more wins. Both are
reasonable reference points; judge new work against whichever one it's
actually trying to beat.

`high_line_zone` is a new concern: it went from a real mid-pack record
(2-8-2, GF4-GA3, beat `low_block` 1-0 and `zone_fluid` 3-1) in the original
backfill to **0-13-0, GF0-GA0** — zero goals scored or conceded in any of
its 13 matches post-fix, despite still showing real possession swings and
occasional shots (e.g. 2 shots vs `zone_fluid`, 1 vs `high_press`/`switch_of_play`).
Spot-checked one match (`vs low_block`, previously a 1-0 win, now 0-0): the
picker cycles through `block`/`switch`/`overload` normally and does reach
the finishing tactic for real stretches, so this isn't the same "never
reaches a shoot phase" bug as `counter_press`/`decoy_and_overload` — more
likely `DecoyOverloadTactic`'s actual finishing conversion got worse now
that `go_to_ball`'s approach angles changed, which is plausible but not
confirmed. Flagged, not yet root-caused; see Known open bugs.

## Baselines — don't fix, don't judge by these

Several of these don't have both a real attack and a real defense answer, or
use a fixed split that ignores game state — they exist to exercise kernel
machinery, not to compete. Their round-robin record above is not a quality
signal; do not treat a loss here as something to fix.

| Strategy | Added | Status | Description |
|---|---|---|---|
| `default` | pre-2026-08-19 | baseline | Single-tactic pool: everyone runs `PassAndShootTactic`, only ever commands the first 2 outfield robots (robots 3-5 are zombies — known bug, see [investigation below](#known-open-bugs)). Exists to exercise minimal kernel scheduling, not to be a real team. Excluded from the round-robin/catalog discovery entirely (see `tournament.py`). |
| `split_shape` | pre-2026-08-19 | baseline | `LeadAndSupportTactic` (attack) + `ShadowAndMarkTactic` (defense), split by possession edge. First concrete forcing case for the kernel's splitting policy. 2-10-0 in the backfill — undefeated but mostly draws, consistent with a reactive-but-not-tuned baseline. |
| `press_and_pass` | pre-2026-08-19 | baseline | `GiveAndGoTactic` (attack) + `PressAndContainTactic` (defense), possession-edge split. Same shape as `split_shape`, newer tactic pair. 2-9-1. |
| `high_press` | pre-2026-08-19 | baseline | Same tactic pair as `press_and_pass` but a fixed 80/20 attack-heavy split, ignoring possession — demonstrates a non-reactive `Partitioner` on the same tactics. Best raw W-D-L in the backfill (4-8-0) purely from being relentlessly attack-heavy against weaker baselines; drew `tiki_taka` 0-0. Not a signal to build more strategies this way — see the backfill note above. |
| `low_block` | pre-2026-08-19 | baseline | `PassAndShootTactic` (attack, floored at 2 robots) + `DefenseTactic` (defense), fixed 20/80 defense-heavy split. The original minimal-risk pairing. Worst record in the backfill (0-7-5, GF0-GA6) — consistent with its defense-heavy split giving it almost no attacking output. **Known bug: draws 0-0 against `default`** — see [investigation](#known-open-bugs). |
| `three_slot` | pre-2026-08-19 | baseline | First 3-concurrent-slot config: `PressAndContainTactic` + `ShadowAndMarkTactic` + `GiveAndGoTactic`. Exercises N>2 scheduling, not tuned for strength. 2-9-1. |
| `decoy_and_overload` | pre-2026-08-19 | baseline | `DecoyOverloadTactic` (attack, floored at 2) + `ShadowAndMarkTactic` (defense), fixed 50/50 split. Exercises the lure/overload tactic in isolation. 0-11-1, GF0 — never scored across 12 matches; the isolated lure/overload pairing doesn't generate offense on its own. |
| `give_and_go_solo` | pre-2026-08-19 | experimental | Entire pool always runs `GiveAndGoTactic`, no defense slot at all. Isolated benchmark for tuning give-and-go internals without a defensive confound — not a fielding-ready config. 1-10-1. |
| `switch_of_play` | pre-2026-08-19 | baseline | `SwitchOfPlayTactic` (attack, floored at 3 for the carrier/pivot/runner relay) + `DefenseTactic`, fixed 50/50 split. Exercises the switch tactic in isolation. 0-11-1, GF0 — same never-scores pattern as `decoy_and_overload`; isolated attack tactics don't produce goals without a complementary posture. |

## Competitive — real playable teams, use these for comparison going forward

| Strategy | Added | Status | Description |
|---|---|---|---|
| `counter_flow` | 2026-08-20 | competitive, **strongest by losses (0 across 91 matches)** | Third anti-tiki_taka strategy, after `overload_press`/`high_line_zone` were parked. Instead of trying to out-number tiki_taka's 2-robot shadow defense (the bet both prior attempts made and lost before ever getting to test it), fights tiki_taka on its own ground: `GiveAndGoTactic` for attack (tiki_taka's own proven-undefeated engine), `PressAndContainTactic` for our own press on turnover, `BlockShapeTactic` for the defensive screen. 3/2 split in both postures, with possession-edge hysteresis (same fix `high_line_zone` needed). Building this surfaced two shared bugs, both now fixed (see Known open bugs): `block_attacker` never contesting the ball, and `go_to_ball` having no opponent-awareness. Beat `tiki_taka` 2-1 (reproducible) at just 22% possession — the first and still only loss `tiki_taka` has anywhere in the catalog. In the full 91-match re-run backfill, `counter_flow` is 3W-10D-**0L**, the only strategy with zero losses. |
| `tiki_taka` | 2026-08-20 | competitive, most wins, one loss (**pre-2026-08-21-fix numbers — stale, see Known open bugs**) | Possession team: 3 give-and-go attackers + 2 shadow-and-mark cover when we have the ball; 3 pressers + 2 shadow when we don't. Best raw record in the 91-match re-run backfill (4W-8D-1L, GF8-GA4) — but no longer undefeated; its one loss is to `counter_flow` (2-1). A `GiveAndGoTactic` orientation-discontinuity bug (fixed 2026-08-21, see Known open bugs) was suppressing this strategy's own scoring the whole time these numbers were recorded — e.g. `tiki_taka vs zone_fluid` is now a reproducible 1-0 rather than the 2-1 shown here, and `vs counter_flow` is now 2-2 rather than a 1-2 loss. Full backfill not yet re-run post-fix; treat this row's numbers as an undercount of `tiki_taka`'s real record until it is. |
| `zone_fluid` | 2026-08-20 | competitive, real but weaker (**pre-2026-08-21-fix numbers — stale, see Known open bugs**) | Zone-adaptive team: man-shape defense throughout; give-and-go trio builds through the middle thirds, hands off to the decoy/overload duet in the final third. 1-7-5 in the re-run backfill, GF3-GA9 — genuinely reactive (unlike the baselines) but loses more than it draws or wins, including 1-2 to `tiki_taka` (recorded as unchanged across both 2026-08-20 backfills, but see below — that pairing wasn't actually stable once the `GiveAndGoTactic` bug is fixed). Also runs `GiveAndGoTactic`, so this row shares the same staleness as `tiki_taka`'s; full backfill not yet re-run post-fix. |
| `counter_press` | 2026-08-20 | **5 layered bugs root-caused and fixed 2026-08-21; scoring still rare — see Known open bugs** | Transition team: full press when the ball is lost and pressable, low block (`BlockShapeTactic`) when it isn't, 4-up switch-of-play attack the moment the ball is won. Zero goals scored in every match across both backfills (0-11-2 in the re-run, GF0). Neither the `block_attacker` fix nor the `go_to_ball` fix resolved this — the actual cause was 5 separate bugs in `SwitchOfPlayTactic`'s relay/finish path, all now fixed and individually trace-verified (see Known open bugs). Post-fix: went from *never* scoring (0 goals across 91+ matches) to scoring at least once (1-0 vs `low_block` in one traced match), but still mostly draws — the remaining blocker is a turn-budget-vs-window-duration mismatch, not a bug in these 5 fixes. Full backfill not yet re-run post-fix. |

## Parked — tried against tiki_taka, didn't win, not being iterated further

| Strategy | Added | Status | Description |
|---|---|---|---|
| `overload_press` | 2026-08-20 | parked, improved post-fix but still not competitive-tier | Built to outnumber tiki_taka's 2-robot shadow line (4-robot overload/switch attack + 1 block insurance) and hit direct on turnovers before tiki_taka's press organizes. Original result: lost 3%/97% possession to tiki_taka before either shared fix, and was genuinely weak overall (1-10-1). Post-fix re-run backfill: 2-11-0, GF3-GA1, zero losses — a real improvement, worth another look before `high_line_zone` if the anti-tiki_taka thread is picked back up again, though not yet re-tested head-to-head against `tiki_taka` itself. |
| `high_line_zone` | 2026-08-20 | parked, **regression root-caused and fixed 2026-08-21** | Built to deny tiki_taka's give-and-go trio 1v1s via a zone screen (`BlockShapeTactic`) instead of man-marking, switching the ball past its thin 2-robot cover. Original backfill: real mid-pack strategy (2-8-2, GF4-GA3) — beat `low_block` 1-0 and `zone_fluid` 3-1. Post-`block_attacker`/`go_to_ball` re-run backfill regressed to 0-13-0, GF0-GA0 (see Known open bugs for the root cause and fix) — verified fixed: re-run vs `low_block` scores 1-0 again. Full 91-match backfill not yet re-run post-fix; treat the pre-regression 2-8-2 record as the best current estimate until it is. |

## Known open bugs

- **`block_attacker` didn't contest the ball (fixed 2026-08-20)** — `PressAndContainTactic`'s
  presser only ever held a fixed 10%-of-the-way shot-line standoff point,
  never converging on the ball itself, so every press-using strategy could
  be held off indefinitely once tiki_taka had the ball. Root cause behind
  the ~5-7% possession ceiling every anti-tiki_taka strategy hit
  independently (`overload_press`, `high_line_zone`, `counter_press`,
  `high_press` in the backfill, and `counter_flow` before this fix). Fixed
  in `utama_core/skills/src/block.py`: within 0.5 m the presser now drives
  at the ball directly. See `counter_flow`'s and `counter_press`'s rows
  above for before/after numbers.
- **`go_to_ball` had no opponent-awareness (fixed 2026-08-20)** — always approached
  the ball from the robot's own current position, so two robots converging
  from opposite sides wedged at a fixed distance short of the ball instead of
  either one reaching it (the exact mechanism behind the `default_vs_lowblock`
  stalemate below). Diagnosed for a second, unrelated reason this session:
  direct `has_ball` instrumentation on `counter_flow`'s carrier showed it held
  the ball only ~0.25s out of 29+ seconds sampled after the `block_attacker`
  fix restored possession — it was winning the territorial battle but losing
  every actual ball-retrieval race. Fixed in `utama_core/skills/src/go_to_ball.py`:
  within 0.5m of a contesting enemy, approach from the far side of the ball
  relative to that enemy (a shield) instead of a straight line from wherever
  we currently are. Result: `counter_flow` went from a 0-0 draw (93%
  possession, 0 shots) to a **2-1 win over `tiki_taka`** at just 22%
  possession — ball control, not territory, was the real bottleneck.
  `tiki_taka`'s own matches against `zone_fluid`/etc. are unchanged, so this
  didn't regress the strategy that was already working.
- **`counter_press` never scored, even after both fixes above (5 layered bugs
  root-caused and fixed 2026-08-21, scoring still rare — see below)** — not
  explained by either shared-skill issue above. All 5 live in
  `SwitchOfPlayTactic`'s relay/finish path
  (`utama_core/tactics/switch_of_play.py`) and were found sequentially, each
  fix revealing the next bug that had been masked behind it:
  1. **Ball-ejection noise in "relay" treated as a real loss.** rsim has a
     known dribble-physics quirk where holding a ball for an extended period
     can eject it for a tick with no tactic-level cause (see the rsim
     dribble-issues project note). A single ejection tick sent the relay
     source robot straight into `go_to_ball`, discarding hold/aim progress.
     Mitigated (not root-cause-fixed — the real fix belongs in rsim's
     dribbler physics) via a `_BALL_RECOVERY_RADIUS` grace window: treat the
     ball as still held if within 0.3m even on a `has_ball=False` tick.
  2. **`_ARRIVAL_SPEED_THRESHOLD` too tight.** 0.05 m/s flickered on ordinary
     station-keeping jitter (observed: speed oscillating 0.03-0.07 m/s around
     a robot that had, for practical purposes, arrived), flapping
     `runner_ready` and never letting the relay pass leg start. Widened to
     0.1 m/s.
  3. **`_find_best_shot`'s largest-gap selection had no hysteresis.** Same
     class of bug as `high_line_zone`'s regression below and `_weak_side`'s
     pre-existing margin gate: recomputing the best shot gap fresh every tick
     let ordinary defender jitter flip which gap "won" between near-equal
     candidates, swinging `target_oren` tick to tick even when the true
     defensive picture was near-static. This fed a PID derivative-term kick
     (see #4) every time it flipped. Fixed via `prev_best_shot_y`/
     `switch_margin` hysteresis threaded through `_find_best_shot`/
     `find_best_shot`/`_score_goal` and all 4 real call sites (`switch_of_play`,
     `pass_and_shoot`, `decoy_and_overload` x2).
  4. **No PID/motion-controller reset across a target-orientation
     discontinuity.** Confirmed via direct code reading:
     `motion_controller.reset(robot_id)` — which clears the angular PID's
     per-robot `pre_errors`/`integrals` (`utama_core/motion_planning/src/pid/pid.py`)
     and the acceleration limiter's per-robot `_last_values`
     (`utama_core/motion_planning/src/common/acceleration_limiter.py`) — was
     never called anywhere in the codebase before this fix. The runner's
     commanded orientation jumps discontinuously on the "relay"->"finish"
     transition (relay's ball-holding aim -> finish's shot-aim); with no
     reset, the derivative term computed against a stale, unrelated
     `pre_errors[robot_id]` produced a large wrong-signed angular command
     that the acceleration limiter could then only unwind gradually — a
     multi-second non-convergent spin. Root-caused via direct match trace:
     63 consecutive "finish" ticks, every one `"turning"`, zero reaching
     `"kick"`. Fixed by calling `ctx.motion_controller.reset(runner_id)`
     exactly once, on the "relay"->"finish" transition tick.
  5. **Same discontinuity, one level down, inside "finish" itself.** Fixing
     #4 revealed that `has_ball`/shot-lane-open can flicker tick to tick
     *within* "finish" (a marker stepping in/out of the shot lane, or a
     momentary `has_ball` miss), bouncing the runner between the
     shoot-attempt branch and the chase-ball/hold branches — each driving a
     different commanded orientation, so every re-entry into shooting poisoned
     the PID with stale state the same way the phase transition did. Fixed
     by tracking re-entry (`SwitchOfPlayMem.was_shooting`) and resetting the
     motion controller only on that edge, mirroring fix #4's pattern.

  All 5 fixes individually trace-confirmed: target orientation stays stable
  across a shooting window (drift <0.1 rad/s), the runner turns in the
  correct (shortest-path) direction at up to max angular velocity, and at
  least one match reached `kick()` with `scored=True` (1-0 vs `low_block`).
  However, **scoring is still rare post-fix** (4 further matches in this
  session's regression batch were 0-0 draws) — root cause, also trace-confirmed,
  is a genuine turn-budget-vs-window-duration mismatch, not a 6th instance of
  the discontinuity bug class above: turning ~3.8 rad (near half a full
  rotation) to aim at goal takes over a second at max angular velocity
  (4 rad/s, `MAX_ANGULAR_ACCELERATION`=50 rad/s²), but the shooting window
  (`has_ball and _shot_open` staying continuously true against a live
  defender) was observed lasting only 0.1-2.6s per attempt across 8 windows
  in one traced match — consistently too short to complete a large re-aim
  before the next interruption resets progress. Not fixed this session.
  One fix direction was investigated and ruled out: pre-turning toward goal
  during "relay" (before the catch) isn't safely implementable without
  either breaking the catch itself (`_pass_exec`'s `receiver_facing_pass`
  gates `receiver_ready`/`ready_to_kick` continuously on facing the
  *passer*, with no idle window to preempt) or reworking that shared
  contract (used by other tactics too, out of scope for a `counter_press`
  fix). Repositioning `_runner_target`/`_pivot_target` to shrink the angle
  was also considered and ruled out: the ~112° turn measured in one traced
  instance is a structural consequence of the tactic's own design (runner
  deep on the weak-side flank, source robot central/back) — that's the
  entire tactical point of "switch of play," not an incidental parameter to
  tune away. A real fix would mean either reworking `_pass_exec`'s
  receiver-orientation ownership, or accepting a tactic-shape trade-off
  (e.g. relay to a more goal-aligned source position, or catch-then-repass
  to a second runner already facing goal instead of shooting from the same
  catch) — a design decision, not a bug fix, and not attempted this session.
- **`GiveAndGoTactic` had the same "no `motion_controller.reset()` across an
  orientation discontinuity" bug as `counter_press`'s bugs #4/#5, in three
  places at once (found and fixed 2026-08-21)** — surfaced not by testing
  `GiveAndGoTactic` directly but by re-running the full 91-match backfill
  after the `counter_press` fixes: `tiki_taka` (untouched by any fix this
  session) collapsed from 4W-8D-1L to 0W-12D-1L, and nearly every strategy's
  win count swung, not just the two actually fixed. Confirmed via repeated
  re-runs with no code changes that this was NOT sampling noise (`tiki_taka
  vs zone_fluid` gave the identical 0-0 result 4 times in a row) — a real
  regression, hiding in plain sight because every fix so far only touched
  `_pass_and_score.py`'s shared `_score_goal`/`_pass_exec`, and
  `GiveAndGoTactic` (`utama_core/tactics/give_and_go.py`) has its own inline
  shot-aiming logic that never went through those call sites. `tiki_taka`
  and `zone_fluid` both use `GiveAndGoTactic` as their attack engine, which
  is why the swing wasn't confined to `counter_press`. Traced three distinct
  orientation-discontinuity edges in `GiveAndGoTactic.tick()`, all needing
  the same fix:
  1. **Fresh carrier assignment** (`mem.carrier_id is None or ... not in
     robot_ids`) — a robot newly given this tactic's carrier role may have
     spent the prior stretch running a completely different tactic with an
     unrelated commanded orientation.
  2. **Hop hand-off** (`pass_complete` in the mid-hop branch) — the new
     carrier just spent the whole hop facing the *old* carrier
     (`_pass_exec`'s `intercept_oren`, required to catch the pass) and is
     about to be re-aimed at goal instead.
  3. **Ball-chase catch, found to be the dominant case in practice** — a
     carrier still running `go_to_ball` (which continuously faces the ball
     itself while approaching) the tick before it catches it. Traced
     directly on `tiki_taka`: a reset at edge #1 fired ~0.45s before the
     actual catch, far too early to help — the real discontinuity was this
     catch edge, not the assignment edge that preceded it. Fixed by adding
     `GiveAndGoMem.was_carrying` (mirroring `SwitchOfPlayMem.was_shooting`'s
     pattern exactly) and resetting on the `has_ball` False->True edge.
  All three fixed with `ctx.motion_controller.reset()` calls at each edge.
  Trace-confirmed: orientation now converges monotonically (no more
  wrong-direction, multi-second spins) and reaches `kick()` reliably.
  `tiki_taka vs zone_fluid` went from a reproducible 0-0 (post-regression) to
  a reproducible 1-0 win (4/4 repeated runs, deterministic sim). `tiki_taka
  vs counter_flow` improved from a 1-2 loss to a 2-2 draw. Full test suite
  green (714 passed) after the fix. Full 91-match backfill re-run in
  progress as of this writing — the win/draw/loss numbers on `tiki_taka`'s
  and `zone_fluid`'s rows above predate this fix and should be treated as an
  undercount until that backfill lands.
- **`high_line_zone` went from a real mid-pack record to 0 goals in any match,
  post-fix (root-caused and fixed 2026-08-21)** — was 0-13-0, GF0-GA0 in the
  2026-08-20 re-run backfill, down from 2-8-2/GF4-GA3 pre-fix. Root cause,
  confirmed via instrumented match trace (`DecoyOverloadTactic`'s "finish"
  phase, `utama_core/tactics/decoy_and_overload.py`): `go_to_ball`'s
  shield-approach logic (`utama_core/skills/src/go_to_ball.py`) recomputes
  its shield target every tick from the contesting enemy's *live* position,
  with no hysteresis. Against a defender racing for a genuinely loose ball
  this converges fine (the case it was built for — see
  `docs/investigation_default_vs_lowblock_stalemate.md`), but against a
  defender actively covering the shot lane — exactly what a real opponent
  does in `DecoyOverloadTactic`'s finish phase — the shield target keeps
  sliding as the defender moves to keep covering, and the decoy's approach
  oscillates instead of converging: traced closing to 0.34m then drifting
  back out to 0.62m, a 7.5s stall that ate the entire scoring window every
  time. Fixed by adding `_COMMIT_RANGE = 0.2` to `go_to_ball.py`: once the
  approaching robot is within that range of the ball itself, it stops
  tracking the enemy's position and commits to a direct approach — no new
  state needed, since proximity to the ball is already known each tick.
  Secondary, minor fix in the same pass: `decoy_and_overload.py`'s "finish"
  phase used the strict (non-`visual`) `has_ball` check where every other
  call site in `_pass_and_score.py` uses `visual=True`; brought in line.
  Verified fixed: `high_line_zone` vs `low_block` now scores 1-0 (was 0-0,
  0 shots). Full 91-match backfill not yet re-run post-fix.
- **`default` vs `low_block` still draws 0-0 after the `go_to_ball` fix** — partial
  improvement only (possession moved from a near-total pin to 55%/44%, ball
  travel from ~6.8m to 8.78m) — see
  [`docs/investigation_default_vs_lowblock_stalemate.md`](investigation_default_vs_lowblock_stalemate.md)
  for the updated status. The match stays scoreless because `default`'s
  separate zombie-robot bug (below) is still unfixed — `default` is still
  effectively playing 2v6 regardless of how well the 2 active robots can now
  hold the ball. Since both configs are `baseline`-status, finishing this
  fix is not worth doing for its own sake.
- **`default` hands 5 robots to a 2-robot tactic** (robots 3-5 are zombies all
  match) — real defect, same investigation doc, fix candidate #3.

## Updating this file

- New strategy added → add a row here in the appropriate section (or a new
  section) with today's date and a one-line description pulled from its
  docstring.
- Strategy run against another → update the "Description" cell (or add a result
  note) with the outcome once you have one; don't leave a `competitive` strategy
  marked "untested" after it's actually been tested.
- Strategy stops being iterated on after a loss → move it to `Parked` with the
  result and why work stopped, rather than deleting the row.

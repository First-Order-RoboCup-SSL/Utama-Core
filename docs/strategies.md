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
| `tiki_taka` | 2026-08-20 | competitive, most wins, one loss | Possession team: 3 give-and-go attackers + 2 shadow-and-mark cover when we have the ball; 3 pressers + 2 shadow when we don't. Best raw record in the 91-match re-run backfill (4W-8D-1L, GF8-GA4) — but no longer undefeated; its one loss is to `counter_flow` (2-1). Still the strategy with the most wins and a reasonable default reference; `counter_flow` is the more defensible pick specifically for "hardest to beat." |
| `zone_fluid` | 2026-08-20 | competitive, real but weaker | Zone-adaptive team: man-shape defense throughout; give-and-go trio builds through the middle thirds, hands off to the decoy/overload duet in the final third. 1-7-5 in the re-run backfill, GF3-GA9 — genuinely reactive (unlike the baselines) but loses more than it draws or wins, including 1-2 to `tiki_taka` (unchanged across both backfills). A real second data point, not an artifact, but needs work before it's a useful comparison target. |
| `counter_press` | 2026-08-20 | **broken — do not use for comparison yet** | Transition team: full press when the ball is lost and pressable, low block (`BlockShapeTactic`) when it isn't, 4-up switch-of-play attack the moment the ball is won. Zero goals scored in every match across both backfills (0-11-2 in the re-run, GF0). Neither the `block_attacker` fix nor the `go_to_ball` fix resolved this. Whatever is actually broken here (most likely `SwitchOfPlayTactic`'s attack path never reaching a shoot phase) is separate from both shared-skill bugs and still unfixed. |

## Parked — tried against tiki_taka, didn't win, not being iterated further

| Strategy | Added | Status | Description |
|---|---|---|---|
| `overload_press` | 2026-08-20 | parked, improved post-fix but still not competitive-tier | Built to outnumber tiki_taka's 2-robot shadow line (4-robot overload/switch attack + 1 block insurance) and hit direct on turnovers before tiki_taka's press organizes. Original result: lost 3%/97% possession to tiki_taka before either shared fix, and was genuinely weak overall (1-10-1). Post-fix re-run backfill: 2-11-0, GF3-GA1, zero losses — a real improvement, worth another look before `high_line_zone` if the anti-tiki_taka thread is picked back up again, though not yet re-tested head-to-head against `tiki_taka` itself. |
| `high_line_zone` | 2026-08-20 | parked, **now flagged as a possible regression** | Built to deny tiki_taka's give-and-go trio 1v1s via a zone screen (`BlockShapeTactic`) instead of man-marking, switching the ball past its thin 2-robot cover. Original backfill: real mid-pack strategy (2-8-2, GF4-GA3) — beat `low_block` 1-0 and `zone_fluid` 3-1. **Post-fix re-run backfill: 0-13-0, GF0-GA0 — zero goals scored or conceded in any of its 13 matches**, despite still showing real possession swings and occasional shots. Not yet root-caused; see Known open bugs. Do not treat the pre-fix 2-8-2 record as current. |

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
- **`counter_press` never scores, even after both fixes above** — see the
  Competitive section above. Confirmed as its own separate bug (low ball
  travel, no shots, even with high possession and real ball control
  elsewhere in the catalog now), not explained by either shared-skill issue.
  It's the only `competitive`-tagged strategy with a likely correctness bug
  (probably `SwitchOfPlayTactic`'s finishing path) rather than a genuine
  strength gap.
- **`high_line_zone` went from a real mid-pack record to 0 goals in any match,
  post-fix** — see the Parked section above (2026-08-20 re-run backfill:
  0-13-0, GF0-GA0, down from 2-8-2/GF4-GA3 pre-fix). Not the same shape as
  `counter_press`/`decoy_and_overload`: it still shows real possession
  swings and does reach its finishing tactic (`overload`) for genuine
  multi-second stretches in a spot-checked match, so this isn't "never
  reaches a shoot phase." More likely `DecoyOverloadTactic`'s finishing
  conversion specifically got worse now that `go_to_ball`'s approach angles
  changed — plausible, not confirmed. Highest-priority open item: this is
  the one place a fix may have made something worse rather than just failing
  to fix something else.
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

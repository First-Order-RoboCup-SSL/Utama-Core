# Strategy catalog

Every `build_*_kernel_strategy` factory in `utama_core/kernel/kernel_strategy.py`,
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

## Backfill round-robin (2026-08-20)

Every non-`default` config played every other once (78 matches, 60s sim time,
6v6 headless rsim, 8 concurrent workers) specifically to answer "which of
these should be used for comparison going forward, and which are early
artifacts" — not to rank baselines as if they were trying to win.
Run: `replays/tournament_20260820_104203/summary.json`. **Predates the
`block_attacker` fix below** (2026-08-20, same day but later) — every
press-using strategy's possession numbers here (`counter_press`, `high_press`,
and indirectly anything facing them) are understated relative to current
behavior; re-run before trusting these for a press-heavy comparison.

| Strategy | W-D-L | GF-GA |
|---|---|---|
| `high_press` | 4-8-0 | 7-3 |
| `tiki_taka` | 3-9-0 | 5-1 |
| `split_shape` | 2-10-0 | 2-0 |
| `press_and_pass` | 2-9-1 | 3-2 |
| `three_slot` | 2-9-1 | 2-1 |
| `high_line_zone` | 2-8-2 | 4-3 |
| `give_and_go_solo` | 1-10-1 | 3-2 |
| `overload_press` | 1-10-1 | 2-2 |
| `zone_fluid` | 1-6-5 | 3-9 |
| `counter_press` | 0-12-0 | 0-0 |
| `decoy_and_overload` | 0-11-1 | 0-1 |
| `switch_of_play` | 0-11-1 | 0-1 |
| `low_block` | 0-7-5 | 0-6 |

`tiki_taka` is undefeated across all 12 matches (drew `high_press` 0-0,
otherwise won or drew everyone else) and has the tightest goal difference of
any strategy that actually won matches — confirmed as the strongest,
most-robust strategy in the catalog and the right one to compare future work
against. `high_press` has a better raw record, but it's a `baseline` (fixed,
non-reactive 80/20 split) padding its total against weak baselines, not a
signal that it's a better team to build on.

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
| `counter_flow` | 2026-08-20 | competitive, **first strategy to beat tiki_taka** | Third anti-tiki_taka strategy, after `overload_press`/`high_line_zone` were parked. Instead of trying to out-number tiki_taka's 2-robot shadow defense (the bet both prior attempts made and lost before ever getting to test it), fights tiki_taka on its own ground: `GiveAndGoTactic` for attack (tiki_taka's own proven-undefeated engine), `PressAndContainTactic` for our own press on turnover, `BlockShapeTactic` for the defensive screen. 3/2 split in both postures, with possession-edge hysteresis (same fix `high_line_zone` needed). Building this surfaced two shared bugs, both now fixed (see Known open bugs): `block_attacker` never contesting the ball (fixed first — possession went 6%→93%, but still 0-0/0 shots, ball genuinely held only ~0.25s at a time) and `go_to_ball` having no opponent-awareness (fixed second, via direct `has_ball` instrumentation showing the carrier held the ball 431/445 sampled ticks = **False**). After both fixes: **`counter_flow` 2-1 `tiki_taka`**, 22% possession, 4-3 shots — reproducible (rerun deterministic, identical result). Low possession but real conversion; territorial dominance was never the actual bottleneck, ball control was. |
| `tiki_taka` | 2026-08-20 | competitive, no longer undefeated | Possession team: 3 give-and-go attackers + 2 shadow-and-mark cover when we have the ball; 3 pressers + 2 shadow when we don't. Undefeated in the original 12-match backfill (3W-9D-0L, GF5-GA1) and still beats `zone_fluid` 2-1 post-fix (unchanged) — but lost its first match ever to `counter_flow` 1-2 after the `block_attacker`/`go_to_ball` fixes below. Still the strongest-established strategy and the right one to judge new work against; just not literally unbeaten anymore. |
| `zone_fluid` | 2026-08-20 | competitive, real but weaker | Zone-adaptive team: man-shape defense throughout; give-and-go trio builds through the middle thirds, hands off to the decoy/overload duet in the final third. 1-6-5 in the backfill, GF3-GA9 — genuinely reactive (unlike the baselines) but loses more than it draws or wins, including 1-2 to `tiki_taka` (confirmed unchanged post-fix). A real second data point, not an artifact, but needs work before it's a useful comparison target. |
| `counter_press` | 2026-08-20 | **broken — do not use for comparison yet** | Transition team: full press when the ball is lost and pressable, low block (`BlockShapeTactic`) when it isn't, 4-up switch-of-play attack the moment the ball is won. **Backfill result: 0W-12D-0L, 0 goals scored or conceded in every single match**, including 95% possession vs `tiki_taka` and 98% vs `zone_fluid` with zero shots recorded. Neither the `block_attacker` fix nor the `go_to_ball` fix resolved this: re-run vs `tiki_taka` post-both-fixes is still 0-0, still 0 shots, ball travel still low relative to `counter_flow`'s. Whatever is actually broken here (most likely `SwitchOfPlayTactic`'s attack path never reaching a shoot phase) is separate from both shared-skill bugs and still unfixed. |

## Parked — tried against tiki_taka, didn't win, not being iterated further

| Strategy | Added | Status | Description |
|---|---|---|---|
| `overload_press` | 2026-08-20 | parked | Built to outnumber tiki_taka's 2-robot shadow line (4-robot overload/switch attack + 1 block insurance) and hit direct on turnovers before tiki_taka's press organizes. **Result: lost 3%/97% possession to tiki_taka** — got possession-dominated by the press before the overload theory ever got tested. Backfill confirms it's genuinely weak overall (1-10-1), not just vs `tiki_taka`. Root cause: never sustained clean possession long enough to exploit the numbers edge. Not scheduled for further iteration ("latter option" — user chose to park rather than investigate the press-dominance further). |
| `high_line_zone` | 2026-08-20 | parked, but stronger than the parking implied | Built to deny tiki_taka's give-and-go trio 1v1s via a zone screen (`BlockShapeTactic`) instead of man-marking, switching the ball past its thin 2-robot cover. Original result: 0-0 draw, 94%/6% possession to tiki_taka, 0 shots, too passive to threaten. Backfill shows it's a real mid-pack strategy (2-8-2, GF4-GA3) — beat `low_block` 1-0 and `zone_fluid` 3-1, only drew (not lost) `tiki_taka` again at 94/6%. Passivity against `tiki_taka` specifically is still the open problem, but the strategy itself isn't as weak as `overload_press`; worth revisiting before `overload_press` if anyone picks the anti-tiki_taka thread back up. |

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
  Highest-priority remaining open item: it's the only strategy with a likely
  correctness bug (probably `SwitchOfPlayTactic`'s finishing path) rather
  than a genuine strength gap.
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

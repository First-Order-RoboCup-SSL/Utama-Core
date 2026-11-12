# Strategy catalog

Every strategy in `utama_core/strategy/` (one module each, named after the strategy), whether it
is core or reserve, and the latest results.

Run:

- `pixi run python tools/evaluation/evaluate.py` — one round robin of the core and the champion
  at the last release: stalls and signals (`--all` plays every strategy).
- `pixi run python tools/evaluation/evaluate.py <name>` — judges a candidate against the champion
  at the last release. The way to judge a strategy. `--help` lists the flags.
- `pixi run python tools/debug_match.py --strategy <a> --opponent <b> [--dump-ticks PATH]` — one
  instrumented matchup.

rsim is deterministic: re-running an identical match gives an identical result. Variance comes
from the seeded sim noise: `evaluate.py`'s round k plays with seed k.

**The catalog.** Core strategies play every round of `evaluate.py`; reserve ones play only with
`--all` (`tools/evaluation/reference/players.json`). Names changed on 2026-11-12; the results
below were played under the old names (column "Was"), which `catalog.RENAMED` still loads.

**Recording a verdict.** Write it as `evaluate.py` gives it: better / dropped / undecided, with
P(better), the margin ± SD in goals per match, the seeds (1-6, or 101-106 for `--confirm`) and the
run id. "Dropped" means not promising, not worse: an equal strategy is dropped about a third of the time.

| Strategy | Was | Role | Description |
|---|---|---|---|
| `avalanche` | `overload_press` | core | 4-robot overload/switch attack + 1 block, built to outnumber a shadow line. |
| `breakwater` | `clear_press_plus` | core | `firebreak` plus `weaver`'s final-third overload. |
| `mirage` | `decoy_and_overload` | core | `DecoyOverloadTactic` (min 2) + `ShadowAndMarkTactic`, fixed 50/50. |
| `pendulum` | `split_shape` | core | `LeadAndSupportTactic` + `ShadowAndMarkTactic`, split by possession edge. |
| `wolfpack` | `high_press` | core | Same pair as `terrier`, fixed 80/20 attack-heavy split ignoring possession. |
| `chameleon` | `score_aware_counter_flow` | reserve | `riptide` plus a late-half scoreline shift (protect a lead / chase). |
| `firebreak` | `clear_danger` | reserve | `riptide` postures plus an own-third `ClearBallTactic` danger-clearance valve. |
| `ricochet` | `give_and_go_solo` | reserve | Whole pool runs `GiveAndGoTactic`, no defense slot. |
| `riptide` | `counter_flow` | reserve | Give-and-go attack, press on the ball, `BlockShapeTactic` screen; 3/2 with possession hysteresis. |
| `terrier` | `press_and_pass` | reserve | `GiveAndGoTactic` + `PressAndContainTactic`, possession-edge split. |
| `tripwire` | `press_trigger_flow` | reserve | `riptide`, but an all-in press when the ball is lost in our own third. |
| `weaver` | `tiki_taka_plus` | reserve | Short passing, handing off to `DecoyOverloadTactic`'s duet in the final third. |

**Deleted** (2026-11-12): the ten strategies retired on 2026-10-10 (`tiki_taka`, `counter_press`,
`high_line_zone`, `three_slot`, `switch_of_play`, `low_block`, `zone_fluid`, `score_aware_zone_flow`,
`overload_flow`, `shadow_switch`), because nobody knew whether they still worked. Git history keeps them.

`default`: Everyone runs `PassAndShootTactic`; only 2 robots are ever commanded (see bugs). Excluded from the catalog `evaluate.py` plays.

Strategies are judged with `evaluate.py` against the champion at the last release
(`docs/STRATEGY_DEVELOPMENT.md`, "Evaluating a strategy"). The round-robin results below are a
snapshot of the catalog, not the verdict on any one strategy.

## Latest results

**Round-robin** (2026-10-17, sim-realism-20261016 at 423f63a0, the 12 strategies above, 66
matches, two halves of 300 s of playing time, seeded sim noise `v3_seed2`,
`replays/tournament_20261017_151052/summary.json`). Points are 3 a win, 1 a draw; 14 of 66
matches were draws. 4 matches stalled (three committed robots frozen about 10 s, and one kick-off
that waited the referee's 15 s cap, fixed after this run in 1731e904); 97% of 3232 restarts
reached NORMAL_START.

Against the 2026-10-10 run: rsim is our fork with the v3 sim noise (latency, motion limits, ball
friction, kick direction), and restarts, ball placement and the carry limit were fixed, so goals
fell from 7.1 a match to 4.0 and the results are not comparable with earlier runs.
`pendulum` is still first, unbeaten; `riptide`, second before, is eleventh.
What each strategy does well and badly in this run, in figures: [`signal_report.md`](signal_report.md).

| Strategy | W-D-L | GF-GA | Points per match |
|---|---|---|---|
| `pendulum` | 10-1-0 | 66-14 | 2.82 |
| `breakwater` | 5-5-1 | 14-9 | 1.82 |
| `avalanche` | 6-2-3 | 22-18 | 1.82 |
| `firebreak` | 5-2-4 | 20-24 | 1.55 |
| `weaver` | 5-1-5 | 23-21 | 1.45 |
| `wolfpack` | 4-3-4 | 24-27 | 1.36 |
| `tripwire` | 4-1-6 | 13-19 | 1.18 |
| `ricochet` | 3-3-5 | 20-30 | 1.09 |
| `terrier` | 3-3-5 | 14-28 | 1.09 |
| `mirage` | 3-2-6 | 16-27 | 1.00 |
| `riptide` | 2-4-5 | 17-21 | 0.91 |
| `chameleon` | 2-1-8 | 16-27 | 0.64 |

## Known open bugs

Writeups of bugs since fixed were removed from this file; code comments that cite "Known open
bugs" refer to them — read them with `git log -p -- docs/strategies.md`. Still open:

- **`default` commands only 2 of 5 robots** (robots 3-5 idle) and draws 0-0 with
  `low_block`; see `docs/investigation_default_vs_lowblock_stalemate.md`. `default` is the
  kernel's smoke-test scaffold and `low_block` is retired, so not worth fixing for its own sake.

## Updating this file

- New strategy → add a row with a one-line description to the table (it starts as reserve).
- New tournament → replace the matching results table (one table per tournament type) and
  record the run directory; don't append narrative. Findings belong in commit messages.
- Strategy abandoned after losing → delete its module and its row.
- Only single matches, no evaluation → put them in the pull request, not here: this
  file holds round-robin and evaluation results only.

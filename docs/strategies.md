# Strategy catalog

Every `build_*_kernel_strategy` factory in `utama_core/strategy/` (one module each, re-exported by `kernel_strategy.py`), whether a
round-robin plays it, and the latest results. Short config names drop `build_`/`_kernel_strategy`.

Run:

- `pixi run python tools/evaluation/round_robin.py [name ...]` — full-match round-robin (two halves of 300 s of playing time) over every config (or the
  named ones). Common flags: `--reuse`, `--strict`, `--pair A B` (one match); `--help` lists all.
- `pixi run python tools/evaluation/ladder.py <name>` — one candidate against the top of the
  newest round-robin, over both sides and both kickoffs.
- `pixi run python tools/debug_match.py --strategy <a> --opponent <b> [--dump-ticks PATH]` — one
  instrumented matchup.

rsim is deterministic: re-running an identical pair gives an identical result. Variance comes
only from `--both-sides`, side/kickoff cells, or `--fuzz-restarts`.

**In the round-robin**, in order of the 2026-10-19 standings (below). A round-robin or a
ladder with no strategies named plays these.

| Strategy | Description |
|---|---|
| `split_shape` | `LeadAndSupportTactic` + `ShadowAndMarkTactic`, split by possession edge. |
| `flat_rule` | The OpenJev flat engine's scripted rule (`tactics/flat.py`, `docs/openjev.md`): one tactic gives each robot an action from its menu at 1 Hz and when play changes; the carrier shoots unless blocked, else passes or dribbles; without the ball two chase, two block, one outlet (three chase in our third); with it three forward, one back; own free-kick formations. |
| `give_and_go_solo` | Whole pool runs `GiveAndGoTactic`, no defense slot. |
| `counter_flow` | Give-and-go attack, press on the ball, `BlockShapeTactic` screen; 3/2 with possession hysteresis. |
| `score_aware_counter_flow` | `counter_flow` plus a late-half scoreline shift (protect a lead / chase). |
| `overload_press` | 4-robot overload/switch attack + 1 block, built to outnumber tiki_taka's shadow line. |
| `tiki_taka_plus` | `tiki_taka`, handing off to `DecoyOverloadTactic`'s duet in the final third. |
| `high_press` | Same pair as `press_and_pass`, fixed 80/20 attack-heavy split ignoring possession. |
| `clear_danger` | `counter_flow` postures plus an own-third `ClearBallTactic` danger-clearance valve. |
| `decoy_and_overload` | `DecoyOverloadTactic` (min 2) + `ShadowAndMarkTactic`, fixed 50/50. |
| `press_and_pass` | `GiveAndGoTactic` + `PressAndContainTactic`, possession-edge split. |
| `press_trigger_flow` | `counter_flow`, but an all-in press when the ball is lost in our own third. |
| `clear_press_plus` | `clear_danger` plus `tiki_taka_plus`'s final-third overload. |

**Retired** (2026-10-10): the bottom ten of the 2026-10-05 round-robin. Left out of a round-robin
or a ladder pool unless named (`RETIRED` in `tools/evaluation/round_robin.py`). Their code stays:
kept strategies and tests build on some of them. A retired strategy comes back by laddering
against the round-robin's top.

| Strategy | Description |
|---|---|
| `tiki_taka` | 3 give-and-go + 2 shadow-and-mark with the ball; 3 press + 2 shadow without. |
| `counter_press` | Full press on loss, low block otherwise, 4-up switch-of-play on winning the ball. |
| `high_line_zone` | Zone screen (`BlockShapeTactic`) + switch attack, built to deny tiki_taka 1v1s. |
| `three_slot` | Three concurrent slots: press + mark + give-and-go. Exercises N>2 scheduling. |
| `switch_of_play` | `SwitchOfPlayTactic` (min 3) + `DefenseTactic`, fixed 50/50. |
| `low_block` | `PassAndShootTactic` (min 2) + `DefenseTactic`, fixed 20/80 defense-heavy split. |
| `zone_fluid` | Man-shape defense; give-and-go through the middle, decoy/overload in the final third. |
| `score_aware_zone_flow` | `zone_fluid` plus a late-half scoreline shift (protect a lead / chase). |
| `overload_flow` | `zone_fluid`; give-and-go grows 3→4 only after the possession edge holds ~1.5s. |
| `shadow_switch` | `SwitchOfPlayTactic` relay attack + `ShadowAndMarkTactic` defense, 3/2. |

`default`: Everyone runs `PassAndShootTactic`; only 2 robots are ever commanded (see bugs). Excluded from round-robin discovery.

## Latest results

**Round-robin** (2026-10-19, `jev-engine` at 64a0eed8 plus `flat_rule`, the 13 strategies above, 78
matches, two halves of 300 s of playing time, `replays/tournament_20261019_121857/summary.json`). Points
are 3 a win, 1 a draw; 14 of 78 matches were draws. 5 matches stalled, none of them `flat_rule`'s
(`--strict` fails the run): three restarts with no taker for STOP and two frozen commitments, all
between strategies that ran without a stall on 2026-10-10; 98% of 2241 restarts reached NORMAL_START.

Against the 2026-10-10 run: `flat_rule` joins, and the code has since taken in v2.1.0-v2.2.0 (planner
and sim speed-ups, referee fixes). `split_shape` and `flat_rule` both won 11 of 12: `flat_rule` beat
`split_shape` 4-3, its only loss, and lost only to `counter_flow`, 2-3.

| Strategy | W-D-L | GF-GA | Points per match |
|---|---|---|---|
| `split_shape` | 11-0-1 | 79-18 | 2.75 |
| `flat_rule` | 11-0-1 | 64-20 | 2.75 |
| `give_and_go_solo` | 6-3-3 | 37-34 | 1.75 |
| `counter_flow` | 5-4-3 | 22-21 | 1.58 |
| `score_aware_counter_flow` | 5-3-4 | 18-20 | 1.50 |
| `overload_press` | 6-0-6 | 31-34 | 1.50 |
| `tiki_taka_plus` | 5-1-6 | 30-40 | 1.33 |
| `high_press` | 3-3-6 | 16-28 | 1.00 |
| `clear_danger` | 3-3-6 | 23-36 | 1.00 |
| `decoy_and_overload` | 3-2-7 | 20-35 | 0.92 |
| `press_and_pass` | 3-2-7 | 16-34 | 0.92 |
| `press_trigger_flow` | 2-4-6 | 16-35 | 0.83 |
| `clear_press_plus` | 1-3-8 | 13-30 | 0.50 |

## Known open bugs

Writeups of bugs since fixed were removed from this file; code comments that cite "Known open
bugs" refer to them — read them with `git log -p -- docs/strategies.md`. Still open:

- **`default` commands only 2 of 5 robots** (robots 3-5 idle) and draws 0-0 with
  `low_block`; see `docs/investigation_default_vs_lowblock_stalemate.md`. `default` is the
  kernel's smoke-test scaffold and `low_block` is retired, so not worth fixing for its own sake.

## Updating this file

- New factory → add a row with a one-line description to the round-robin table.
- New tournament → replace the matching results table (one table per tournament type) and
  record the run directory; don't append narrative. Findings belong in commit messages.
- Strategy abandoned after losing → add it to `RETIRED` and move its row to the retired table.
- Only single matches (`--pair`), no round-robin → put them in the pull request, not here: this
  file holds round-robin and full-match results only.

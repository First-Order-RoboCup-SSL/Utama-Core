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

**In the round-robin**, in order of the 2026-10-05 standings (below). A round-robin or a
ladder with no strategies named plays these.

| Strategy | Description |
|---|---|
| `split_shape` | `LeadAndSupportTactic` + `ShadowAndMarkTactic`, split by possession edge. |
| `high_press` | Same pair as `press_and_pass`, fixed 80/20 attack-heavy split ignoring possession. |
| `give_and_go_solo` | Whole pool runs `GiveAndGoTactic`, no defense slot. |
| `clear_danger` | `counter_flow` postures plus an own-third `ClearBallTactic` danger-clearance valve. |
| `press_trigger_flow` | `counter_flow`, but an all-in press when the ball is lost in our own third. |
| `press_and_pass` | `GiveAndGoTactic` + `PressAndContainTactic`, possession-edge split. |
| `counter_flow` | Give-and-go attack, press on the ball, `BlockShapeTactic` screen; 3/2 with possession hysteresis. |
| `score_aware_counter_flow` | `counter_flow` plus a late-half scoreline shift (protect a lead / chase). |
| `overload_press` | 4-robot overload/switch attack + 1 block, built to outnumber tiki_taka's shadow line. |
| `clear_press_plus` | `clear_danger` plus `tiki_taka_plus`'s final-third overload. |
| `tiki_taka_plus` | `tiki_taka`, handing off to `DecoyOverloadTactic`'s duet in the final third. |
| `decoy_and_overload` | `DecoyOverloadTactic` (min 2) + `ShadowAndMarkTactic`, fixed 50/50. |

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

**Round-robin, full catalog** (2026-10-05, strategy-guard at b26f5b37, 22 configs, 231 matches,
600s, `replays/tournament_20261005_170958/summary.json`). Points are 3 a win, 1 a draw; 33 of 231
matches were draws. 10 matches stalled (9 committed-frozen, 1 restart stall).

Against the previous run (tournament_20261004_204810, same configs, before this PR's referee and
tactic fixes): goal-line exits now restart in the corner instead of 2 m in front of goal,
`no_progress` restarts fell from 699 to 296 and keep-out fouls from 245 to 26, and
`split_shape`'s lead over second place fell from 0.52 to 0.28 points per match.
Out-of-bounds restarts rose from 2446 to 2938: a goal kick from the corner was cleared at full
power over the far goal line, giving the other side a goal kick at its corner (388 of 1377 goal
kicks were followed by one at the other end, median 4 s). The aimless-kick rule (§6.2.3, added
after this run) makes such a clearance a free kick at the kick point. What each strategy does
well and badly in this run, in figures: [`signal_report.md`](signal_report.md).

| Strategy | W-D-L | GF-GA | Points per match |
|---|---|---|---|
| `split_shape` | 19-1-1 | 103-23 | 2.76 |
| `high_press` | 17-1-3 | 89-40 | 2.48 |
| `give_and_go_solo` | 16-2-3 | 84-28 | 2.38 |
| `clear_danger` | 14-4-3 | 59-28 | 2.19 |
| `press_trigger_flow` | 15-1-5 | 64-41 | 2.19 |
| `press_and_pass` | 13-5-3 | 76-29 | 2.10 |
| `counter_flow` | 14-2-5 | 62-27 | 2.10 |
| `score_aware_counter_flow` | 13-3-5 | 64-29 | 2.00 |
| `overload_press` | 13-2-6 | 55-40 | 1.95 |
| `clear_press_plus` | 10-5-6 | 51-30 | 1.67 |
| `tiki_taka_plus` | 9-6-6 | 48-32 | 1.57 |
| `decoy_and_overload` | 10-0-11 | 59-52 | 1.43 |
| `tiki_taka` | 6-5-10 | 34-41 | 1.10 |
| `high_line_zone` | 5-3-13 | 34-54 | 0.86 |
| `counter_press` | 5-3-13 | 35-69 | 0.86 |
| `three_slot` | 3-8-10 | 10-33 | 0.81 |
| `switch_of_play` | 4-4-13 | 23-58 | 0.76 |
| `low_block` | 4-3-14 | 21-40 | 0.71 |
| `zone_fluid` | 3-3-15 | 33-77 | 0.57 |
| `score_aware_zone_flow` | 2-2-17 | 24-84 | 0.38 |
| `overload_flow` | 1-3-17 | 21-91 | 0.29 |
| `shadow_switch` | 2-0-19 | 19-122 | 0.29 |

## Known open bugs

Writeups of bugs since fixed were removed from this file; code comments that cite "Known open
bugs" refer to them — read them with `git log -p -- docs/strategies.md`. Still open:

- **`kick_upfield` clears blind at fixed power.** The keeper, `PressAndContainTactic`'s presser
  and `ShadowAndMarkTactic`'s markers kick straight upfield whenever they hold the ball, at the
  kicker's one speed (about 4.7 m/s; the ball would roll about 17 m unobstructed, longer than the
  pitch). Measured 2026-10-10 (8 matches of 300 s between round-robin strategies): 83 such kicks
  (keeper 55, presser 21, marker 7), of which 14 (17%) were followed by the ball leaving the pitch
  within 4 s; most hit a robot first. They explain at most 14 of the 58 times the ball went out,
  so the larger sources of out-of-bounds restarts are elsewhere: round-robins after
  tournament_20261005_170958 charge an out-of-bounds foul to the last robot to touch the ball
  (that run charged the nearest one), so the next run's foul table names them. `ClearBallTactic` picks
  landing points 4.5 m upfield (`_CLEAR_DISTANCE`). The kicker is fixed power on the real robots
  too (`docs/roadmap.md` 10c).
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

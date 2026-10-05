# Strategy catalog

Every `build_*_kernel_strategy` factory in `utama_core/strategy/` (one module each, re-exported by `kernel_strategy.py`), its
status, and the latest results. Short config names drop `build_`/`_kernel_strategy`.

Run:

- `pixi run python tools/tournament/round_robin.py [name ...]` — full-length (600s) round-robin over every config (or the
  named ones). Common flags: `--reuse`, `--strict`, `--pair A B` (one match); `--help` lists all.
- `pixi run python tools/tournament/full_match_tournament.py` — 600s, side x kickoff decoupled round-robin over
  the `COMPETITIVE` list in that file.
- `pixi run python tools/debug_match.py --strategy <a> --opponent <b> [--dump-ticks PATH]` — one
  instrumented matchup.

rsim is deterministic: re-running an identical pair gives an identical result. Variance comes
only from `--both-sides`, side/kickoff cells, or `--fuzz-restarts`.

**Status legend**

- `baseline` — exercises kernel machinery or serves as a minimal-risk reference; not meant to
  win. Don't judge or fix by its record.
- `competitive` — reads live game state, has attack and defense answers, allocation reacts to
  possession/zone. Strategy-quality effort goes here.
- `parked` — a competitive attempt that lost to a strong opponent; work redirected elsewhere.
- `experimental` — isolated benchmark config, not a realistic match posture.
- `unclassified` — exists but never catalogued or tournament-tested as its own entry.

## Catalog

| Strategy | Status | Description |
|---|---|---|
| `default` | baseline | Everyone runs `PassAndShootTactic`; only 2 robots are ever commanded (see bugs). Excluded from round-robin discovery. |
| `split_shape` | baseline | `LeadAndSupportTactic` + `ShadowAndMarkTactic`, split by possession edge. |
| `press_and_pass` | baseline | `GiveAndGoTactic` + `PressAndContainTactic`, possession-edge split. |
| `high_press` | baseline | Same pair as `press_and_pass`, fixed 80/20 attack-heavy split ignoring possession. |
| `low_block` | baseline | `PassAndShootTactic` (min 2) + `DefenseTactic`, fixed 20/80 defense-heavy split. |
| `three_slot` | baseline | Three concurrent slots: press + mark + give-and-go. Exercises N>2 scheduling. |
| `decoy_and_overload` | baseline | `DecoyOverloadTactic` (min 2) + `ShadowAndMarkTactic`, fixed 50/50. |
| `switch_of_play` | baseline | `SwitchOfPlayTactic` (min 3) + `DefenseTactic`, fixed 50/50. |
| `give_and_go_solo` | experimental | Whole pool runs `GiveAndGoTactic`, no defense slot. |
| `tiki_taka` | competitive | 3 give-and-go + 2 shadow-and-mark with the ball; 3 press + 2 shadow without. |
| `tiki_taka_plus` | competitive | `tiki_taka`, handing off to `DecoyOverloadTactic`'s duet in the final third. |
| `counter_flow` | competitive | Give-and-go attack, press on the ball, `BlockShapeTactic` screen; 3/2 with possession hysteresis. |
| `zone_fluid` | competitive | Man-shape defense; give-and-go through the middle, decoy/overload in the final third. |
| `counter_press` | competitive | Full press on loss, low block otherwise, 4-up switch-of-play on winning the ball. |
| `score_aware_zone_flow` | competitive | `zone_fluid` plus a late-half scoreline shift (protect a lead / chase). |
| `score_aware_counter_flow` | competitive | `counter_flow` plus the same late-half scoreline shift. |
| `clear_danger` | unclassified | `counter_flow` postures plus an own-third `ClearBallTactic` danger-clearance valve. |
| `clear_press_plus` | competitive | `clear_danger` plus `tiki_taka_plus`'s final-third overload. |
| `shadow_switch` | competitive | `SwitchOfPlayTactic` relay attack + `ShadowAndMarkTactic` defense, 3/2. |
| `overload_flow` | competitive | `zone_fluid`; give-and-go grows 3→4 only after the possession edge holds ~1.5s. |
| `press_trigger_flow` | competitive | `counter_flow`, but an all-in press when the ball is lost in our own third. |
| `overload_press` | parked | 4-robot overload/switch attack + 1 block, built to outnumber tiki_taka's shadow line. |
| `high_line_zone` | parked | Zone screen (`BlockShapeTactic`) + switch attack, built to deny tiki_taka 1v1s. |

## Latest results

**Round-robin, full catalog** (2026-10-05, strategy-guard at b26f5b37, 22 configs, 231 matches,
600s, `replays/tournament_20261005_170958/summary.json`). Points are 3 a win, 1 a draw; 33 of 231
matches were draws. 10 matches stalled (9 committed-frozen, 1 restart stall).

Against the previous run (tournament_20261004_204810, same configs, before this PR's referee and
tactic fixes): goal-line exits now restart in the corner instead of 2 m in front of goal,
`no_progress` restarts fell from 699 to 296 and keep-out fouls from 245 to 26, and
`split_shape`'s lead over second place fell from 0.52 to 0.28 points per match.
Out-of-bounds restarts rose from 2446 to 2938; not yet looked into.

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

The full-match tables below predate the planner, referee and tactic fixes since 2026-09-02.
Treat them as rough ordering, not current truth.

**Full match, competitive tier** (2026-09-01, 5 configs x 4 side/kickoff cells, 40 matches,
600s, `replays/tournament_20260901_193644/summary.json`):

| Strategy | W-D (16 matches each) |
|---|---|
| `tiki_taka_plus` | 10-6 |
| `counter_flow` | 5-8 |
| `tiki_taka` | 4-7 |
| `zone_fluid` | 1-10 |
| `counter_press` | 0-9 |

**Full match, 2026-09-02 additions** (5 configs, `--both-sides`, 40 matches, 600s,
`tournament_20260902_065314`):

| Strategy | W-D-L | GF-GA |
|---|---|---|
| `clear_press_plus` | 9-7-0 | 14-4 |
| `overload_flow` | 3-10-3 | 12-11 |
| `press_trigger_flow` | 2-9-5 | 5-10 |
| `shadow_switch` | 1-12-3 | 3-5 |
| `score_aware_counter_flow` | 0-12-4 | 1-5 |

65s samples of the same five were ~80% draws and showed no spread; only full-length matches
separated them. Prefer full-length results when ranking.

## Known open bugs

Writeups of bugs since fixed were removed from this file; code comments that cite "Known open
bugs" refer to them — read them with `git log -p -- docs/strategies.md`. Still open:

- **Two markers can chase each other off the pitch.** `ShadowAndMarkTactic` targets 0.6 m
  goal-side of its opponent and `man_mark` (in `PressAndContainTactic`) 0.5 m beside its own;
  when each marks the other, both targets move with the robots and the pair walks to the sim's
  wall outside the field (42 of 231 matches in tournament_20261004_204810, mostly during
  wedges; 5 of 15 after the wedge fix, the longest 22 s). Clamping the targets only moves where
  they stick; the fix is markers that don't define their targets from each other.
- **`default` commands only 2 of 5 robots** (robots 3-5 idle) and draws 0-0 with
  `low_block`; see `docs/investigation_default_vs_lowblock_stalemate.md`. Both are baselines, so
  not worth fixing for its own sake.

## Updating this file

- New factory → add a catalog row with status and a one-line description.
- New tournament → replace the matching results table (one table per tournament type) and
  record the run directory; don't append narrative. Findings belong in commit messages.
- Strategy abandoned after losing → mark it `parked`, don't delete the row.
- Only single matches (`--pair`), no round-robin → put them in the pull request, not here: this
  file holds round-robin and full-match results only.

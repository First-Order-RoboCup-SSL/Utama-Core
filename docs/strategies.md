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

**Round-robin, full catalog** (2026-10-04, strategy-guard at 1c29a4e8, 22 configs, 231 matches,
600s, `--strict`, about 2h45m wall on 15 workers sharing the machine,
`replays/tournament_20261004_204810/summary.json`). Points are 3 a win, 1 a draw; 24 of 231
matches were draws (65s matches were 55%).

This run predates three fixes in the same PR (#141), so a rerun will move places:
`DecoyOverloadTactic` now releases its robots once the enemy has held the ball 1 s (used by
`decoy_and_overload`, `overload_press`, `clear_press_plus`, `tiki_taka_plus`), and a defender
standing inside 0.5 m can no longer hold a free kick forever. That freeze ran to full time in
`counter_flow` 1-2 `split_shape` and `decoy_and_overload` 3-6 `split_shape` (frozen with
`split_shape` ahead) and for the last minute of `low_block` 1-1 `score_aware_zone_flow`. Above all, a ball over a
goal line now restarts in the corner (corner or goal kick, as the rulebook says) instead of as
a free kick 2 m in front of goal: 521 such kicks gave 150 goals in this run, and `split_shape`
converted 69% of its own. Without them its lead over `counter_flow` drops from 11 points to 1.

| Strategy | W-D-L | GF-GA | Points per match |
|---|---|---|---|
| `split_shape` | 21-0-0 | 135-29 | 3.00 |
| `give_and_go_solo` | 17-1-3 | 86-32 | 2.48 |
| `counter_flow` | 17-1-3 | 63-21 | 2.48 |
| `high_press` | 16-1-4 | 57-20 | 2.33 |
| `score_aware_counter_flow` | 15-1-5 | 70-30 | 2.19 |
| `press_and_pass` | 13-6-2 | 50-21 | 2.14 |
| `clear_danger` | 14-1-6 | 63-27 | 2.05 |
| `press_trigger_flow` | 12-1-8 | 56-34 | 1.76 |
| `overload_press` | 11-4-6 | 65-43 | 1.76 |
| `clear_press_plus` | 11-3-7 | 60-29 | 1.71 |
| `decoy_and_overload` | 11-1-9 | 58-48 | 1.62 |
| `tiki_taka_plus` | 11-1-9 | 47-41 | 1.62 |
| `three_slot` | 8-3-10 | 29-34 | 1.29 |
| `low_block` | 7-5-9 | 30-30 | 1.24 |
| `tiki_taka` | 6-5-10 | 45-52 | 1.10 |
| `high_line_zone` | 6-1-14 | 34-72 | 0.90 |
| `counter_press` | 4-1-16 | 19-70 | 0.62 |
| `switch_of_play` | 3-2-16 | 17-66 | 0.52 |
| `score_aware_zone_flow` | 1-5-15 | 47-97 | 0.38 |
| `overload_flow` | 2-2-17 | 31-85 | 0.38 |
| `zone_fluid` | 1-3-17 | 43-100 | 0.29 |
| `shadow_switch` | 0-0-21 | 12-136 | 0.00 |

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

- **`go_to_ball`'s dribbler-back approach angle is a no-op**
  (`skills/src/go_to_ball.py:130`): `(approach_oren + pi) % (2pi) - pi` is an identity on an
  already-normalized angle, so every approach faces the ball head-on. A correct flip was
  reverted because it broke `test_out_of_bounds_restart_spot_is_capturable_by_go_to_ball`.
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

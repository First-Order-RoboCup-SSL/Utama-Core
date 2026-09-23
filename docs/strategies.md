# Strategy catalog

Every `build_*_kernel_strategy` factory in `utama_core/strategy/kernel_strategy.py`, its
status, and the latest results. Short config names drop `build_`/`_kernel_strategy`.

Run:

- `pixi run python smoke_tournament.py [name ...]` — 65s round-robin over every config (or the
  named ones). Flags: `--both-sides`, `--strict`, `--stop-at-first-stall`,
  `--fuzz-restarts SEED`, `--control-scheme {fpp,trajsample,...}`, `--no-save`, `-v`.
- `pixi run python full_match_tournament.py` — 600s, side x kickoff decoupled round-robin over
  the `COMPETITIVE` list in that file.
- `pixi run python arena_tournament.py <a> <b>` or `... round_robin` — instrumented matchup
  runner.

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
| `counter_press` | competitive | Full press on loss, low block otherwise, 4-up switch-of-play on winning the ball. Scores in no match (see bugs). |
| `score_aware_zone_flow` | competitive | `zone_fluid` plus a late-half scoreline shift (protect a lead / chase). Untested at full length. |
| `score_aware_counter_flow` | competitive | `counter_flow` plus the same late-half scoreline shift. Winless at full length (0-12-4). |
| `clear_danger` | unclassified | `counter_flow` postures plus an own-third `ClearBallTactic` danger-clearance valve. |
| `clear_press_plus` | competitive | `clear_danger` plus `tiki_taka_plus`'s final-third overload. Best full-length record among the 2026-09-02 additions. |
| `shadow_switch` | competitive | `SwitchOfPlayTactic` relay attack + `ShadowAndMarkTactic` defense, 3/2. |
| `overload_flow` | competitive | `zone_fluid`; give-and-go grows 3→4 only after the possession edge holds ~1.5s. |
| `press_trigger_flow` | competitive | `counter_flow`, but an all-in press when the ball is lost in our own third. |
| `overload_press` | parked | 4-robot overload/switch attack + 1 block, built to outnumber tiki_taka's shadow line. |
| `high_line_zone` | parked | Zone screen (`BlockShapeTactic`) + switch attack, built to deny tiki_taka 1v1s. |

## Latest results

All results below predate later planner/tactic fixes and the current stall re-measurement
(`docs/roadmap.md` item 1). Treat them as rough ordering, not current truth.

**Smoke round-robin, full catalog** (2026-08-21, 14 configs, 91 matches, 60s,
`replays/tournament_20260821_171924/summary.json`):

| Strategy | W-D-L | GF-GA |
|---|---|---|
| `press_and_pass` | 6-5-2 | 7-2 |
| `counter_flow` | 5-8-0 | 8-3 |
| `split_shape` | 5-8-0 | 6-0 |
| `high_press` | 5-6-2 | 10-3 |
| `overload_press` | 4-9-0 | 7-1 |
| `give_and_go_solo` | 3-8-2 | 5-3 |
| `three_slot` | 2-8-3 | 3-4 |
| `decoy_and_overload` | 2-7-4 | 2-5 |
| `high_line_zone` | 2-7-4 | 2-5 |
| `zone_fluid` | 2-6-5 | 3-8 |
| `switch_of_play` | 1-10-2 | 1-2 |
| `tiki_taka` | 1-8-4 | 5-9 |
| `low_block` | 0-10-3 | 0-6 |
| `counter_press` | 0-6-7 | 0-8 |

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
- **`counter_press` scores in no match.** Five `SwitchOfPlayTactic` relay/finish bugs were
  fixed (2026-08-21); the remainder is a design tension — the runner needs ~3.8 rad of turning
  but the shooting window is shorter than the turn. Not a small patch.
- **`default` commands only 2 of 5 robots** (robots 3-5 idle) and draws 0-0 with
  `low_block`; see `docs/investigation_default_vs_lowblock_stalemate.md`. Both are baselines, so
  not worth fixing for its own sake.
- **`low_block`'s `PassAndShootTactic` stays in "setup" all match**: `run_setup_phase` only
  moves the passer once it holds the ball, and 2 attackers can't win sustained possession
  against a heavier opponent. Needs a design pass (setup without possession, or help winning
  the ball).

## Updating this file

- New factory → add a catalog row with status and a one-line description.
- New tournament → replace the matching results table (one table per tournament type) and
  record the run directory; don't append narrative. Findings belong in commit messages.
- Strategy abandoned after losing → mark it `parked`, don't delete the row.

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

## Backfill round-robin (2026-08-21, re-run after the full motion-controller-reset fix pass)

Every non-`default` config played every other once (91 matches — 14 configs —
60s sim time, 6v6 headless rsim), re-run after fixing every then-known
instance of the "no `motion_controller.reset()` across an orientation
discontinuity" bug class (see Known open bugs): `SwitchOfPlayTactic` (5 bugs,
`counter_press`), `GiveAndGoTactic` (3 edges, `tiki_taka`/`zone_fluid`), and
`PassAndShootTactic`/`LeadAndSupportTactic`/`DecoyOverloadTactic` (this pass).
Supersedes the 2026-08-20 re-run (`replays/tournament_20260820_113853/`) and
the two intermediate 2026-08-21 runs taken between individual fixes, none of
which reflect the fully-fixed codebase. Current run:
`replays/tournament_20260821_171924/summary.json`.

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

The picture reshuffled substantially from the 2026-08-20 backfill, and not
just for the strategies actually touched — every strategy sharing a fixed
tactic with one of this session's bug fixes moved. `press_and_pass` (uses
`GiveAndGoTactic`) now leads on wins; `counter_flow`/`split_shape` remain
undefeated on losses (0L). `decoy_and_overload` and `high_line_zone` (both use
`DecoyOverloadTactic`) went from 0 wins to 2 each — direct evidence the
`decoy_and_overload.py` reset fix (below) was live and load-bearing, not
just theoretically correct.

`tiki_taka` (1W-8D-4L, GF5-GA9) is still well below its original 4W-8D-1L —
but all 4 losses are to strategies that themselves got materially stronger
this session via the very same shared-tactic fixes (`high_press`,
`press_and_pass`, `split_shape`, `give_and_go_solo` — all four either use
`GiveAndGoTactic` directly or a sibling tactic whose call sites were fixed in
this pass), so a weaker relative record is the expected shape of "everyone
who shared the bug got fixed at once," not a sign `tiki_taka` itself
regressed further. `counter_press` (0W-6D-7L, GF0) still scores in no match
at all — expected and already explained: its remaining blocker is the
turn-budget-vs-window-duration mismatch documented below, a design tension,
not a bug this pass touches.

## Full-match, both-sides round-robin (2026-08-22, competitive tier only)

Every prior backfill ran 60s per match — enough to compare tactic-level
behaviour, but short of a real match (regulation half is 300s per
`half_duration_seconds` in `utama_core/custom_referee/profiles/simulation.yaml`,
so a full match is 600s of sim time across both halves). This run played just
the 4 `competitive`-tier strategies (`counter_flow`, `tiki_taka`, `zone_fluid`,
`counter_press`) at full 600s length, every pair both ways (each config once
as yellow/right, once as blue/left — see `tournament.py --both-sides`'s
docstring for why this is a real second data point, not a duplicate): 12
matches total. Driver: `full_match_tournament.py` (new, thin wrapper around
`tournament.py`'s `run_match`/round-robin machinery with `MATCH_DURATION_SECONDS`
overridden to 600.0 and the config pool restricted to the competitive subset).
Run: `replays/tournament_20260822_224929/summary.json`.

**This run only exists because the first attempt at it (yellow-side-only,
`replays/tournament_20260822_221958/`) uncovered a real deadlock bug — every
one of those 6 matches froze in a `DIRECT_FREE_*` restart within the first two
minutes and sat stalled for the rest of the 600s, so the scores it produced
were meaningless. See Known open bugs below for the root cause and fix; this
section's numbers are from the re-run after that fix, and are the first
full-length numbers for these 4 strategies that actually reflect complete
matches.**

| Strategy | W-D-L | Elo (start 1000, K=24) |
|---|---|---|
| `counter_flow` | 3W-2D-1L | 1022.4 |
| `tiki_taka` | 3W-2D-1L | 1020.9 |
| `zone_fluid` | 0W-5D-1L | 988.1 |
| `counter_press` | 0W-3D-3L | 968.6 |

`counter_flow` and `tiki_taka` are effectively co-leaders (tied on W-D-L, ~2
points apart on Elo) — a change from the 60s-backfill picture where
`tiki_taka` was mid-pack; a full match apparently lets its possession-based
approach compound its advantage rather than getting cut short. Side matters:
`counter_flow` vs `tiki_taka` flips outright by which side each plays (`counter_flow`
loses 0-2 as yellow/right, wins 1-0 as blue/left), and `counter_press`'s only
non-loss against `counter_flow` (a scoreless draw feel) disappears on the
other side (loses 0-5) — a single-sided run would have materially
misrepresented at least these two matchups. `counter_press` again never won
and scored only 1 goal across all 6 of its matches, consistent with its
already-documented turn-budget-vs-window-duration mismatch, not a new issue.

Elo/plots: `pixi run python elo.py <summary.json>` then
`pixi run python plot_elo.py <run_dir>/elo_history.json <run_dir>/summary.json`
(both pre-existing, unchanged this session) — writes `elo_history.json` plus
`elo_history.png`/`wdl_matrix.png`/`goal_diff.png` alongside the summary.
Elo here is the textbook rating (no Glicko/TrueSkill uncertainty modelling,
see `elo.py`'s docstring), so treat the numbers as ballpark separation, not
a precise strength measurement from only 12 games.

## Full-match, decoupled side x kickoff x seed round-robin (2026-08-22/23, competitive tier only)

The both-sides run above conflates side, colour, and kickoff into one combined
swap (colour and kickoff are pinned together via the `simulation` referee
profile's `kickoff_team: "yellow"` default, and `config_a` is always yellow),
so a 2-game "both sides" sample can't attribute an outcome difference to any
one cause — confirmed live: `counter_flow` vs `tiki_taka` flips (0-2 -> 1-0)
under that combined swap alone. `full_match_tournament.py` decouples side and
kickoff into independent axes (colour stays tied to "config_a is yellow" — no
tactic reads raw colour, only `my_team_is_right`) and adds seeded per-robot
formation jitter (`JITTER_POS_STD=0.15m`, `JITTER_THETA_STD=0.2rad`) as a
third axis, aiming for more independent samples than side x kickoff's 4
combinations alone. Per pair: all 4 side x kickoff cells x 2 jitter seeds = 8
matches, still full 600s length. 6 pairs x 8 cells = 48 matches total. Run:
`replays/tournament_20260822_234112/summary.json`.

| Strategy | W-D-L (24 matches each) |
|---|---|
| `tiki_taka` | 12W-12D-0L |
| `counter_flow` | 8W-16D-0L |
| `zone_fluid` | 4W-20D-0L |
| `counter_press` | 0W-24D-0L |

**Two findings, both load-bearing for how to read this table and for any
future tournament methodology:**

1. **Formation jitter added zero variance.** Every pair's `seed=0` and
   `seed=1` runs produced byte-identical scores, in every one of the 4
   side x kickoff cells, across all 6 pairs — 48/48 matches confirm this,
   zero exceptions. A ±0.15m/±0.2rad per-robot starting-position perturbation
   is not enough to change which branch any of these deterministic matches
   takes. Practical upshot: the real independent sample count per pair here
   is **4** (side x kickoff), not 8 — seed was not, in practice, a third
   axis. Getting genuinely more independent samples would need either much
   larger jitter, or randomness injected somewhere that actually changes
   early tactical branching (e.g. randomizing which side starts with
   possession), not sub-half-meter formation noise.
2. **Side is the dominant variable; kickoff contributes nothing on its own.**
   Every single pair shows the identical shape: the outcome is constant
   across both kickoff states for a given side, and changes (or doesn't)
   only when side flips:
   - `counter_flow` vs `tiki_taka`: right -> `tiki_taka` wins 0-2 (both
     kickoff states, both seeds); left -> 2-2 draw (both kickoff states,
     both seeds). Confirms the earlier combined-swap flip was a side effect,
     not a kickoff or colour effect.
   - `counter_flow` vs `zone_fluid`: 0-0 in all 8 cells — fully deadlocked
     regardless of side, kickoff, or seed.
   - `counter_press` vs `tiki_taka`: right -> 0-0 draw; left -> `tiki_taka`
     wins 0-1.
   - `counter_press` vs `zone_fluid`: right -> 0-0 draw; left -> `zone_fluid`
     wins 0-1.
   - `tiki_taka` vs `zone_fluid`: right -> `tiki_taka` wins 2-0; left -> 0-0
     draw.
   - `counter_flow` vs `counter_press`: `counter_flow` wins every cell
     (2-0 right, 1-0 left) — the one pair where side changes the margin, not
     the winner.

   Right-side play is consistently more decisive (wins or clean losses);
   left-side play consistently trends toward a draw. This holds with zero
   exceptions across all 6 pairs. Not yet root-caused — plausibly a
   `my_team_is_right`-dependent asymmetry somewhere in geometry/goal-side
   logic shared across tactics, since the pattern is uniform across
   architecturally different strategies rather than isolated to one tactic.
   Worth a real investigation before trusting any single-side tournament
   result again, but out of scope for this run.

**Read on statistical robustness**: 4 truly independent structural samples
per pair (not 8) is thin, and the side effect above means those 4 aren't even
symmetric — they're 2 conditions (right, left) each duplicated by kickoff,
and kickoff didn't move anything. Treat any pair's record here as "this is
what happens on the right side" plus "this is what happens on the left side,"
not as one converged number.

## Full-match, decoupled side x kickoff round-robin, both kickoff-ceremony + hysteresis fixes active (2026-08-23, competitive tier only)

Re-run of the same 4-cell decoupled methodology above (jitter axis dropped —
see finding 1 above), now with both fixes from Known open bugs' "kickoff
tie" entry active: the real `PREPARE_KICKOFF_YELLOW`/`_BLUE` ceremony (this
was already active in the 2026-08-22/23 run above) plus
`kernel_strategy.py`'s `_CLOSER_TO_BALL_MARGIN = 0.05` hysteresis margin
(new this run — not active in the run above). 6 pairs x 4 cells = 24
matches, 600s each. Run: `replays/tournament_20260823_173432/summary.json`.

| Strategy | W-D-L (12 matches each) | Points (3/1/0) |
|---|---|---|
| `counter_flow` | 5W-6D-1L | 21 |
| `tiki_taka` | 2W-8D-2L | 14 |
| `zone_fluid` | 2W-7D-3L | 13 |
| `counter_press` | 2W-5D-5L | 11 |

Compared to the pre-hysteresis-fix run: `counter_press` goes from 0W-24D-0L
(never lost, never won, never scored across 48 matches) to 2W-5D-5L over half
as many matches — it now both wins and loses, and (per the raw per-cell log)
scores real goals in several matches, e.g. `counter_press` 2-0 and 1-0 wins
over `zone_fluid`. That's a materially different qualitative picture than
"can never break a tie," consistent with the hysteresis margin actually
letting the tie-break resolve on real kinematics rather than rsim noise in at
least some matches. The overall draw-heavy shape persists (13 of 24 matches
this run were draws), so the tie is reduced, not eliminated, matching the
residual-edge-case caveat already documented for `_CLOSER_TO_BALL_MARGIN`
(first-contact can still be a near-zero-distance crossing). Not re-tested this run: whether the
right/left decisiveness-vs-draw asymmetry (finding 2 above) persists —
would need the same per-cell breakdown this run didn't isolate before
overwriting; worth a follow-up if that asymmetry is investigated directly.

## Full-match, decoupled side x kickoff round-robin, tiki_taka_plus added (2026-08-23, competitive tier)

First round-robin including `tiki_taka_plus` alongside the existing 4
competitive strategies: 5 configs, 10 pairs x 4 cells (side x kickoff) = 40
matches, 600s each, same decoupled methodology as the run above. Run:
`replays/tournament_20260823_202520/summary.json`.

| Strategy | W-D (16 matches each) |
|---|---|
| `counter_flow` | 6W-8D |
| `tiki_taka_plus` | 6W-7D |
| `tiki_taka` | 3W-10D |
| `zone_fluid` | 3W-8D |
| `counter_press` | 2W-7D |

`tiki_taka_plus` ties `counter_flow` for most wins and clearly outperforms its
own base `tiki_taka` (3W-10D) despite differing from it by exactly one
change: a final-third handoff from the give-and-go/press/shadow trio to
`DecoyOverloadTactic`'s 2-robot overload duet (see `_tiki_taka_plus_picker`,
`utama_core/strategy/kernel_strategy.py`). Direct head-to-head vs `tiki_taka`
across the 4 cells: 2W-2D-0L for `tiki_taka_plus`. Promoted into
`COMPETITIVE` (`full_match_tournament.py`) on this result — worth a second
independent run to confirm before leaning on it further, since one 40-match
tournament is not yet enough to rule out variance at this sample size.

## Baselines — don't fix, don't judge by these

Several of these don't have both a real attack and a real defense answer, or
use a fixed split that ignores game state — they exist to exercise kernel
machinery, not to compete. Their round-robin record above is not a quality
signal; do not treat a loss here as something to fix.

| Strategy | Added | Status | Description |
|---|---|---|---|
| `default` | pre-2026-08-19 | baseline | Single-tactic pool: everyone runs `PassAndShootTactic`, only ever commands the first 2 outfield robots (robots 3-5 are zombies — known bug, see [investigation below](#known-open-bugs)). Exists to exercise minimal kernel scheduling, not to be a real team. Excluded from the round-robin/catalog discovery entirely (see `tournament.py`). |
| `split_shape` | pre-2026-08-19 | baseline | `LeadAndSupportTactic` (attack) + `ShadowAndMarkTactic` (defense), split by possession edge. First concrete forcing case for the kernel's splitting policy. 2026-08-21 backfill: 5-8-0, GF6-GA0 — undefeated with a real conceded-nothing record, benefiting from this session's `LeadAndSupportTactic` reset fix (see Known open bugs). |
| `press_and_pass` | pre-2026-08-19 | baseline | `GiveAndGoTactic` (attack) + `PressAndContainTactic` (defense), possession-edge split. Same shape as `split_shape`, newer tactic pair. 2026-08-21 backfill: 6-5-2, GF7-GA2 — best raw win count in the whole catalog, benefiting from the `GiveAndGoTactic` reset fix (see Known open bugs). |
| `high_press` | pre-2026-08-19 | baseline | Same tactic pair as `press_and_pass` but a fixed 80/20 attack-heavy split, ignoring possession — demonstrates a non-reactive `Partitioner` on the same tactics. 2026-08-21 backfill: 5-6-2, GF10-GA3, highest goals-for in the catalog — still purely from being relentlessly attack-heavy against weaker opponents, not a signal to build more strategies this way. |
| `low_block` | pre-2026-08-19 | baseline | `PassAndShootTactic` (attack, floored at 2 robots) + `DefenseTactic` (defense), fixed 20/80 defense-heavy split. The original minimal-risk pairing. 2026-08-21 backfill: 0-10-3, GF0-GA6 — still the weakest record in the catalog. Root-caused this session: with only 2 of 5 outfield robots on attack, the passer can never secure sustained possession against a heavier opponent, so `run_setup_phase` never completes — a possession/robot-allocation design gap, not a bug (see Known open bugs). **Known bug: draws 0-0 against `default`** — see [investigation](#known-open-bugs). |
| `three_slot` | pre-2026-08-19 | baseline | First 3-concurrent-slot config: `PressAndContainTactic` + `ShadowAndMarkTactic` + `GiveAndGoTactic`. Exercises N>2 scheduling, not tuned for strength. 2026-08-21 backfill: 2-8-3, GF3-GA4. |
| `decoy_and_overload` | pre-2026-08-19 | baseline | `DecoyOverloadTactic` (attack, floored at 2) + `ShadowAndMarkTactic` (defense), fixed 50/50 split. Exercises the lure/overload tactic in isolation. 2026-08-21 backfill: 2-7-4, GF2-GA5 — up from never scoring at all (0-11-1, GF0) once `DecoyOverloadTactic`'s own reset-discontinuity bug was fixed (see Known open bugs). |
| `give_and_go_solo` | pre-2026-08-19 | experimental | Entire pool always runs `GiveAndGoTactic`, no defense slot at all. Isolated benchmark for tuning give-and-go internals without a defensive confound — not a fielding-ready config. 2026-08-21 backfill: 3-8-2, GF5-GA3, benefiting from the `GiveAndGoTactic` reset fix like every other user of that tactic. |
| `switch_of_play` | pre-2026-08-19 | baseline | `SwitchOfPlayTactic` (attack, floored at 3 for the carrier/pivot/runner relay) + `DefenseTactic`, fixed 50/50 split. Exercises the switch tactic in isolation. 2026-08-21 backfill: 1-10-2, GF1-GA2 — still barely scores in isolation despite `SwitchOfPlayTactic`'s own 5 bugs being fixed (see `counter_press`'s row), consistent with the documented turn-budget-vs-window-duration mismatch affecting this tactic regardless of which strategy wraps it. |

## Competitive — real playable teams, use these for comparison going forward

| Strategy | Added | Status | Description |
|---|---|---|---|
| `score_aware_zone_flow` | 2026-08-23 | competitive, new — not yet tournament-tested | `zone_fluid`'s exact tactic set (`GiveAndGoTactic`/`DecoyOverloadTactic`/`ShadowAndMarkTactic`) with one added decision input nothing else in the catalog reads: the scoreline. Every existing picker reads possession/ball-zone only — `game.referee.{yellow,blue}_team.score` is already populated in any refereed match but was unused. In the last 60s of a playing half (`game.referee.stage in {NORMAL_FIRST_HALF, NORMAL_SECOND_HALF}` and `stage_time_left <= 60`), the give-and-go attack/defense split shrinks from 3/2 to 2/3 when ahead (protect the lead) and grows to 4/1 when behind (chase an equaliser); tied, early, or with no referee data (`game.referee is None`) it's identical to `_zone_flow_picker`. New helpers: `_friendly_score_diff`, `_is_late_in_half` (`utama_core/strategy/kernel_strategy.py`). Unit-verified directly against a mocked `Game` (ahead-late → 2 givego/3 defense, behind-late → 4 givego/1 defense, not-late → 3/2 matching `zone_fluid`); smoke-tested via `tournament.py score_aware_zone_flow zone_fluid` (1-0, 60s match — too short to exercise the late-game branch, since that only fires in a half's final 60s). Not yet run in a full-length round-robin against the other 3 competitive strategies. |
| `counter_flow` | 2026-08-20 | competitive, undefeated (0L across 91 matches) | Third anti-tiki_taka strategy, after `overload_press`/`high_line_zone` were parked. Instead of trying to out-number tiki_taka's 2-robot shadow defense (the bet both prior attempts made and lost before ever getting to test it), fights tiki_taka on its own ground: `GiveAndGoTactic` for attack (tiki_taka's own proven engine), `PressAndContainTactic` for our own press on turnover, `BlockShapeTactic` for the defensive screen. 3/2 split in both postures, with possession-edge hysteresis (same fix `high_line_zone` needed). Building this surfaced two shared bugs, both now fixed (see Known open bugs): `block_attacker` never contesting the ball, and `go_to_ball` having no opponent-awareness. Beat `tiki_taka` 2-1 (reproducible) at just 22% possession, the first loss `tiki_taka` had anywhere in the catalog — now a 2-2 draw post-`GiveAndGoTactic`-fix (below). In the 2026-08-21 backfill (post full reset-fix pass), `counter_flow` is 5W-8D-**0L**, still undefeated. |
| `tiki_taka` | 2026-08-20 | competitive, but now mid-pack on wins | Possession team: 3 give-and-go attackers + 2 shadow-and-mark cover when we have the ball; 3 pressers + 2 shadow when we don't. In the 2026-08-21 backfill (post full reset-fix pass): 1W-8D-4L, GF5-GA9 — down from the pre-fix-era 4W-8D-1L. Not a regression: a `GiveAndGoTactic` orientation-discontinuity bug (fixed 2026-08-21, see Known open bugs) had been suppressing this strategy's own scoring the whole time the earlier numbers were recorded (`vs zone_fluid` is now a reproducible 1-0 rather than 2-1; `vs counter_flow` is 2-2 rather than a 1-2 loss), but all 4 of its current losses are to strategies that themselves got materially stronger from the *same* shared-tactic fixes this session (`high_press`, `press_and_pass`, `split_shape`, `give_and_go_solo`) — the whole `GiveAndGoTactic`-using cohort moved together, so `tiki_taka`'s *relative* standing dropped even though its own play improved. Worth another look if the anti-tiki_taka thread continues, but not an open bug. |
| `zone_fluid` | 2026-08-20 | competitive, real but weaker | Zone-adaptive team: man-shape defense throughout; give-and-go trio builds through the middle thirds, hands off to the decoy/overload duet in the final third. 2-6-5 in the 2026-08-21 backfill, GF3-GA8 — genuinely reactive (unlike the baselines) but loses more than it draws or wins. Also runs `GiveAndGoTactic`, so benefited from the same fix as `tiki_taka`; `vs tiki_taka` is now a reproducible 1-0 rather than the earlier 2-1 loss. |
| `counter_press` | 2026-08-20 | **5 layered bugs root-caused and fixed 2026-08-21; still scores in no match — turn-budget mismatch, see Known open bugs** | Transition team: full press when the ball is lost and pressable, low block (`BlockShapeTactic`) when it isn't, 4-up switch-of-play attack the moment the ball is won. In the 2026-08-21 backfill (post full reset-fix pass): 0W-6D-7L, GF0 — zero goals scored in any match, same as every prior backfill. Neither the `block_attacker` fix nor the `go_to_ball` fix resolved this — the actual cause was 5 separate bugs in `SwitchOfPlayTactic`'s relay/finish path, all fixed and individually trace-verified (see Known open bugs), and confirmed capable of scoring in isolated traces (1-0 vs `low_block`). The remaining, still-open blocker is a turn-budget-vs-window-duration mismatch (a design tension, not a bug in these 5 fixes) — see Known open bugs for why it isn't cleanly fixable as a small patch. |
| `tiki_taka_plus` | 2026-08-23 | competitive, promoted — final-third overload variant of `tiki_taka` | Same 3/2 give-and-go/press/shadow base as `tiki_taka`, with one change: in the final third while holding the possession edge, hands off from the give-and-go trio to `DecoyOverloadTactic`'s 2-robot overload duet (mirroring `zone_fluid`'s existing final-third handoff, grafted onto `tiki_taka`'s base instead of the man-shape-defense-throughout shape). See `_tiki_taka_plus_picker`, `utama_core/strategy/kernel_strategy.py`. A second candidate change (giving cover robots explicit press applicability) was investigated and found to already be covered by `PressAndContainTactic.applicable()`'s existing global range check — correctly not implemented as a separate change. First round-robin (2026-08-23, 5-config decoupled round-robin above): 6W-7D, tied with `counter_flow` for most wins, 2W-2D-0L head-to-head against plain `tiki_taka`. Not yet re-tested independently to confirm the result holds beyond one 40-match sample. |

## Parked — tried against tiki_taka, didn't win, not being iterated further

| Strategy | Added | Status | Description |
|---|---|---|---|
| `overload_press` | 2026-08-20 | parked, improved post-fix but still not competitive-tier | Built to outnumber tiki_taka's 2-robot shadow line (4-robot overload/switch attack + 1 block insurance) and hit direct on turnovers before tiki_taka's press organizes. Original result: lost 3%/97% possession to tiki_taka before either shared fix, and was genuinely weak overall (1-10-1). 2026-08-21 backfill (post full reset-fix pass): 4-9-0, GF7-GA1, still zero losses — a real, growing improvement each fix pass, worth another look before `high_line_zone` if the anti-tiki_taka thread is picked back up again, though not yet re-tested head-to-head against `tiki_taka` itself. |
| `high_line_zone` | 2026-08-20 | parked, **regression root-caused and fixed 2026-08-21** | Built to deny tiki_taka's give-and-go trio 1v1s via a zone screen (`BlockShapeTactic`) instead of man-marking, switching the ball past its thin 2-robot cover. Original backfill: real mid-pack strategy (2-8-2, GF4-GA3) — beat `low_block` 1-0 and `zone_fluid` 3-1. Post-`block_attacker`/`go_to_ball` re-run backfill regressed to 0-13-0, GF0-GA0 (see Known open bugs for the root cause and fix) — verified fixed via direct re-run (vs `low_block` scores 1-0 again), and the 2026-08-21 backfill (after also fixing `DecoyOverloadTactic`'s own reset gap, see Known open bugs) confirms it at scale: 2-7-4, GF2-GA5 — real wins again, though not yet back to the original 2-8-2 mid-pack form. |

## Known open bugs

- **Every sim-mode tournament match started at `FORCE_START`, skipping
  `PREPARE_KICKOFF` entirely — so "kickoff" possession was a simultaneous
  release-and-race, and the resulting tie was broken by sub-millimetre rsim
  floating-point noise, which then locked in a whole match's shape
  (root-caused 2026-08-23; both fixes below implemented and verified
  2026-08-23 — real, substantial improvement, not a total elimination, see
  the "Conclusion" paragraph for why)** — root cause of the "side is the
  dominant variable" finding in the decoupled tournament above. First
  correction to an earlier framing of this bug: it is **not** a
  `PREPARE_KICKOFF_*`-stage issue — `StrategyRunner` seeds `CustomReferee`
  with `RefereeCommand.FORCE_START` by default for every sim-mode match
  (`strategy_runner.py:389-401`, `self.mode != Mode.REAL`; both
  `tournament.py` and `full_match_tournament.py` never pass
  `referee_initial_command`, so every tournament match to date has used this
  default), by design ("sim matches... releas[e] both teams at the ball at
  the same instant" rather than going through a real kickoff ceremony with
  `RefereeOverride`/`PrepareKickoffOursStep` and encroachment enforcement).
  Confirmed via direct trace: `game.referee.referee_command` reads
  `FORCE_START` from tick 0, not `PREPARE_KICKOFF_YELLOW`/`BLUE`, for the
  whole match. So `RefereeOverride` is never engaged at the start of these
  matches at all — both teams' normal pickers are live and racing for the
  ball from tick 0, which is itself a real rules deviation from a proper
  kickoff worth knowing about independent of the bug below.

  Given that framing, the mechanism: at force-start the ball sits exactly at
  the centre and both teams' formations are true mirror images
  (`formations.py`'s `_mirror` verified exact: `2*cx - x`, `theta -> pi -
  theta`) — so `_friendly_closer_to_ball` (`kernel_strategy.py:513-527`)
  should read as a genuine tie either way `my_team_is_right` is set. It
  doesn't: traced directly (`counter_flow` vs `tiki_taka`, tick 1), rsim's
  physics resolves the two teams' near-identical starting robots to positions
  ~0.1-0.3mm off their true mirror point (e.g. right-side friendly robot 3 at
  `x=0.71999`, its exact mirror opponent at `x=-0.71986` — a 0.00012m gap that
  should be zero under perfect mirror symmetry). `_friendly_closer_to_ball`'s
  bare `friendly_dist < enemy_dist` comparison has no tie margin, so this
  sub-millimetre noise deterministically resolves to `False` on the right and
  `True` on the left — confirmed via direct trace, not inferred. Both
  `counter_flow` and `tiki_taka`'s pickers branch hard on this single boolean
  (attack-heavy split vs press-heavy split) and neither ever revisits the
  choice once robots commit to their first tactic, so one coin-flip-width
  difference at tick 1 cascades into a completely different match: by tick 60
  the same nominal robot (id 3) is at `(0.399, -0.874)` on the right and
  `(0.028, 0.504)` on the left — not a small drift, a different tactical
  branch entirely. This is the same bug *class* already fixed twice elsewhere
  in this codebase (`_find_best_shot`'s hysteresis, `_score_goal`'s
  `switch_margin` — see below) — a fresh-every-evaluation comparison with no
  tie tolerance — just triggered once at force-start instead of flickering
  tick-to-tick.

  **Tested both candidate fixes directly, so this is verified, not
  speculated.** Seeding `StrategyRunner(referee_initial_command=
  RefereeCommand.PREPARE_KICKOFF_YELLOW)` (auto-advance state machine
  confirmed sound for this: all 5 `auto_advance` stages plus the legacy
  `force_start_after_goal` path each gate on a real physical readiness
  condition with a sustained-delay debounce — `_all_robots_clear`,
  `_kicker_in_centre_circle`, `_penalty_kicker_ready`, `_free_kick_ready`,
  `_ball_placement_done` — none silently skips a stage; the only shortcut was
  `StrategyRunner`'s initial-command default) does produce a real kickoff:
  `PREPARE_KICKOFF_YELLOW` for `prepare_duration_seconds` (3s) plus the
  kicker's walk to the centre circle (~5s total observed), then `NORMAL_START`
  with the kicker correctly mirror-positioned on both sides (traced:
  `(0.797, -0.010)` right vs `(-0.797, 0.005)` left — a true mirror, unlike
  force-start's already-diverged formation). **This measurably improves
  things — `_friendly_closer_to_ball` now agrees between sides through the
  entire approach phase, not just at tick 1 — but does not fully eliminate
  the bug.** Traced past `NORMAL_START`: the edge stays `True` (friendly
  closer) identically on both sides until the kicker actually reaches the
  ball, then flips to `False` at a slightly different tick on each side
  (tick 320 left vs 321 right in one trace) and the same cascade as before
  follows. Root cause: reaching the ball is inherently a near-zero-distance
  moment regardless of how the approach itself started, so the same
  sub-millimetre rsim tie recurs right at first touch — the real kickoff
  ceremony narrows the window this bug can fire in (no longer exposed for
  the entire match, only right at first contact) but doesn't close it.
  **Conclusion: both fixes were needed, not either one alone — both are now
  implemented and verified.**

  1. **Real kickoff ceremony.** `tournament.py` and `full_match_tournament.py`
     now seed `referee_initial_command=PREPARE_KICKOFF_YELLOW`/`_BLUE`
     (`tournament.py:127-137`, always `_YELLOW` since `config_a` is always
     yellow and the `simulation` profile's `kickoff_team` defaults to
     `"yellow"`; `full_match_tournament.py`'s `run_match_cell` picks
     `_YELLOW`/`_BLUE` matching its own `a_kicks_off` parameter). Verified
     the state machine itself has no other shortcut hiding nearby: all 5
     `auto_advance` stages plus the legacy `force_start_after_goal` path
     (`state_machine.py:198-397`) each gate on a real physical readiness
     condition with a sustained-delay debounce (`_all_robots_clear`,
     `_kicker_in_centre_circle`, `_penalty_kicker_ready`, `_free_kick_ready`,
     `_ball_placement_done`) — the only shortcut was `StrategyRunner`'s
     initial-command default, now overridden. `tournament.py`'s
     `MATCH_DURATION_SECONDS` bumped 60.0 -> 65.0 to compensate for the ~5s
     ceremony overhead a "60s" match now spends before live play starts
     (`prepare_duration_seconds=3.0` plus kicker walk time), so total live
     play time is unchanged.
  2. **Hysteresis margin on `_friendly_closer_to_ball`.** Added
     `_CLOSER_TO_BALL_MARGIN = 0.05` (`kernel_strategy.py`, right above the
     function): `friendly_dist < enemy_dist - margin` instead of a bare `<`.
     0.05m is ~500-1500x the observed noise floor (~0.1-0.3mm) but well below
     `ROBOT_RADIUS` (0.09m), so a genuine near-tie between two robots
     actually converging on the same loose ball still resolves by real
     distance, not by which side the deadzone happens to favour. All 8
     call sites already treat `is not True` (i.e. `False` or `None`) as
     "not clearly ours, play conservative," so a margin-induced `False` on a
     noise-level tie needed no caller changes — verified by reading every
     call site.

  **Verified effect, and why it's a real improvement without being a total
  fix.** Traced `counter_flow` vs `tiki_taka`, both fixes active: the tick-1
  noise-only tie (previously `False` right / `True` left, a 0.00012m gap) is
  now correctly suppressed to `False`/`False` on both sides. Across a 400-tick
  post-kickoff window, disagreement between mirrored right/left runs dropped
  from 233/400 ticks (before either fix, `FORCE_START` only) to 0 ticks during
  the actual approach phase — but one genuine disagreement remains right at
  first ball contact (tick ~320): both sides' friendly-enemy distance gap
  crosses the 0.05m deadzone boundary within one tick of each other (real,
  non-noise gaps of several centimetres, not sub-millimetre), so one side
  crosses first purely from reaching the ball a few milliseconds sooner. A
  margin cannot fix this — it suppresses noise-level ties, but two independent
  physics runs converging on a moving threshold will not always cross a fixed
  discrete tick boundary simultaneously even with a real (if tiny) kinematic
  difference between them. That single-tick edge case can still cascade over
  a long match, the same way the original bug did, just far more rarely (one
  narrow real-crossing window instead of the entire match being exposed to
  noise). Full test suite green after both fixes (736 passed, 4 skipped, 2
  xfailed, unchanged from before). Not attempted this session: re-running the
  48-match decoupled tournament with both fixes active to measure how much
  the side-dependence pattern actually shrinks in aggregate — the per-tick
  trace above shows the mechanism is real and improved, but only a fresh
  tournament run would show the practical size of the remaining effect on
  match outcomes.

- **Out-of-bounds ball froze every full-match `DIRECT_FREE_*`/ball-placement
  restart forever (fixed 2026-08-22)** — found via the first full-length
  (600s) tournament run (`replays/tournament_20260822_221958/`): all 6
  matches froze in a restart within the first two minutes and stayed frozen
  for the rest of the 600s (confirmed via the replay trail: the referee
  command genuinely never changed again after the freeze — not a slow
  strategy, a real deadlock). Root cause, two compounding bugs:
  1. `DirectFreeOursStep`/`BallPlacementOursStep` (`utama_core/custom_referee/actions.py`)
     computed the kicker/placer's approach point relative to the ball, then
     clamped it to `_clamp_to_field`'s 0.1m in-bounds inset — fine when the
     ball is in play, but a `DIRECT_FREE`/ball-placement restart is routinely
     awarded *because* the ball went out of bounds, so the true approach
     point sits outside the line too. The clamp then stranded the robot 0.1m
     inside the line while the ball sat farther out — confirmed via direct
     per-tick trace: the kicker converged smoothly to a dead stop exactly
     0.307m from a ball resting 0.307m past the boundary (`_KICKER_READY_DIST`
     is 0.3m), so `CustomReferee`'s auto-advance-3 condition could never
     fire. Fixed with a new `_clamp_to_field_or_ball` helper: only clamp if
     doing so doesn't push the target *farther* from the ball than it already
     is (used by both steps' ball-approach targets; `_clear_to_legal_positions`,
     used for parking clear robots away from the ball, keeps the plain clamp).
  2. Fixing (1) exposed a second bug: `FastPathPlanner` still treated the
     field-boundary wall as a real obstacle to detour around even once the
     target was legitimately allowed to sit beyond it (`sanitize_target`
     already exempted boundaries from *target* repulsion, but not from *path*
     routing) — the kicker then orbited the boundary forever, swinging
     between ~0.13m and ~0.44m from the ball every cycle, confirmed via
     per-tick trace with zero real obstacles anywhere nearby (ball
     stationary, nearest enemy >1m away) to rule out the detour-hysteresis
     mechanism instead. Fixed in `FastPathPlanner._path_to`: field-boundary
     segments are now dropped from the path-routing obstacle list whenever
     the (pre-projection) target itself is outside the field, matching
     `sanitize_target`'s existing boundary exemption so the two don't
     disagree about whether crossing the line is allowed.
  Verified: the exact frozen match now resolves the restart in ~3 seconds and
  plays the rest of the 600s normally; full test suite unaffected (736
  passed, 4 skipped, 2 xfailed). Re-run: `replays/tournament_20260822_224929/`
  (see the full-match section above) — all 12 matches completed with no
  further freeze.

**Note (2026-08-22): the manual `ctx.motion_controller.reset()` fixes described
throughout this section no longer exist in the tactic files.** They've been
superseded by automatic discontinuity detection at the PID level
(`AbstractPID.calculate()`, commit `91100ff`) — the underlying bug class and
root-cause analysis below is still accurate and worth reading, but if you go
looking for the `reset()` calls themselves in `give_and_go.py`/
`switch_of_play.py`/etc. to use as a template for a new tactic, they're gone;
new tactics don't need to call `motion_controller.reset()` for this at all.
See `docs/roadmap.md`'s "Motion-controller discontinuity handling" entry for
the full account of what replaced them and why.

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
  green (714 passed) after the fix.
- **Three more tactics had the same "no `motion_controller.reset()` across an
  orientation discontinuity" bug, found by a dedicated architecture audit
  rather than by another backfill surprise (fixed 2026-08-21)** — after the
  `GiveAndGoTactic` fix above revealed the bug class could hide in any tactic
  with a discrete phase/role state machine, a research pass specifically
  checked every remaining tactic's phase transitions for the same shape.
  Confirmed exposed and fixed:
  1. **`PassAndShootTactic`** (`utama_core/tactics/pass_and_shoot.py`) — the
     `pass_then_score`->`score` transition: the receiver spent the pass leg
     facing the passer (`_pass_exec`'s `intercept_oren`) and was immediately
     re-aimed at goal. Fixed with `ctx.motion_controller.reset(receiver_id)`
     right on the transition, mirroring `give_and_go.py`'s hop-hand-off fix
     exactly.
  2. **`LeadAndSupportTactic`** (`utama_core/tactics/lead_and_support.py`) —
     the `has_ball(leader_id)` catch edge: the leader's orientation source
     jumps from `go_to_ball`'s ball-facing approach to `turn_on_spot`'s
     goal-facing shot aim the instant it catches the ball. This directly
     contradicted an earlier in-session claim that this tactic didn't need
     the fix — that check had only considered the leader-reassignment edge,
     not this has_ball-flip edge, which has the identical shape to
     `give_and_go.py`'s already-fixed catch edge. Fixed via a new
     `LeadAndSupportMem.was_carrying` bool, reset only on the False->True
     edge (mirroring `SwitchOfPlayMem.was_shooting`'s pattern), with a guard
     so a freshly-chosen leader can't inherit a stale `True` from whoever
     held the role before.
  3. **`DecoyOverloadTactic`** (`utama_core/tactics/decoy_and_overload.py`) —
     two separate discontinuities in the same tactic: (a) `"lure"`->`"finish"`,
     where the decoy's orientation source jumps from the lateral
     touchline-lure target to `_score_goal`'s goal-facing aim; (b) within
     "finish" itself, the overloader's `_pass_exec`->`_score_goal` hand-off
     (same shape as every other pass-then-shoot transition fixed this
     session). Both fixed with `ctx.motion_controller.reset()` at their
     respective transition ticks.

  Trace-verified via full-match runs for `DecoyOverloadTactic` (both edges:
  orientation converges monotonically toward a slowly-drifting shot target
  over hundreds of ticks, diff shrinking from >6 rad, i.e. a wrapped near-zero
  gap, down to 0.03 rad by kick time — no non-convergent spin) and
  `LeadAndSupportTactic` (13-tick window, diff shrinking monotonically from
  0.12 to 0.05 rad). `PassAndShootTactic`'s fix could not be exercised via a
  full match this session — `low_block` (the only strategy using this
  tactic) was observed getting stuck in "setup" for entire 60s matches
  against every opponent tried, a separate, likely pre-existing issue
  unrelated to this fix (see `low_block`'s row) — but the fix itself is a
  byte-identical call to the same `MotionController.reset()` API already
  confirmed correct at 6 other call sites, placed at the exact tick analogous
  to `give_and_go.py`'s already-verified hop-hand-off fix, so it is applied
  with high confidence pending a future trace once `low_block`'s setup-phase
  issue is separately resolved. Full test suite green (714 passed) after all
  three fixes. Full 91-match backfill re-run — see the top of this doc for
  the fresh numbers, which supersede every earlier backfill in this file.
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
- **`low_block`'s `PassAndShootTactic` observed stuck in "setup" for entire
  60s matches — root-caused 2026-08-21, a design tension not a quick patch**
  — while trace-verifying the `pass_and_shoot.py` reset fix above, every
  match tried against `low_block` (vs `switch_of_play`, `three_slot`,
  `overload_press`, `press_and_pass`, `high_press`) showed
  `PassAndShootMem.phase` staying `"setup"` for the tactic's full 2-robot
  attack-slot lifetime. Direct trace (standalone script instrumenting
  `run_setup_phase`, `low_block` vs `three_slot`, 20s of sim time) found the
  actual mechanism: the **receiver has no problem at all** — it ignores
  ball state and walks straight to its fixed `receiver_position`, settling
  there (`at_target=True`) within ~6s and staying. The **passer never gets
  there** — `run_setup_phase` only calls `_move_to(passer_position)` once
  the passer already has the ball (`has_ball(..., visual=True)`, plus a
  10-tick flicker-grace window), and with only 2 of 5 outfield robots on
  attack, the passer can spend the whole match contesting the ball with a
  heavier opponent, briefly touching it and losing it again inside the
  grace window, over and over, without ever accumulating the sustained
  possession needed to leave the contested area and start walking toward
  `passer_position`. Confirmed directly: `has_ball_v` stayed `False` for the
  passer through the entire first 10s of the traced match while it
  wandered near midfield rather than progressing toward its target 5m
  away. This is a possession/robot-allocation design gap in
  `run_setup_phase` itself (it has grace periods for possession *flicker*,
  but no fallback for genuinely being unable to win the ball at all) — the
  same shape of open question as `counter_press`'s turn-budget mismatch
  below, not a bug fixable as a small patch. Not attempted this session;
  worth a real design pass (e.g. letting setup complete without possession
  and deferring ball acquisition to the pass phase, or giving the passer
  help winning the ball back) before revisiting `low_block`'s backfill
  record.

## Updating this file

- New strategy added → add a row here in the appropriate section (or a new
  section) with today's date and a one-line description pulled from its
  docstring.
- Strategy run against another → update the "Description" cell (or add a result
  note) with the outcome once you have one; don't leave a `competitive` strategy
  marked "untested" after it's actually been tested.
- Strategy stops being iterated on after a loss → move it to `Parked` with the
  result and why work stopped, rather than deleting the row.

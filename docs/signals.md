# Signals

Every signal we record about how a strategy plays, grouped by the question it answers. Use it
when designing or debugging a strategy: find the question, read the signal, compare with the
reference range, and follow "if it's off". Add a signal here when you add one anywhere else.

Signals are diagnostics, not objectives. Rank strategies by results (W-D-L, goals); use these to
explain a result, to find what a strategy does badly, and to notice a shared primitive
(reception, carrying, the planner, the referee) failing every strategy at once. A strategy that
never passes or shoots loses the ball least, so no signal is "better" on its own.

## Where they come from

Every saved `tools/tournament/round_robin.py` run prints its sections and writes the same data to
`replays/<run>/summary.json`:

| Printed section | `summary.json` key | Computed by |
|---|---|---|
| Standings, STRATEGIES | `standings`, `strategies[*]` | `MatchStats` (live, every match) |
| LOSS KINDS, BALL LOSSES, `passes:` line | `strategies[*].real_loss_kinds_as_a`, `ball_losses` (tables in `ball_losses.md`) | `analysis/turnover_breakdown.py` |
| CHANCES | `strategies[*].chances`, `pass_progress_m`, `forward_pass_share` | `analysis/chances.py` |
| FOULS | `fouls` | `MatchStats` |
| STALLS | `stalled_match_count`, `stall_incidents`, per-match `stats.stall_events` | `MatchStats` stall watchdog |
| RESTARTS | `restarts` | `analysis/restart_outcomes.py` |
| — | `results[*].stats` | per-match `MatchStats`, everything above per match |
| — | `run` | git commit, dirty flag, argv: compare runs only at clean commits |

**Offline only** (not in a round-robin's output): `tools/metric_correlation.py <run_dir> ...`
replays a run at 10 Hz for the signals marked *offline* below.

**Sides.** Per-match stats are config_a's view ("friendly"); a strategy's row reads its own
side of every match. Signals marked *config_a* need the intentions log, which only config_a
writes, so they cover the matches a strategy played as config_a (`matches_as_a`).

**Reference ranges** are per match over the 22 strategies of `tournament_20261005_170958`
(600 s matches, strategy-guard at `b26f5b37`): min · median · max, with who holds the extremes.
They show what normal looks like now; they move as strategies and shared code change.

## Is something broken?

Check these first: a broken strategy's other signals mean nothing.

| Signal | Where | Meaning | Reference | If it's off |
|---|---|---|---|---|
| Stalls | STALLS; `stats.stall_events` | `RESTART_STALL`: a restart held > 15 s, with a one-line diagnosis. `COMMITTED_FROZEN`: ball moved < 5 cm for > 10 s in live play while a slot was committed, with who holds the ball | 10 of 231 matches | A bug to fix before anything else. Rerun with `round_robin.py --pair A B`, look with `tools/replay_trace.py` and `render_window()` |
| Possession backstop | `results[*].possession_backstop` | possession pinned 100/0 with the ball barely moving: a freeze the watchdog missed | 0 | Same as a stall |
| Restarts not taken | RESTARTS; `restarts.by_kind[*].outcomes` | a kickoff, free kick or penalty `voided`, `timeout`, `stopped_before_kick` instead of `taken` | free kicks: 3414 taken of 3467 | The taker can't reach or kick the ball: restart positioning, a planner clamp, a keep-out circle |
| Fouls by rule | FOULS; `fouls` | every foul by rule, then strategy/tactic of the offending robot | per strategy 6.3 (`score_aware_counter_flow`) · 7.9 · 23.3 (`three_slot`); run: out_of_bounds 2938, excessive_dribbling 448, defense_area 310, crashing 307 | One tactic dominating a rule is its bug: dribbling too far, entering the box, driving into robots |
| `no_progress` restarts | `results[*].stats.rule_event_counts` | the referee stopped play: nobody moved the ball for 10 s | 296 in the run | Two robots wedged on the ball, or a carrier holding without a pass |
| Near stalls *(offline)* | `near_stall_events` | a side's nearest robot stuck within 0.5 m of the ball for 2 s without gaining it: an early warning before the watchdog | not checked on current code | Wedges, a robot that can't take the ball |
| Kicker thrash *(offline)* | `kicker_identity_thrash` | during a restart, how often a side's nearest-to-ball robot changes | not checked on current code | Restart takers swapping on position noise (a sticky pick missing) |
| Retreat-reapproach *(offline)* | `retreat_reapproach_oscillations` | a side's nearest robot closes on the ball, backs off, closes again (> 0.3 m each way) | not checked on current code | Planner local minimum, congestion, a target that flips |
| Longest carry *(offline)* | `ball_carrier_hold_max_s` | the longest one robot held the ball in a match | not checked on current code | A carrier dragging the ball or holding for a pass that never comes |

## Does it create chances?

| Signal | Where | Meaning | Reference | If it's off |
|---|---|---|---|---|
| Shots | STRATEGIES `shots`; `chances.shots` | hard balls from the attacking third heading into the goal mouth (on target by construction) | 1.4 (`three_slot`) · 6.1 · 11.2 (`split_shape`) | Few shots with many entries: attacks stall in the final third |
| Attacking-third entries | STRATEGIES `entries` | times the ball crossed into the attacking third | 6.6 (`shadow_switch`) · 11.9 · 17.2 (`split_shape`) | Few: the build-up never reaches the final third |
| Regain to shot | CHANCES `regain>shot`, `secs`; `chances.regain_to_shot`, `regain_to_shot_s` | share of open-play regains (kept ≥ 1 s, not just after a restart) followed by a shot within 10 s, and how fast | 1% (`shadow_switch`) · 9% · 27% (`split_shape`); 2.0 s · 4.4 · 9.3 | Low: winning the ball leads nowhere. Slow: too many passes before the shot |
| Free kick to shot | CHANCES `fk>shot`; `chances.free_kick_to_shot`, `attacking_free_kick_to_shot` | share of the side's free kicks (corner and goal kicks included) followed by a shot within 10 s | 0% (`three_slot`) · 12% · 38% (`split_shape`) | Low in the attacking third: set pieces are wasted |
| Shot distance | CHANCES `dist`; `chances.shot_distance_m` | metres from the goal centre where the shooting side last held the ball | 2.2 · 2.5 · 3.0 m | Far: shooting because nothing better was found |
| Open goal mouth | CHANCES `open`; `chances.shot_open_goal` | share of the goal mouth no opponent blocked at the shot. Tops out near 75%, the keeper always covers some | 43% · 50% · 68% | Low: shooting into bodies; the attack never opens a lane |
| Conversion | CHANCES `conv`; `chances.conversion` | goals per shot. High for everyone, since a shot is already on target | 21% · 35% · 60% | Read with distance and open goal, not alone |
| Goals without a shot | `chances.unshot_goals` | goals no detected shot preceded: deflections, slow rolls, scrambles | 73 of 1068 goals | Many: goals by accident, not by design |
| Rebound support *(offline)* | `shot_backed_up_rate` | share of shots with a teammate within 1 m of the ball | not checked on current code | Low: nobody follows up a save |

## Does it keep and move the ball?

| Signal | Where | Meaning | Reference | If it's off |
|---|---|---|---|---|
| Completed passes | STRATEGIES `passes` | passes from one teammate's control to another's | 15 (`split_shape`) · 38 · 61 (`overload_press`) | Many passes with few shots: circulation without progress |
| Pass progress | CHANCES `progress`, `fwd`; `pass_progress_m`, `forward_pass_share` | mean metres gained toward goal per completed pass, and the share gaining ≥ 1 m | 0.3 (`overload_press`) · 0.65 · 1.7 m (`split_shape`); 10% · 23% · 56% | Near 0: sideways passing. Fine if the strategy wins by regaining instead |
| Pass outcomes, catch rate *(config_a)* | `passes:` line; `ball_losses.receptions` | every pass followed to `received` / `missed_reception` (reached a teammate, no contact) / `intercepted` / `off_target`; catch rate of reachable passes, by receiver facing and by tactic | run: 79% caught; 96% when the receiver faces the ball within 10° | Low catch rate: reception or passing into a turned receiver. ±4–8 points between runs is noise |
| Real ball losses *(config_a)* | LOSS KINDS; `real_losses_as_a`, `real_loss_kinds_as_a` | losses not won back within 1 s, by kind: `tackled`, `ball_out_after_kick`, `shot_saved_or_blocked`, `pass_intercepted`, `loose_ball_lost`, `foul`, `ball_out_other` | total 25 (`tiki_taka_plus`) · 34.5 · 42.7 (`three_slot`); tackled 2 · 12.3 · 23.8 | The dominant kind names the fix: tackled (carrying into pressure), kicked out (aim), intercepted (pass choice) |
| Losses by tactic *(config_a)* | BALL LOSSES; `ball_losses.by_tactic`, `ball_losses.md` | which tactic held the ball when it was lost | — | One tactic with most losses is where to look |
| Possession under pressure | `results[*].stats.possession_under_pressure_s` | seconds the side's ball holder had an opponent within 0.5 m | — | High: carrying into opponents |

## Does it defend?

| Signal | Where | Meaning | Reference | If it's off |
|---|---|---|---|---|
| Goals against | STRATEGIES `GA` | | 23 (`split_shape`) · 40 · 122 (`shadow_switch`) per 21 matches | |
| Danger conceded | CHANCES `danger`; `chances.danger_s_per_match`, `danger_spells_per_match` | seconds the enemy held the ball in the side's defensive third in live play, and how many separate spells | 50 (`overload_press`) · 78 · 167 s (`high_line_zone`) | High: the defense doesn't win the ball back near its goal or lets the enemy settle there. One stalled match can add hundreds of seconds |
| Shots faced, open goal faced | `chances.shots_faced`, `faced_open_goal` | the enemy's shots and how open they were | open 37% · 49% · 66% (`give_and_go_solo`) | Open shots faced: nobody blocks the lane; usually everyone is upfield |
| Save rate | CHANCES `save`; `chances.save_rate` | 1 - goals from shots / shots faced. Every strategy fields the same keeper, so this measures the shots the defense allows | 49% (`high_press`) · 64% · 74% (`split_shape`) | Low: see open goal faced |
| Regains | `chances.regains` | open-play regains kept ≥ 1 s | 18 (`shadow_switch`) · 25 · 29 (`high_press`) | Few: the press or the marking doesn't win the ball |

## Restarts

| Signal | Where | Meaning | Reference | If it's off |
|---|---|---|---|---|
| Free kick to shot | see "Does it create chances?" | | | |
| Restart to first entry | `results[*].stats.restart_to_first_entry_s`, `n_restarts`, `n_restarts_with_entry` | seconds from a restart until the possessing side's ball reaches its attacking third; restarts with no entry are counted, not averaged | — | Slow or rarely entering: restarts are played backward or lost |
| Restart outcomes | RESTARTS | see "Is something broken?" | | |

## Adding a signal

Check this page first: it probably exists. A new signal should be interpretable on its own,
in game terms, and say what to do when it's off; correlation with results is not required.
Put it where it's cheapest:

- a count the match can keep as it plays: `MatchStats` (`engine/match_stats.py`). Changing it
  changes every match's fingerprint, so the next `--reuse` round-robin plays everything again;
- anything else derived from replays, both sides: `ChanceTracker` (`analysis/chances.py`), fed
  from the replay pass the ball-loss breakdown already makes. Cached records keep the analysis
  they were stored with: refresh them after changing it (`utama_core/replay/match_cache.py`);
- needing the intentions log (which tactic, which robot's role): `analysis/turnover_breakdown.py`,
  config_a only.

Then add its row here with a reference range from a full round-robin.

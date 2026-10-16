# Sim realism: closing the gap between rsim and real matches

Status: plan, 2026-10-14 (priorities added 2026-10-16). Nothing below is built yet unless it
says so.

## Where this leads

The end goal is **search over the structured strategy space**: tactics, their parameters, and
the scheduler (partitioner) that combines them. Candidates are scored with a **dense value
signal** instead of goals alone. Three pieces, in priority order:

| Priority | Work | Why this order |
|---|---|---|
| P0 | Sim realism, Phases 0–3 below: scorecard, league ranges, seeded per-match physics draws, calibration | Search finds whatever wins in the sim. If the sim rewards an unrealistic detail (a perfect keeper, kicks that never miss), stronger search exploits it harder. The seeded draws are also the controlled-noise framework every later evaluation uses |
| P1 | [Value model fitted on real logs](#p1-value-model-on-real-data) | A model trained on sim outcomes learns the sim's errors. Fitted on real matches, it values situations the way real matches do. Safe to apply to sim strategies only once the sim varies over realistic ranges |
| P2 | [Structured search](#p2-structured-strategy-search) scored by the value model in the randomized sim | Needs both of the above |

A "novel" winner from search looks exactly like a sim exploit until checked. Every one gets
the per-strategy scorecard: an edge from parts the sim gets right (positioning, chance
creation) is kept; an edge from parts it gets wrong is suspect.

## Why

Strategies are developed and ranked in rsim. A ranking is only useful if a strategy that wins
in the sim would also win on a real field. That needs two things:

- **Sim physics in the right range.** The sim must not reward behaviour that only works
  because of an unrealistic detail, such as a perfect keeper or a perfectly accurate kick.
- **Robustness.** Real parameters drift (floor friction, kick speed as the capacitor drains,
  dribbler wear, one robot slower than another). A strategy must win across that drift, not
  at one tuned value.

The goal is therefore not a perfect replica of one real robot. It is a sim whose behaviour
matches real matches on the measures that decide goals, and that varies its physics over the
ranges real robots vary over.

We assume no measurements of our own robots for the next month. Ranges come from the official
match logs of other teams (see [Official logs](#what-the-official-logs-can-and-cannot-give)),
and are replaced by our own measurements once hardware tests are possible.

## What we found (2026-10-14)

Real: 54 official Division B matches from 2025–2026 (`tools/convert_ssl_log.py`). Sim: the
baseline round-robin `tournament_20261014_144456` (66 matches). An expected-goals (xG) model
was fitted on real shots. On held-out matches it ranks shots with AUC 0.82 and its predicted
rates agree with observed ones.

Goals per team per live minute are the same in sim and real. That agreement hides two
differences that cancel:

| Shots of equal quality (xG band) | Real | Sim |
|---|---|---|
| Saved by the keeper, all shots | 17% | 39% |
| Saved by the keeper, xG > 0.35 | 20% | 52% |
| Wide or off the post | 22% | 6% |
| Goal rate, xG < 0.05 | 2% | 14% |
| Goal rate, xG > 0.35 | 53% | 35% |

Sim keepers save far more good chances; sim shooters almost never miss.

Other differences:

- **Ball slow-down:** the sim ball keeps 57% of its speed by 1 m from goal; real balls keep 67%.
- **Kick speed:** every sim kick is 4.75 m/s; real kicks range from 3.9 to 5.5 m/s (interquartile).
- **Keeper position:** the sim keeper always stands on the goal line, 0.09 m deep; real keepers stand about 0.2 m out.
- **Shot count:** with passes and clearances counted, sim teams make 1.8× more goalward kicks than real teams.
- **Live time:** real matches have about 5.5 live minutes; the sim has about 9.1. This, not finishing, is why sim matches have more goals per match.

Lesson: **never judge realism by goal totals.** They can match for the wrong reasons.
Compare the components, as in the [scorecard](#the-realism-scorecard).

## Parameters

Ordered by how much each one is likely to change match outcomes.

**Between** means the value differs per robot or per match (one robot's top speed, a field's
friction). **Within** means kick-to-kick noise for the same robot.

| Parameter | Why it matters | From official logs? | rsim today |
|---|---|---|---|
| Sensing-to-action latency (vision, decision, radio, motor response) | Likeliest cause of the keeper gap | Only as other teams' total reaction time: from a kick to their keeper accelerating | None: commands act on the next tick |
| Ball rolling deceleration (and the switch from sliding to rolling) | Pass and shot timing; keeper reaction time | Yes: fit free-rolling segments, per match (venue carpet) | Fixed in the C++ physics (`vendor/rSim`) |
| Kick speed: between and within | How much time the keeper has; pass weight | Yes: peak ball speed after each kick, per team | Fixed |
| Kick direction error | Wide rate (6% vs 22%) | Partly: shot miss distance, pass error relative to the receiver | None: exact |
| Robot top speed, acceleration, braking, turn rate | Who reaches the ball first | Yes: from the tracker, per robot | One limit for all robots (2 m/s for evaluation) |
| Vision noise and dropped frames (ball especially, also when blocked) | Every decision; keeper tracking | Yes from raw vision packets in the logs; the converter currently reads the filtered tracker | `rsim_noise`, `rsim_vanishing` in `StrategyRunner`, unused by the round-robin |
| Dribbler: hold strength, ball loss when turning or touched, release speed | Possession, passing | Weak: only possession-loss rates as a proxy | Fixed, with the fork's dribbler fixes (`vendor/rSim/FORK_NOTES.md`) |
| Ball bounce off robots and walls | Rebounds, deflections | Partly: speed before and after contact | Fixed |

### What the official logs can and cannot give

- **They describe the league, not our robots.** That's still the right range for opponents, and
  a reasonable starting range for our own robots until we measure them. A strategy robust
  across the league's range is unlikely to break on ours.
- **Robots and ball come from an autoref's tracker**, which filters positions. Peaks
  (kick speed, acceleration) are smoothed, so tracker-based estimates are lower bounds. Raw
  camera detections are also in the logs, for vision noise.
- **Latency cannot be separated from decision time** for other teams. We get their total
  reaction time, which is what matters for keeper saves anyway.
- **Kick direction error needs an intended target**, which the logs don't record. Proxies are a
  shot's miss distance, and a pass's distance from the nearest teammate when it arrives.

## How varying parameters are handled in strategy development

**Domain randomization:** each parameter gets a range, and the sim draws from it.

1. **Between-values are drawn per match and per robot; within-noise per kick.** For example:
   - friction once per match;
   - a top speed and kick-speed mean per robot;
   - kick speed and direction noise on every kick.
2. **Draws come from the match seed.** They join the seeded vision noise already planned for
   the round-robin. When strategy A and strategy B each play the baseline,
   they see the same draws, so the comparison is paired and the extra randomness costs few
   extra matches.
3. **Ranges are measured, not guessed.** Use the spread of team-level medians for between-
   values, and each team's own spread for within-noise. Too narrow overfits to the sim; too wide
   rewards overcautious play.
4. **Randomize only what changes standings.** A sensitivity sweep decides which parameters
   are randomized; the rest stay fixed at the league median.
5. **Edge cases are a separate diagnostic, not part of the ranking.** Run candidates at the
   edges (slow robot, weak kick, worn dribbler) and report how far they fall.
6. **Prefer strategies that measure parameters over ones that assume them.** For example, a
   pass speed chosen from the kick speeds actually observed. The sim then checks that the code
   copes with the whole range.

## The realism scorecard

One report that compares sim with real on the components that decide goals. Each row is a
real value with a 95% interval, the sim value, and a pass mark (sim inside the real
interval):

- shots per team per live minute (passes and kicks from beyond 6 m excluded);
- real xG per shot;
- finishing (goals ÷ real xG), overall and in each xG band;
- shot outcome shares: keeper, defender, wide or post, goal;
- ball speed kept to 1 m from the goal;
- kick speed: median and spread;
- keeper depth and keeper reaction time;
- live minutes per match and restarts per match (referee-driven, so they are checked but not targeted).

Real values are recomputed when new logs are converted. The sim side reruns on any
round-robin.

## Plan

Each phase ends with a checkpoint: results reported, decision taken, then the next phase
starts. Evaluation changes and strategy changes go in separate commits.

### Phase 0: Bring the analysis into the repo (2–3 days)

The xG and shot-outcome analysis currently lives outside the repo (`~/dev/ssl/xg-analysis/`).

- Move it into `utama_core/analysis/`:
  - `shots.py`: the kick-based shot detector. Real goals are linked through the game
    controller's GOAL events, because real goals are confirmed by a human operator seconds to
    minutes late.
  - `xg.py`: the model fit and the decomposition.
  - `shot_outcomes.py`: what happened to each shot.
- Add a `tools/realism_scorecard.py` entry point, and unit tests on the shot detector and goal
  linking using small synthetic frames.
- Commit the fitted real xG coefficients as data, so the sim side doesn't need the real logs.
- Done when: the scorecard reproduces the numbers above for the baseline.

### Phase 1: League ranges from the official logs (1 week, analysis only)

One script per parameter, each giving a between-range and a within-spread with intervals:

1. **Ball deceleration:** free-rolling segments (no robot within 0.3 m, speed below the
   sliding threshold), fitted per match.
2. **Kick speed:** peak speed after each detected kick. Spread across teams, and within each team.
3. **Robot motion:** per robot, the 95th-percentile speed and acceleration, and braking.
4. **Reaction time:** from a shot's kick to the keeper's acceleration onset, and to its first
   movement toward the shot line. The same for the nearest defender.
5. **Kick direction error, by proxy:** miss distance of shots, and pass arrival error relative
   to the nearest teammate.
6. **Vision noise and dropped frames:** from raw camera detections of a ball at rest, and the
   share of frames missing each object. This needs the converter to keep raw detections (a
   small converter option).

Done when: a table with each range and its caveats has been added to this doc.

**Checkpoint:** review which ranges are trustworthy.

### Phase 2: Knobs in the sim (1–2 weeks)

Each knob is off by default, so current results are unchanged, and each comes with a test that
pins its effect:

- **Latency:** a delay buffer between the strategy's command and rsim (Python). This is the cheapest and probably biggest.
- **Vision noise and dropped frames:** wire the existing `rsim_noise` and `rsim_vanishing` through `run_match` and the round-robin.
- **Kick speed noise:** perturb the commanded kick speed (Python).
- **Kick direction noise:** perturb the direction inside the C++ kicker (fork change, recorded in `FORK_NOTES.md`).
- **Ball deceleration:** expose the ball's damping or friction in the C++ fork.
- **Per-robot motion limits:** speed and acceleration caps per robot in the sim controller.
- **One seeded draw** per match, and per robot, from the ranges, recorded in the replay so any match can be reproduced.

Done when: a match reproduces exactly from its seed, and every knob at zero reproduces today's results.

**Physics substeps (gated, not assumed).** rsim runs 5 substeps per tick, hardcoded in
`vendor/rSim/src/robosim/sslworld.cpp` (`SSLWorld::step`); making it a parameter is a fork change. Fewer would be
faster, but substep error is systematic (every kick, contact and dribble wrong the same way),
so it does not average out like vision noise. A fast ball at 6.5 m/s moves about 22 mm per
substep at 5 and 36 mm at 3, against a 43 mm ball, and the fork's kicker and dribbler fixes
were tuned at 5. Physics was 46% of match time (2026-10-10), so 3 substeps saves at most about
15–20% of a round-robin. Steps:

1. Measure match time at 3, 4 and 5 substeps.
2. Measure kick speed and direction, ball slow-down, dribbler hold and release, and collisions
   at 3, 4, 5 and a 10-substep reference.
3. Adopt fewer substeps only if the error against the reference sits well inside the Phase 1
   real ranges, and the scorecard and standings don't move beyond match noise. Adoption
   reruns every cached match, so it goes in with the Phase 4 baseline.

### Phase 3: Calibrate and find what matters (1 week)

- **Sensitivity sweep:** for each knob, rerun a small fixed set of matches at the low, median
  and high end of its range. Measure the change in the scorecard and in standings between a few
  strategy pairs.
- **Calibrate:** set each kept knob to its league range and check the scorecard moves into the
  real intervals, keeper saves and the wide rate above all. Where a knob at the league range
  overshoots or does nothing, record it. That usually means a modelling gap, not a tuning problem.
- Done when: the scorecard passes on its main rows, or the remaining gaps are named with a cause.

**Checkpoint:** decide which knobs become the evaluation default.

### Phase 4: Adopt in evaluation (3–5 days)

- The round-robin uses randomized physics by default, with several seeds per pairing (number
  set by the per-match noise: goal difference varies by about 3 goals per match) and paired seeds for A/B comparisons.
- Run a new baseline round-robin. Every cached match reruns: a cache invalidation, so batch it
  with any pending renames.
- Update `docs/STRATEGY_DEVELOPMENT.md` (evaluation), `docs/signals.md` (scorecard rows that
  become signals) and `docs/strategies.md` (new results).
- Done when: the strategy catalog has been re-ranked under the new evaluation, and the
  changes in ranking are explained.

### Phase 5: Our own robots (when hardware tests are possible)

Replace league ranges with measurements of our robots. Tests, each short and repeatable:

| Test | Measures |
|---|---|
| Step command, timed from the vision log | End-to-end latency |
| Repeated kicks at a target, at several charge levels | Kick speed mean and spread, direction error, drop with charge |
| Rolling and kicked balls on the competition carpet (and our lab floor) | Deceleration, sliding-to-rolling switch |
| Speed and acceleration ramps, per robot | Motion limits, spread across robots |
| Dribble while turning or being touched; release | Hold strength, loss rate, release speed |
| Ball at rest under the cameras | Vision noise and dropped frames for our setup |

Ranges for our robots then replace the league ranges for our side only. Opponents keep the
league ranges.

## P1: Value model on real data

The xG model is already a small value model fitted on reality, for shots only. The extension
values every situation: V = P(we score within 10 s) − P(they score within 10 s), from a dozen
interpretable features (ball position and zone, who has the ball, defenders goal-side,
attackers ahead of the ball, keeper position, the restart it came from). Football analytics
calls these expected threat and VAEP.

- **Fit and check** on real logs, held out by match, like the xG model: calibration and AUC.
- **Score sim strategies** by crediting each action (pass, dribble, interception, shot) with
  the change in V it caused. Reported next to standings in the round-robin summary.
- **Validate** that ranking by it agrees with match results once the sim is randomized.
- **Limits:**
  - 54 real matches (118 with 2022–2024) give about 1,300 shots and 5 live hours: enough for
    a compact model, not for a learned world model or policy.
  - Where sim strategies reach situations the real data rarely contains, the model is
    guessing. Measure the overlap, and fall back to match results there.
  - V assumes league-average play from each situation on, so it cannot credit a tactic whose
    payoff comes after the situation it scores.
- The 2026-10-12 sim-trained value model failed as a cheaper outcome partly because it was
  trained on sim outcomes; this one is not.

## P2: Structured strategy search

Search over what the repo already composes:
- which tactics fill each slot;
- tactic parameters;
- the partitioner that splits robots between slots (for example `split_shape`'s
  possession split).

Candidates are scored by the value signal plus match results in the randomized sim, with paired
seeds. The 2026-10-11 search (PR #142) is the manual version of this loop: candidates
written by hand, screened by round-robin, held out on a ladder. Its lesson carries over: a lead
from one match per pairing is not evidence.

## Risks and rules

- **Don't tune the sim toward goal totals.** Only components, with a named physical cause each.
- **League ranges are a prior.** Don't treat the league median as our robot.
- **More randomness means more matches** for the same confidence. Paired seeds and sequential
  stopping keep that affordable. Measure the cost in Phase 3 before adopting.
- **Real logs are other teams under real referees.** Live time and restarts differ for
  referee reasons, not physics. Compare per live minute.

## Related

- [`signals.md`](signals.md): how strategies are measured.
- [`roadmap.md`](roadmap.md): the workstream entry.
- `tools/convert_ssl_log.py`: official logs as replays.
- `vendor/rSim/FORK_NOTES.md`: physics fork changes.

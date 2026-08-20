# Investigation — `default_vs_lowblock` stays 0-0: passer/opponent tangle at the center circle

Branch: `investigate/default-vs-lowblock-stalemate`
Date: 2026-08-19
Status: **Root-caused, not fixed.** Fix candidates at the bottom are recommendations
for follow-up work, not implemented changes.

## Summary

`build_default_kernel_strategy` vs `build_low_block_kernel_strategy` (60 s, headless
rsim, 6v6, yellow = default, right team) ends **0-0** — reproduced consistently, and
the render windows look exactly as reported: at t=30-38 s and t=50-58 s the two teams'
**passers** (yellow robot 1 and blue robot 1 — not a defender) orbit each other in a
small area around the halfway line / center-circle edge, with the ball pinned between
them. The pattern is "smaller and more mobile than the pre-fix frozen-in-place
pattern" precisely because the Aug-19 setup-phase timeout (`3a17afc`) unfreezes the
passer every 12 s and re-samples setup targets — but the underlying pin never breaks.

This is **not** a referee-rule issue (zero rule events all match), not a planning bug
(the robots do exactly what they're asked), and not a regression from the Aug-16
pass-and-shoot bug fixes. It is a **tactic-level deadlock**: `PassAndShootTactic`'s
setup phase is opponent-blind, and two identical passers converging on the same ball
simultaneously have no escape mechanism.

## Reproduction

```
pixi run python arena_tournament.py                # 60 s, dumps per-tick JSONL to /tmp/opencode/
pixi run python demo_tournament.py default low_block
```

`arena_tournament.py` (renamed from the `probe_default_vs_lowblock.py` committed
with this doc) reruns the exact tournament match while dumping, every tick:
referee command, ball pos/vel, all robot positions, both teams' kernel slot
state (phase, pair, setup targets, phase ticks), and each robot's commanded
motion target. Replays are written via the standard replay writer;
`utama_core/replay/render_window.py` reproduces the t=30-38 / t=50-58 renders
from `replays/probe_default_vs_lowblock.pkl`.

## Evidence

All numbers from one 60 s probe run of the exact tournament match (3600 ticks).

| Metric | Value |
|---|---|
| Final score | 0-0 |
| Referee command | `FORCE_START` for the entire match (see "Contributing factors") |
| Rule events (stats accumulator) | none |
| **Both teams' tactic phase** | `setup` for all 3600 ticks — `pass_then_score`/`score` never reached |
| Longest clean possession (passer ≤0.16 m, other passer >0.4 m from ball) | yellow 0.7 s, blue 0.9 s |
| Ticks both passers within 0.4 m of the ball | 10 % of all ticks |
| Total ball path length, 60 s | **6.8 m** (~0.11 m/s average — a jostle, not play) |
| Max ball speed ever | ~1.0 m/s — **no kick ever fired** (kicks are 4-6.5 m/s; `kick()` is only reachable from `pass_then_score`/`score` phases) |
| Yellow passer distance to its own setup target | never < 0.2 m in 3600 ticks |
| Nearest blue robot to yellow r1, t=48-60 | blue robot 1 on **all 721 ticks** — a strict 1v1 tangle, no third party |

## Root-cause chain

1. **Perfectly symmetric race to the kickoff ball.** Ball starts at (0,0). Yellow r1
   starts at (+3.38, −0.60) and blue r1 at (−3.38, −0.60) — mirrored formation spots,
   both 3.43 m from the ball. Both are `pass_and_shoot` passers («closest of the pair
   to the ball»; both pairs are (1,2)). Same distance, same speed profile, **no head
   start**: the two robots reach capture distance *simultaneously* (~t=2.8-3.0 s).

2. **`go_to_ball` has no «opponent on the ball» concept.** The skill drives the robot
   to the ball's exact position, dribbler on, with an overshoot of only ~9 mm
   (`go_to_ball.py`). Two robots commanded to the same point `(ball.x, ball.y)` from
   opposite sides converge on *each other* around the ball; each one's path planner
   treats the other as an obstacle, so both wedge at 0.11-0.4 m from the ball — the
   physical pin distance (ROBOT_RADIUS+BALL_RADIUS ≈ 0.11 m). The ball sits between
   the two dribbler lines with near-zero velocity for seconds at a time (measured:
   ball velocity ≈ 0 m/s for stretches of 5+ s; occasional shoves of 0.1-0.3 m every
   few seconds as one robot darts around the other — the "smaller and more mobile"
   texture; the shoves are what ratchet the pin point from (0,0) up to (~−0.7, ~1.5)
   over the match).

3. **`PassAndShootTactic`'s setup phase is opponent-blind.** Setup completes only when
   the passer reaches its sampled setup target (~2.5 m from the ball) *while holding
   the ball* (`run_setup_phase`). With an identical robot glued to the ball from the
   other side, the longest possession either side ever sustains is 0.9 s — never
   enough to traverse 2.5 m while being body-blocked. The phase machine has exactly
   one exit from setup (both robots at target) and no way to manufacture that exit:
   no kick, no clear, no dribble-past, no shield, no approach-from-the-free-side.

4. **The 12 s timeout cannot break the pin.** The Aug-19 fix (`3a17afc`) resets
   `PassAndScoreMem` and re-samples setup positions every 720 ticks. But the
   assignment pair stays (1,2), the ball is still pinned at the same point, and the
   re-sampled target is still unreachable — so the dance simply restarts on a ~12 s
   cycle. That is exactly the recurring pattern seen at t=30-38 and t=50-58 (and,
   in this probe, at every ~12 s interval throughout).

5. **The kernel scheduler cannot resolve it either.** `is_committed()` is False the
   whole match (setup is exempt by design), so the free pool never shrinks — but the
   configs have no alternative anyway: default is single-tactic, low_block splits by
   a fixed 0.2 ratio with `min_attack=2`. No barrier reset ever occurs (no restarts),
   so nothing ever clears even the setup-phase state.

## Contributing factors (not root causes, but real)

- **Sim matches start in FORCE_START with no kickoff ceremony.**
  `StrategyRunner` seeds `CustomReferee` with `FORCE_START` in sim modes ("play
  begins immediately", `strategy_runner.py:382-385`). In a real SSL match the
  kickoff gives one team's kicker exclusive access to the center circle and starts
  the race with an asymmetry — here both teams charge the ball at the same instant,
  which is what manufactures the symmetric simultaneous arrival in step 1.
- **`build_default_kernel_strategy` plays with 3 zombie robots.** The config hands
  all 5 outfield robots to `PassAndShootTactic`, which only ever emits commands for
  `robot_ids[0:2]` (pair (1,2)). Yellow robots 3, 4, 5 received **no command for the
  entire match** and stand pixel-immobile at their formation spots (measured bbox
  < 4 cm all match) — e.g. yellow r3 sat at (0.76, 0.03), literally closer to the
  kickoff ball than the passer was. The default config effectively plays
  2v4+keeper. This does not cause the tangle (blue low_block's defenders also never
  converge on the ball), but it is the reason the tangle is a strict 1v1 and why the
  deadlock is so stable — nobody is ever on hand to break it, on either side.

## Fix candidates (recommendations, not implemented)

Ranked by minimalism-first fit with the repo's standing rules (design doc §0/§15,
AGENTS.md "minimalism discipline"):

1. **Skill-level, smallest surface: teach `go_to_ball` (or `run_setup_phase`) to
   deal with an opponent on the ball.** When an enemy robot is within ~0.35 m of the
   ball, approach along the line *through* the ball in the direction that puts our
   body between the opponent and the ball (shield), or approach the free side. This
   is a single shared primitive — every tactic that uses `go_to_ball` inherits it,
   and no tactic interface changes. Caveat: in the 1v1-with-identical-tactic match
   this can still degenerate (both shield) — but it converts a silent deadlock into
   at least a contest, and it fixes the genuinely common case (approaching a loose
   ball guarded by ONE opponent).

2. **Tactic-level: a setup-phase possession budget with a kick-out.** `setup` already
   has a 720-tick overall timeout; add a narrower rule: if the passer has held the
   ball (visual) for ≥N ticks while an opponent is within ~0.4 m, kick/push the ball
   past the opponent into open space (aim behind the defender, toward our own half
   or along the sideline) and re-approach. `kick()` and `find_best_shot`-style
   geometry already exist; this is a few lines in `run_setup_phase`, no new
   concepts. It is also the honest fix for the deadlock specifically: the phase
   machine currently *cannot express* "I have the ball and an opponent is on me,"
   so nothing ever breaks the pin.

3. **Config-level: `build_default_kernel_strategy` should not hand 5 robots to a
   2-robot tactic.** Either restrict the outfield pool to the tactic's real needs
   (the kernel's exhaustive-cover invariant would need a config-side answer to
   "what do the other 3 robots do"), or give the extra robots *some* simple default
   behavior in `PassAndShootTactic.tick()` (e.g. a fixed holding formation around
   the pair — not a new Tactic, just commands for the leftover ids). The zombie-
   robot state is a real defect independent of the tangle: it makes the config's
   "everyone attacks" docstring false and leaves 60 % of the team inert.

4. **Environment: give sim matches a kickoff ceremony (or asymmetric start).**
   Starting the ball at center with both teams free to charge is an artifact of the
   FORCE_START seed; the SSL kickoff rules exist precisely to prevent the symmetric
   race. Before treating tournament 0-0 rates as strategy quality metrics, check
   whether the no-ceremony start is distorting results — e.g. a short PREPARE_KICKOFF
   phase for one team with the other held out of the circle would make `default vs
   low_block` exercise what a real match looks like.

## What was explicitly checked and ruled out

- **rSim dribble/physics flakiness** — not a factor: the pattern is a stable,
  deterministic-looking equilibrium, reproduced across runs, and no dribble-release
  path is even involved (the ball is never released, it is pinned).
- **FastPathPlanner / motion bugs** — robots reach exactly the waypoints the tactics
  request; the planner's obstacle avoidance is what produces the orbit, and it works
  as designed.
- **Referee interference** — zero rule events; FORCE_START is live play, no
  keep-out/defense-area constraints during the tangle (both tanglers stay clear of
  all areas).
- **Defense tactics** — the tangle partner is blue's *passer*, not a low_block
  defender; blue defenders sit at x ≈ −3 all match shadowing the shot line, exactly
  as designed.
- **The Aug-16 five-bug pass_and_shoot fixes regressing** — phase/assignment logic
  works (pair stable, has_ball free of flicker lockups since the timeout fix); the
  tactic is stuck in setup by *physics of the pin*, not by a state bug.

## Artifacts

- `probe_default_vs_lowblock.py` — instrumented reproduction (committed).
- `/tmp/opencode/probe_default_vs_lowblock.jsonl` — per-tick probe rows from the run
  analyzed here.
- `/tmp/opencode/window_30_38.png`, `/tmp/opencode/window_50_58.png` — the two
  render windows the investigation started from.
- `replays/probe_default_vs_lowblock.pkl` — replay for re-rendering.
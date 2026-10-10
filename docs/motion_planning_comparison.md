# Comparing motion-planning algorithms

`tools/motion_planning_benchmark.py` runs `fpp`, `dwa` and `trajsample` through the same
`StrategyRunner` + headless rsim path and writes raw JSON plus a Markdown table to
`benchmark_results/` (gitignored). It is a benchmark/regression report, not a replacement for unit tests of
algorithm internals (bang-bang endpoint invariants, FPP collision-kernel equivalence, ...).

```bash
pixi run python tools/motion_planning_benchmark.py                      # full matrix, once
pixi run python tools/motion_planning_benchmark.py --scenarios direct static_slalom --allow-failures
pixi run python tools/motion_planning_benchmark.py --repeats 5          # expose instability
pixi run python tools/motion_planning_benchmark.py --schemes fpp trajsample --scenarios mirror_swap --repeats 3
pixi run python tools/motion_planning_benchmark.py --list-scenarios
```

Runs are sequential (concurrent cells would contend for CPU and corrupt latency numbers).
Geometry, colours/sides, rsim noise and the sampler seed are fixed; `--repeats` checks that the
deterministic setup stays repeatable. Exits non-zero if any cell fails (after writing reports)
unless `--allow-failures`; `--output-dir PATH` redirects output.

## Scenarios

| Name | Exercises | Timeout | Tolerance |
|---|---|---:|---:|
| `direct` | Unobstructed 6m traversal and braking | 12s | 0.15m |
| `static_slalom` | Three staggered stationary robots | 20s | 0.15m |
| `crossing` | One perpendicular crossing robot | 20s | 0.20m |
| `crossing_oblique_45` | Crossing at 45°, both heading the same general way | 20s | 0.20m |
| `crossing_oblique_135` | Crossing at 135°, heading broadly towards each other | 20s | 0.20m |
| `crossing_offset` | Perpendicular crossing off-centre, reaching it at different times | 20s | 0.20m |
| `crossing_steady_runner` | Opponent follows a point crossing our path at 1 m/s | 20s | 0.20m |
| `overtaking` | Passing an opponent that follows a point ahead on the same line at 0.5 m/s | 20s | 0.20m |
| `grid_intersection` | Four moving robots, four crossing points | 30s | 0.25m |
| `mirror_swap` | Dense 6v6 yielding and convergence (2cm symmetry-breaking offset) | 45s | 0.30m |
| `ball_scrum` | A run to a support spot just past three robots circling the ball at 0.5 m/s, a marker crossing the lane | 15s | 0.20m |
| `recovery_run` | A run back past a 2v2 pack circling the ball, an opponent runner crossing in front | 15s | 0.20m |
| `wing_switch` | Two teammates cross just ahead of a teammate on the ball, an opponent shuttling across | 15s | 0.20m |
| `kickoff_reset` | Ten robots from a corner scramble to the kick-off formation (real positions from a replay) | 20s | 0.20m |
| `narrow_passage` | Threading a 0.24m gap between two stationary robots | 10s | 0.15m |
| `head_on_swap` | Two robots swap positions driving straight at each other | 10s | 0.20m |
| `field_boundary_corner` | Target just inside a field corner | 10s | 0.15m |
| `defense_area_boundary` | Target just outside the defense-area keep-distance (fpp's clamp settles ~0.2m short) | 10s | 0.25m |
| `interception` | Meeting an opponent that follows a point at constant velocity | 8s | 0.20m |
| `sudden_obstacle` | Clear corridor; an obstacle appears mid-path at 2s | 12s | 0.15m |
| `disturbance_recovery` | Robot is teleported off its path at 1.5s and must replan | 12s | 0.15m |
| `start_inside_obstacle` | Starts overlapping a stationary robot; only a new collision fails it | 10s | 0.15m |
| `jittering_target` | `direct`'s geometry with ~1mm per-tick target jitter; compare its time with `direct`'s | 12s | 0.15m |

Opponents with a target or a moving target point drive with the selected scheme; the others stay
put (`sudden_obstacle`'s is teleported into the corridor).

The four crowd scenarios (`ball_scrum` to `kickoff_reset`) are built on where crowds form in a
match. In the 2026-10-10 round-robin a moving robot had 3 or more robots within 1 m 40% of the
time; those crowds were within 1.5 m of the ball 66% of the time, with the robot at about
0.8 m/s and its neighbours at 0.5 m/s. `mirror_swap`, twelve robots head-on at full speed, is a
stress test beyond that. In them robots circling the ball keep moving after they count as
arrived, so the path ratio is not meaningful there; compare time and passes.

The trajsample planner's random sampler is seeded with 0, so `--repeats` gives the same run each
time: one seed is one draw. To compare two versions of that planner, run several seeds (the
planner's `random.Random(0)`) and count passes; a single seed can flip a crowded scenario either
way.

Latest full run: [`motion_planning_results.md`](motion_planning_results.md) (about 1.5 min wall for all
57 cells, before the crowd scenarios). fpp and trajsample pass all 19; dwa collides in 9. `mirror_swap` used to time out for
fpp because four targets sat on the opponent's defense-area edge, which planners keep outfield
robots away from; its back-row targets are now at |x| = 2.9. To record a new full run, copy its `.md`
over `motion_planning_results.md`.

## Pass/fail and metrics

A run passes only if every moving robot reaches its target within tolerance before the
simulated timeout and no two robot centres come closer than `2 * ROBOT_RADIUS` (0.18m); a
repeated cell passes only if every repeat does. Reasons: `completed`, `collision`,
`sim_timeout`, `wall_timeout`, `exception`.

Speed above 2 m/s + 0.15 tolerance is reported as `speed_limit_exceeded` but is diagnostic
only. Acceleration is also diagnostic: it is reconstructed from filtered simulator velocity and
can include estimator transients, so it is not an exact commanded-acceleration contract.

The table reports: completion time; path ratio (travelled / straight-line); worst centre
clearance minus both radii (negative = overlap); collision count; largest final error; stopped
fraction (robot-seconds below 0.05 m/s while unreached); mean/p95/max
`MotionController.calculate()` latency (first 5 calls excluded for Numba warm-up,
`--timing-warmup-calls`); max speed and acceleration.

## Reading it

Pass rate first — a fast planner that collides or doesn't converge isn't a winner. Then: lower
time and path ratio = efficient; large clearance with a long path = over-conservative; high
stopped fraction = yielding, oscillation or local minima; p95 latency matters more than the
mean at 60 Hz. There is deliberately no weighted overall score: the speed/clearance/compute
trade-off is a team decision, and one scalar would hide why a planner won. Archive the JSON
(it records the git revision) and compare matching rows across revisions.

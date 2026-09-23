# Comparing motion-planning algorithms

`tools/motion_planning_benchmark.py` runs `fpp`, `dwa` and `trajsample` through the same
`StrategyRunner` + headless rsim path and writes raw JSON plus a Markdown table to
`benchmark_results/`. It is a benchmark/regression report, not a replacement for unit tests of
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
| `grid_intersection` | Four moving robots, four crossing points | 30s | 0.25m |
| `mirror_swap` | Dense 6v6 yielding and convergence (2cm symmetry-breaking offset) | 45s | 0.30m |

Moving scenarios use the selected scheme for both teams; `static_slalom`'s opponents stay put.

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

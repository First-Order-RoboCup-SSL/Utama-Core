# Comparing motion-planning algorithms

`tools/motion_planning_benchmark.py` runs `fpp`, `dwa`, and `trajsample`
through the same `StrategyRunner` + headless rsim path and writes both raw JSON
and a Markdown comparison table. It is a benchmark and regression-reporting
tool, not a replacement for implementation-level unit tests.

## Run it

From the repository root:

```bash
pixi run python tools/motion_planning_benchmark.py
```

The default matrix runs every scenario once for all three schemes and writes
timestamped files under `benchmark_results/`. Useful smaller or more rigorous
runs are:

```bash
# Quick smoke comparison
pixi run python tools/motion_planning_benchmark.py \
  --scenarios direct static_slalom --allow-failures

# Repeat every cell to expose instability
pixi run python tools/motion_planning_benchmark.py --repeats 5

# Compare two schemes on the known dense case
pixi run python tools/motion_planning_benchmark.py \
  --schemes fpp trajsample --scenarios mirror_swap --repeats 3

# See the scenario catalog without starting rsim
pixi run python tools/motion_planning_benchmark.py --list-scenarios
```

Runs are sequential. Running simulator cells concurrently would make controller
latency numbers contend for CPU and weaken comparisons. The scenario geometry,
team colour/side, rsim noise settings, and trajectory-sampler seed are fixed;
`--repeats` checks whether that deterministic setup remains repeatable.

By default the command exits non-zero if any matrix cell fails, after writing
both reports. Use `--allow-failures` for exploratory comparisons where failures
are expected. Use `--output-dir PATH` to put artifacts elsewhere.

## Scenarios

| Name | What it exercises | Timeout | Endpoint tolerance |
|---|---|---:|---:|
| `direct` | Unobstructed six-metre traversal and braking | 12 s | 0.15 m |
| `static_slalom` | Routing around three staggered stationary robots | 20 s | 0.15 m |
| `crossing` | Dynamic prediction for one perpendicular robot crossing | 20 s | 0.20 m |
| `grid_intersection` | Four moving robots and four crossing points | 30 s | 0.25 m |
| `mirror_swap` | Dense 6v6 interaction, yielding, and convergence | 45 s | 0.30 m |

The crossing, grid, and mirror scenarios use the selected scheme for both
teams. The static scenario has no opponent strategy, so its opponent robots
remain fixed. `mirror_swap` includes the same deterministic 2 cm offset used by
the existing regression scenario to break exact symmetry.

## Pass/fail semantics

A run passes only when every moving robot reaches its target within the stated
tolerance before simulated timeout and no pair of robot centres comes closer
than `2 * ROBOT_RADIUS` (0.18 m). A repeated matrix cell passes only if every
repeat passes. This deliberately makes safety and task completion universal;
it does not encode algorithm-specific expectations or mark known failures as
acceptable.

The result reason is one of:

- `completed`: all moving robots reached their targets;
- `collision`: physical robot footprints overlapped;
- `sim_timeout`: the scenario exhausted its simulated-time budget;
- `wall_timeout`: the runner failed to make enough simulation progress;
- `exception`: setup or execution raised an exception.

Observed speed above the rsim 2 m/s limit plus a 0.15 m/s measurement tolerance
is reported as `speed_limit_exceeded`, but is diagnostic rather than a hard
failure. Observed acceleration is also diagnostic: it is reconstructed from
filtered simulator velocity and can include estimator transients, so it should
not be treated as an exact commanded-acceleration contract.

## Metrics

The JSON contains each repeat and an aggregate for every scheme/scenario cell.
The Markdown table focuses on:

- simulated completion time;
- total travelled path divided by total straight-line start-to-target distance;
- worst centre clearance after subtracting both robot radii (negative means overlap);
- collision-event count;
- largest final target error;
- fraction of robot-seconds spent below 0.05 m/s while still unreached;
- mean, p95, and maximum `MotionController.calculate()` latency;
- maximum observed speed and acceleration.

Controller latency includes the selected planner plus its translation/orientation
controller work. It excludes the first five calls per cell by default so Numba
compilation and one-time initialization do not dominate steady-state comparison;
change this with `--timing-warmup-calls`. Wall time still includes startup and
compilation.

## How to compare planners

First compare pass rate: a faster colliding or non-converging planner is not a
viable winner. Among cells that pass, read metrics together:

- lower completion time and path ratio indicate efficient progress;
- larger minimum clearance indicates more safety margin, but very large
  clearance paired with a long path can indicate excessive conservatism;
- a high stopped ratio exposes yielding, oscillation, or local-minimum stalls;
- controller p95 matters more than only the mean for a 60 Hz control loop;
- repeated runs distinguish stable behavior from a one-off success.

There is intentionally no weighted overall score. The trade-off between path
speed, clearance, and compute budget is a team decision, and one scalar would
hide why a planner won. Archive the JSON report with the git revision recorded
inside it, then compare matching scenario/scheme rows across revisions.

The benchmark covers external behavior shared by every algorithm. Keep narrow
unit tests for algorithm internals (for example bang-bang endpoint invariants or
FPP collision-kernel equivalence), while using this report as the common basis
for choosing between `fpp`, `dwa`, and `trajsample`.

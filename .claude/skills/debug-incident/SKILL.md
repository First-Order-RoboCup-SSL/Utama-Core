---
name: debug-incident
description: Take one bad moment in a match (a foul, a stall, a robot thrashing, driving into the ball at a restart, a strange path near an obstacle) from symptom to a fixed cause with a regression test. Use when tools/debug/find_bugs.py or a replay shows a specific incident, or the user points at a time in a match. Not for "why did this strategy lose" in general (that is the explain-result skill).
---

# Debug one incident

A foul or failure is a symptom. The cause may be the robot's tactic, the shared referee
override, the motion planner, the referee itself, or the check that flagged it: of the first
six found this way, two were the referee or the check. The full workflow, with the bugs found
so far, is `docs/STRATEGY_DEVELOPMENT.md`, "Debugging a foul or failure"; read it for anything
below that is unclear.

Inputs: a replay `REPLAY` (`replays/eval_<id>/round_<k>/<a>_vs_<b>.npz`, or any `.npz`) and a
time `S` in seconds. In a parallel worktree, run `tools/dev/setup_worktree.sh` first.

## 1. Find it

`pixi run python tools/debug/find_bugs.py ROUND_DIR --strategy NAME` → `find_bugs/index.md`, then
filter `find_bugs/findings.jsonl` on the kind you care about. Open the finding's picture,
then confirm with the numbers in its message. Only the friendly side (config_a) is checked.

## 2. Look at it

Pick the tool by the question. `tools/debug/match_timeline.py`, `tools/debug/restart_picture.py` and `tools/debug/plan_trace.py` take
`--findings` to write their flags as JSON.

| Question | Command |
|---|---|
| What happened in this window, in text? | `tools/debug/replay_trace.py REPLAY T0 T1` (`--events`: kicks, possession, referee changes, fouls) |
| Which tactic was each robot in, and what was it aiming at? | `tools/debug/repro_from_replay.py REPLAY --t S --summary IDS` (replays the state with tracing on, prints tactic, position, target every 0.25 s) |
| Swimlanes: referee, each robot's tactic, distance to ball | `tools/debug/match_timeline.py REPLAY T0 T1 --side friendly` |
| Where did the referee override send everyone at this restart? Legal? | `tools/debug/restart_picture.py REPLAY --t S` (`--all-restarts`, `--side enemy`) |
| What did the planner decide on this tick? | `tools/debug/plan_snapshot.py REPLAY --t S --robot ID --target X Y` (`--side enemy`) |
| How did the plan evolve over a few seconds? | `tools/debug/plan_trace.py REPLAY --t0 S --t1 S --robot ID [--target X Y]` |
| Run a fresh short match with my own traces | `tools/debug/debug_match.py --strategy A --opponent B --duration 60 --stop-on-goal`, with `ctx.match_log.trace(...)` calls added in the tactic |

All run as `pixi run python tools/<tool>.py ...`. Never conclude from a picture alone.

## 3. Find the cause in the code

From the trace: which tactic or override sent the robot there, with which target, waypoint
and obstacles. Then ask whether the rule was really broken (`docs/referee.md ("Design decisions")`)
and whether the check reads it right. Frame trap: rsim stores y negated; anything written into
the sim negates y and heading.

## 4. Fix it, with a unit test at the boundary

The fix is not done without a test that fails before and passes after, pinning the exact
boundary (a tolerance, a tie, a target off the field). Prefer a pure unit test over an rsim
fixture. Add a row to the bug table in `docs/STRATEGY_DEVELOPMENT.md` if it is a new kind.

## 5. Optionally keep a moment

Only when a field state alone decides the behaviour (restart and placement geometry under the
referee override); open-play fouls almost never reproduce from a captured state.

```
pixi run python tools/debug/moments.py capture REPLAY --t S --name NAME --check CHECK --note "..."
pixi run python tools/debug/moments.py play utama_core/tests/moments/NAME.json --seeds 0-4
```

It must fail on at least 4 of 5 seeds before the fix. If the fix is not in the same change,
set `"known_failure": "<cause>"` (a strict xfail). Moments pin shared-code bugs; they are not a
gate on strategies.

## Rules

- A shared-code fix (skills, tactics, planner, referee) changes every strategy: batch such
  fixes and evaluate once at the end, and ask before starting any `evaluate.py` run.
- Determinism: replaying the same fixture gives the same match. If a rerun differs, suspect
  module-level state in a tactic or skill not keyed by team and cleared at match start.
- Don't chase rsim-only dribble failures as kernel bugs.

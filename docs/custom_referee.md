# Custom Referee

`CustomReferee` (`utama_core/custom_referee/`) is an in-process referee: it steps
synchronously on `GameFrame`s and produces `RefereeData`, which `RefereeOverride`
(`engine/referee_override.py`) reads each tick via `game.referee`. No network, no AutoReferee
process, no simulator-specific code — identical in RSim, grSim and Real.

**Why not the TIGERs AutoReferee?** It is an asynchronous real-time Java/UDP process: unusable
in RSim and for faster-than-real-time RL, its thresholds are hardcoded (break on small test
fields), and strict rules ruin human-vs-robot exhibition play. Profiles solve the last two.

## Architecture

```
CustomReferee.step(game_frame, t)
  ├─ for rule in rules (priority order): rule.check(frame, geometry, command, placement_target)
  │     first *stopping* violation wins; a non-stopping foul (SSL §8.4.2) only feeds the
  │     foul counter and never suppresses a later stopping rule
  └─ GameStateMachine.step(t, violation)   # command, score, stage, next_command,
        → RefereeData(source_identifier="custom_referee")   designated_position
```

- 0.3s transition cooldown (`_TRANSITION_COOLDOWN`) stops one violation applying repeatedly.
- One-frame lag (the frame is from the previous step) — acceptable in every mode.
- `RefereeGeometry` is a frozen dataclass decoupled from `Field`; `StrategyRunner` overrides it
  from `full_field_dims` at startup (default `STANDARD_FIELD_DIMS`). YAML never sets geometry.
  In grSim/Real, `StrategyRunner` raises if the first vision geometry packet disagrees with
  `full_field_dims`.

## Rules

Built in `_build_active_rules` (`custom_referee.py`), in this priority order; defaults live in
`profiles/profile_loader.py`'s config dataclasses.

| Rule | Active during | Notes |
|---|---|---|
| `GoalRule` | live play | Scoring team from `my_team_is_right`/`my_team_is_yellow`; 1s cooldown; `designated_position=(0,0)`. |
| `OutOfBoundsRule` | live play | Free kick to the team that didn't touch last, placed 0.25m infield. Last touch: `rules/last_touch.py`'s colour-blind `infer_last_touch_team` (both teams' contact data; closest robot only when there is no prior attribution; unresolved rather than a default colour). |
| `BallSpeedRule` | live play | Ground speed > 6.5 m/s, edge-detected; same last-touch attribution. |
| `DoubleTouchRule` | `NORMAL_START` after a restart | Only the restart kicker (first toucher after arming) is barred; disarms when any other robot touches. Open-play release-and-reacquire dribbling is legal. Keeps `_prev_command` across `reset()` because `reset()` runs on the very transition it must observe. |
| `DefenseAreaRule` | live play | > `max_defenders` (1) in own area, or an attacker in ours. |
| `KeepOutRule` | `DIRECT_FREE_*`, `PREPARE_KICKOFF_*`, `PREPARE_PENALTY_*` | Non-kicking team within 0.5m for 30 consecutive frames. Excludes bare `STOP` so it can't overwrite `next_command` while robots clear. |
| `PushingRule`, `KeeperHeldBallRule`, `ExcessiveDribblingRule`, `RobotStopSpeedRule`, `CrashingRule`, `DefenseAreaStoppageRule`, `BallPlacementInterferenceRule` | see each rule file | SSL §8.3/8.4 audit (`434ab29`). Rationale per rule: `custom_referee_design_decisions.md`. |

## State machine and auto-advance

`GameStateMachine` starts in `HALT`. Violations move live play to `STOP` with a queued
`next_command`; then, when the profile's `auto_advance` flags allow (constants at the top of
`state_machine.py`):

| # | Transition | Trigger |
|---|---|---|
| 1 | `STOP` → queued restart | all robots ≥0.5m from ball (15s clear timeout) |
| 2 | `PREPARE_KICKOFF_*`/`PREPARE_PENALTY_*` → `NORMAL_START` | prepare timer + kicker in position, held 2s |
| 3 | `DIRECT_FREE_*` → `NORMAL_START` | kicker ≤0.3m from ball, defenders ≥0.5m, held 2s |
| 4 | `BALL_PLACEMENT_*` → next command | ball ≤0.15m from target, held 2s (10s placement timeout) |
| 5 | `NORMAL_START` → `FORCE_START` | `kickoff_timeout_seconds` elapsed and ball unmoved |

`force_start_after_goal` is a legacy path (STOP → FORCE_START after `stop_duration_seconds`).
Scripts resume play with `referee.set_command(RefereeCommand.NORMAL_START, timestamp=...)`.

## Profiles

`CustomReferee.from_profile_name("simulation" | "human" | "/path/to.yaml", n_robots_yellow=,
n_robots_blue=)`.

- **`simulation`** — every rule on, all auto-advances on. Simulator, AI-vs-AI, RL.
- **`human`** — every rule off, auto-advances off; an operator issues every command. Real-field
  testing and human-vs-robot exhibition play.

Give every rule an explicit `enabled:` — an omitted rule block silently falls back to enabled
with sim defaults (`load_profile()` warns listing them; this once ran 9 strict rules in a
relaxed profile). Copy `simulation.yaml`/`human.yaml` to start a new profile.

## StrategyRunner integration

Pass `StrategyRunner(..., referee=CustomReferee(...))`. Then the UDP `RefereeMessageReceiver` is
not started; `referee.step()` runs each tick and feeds `ref_buffer`
(→ `RefereeRefiner` → `game.referee` → `RefereeOverride`); and on the edge into `STOP` with a
`designated_position`, the ball is teleported there when a `sim_controller` exists (skipped in
Real). `StrategyRunner` only checks `isinstance(referee, CustomReferee)`, so subclasses work
unchanged. `CustomReferee.reset()` restores score/stage/command/timers for a new RL episode
(`BaseRule.reset_for_new_episode()` also clears `GoalRule`'s cooldown); call `seed_clock()`
again afterwards.

## Restart fuzzing (`RestartFuzzingReferee`)

`custom_referee/restart_fuzzer.py`: an opt-in subclass that injects legal restarts at
seeded-random sim times during live play (never on top of a queued restart), via the same
public `set_command` path an operator uses, so a stall it exposes is a real one. rsim is
deterministic, so this is how a round-robin explores new restart geometries. Kinds: ball
placement → `DIRECT_FREE_*` (position ≥0.25m infield, ≥0.2m from both defense areas),
`PREPARE_KICKOFF_*`, and `STOP` → `FORCE_START`. `referee.injections` records each one; with a
`MatchLog` attached, they are traced as `restart_fuzzer_injection`. Used by
`smoke_tournament.py --fuzz-restarts SEED`.

```python
referee = RestartFuzzingReferee.from_profile_name("simulation", seed=1, interval_s=(8.0, 20.0),
                                                  n_robots_yellow=6, n_robots_blue=6)
```

## Known gaps

- **Last-touch attribution needs contact data for both teams.** Enemy `has_ball` is filled
  by `RobotInfoRefiner` from sim contact physics (`data_processing/refiners/robot_info.py`).

Resolved (kept here so nobody re-reports them): the old friendly-first, ≤0.15m proximity
last-touch heuristic and its yellow default (replaced by `infer_last_touch_team`), double touch,
ball speed, full-episode reset,
`bt_nodes` → `debug_status` rename, auto-advance after goals/timeouts, keep-out during bare
`STOP`, blue-perspective goal tests, `StrategyRunner` integration tests
(`tests/strategy_runner/test_referee_rsim.py`, `test_ball_placement_rsim.py`),
`force_start_after_goal`.

## Running

```bash
pixi run pytest utama_core/tests/custom_referee/ --headless
pixi run python demo_custom_referee.py     # pygame, 6 scripted rule scenes; SPACE pause, R restart, ←/→ skip
pixi run python demo_referee_gui_rsim.py   # RSim + dashboard referee tab, see custom_referee_gui.md
```

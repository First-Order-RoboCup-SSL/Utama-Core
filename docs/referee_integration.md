# Referee Integration

How referee commands reach robots, and what each command requires. The original py_trees
integration is gone (BT removed in `087ee4b`, `960662c`, `48affd6`); the current mechanism is
`RefereeOverride` (`utama_core/engine/referee_override.py`, design: §13 of
`tactic_model_design_decisions.md`) driving the `*Step` classes in
`utama_core/custom_referee/actions.py`. For the in-process referee itself see
`custom_referee.md`.

## Data path

```
AutoReferee UDP 224.5.23.1:10003 → RefereeMessageReceiver ─┐   (Real/grSim with the official GC)
CustomReferee.step() ──────────────────────────────────────┴→ ref_buffer (deque maxlen=1)
  → StrategyRunner._run_step → RefereeRefiner.refine(game_frame, referee_data)
  → game.referee → RefereeOverride (restart/stop commands) or the Strategy (live play)
```

In WSL, multicast UDP needs `networkingMode=mirrored` in `.wslconfig`.

`RefereeData` carries `referee_command`, `stage`, `stage_time_left`, `blue_team`/`yellow_team`
(`TeamInfo`: score, cards, goalkeeper id, fouls, `can_place_ball`), `designated_position`,
`blue_team_on_positive_half`, `next_command`, `current_action_time_remaining`,
`source_identifier`, plus `game_events`, `match_type`, `status_message` — the last three are
excluded from `__eq__` so they don't trigger spurious re-records in `RefereeRefiner`.

## Required behaviour per command ([SSL rulebook](https://robocup-ssl.github.io/ssl-rules/sslrules.html))

| Command | Our robots must |
|---|---|
| `HALT` | Zero velocity (2s braking grace). Highest priority. |
| `STOP` | ≤1.5 m/s, ≥0.5m from ball, ≥0.2m from opponent defense area, no ball contact. |
| `TIMEOUT_*` | Nothing forced; handled as `STOP`. |
| `PREPARE_KICKOFF` ours / theirs | All but the kicker in own half outside the 0.5m centre circle; kicker approaches, no touch. Theirs: everyone in own half outside the circle. |
| `PREPARE_PENALTY` ours / theirs | Kicker to the mark, no touch; others ≥0.4m behind the mark line. Theirs: keeper on own goal line, others ≥0.4m behind the mark. |
| `DIRECT_FREE` ours / theirs | Kicker approaches and may shoot after the ball moves ≥0.05m. Theirs: all ≥0.5m from ball, full speed allowed. |
| `BALL_PLACEMENT` ours / theirs | One robot places the ball at `designated_position` (±0.15m); others ≥0.5m. If `can_place_ball` is False, behave as `STOP`. Theirs: ≥0.5m from ball and target. |
| `NORMAL_START`, `FORCE_START` | Live play — no override; the Strategy runs. |

Yellow/blue variants are resolved to ours/theirs at tick time against
`game.my_team_is_yellow`, so nothing depends on team colour at construction time.

## Pre-rewrite baseline (BT priority tree)

Code comments cite this: the old tree put a referee `Selector` as the first child of every
strategy's root, one `Sequence` per command in priority order — `HALT`, `STOP`,
`TIMEOUT_YELLOW|BLUE` (dispatched to `StopStep`, same as `STOP`), ball placement, kickoff,
penalty, direct free — each `CheckRefereeCommand(...)` → `*Step`, falling through to the
strategy tree on `NORMAL_START`/`FORCE_START`. `RefereeOverride` keeps that command→Step
mapping as a plain class.

## Open

- **Ball placement before a free kick.** The rulebook sequence after ball-out is
  `STOP → BALL_PLACEMENT_* → DIRECT_FREE_* → NORMAL_START`; `OutOfBoundsRule` still queues
  `DIRECT_FREE_*` directly (in sim the ball is teleported instead). Depends on reliable
  `BallPlacementOursStep` carrying.
- **Placement carry mechanics.** Single-robot dribble tends to push rather than carry; options
  are a dedicated slow "get-behind-ball" approach or the two-robot technique most SSL teams use.
- **Pre-positioning from `next_command`** during `STOP` — an optimisation, not a compliance need.
- **Operator GUI "suggested next step"** based on the current command, so an operator decides
  only *when* to advance, not *what*.

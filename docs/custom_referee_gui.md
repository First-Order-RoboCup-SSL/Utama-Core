# Custom Referee Operator GUI

The browser operator panel for `CustomReferee` is the dashboard's referee tab
(`utama_core/dashboard/views/referee.py`). Attach it with:

```python
server = attach_dashboard()               # utama_core.dashboard
referee_view.attach(server, referee, profile)   # utama_core.dashboard.views.referee
```

`examples/demo_referee_gui_rsim.py` does this with RSim, the `human` profile and
`tiki_taka_plus` (constants `PROFILE`, `N_ROBOTS`, `MY_TEAM_IS_YELLOW`, `MY_TEAM_IS_RIGHT` at
the top of the file):

```bash
pixi run python examples/demo_referee_gui_rsim.py   # then open http://localhost:8080
```

The page shows the score, current/next command, stage and time left, `designated_position`, the
field, the active profile and an event log; a button issues each command.

## Field check (camera runs)

With vision from a camera (real robots or grSim) the Live tab also shows the field check
(`utama_core/run/field_check.py`), one row per question, red / amber / green:

| Row | Fails (red) when | Warns (amber) when |
|---|---|---|
| Camera | no vision packet for 0.5 s | |
| Field size | ssl-vision's geometry differs from `full_field_dims` | no geometry packet yet |
| `<colour> <id>` (each robot we drive) | never seen, lost by vision 0.5 s, or more than 0.3 m outside `field_bounds` | |
| `<colour> <id> radio` (real only) | it answered on the radio, then stopped | it has not answered yet (the firmware may send no feedback) |
| `<colour> <id>` (any other) | | seen inside `field_bounds` but not configured: our planners cannot see it, often a wrong ID |
| Ball | | not seen, or outside `field_bounds` |

Before it gets that far, the runner keeps our robots inside `field_bounds` on camera runs
(`utama_core/run/field_fence.py`): each robot's speed towards an edge is capped to what it can
stop from before the line, whatever tactic sent the command, so a robot chasing a ball out of the
camera's area stops at the line. The planners' walls at `field_bounds` are only obstacles to plan
around, and did not stop it (0.5–0.9 m out in grSim before the fence).

Only the red and amber rows are listed; the green ones fold into an "N ok" line that opens on
click. On real robots any red row forces `HALT`, whatever the command and whether or not the page is
open; the red banner says why. Pressing Force Start while a row is red halts again at once, so
the check is also the pre-start gate. The field is drawn as ssl-vision sees it (+x to the right),
with everything outside `field_bounds` dimmed, so a half-field test shows which half the runner
expects. Space halts; nothing on the keyboard starts the robots. The Kickoff and Free Kick
buttons are hidden when the profile has no rule that sets up a restart (goal detection only).

## Strategy panel

A script can let the page change what each side plays, or restart it, without restarting the
run (`utama_core/dashboard/views/strategy.py`):

```python
strategy_view.attach(server, runner,
    choices={"my": {"lead": make_lead, "press": make_press}},  # name -> builds a fresh AbstractStrategy
    running={"my": "lead"})
```

The Live tab then lists the choices per side with a Run button; Run on the one running restarts
it. `StrategyRunner.replace_strategy` takes the new strategy at the next tick and only while
halted, so the page refuses otherwise. `examples/shooting_drill.py` offers its attacker
and defender tactics this way.

## Commands and what robots do

| Button | Command | Use | Robots |
|---|---|---|---|
| Halt | `HALT` | Emergency / unsafe | Zero velocity |
| Stop | `STOP` | Between incidents, pre-match | ≤1.5 m/s, stay ≥0.5m from the ball |
| Normal Start | `NORMAL_START` | Restart formation is in position | Live play |
| Force Start | `FORCE_START` | Stalled play | Live play from the ball's current position (a tactic barrier reset, see `tactic_model_design_decisions.md` §4) |
| Kickoff Y/B | `PREPARE_KICKOFF_*` | Half start, after a goal | Kicker to centre, others to own half |
| Free Kick Y/B | `DIRECT_FREE_*` | Foul by the other team | Kicker to ball, opponents ≥0.5m |
| Penalty Y/B *(advanced)* | `PREPARE_PENALTY_*` | Manual override | Kicker at mark, others behind the line |
| Ball Placement Y/B *(advanced)* | `BALL_PLACEMENT_*` | Manual placement | One robot carries ball to `designated_position` |

Typical sequences: `Halt → Stop → Kickoff Yellow → Normal Start` to start a half;
`Stop → Free Kick → Normal Start` for a manual free kick. With `human`, nothing auto-advances —
after a goal the referee stays in `STOP` until the operator acts.

For a physical field, keep every `auto_advance` flag false so robots never start moving until
the operator presses a button. Profile keys, auto-advance triggers and making a custom profile:
`custom_referee.md`.

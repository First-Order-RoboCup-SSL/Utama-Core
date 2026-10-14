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

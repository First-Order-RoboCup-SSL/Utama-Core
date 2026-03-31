# Custom Referee Web GUI

`demo_referee_gui_rsim.py` (project root) is a standalone browser-based operator panel for the
`CustomReferee`. It lets you issue referee commands, watch live scores and robot positions,
and inspect the active profile — all from a browser tab. It requires no npm, no build step,
and no dependencies beyond the project's existing Python environment.

---

## Quick start

```bash
pixi run python demo_referee_gui_rsim.py
```

Then open **http://localhost:8080** in any browser.

The referee starts in **HALT**. A typical pre-match sequence is:

```
Halt → Stop → Kickoff Yellow → Normal Start
```

---


## UI areas

The page has five areas.

### 1. Scoreboard

Live yellow / blue scores, updated in real time.

### 2. Status block

Four read-only fields that reflect the current referee state:

| Field | What it shows |
|---|---|
| **Command** | Current referee command, colour-coded: red = HALT, orange = STOP, green = NORMAL/FORCE START, yellow/blue = team-specific commands |
| **Next** | `next_command` — the command that will follow the current stoppage, if known |
| **Stage** | Game stage (e.g. `NORMAL FIRST HALF`) and time remaining (mm:ss) |
| **Designated** | Ball placement target in metres — hidden when `designated_position` is null |

### 3. Command buttons

Clicking a button immediately issues that command to the `CustomReferee`.

The **Advanced** toggle reveals penalty and ball-placement buttons that are rarely needed
manually (they are usually auto-detected). Hover over any button to see a tooltip describing
when to use it.

| Row | Buttons |
|---|---|
| Flow control | Halt · Stop · Normal Start · Force Start |
| Kickoffs | Kickoff Yellow · Kickoff Blue |
| Free kicks | Free Kick Yellow · Free Kick Blue |
| Penalties *(adv)* | Penalty Yellow · Penalty Blue |
| Ball placement *(adv)* | Ball Placement Yellow · Ball Placement Blue |

#### Button reference

| Button | Command issued | When to press | What robots do |
|---|---|---|---|
| **Halt** | `HALT` | Emergency stop; any unsafe situation | Immediately zero velocity — no movement |
| **Stop** | `STOP` | Pause between incidents; pre-match | Slow to ≤1.5 m/s, stay ≥0.5 m from ball |
| **Normal Start** | `NORMAL_START` | After kickoff / free kick robots are in position | Game live — strategy tree takes over |
| **Force Start** | `FORCE_START` | Double-touch infringement; stalled play | Game live — ball at current position, no placement |
| **Kickoff Yellow** | `PREPARE_KICKOFF_YELLOW` | Half-start or after Blue scores | Yellow kicker approaches centre; others to own half |
| **Kickoff Blue** | `PREPARE_KICKOFF_BLUE` | After Yellow scores | Blue kicker approaches centre; others to own half |
| **Free Kick Yellow** | `DIRECT_FREE_YELLOW` | Foul by Blue | Yellow kicker near ball; Blue ≥0.5 m away |
| **Free Kick Blue** | `DIRECT_FREE_BLUE` | Foul by Yellow | Blue kicker near ball; Yellow ≥0.5 m away |
| **Penalty Yellow** *(adv)* | `PREPARE_PENALTY_YELLOW` | Usually auto-detected; manual override only | Yellow kicker at penalty mark; others behind line |
| **Penalty Blue** *(adv)* | `PREPARE_PENALTY_BLUE` | Usually auto-detected; manual override only | Blue kicker at penalty mark; others behind line |
| **Ball Placement Yellow** *(adv)* | `BALL_PLACEMENT_YELLOW` | Manual placement command | Yellow robot moves ball to `designated_position` |
| **Ball Placement Blue** *(adv)* | `BALL_PLACEMENT_BLUE` | Manual placement command | Blue robot moves ball to `designated_position` |

#### Auto-detected vs manual

| Command category | Detection | Notes |
|---|---|---|
| Goal → kickoff | Auto (`GoalRule`) | Operator sets kickoff team before half starts |
| Out-of-bounds → free kick | Auto (`OutOfBoundsRule`) | `free_kick_assigner` in profile controls which team |
| Defense area → penalty | Auto (`DefenseAreaRule`, if enabled) | Penalty buttons are in **Advanced** row; hidden by default |
| Ball placement | Auto (`OutOfBoundsRule`, if enabled) | Manual override via Advanced row if auto fails |

#### Typical sequences

Standard half start:
```
Halt → Stop → Kickoff Yellow → Normal Start
```

After goal (human profile — auto-restart):
```
(Goal auto-detected) → Stop → (auto Force Start after stop_duration_seconds)
```

Manual free kick:
```
Halt → Stop → Free Kick Yellow → Normal Start
```

### 4. Field canvas

A top-down view of the field, updated at ~30 Hz via SSE.

| Element | Colour | Notes |
|---|---|---|
| Field background | Green | Scales automatically to profile geometry |
| Lines / circles | White | Boundary, centre line, centre circle, defence areas |
| Left goal | Yellow (translucent) | Yellow team's goal (negative x side) |
| Right goal | Blue (translucent) | Blue team's goal (positive x side) |
| Friendly robots | Yellow filled circle | ID label above; orientation line shows heading |
| Enemy robots | Blue filled circle | ID label above; orientation line shows heading |
| Ball | Orange filled circle | Minimum 4 px radius |
| Designated × | White cross | Ball placement target; hidden when `designated_position` is null |

### 5. Profile config panel

A read-only display of the active profile loaded at startup, confirming what geometry and
rules are in effect.

| Section | Fields shown |
|---|---|
| Field geometry | half length, half width, half goal width, defense length/width (half), centre circle radius — all in metres |
| Game settings | half duration (minutes), kickoff team, force-start-after-goal flag, stop duration (seconds), auto-advance flags |
| Goal detection | enabled (ON/OFF), cooldown seconds |
| Out of bounds | enabled (ON/OFF), free-kick assigner method |
| Defense area | enabled (ON/OFF), max defenders, attacker infringement flag |
| Keep-out zone | enabled (ON/OFF), radius (metres), violation persistence (frames) |

Boolean fields are shown as green **ON** / red **OFF** pills. The panel title and browser tab
title both show the profile name.

A small green dot at the bottom of the page indicates the SSE connection is live. If it turns
grey, the browser will reconnect automatically.

### Event log

The **Event Log** panel shows the 20 most recent events, newest first.

| Entry type | Colour | Trigger |
|---|---|---|
| Command change | Green | `d.command` differs from the previous message |
| Score change | Yellow | `d.yellow_score` or `d.blue_score` differs |
| Status message | Grey (muted) | `d.status_message` is non-empty and differs from previous value |

---

## Profiles

Built-in profiles live in `utama_core/custom_referee/profiles/`:

| Profile | Rules active | Auto-advances |
|---|---|---|
| `human` | Goal detection only | All off — operator advances commands manually |
| `simulation` | All four rules (goal, out-of-bounds, defense area, keep-out) | Mostly on — simulator workflow auto-progresses when criteria are met |

Use `human` for physical environments where a referee operator must control transitions for
safety and testing. Use `simulation` for simulator, strategy testing, and RL workflows.

---

## Changing field dimensions

All six `geometry` fields are in **metres** and are fully dynamic — every rule reads geometry
on each tick, so changing a value in the profile instantly changes rule behaviour. The field
canvas auto-scales to match.

| Field | Controls | Rules affected |
|---|---|---|
| `half_length` | Distance from centre to goal line | GoalRule, OutOfBoundsRule, DefenseAreaRule |
| `half_width` | Distance from centre to touch line | OutOfBoundsRule |
| `half_goal_width` | Half-width of each goal mouth | GoalRule |
| `half_defense_length` | Depth of defence area | DefenseAreaRule |
| `half_defense_width` | Half-width of defence area | DefenseAreaRule |
| `center_circle_radius` | Centre circle drawn on canvas; reserved for future keep-out rule | — |

---

## Configuring rules

Every tuneable parameter and its effect:

| Key | Type | Effect |
|---|---|---|
| `goal_detection.enabled` | bool | Enables/disables goal scoring |
| `goal_detection.cooldown_seconds` | float | Prevents double-counting within this window |
| `out_of_bounds.enabled` | bool | Enables free-kick on ball exit |
| `out_of_bounds.free_kick_assigner` | `"last_touch"` | Team awarded the free kick |
| `defense_area.enabled` | bool | Enables defender/attacker penalty |
| `defense_area.max_defenders` | int | Max robots allowed inside own defence area |
| `defense_area.attacker_infringement` | bool | Also penalise attackers who enter opponent area |
| `keep_out.enabled` | bool | Enables exclusion zone around ball during stoppages |
| `keep_out.radius_meters` | float | Exclusion radius in metres |
| `keep_out.violation_persistence_frames` | int | Frames robot must stay clear before violation clears |
| `game.half_duration_seconds` | float | Length of each half |
| `game.kickoff_team` | `"yellow"` or `"blue"` | Which team kicks off at the start |
| `game.force_start_after_goal` | bool | Legacy human fast-path; superseded by `auto_advance` flags |
| `game.stop_duration_seconds` | float | STOP hold time before auto FORCE START (legacy human path) |
| `game.auto_advance.stop_to_prepare_kickoff` | bool | Auto STOP → PREPARE_KICKOFF when all robots clear |
| `game.auto_advance.prepare_kickoff_to_normal` | bool | Auto PREPARE_KICKOFF → NORMAL_START when kicker in position (2 s delay) |
| `game.auto_advance.direct_free_to_normal` | bool | Auto DIRECT_FREE → NORMAL_START when kicker ready (2 s delay) |
| `game.auto_advance.ball_placement_to_next` | bool | Auto BALL_PLACEMENT → next command when ball at target (2 s delay) |
| `game.auto_advance.normal_start_to_force` | bool | Auto NORMAL_START → FORCE_START after kickoff timeout if ball hasn't moved |

---

## Auto-advance configuration

The state machine can automatically advance through referee states when certain conditions are
met. Each transition is independently configurable via the `auto_advance` block in the profile.

### The five auto-advances

| # | Transition | Trigger | Delay |
|---|---|---|---|
| 1 | `STOP` → `PREPARE_KICKOFF_*` | All robots ≥ 0.5 m from ball | None |
| 2 | `PREPARE_KICKOFF_*` → `NORMAL_START` | Timer elapsed + kicker inside centre circle | **2 s** |
| 3 | `DIRECT_FREE_*` → `NORMAL_START` | Kicker ≤ 0.3 m from ball + defenders ≥ 0.5 m away | **2 s** |
| 4 | `BALL_PLACEMENT_*` → next command | Ball ≤ 0.15 m from placement target | **2 s** |
| 5 | `NORMAL_START` → `FORCE_START` | Kickoff timeout elapsed + ball hasn't moved | None (uses `kickoff_timeout_seconds`) |

Advances 2, 3, and 4 require the readiness condition to be **sustained for 2 seconds** before
firing. If the condition drops out during that window (e.g. kicker steps back), the countdown
resets. This gives robots time to settle before play begins.

### Choosing settings by environment

**Simulation (default):** All advances on. The state machine drives itself — no operator needed.

```yaml
game:
  auto_advance:
    stop_to_prepare_kickoff: true
    prepare_kickoff_to_normal: true
    direct_free_to_normal: true
    ball_placement_to_next: true
    normal_start_to_force: true
```

**Physical environment (safety):** All advances off. A human operator must explicitly issue
every command. Robots will not start moving until the operator presses a button — ensuring
nobody is on the field when play begins.

```yaml
game:
  auto_advance:
    stop_to_prepare_kickoff: false
    prepare_kickoff_to_normal: false
    direct_free_to_normal: false
    ball_placement_to_next: false
    normal_start_to_force: false
```

---

## Creating a custom profile

1. Copy an existing profile as a starting point:
   ```bash
   cp utama_core/custom_referee/profiles/human.yaml my_field.yaml
   ```

2. Edit `my_field.yaml` to match your field and rule requirements.

3. Run the GUI with your profile:
   ```bash
   pixi run python demo_referee_gui_rsim.py --profile my_field.yaml
   ```

### Annotated YAML template

```yaml
profile_name: "my_profile"

geometry:
  half_length: 4.5          # metres from centre to goal line  (full length = 9.0 m)
  half_width: 3.0           # metres from centre to touch line (full width  = 6.0 m)
  half_goal_width: 0.5      # metres from centre of goal mouth to post
  half_defense_length: 0.5  # depth of defence area from goal line
  half_defense_width: 1.0   # half-width of defence area
  center_circle_radius: 0.5 # used for canvas only (future: keep-out at kickoff)

rules:
  goal_detection:
    enabled: true
    cooldown_seconds: 1.0   # ignore further goals for this many seconds after one is scored

  out_of_bounds:
    enabled: true
    free_kick_assigner: "last_touch"  # only option currently

  defense_area:
    enabled: true
    max_defenders: 1                  # robots allowed inside own defence area
    attacker_infringement: true       # penalise attackers entering opponent defence area

  keep_out:
    enabled: true
    radius_meters: 0.5                # exclusion radius around ball during stoppages
    violation_persistence_frames: 30  # consecutive frames of encroachment before penalty

game:
  half_duration_seconds: 300.0
  kickoff_team: "yellow"
  force_start_after_goal: false
  stop_duration_seconds: 3.0
  auto_advance:
    stop_to_prepare_kickoff: true   # set false for physical/operator-driven environments
    prepare_kickoff_to_normal: true
    direct_free_to_normal: true
    ball_placement_to_next: true
    normal_start_to_force: true
```

### Worked example: small practice field (4 m × 2.67 m)

```yaml
profile_name: "practice_room"

geometry:
  half_length: 2.0
  half_width: 1.335
  half_goal_width: 0.35
  half_defense_length: 0.4
  half_defense_width: 0.7
  center_circle_radius: 0.35

rules:
  goal_detection:
    enabled: true
    cooldown_seconds: 1.0
  out_of_bounds:
    enabled: true
    free_kick_assigner: "last_touch"
  defense_area:
    enabled: false
    max_defenders: 1
    attacker_infringement: false
  keep_out:
    enabled: false
    radius_meters: 0.3
    violation_persistence_frames: 30

game:
  half_duration_seconds: 120.0
  kickoff_team: "yellow"
  force_start_after_goal: true
  stop_duration_seconds: 2.0
  auto_advance:
    stop_to_prepare_kickoff: true
    prepare_kickoff_to_normal: true
    direct_free_to_normal: true
    ball_placement_to_next: true
    normal_start_to_force: true
```

The canvas will render a noticeably smaller field, and the profile panel will confirm the
updated geometry values.

---

## Running with a live game loop

`demo_referee_gui_rsim.py` on its own runs in standalone mode — the canvas shows a static empty field.
To see live robots:

```bash
pixi run python utama_core/tests/referee/demo_referee_gui_rsim.py
# RSim window opens; open http://localhost:8080 in a browser
```

This script starts a `CustomReferee` (default: `simulation` profile) with `enable_gui=True`,
connects it to RSim, and drives `referee.step()` on every tick. Edit the `PROFILE`,
`N_ROBOTS`, `MY_TEAM_IS_YELLOW`, and `MY_TEAM_IS_RIGHT` constants at the top of the file to
configure it.

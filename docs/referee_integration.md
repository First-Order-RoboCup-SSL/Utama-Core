# Referee Integration Design

This document captures the design decisions for integrating the SSL Game Controller referee
into Utama's behaviour tree architecture.

---

## 1. Referee Data

### Fields in `RefereeData` 

| Field | Type | Purpose |
|---|---|---|
| `referee_command` | `RefereeCommand` | Primary driver of the hardcoded tree |
| `stage` | `Stage` | Game phase (first half, penalty shootout, etc.) |
| `stage_time_left` | `float` | Seconds remaining in current stage |
| `blue_team` / `yellow_team` | `TeamInfo` | Scores, cards, goalkeeper ID, foul counts |
| `designated_position` | `Optional[Tuple[float, float]]` | Ball placement target (x, y) in metres |
| `blue_team_on_positive_half` | `Optional[bool]` | Which side blue team defends |
| `next_command` | `Optional[RefereeCommand]` | Command after current stoppage — use to pre-position |
| `current_action_time_remaining` | `Optional[int]` | Microseconds remaining for ball placement / free kick |
| `source_identifier` | `Optional[str]` | Which autoreferee sent this packet |

### Fields recently added

| Field | Protobuf source | Why useful |
|---|---|---|
| `game_events` | `repeated GameEvent game_events = 16` | Why the command was issued (foul type, ball-out side, etc.); useful for logging and future decision-making |
| `match_type` | `optional MatchType match_type = 19` | FRIENDLY / GROUP_PHASE / ELIMINATION_PHASE; may affect strategy aggression |
| `status_message` | `optional string status_message = 20` | Human-readable reason shown by referee UI; shown in live FPS display |

`game_events`, `match_type`, and `status_message` are intentionally **excluded from `__eq__`**
so they do not trigger spurious re-records in `RefereeRefiner`.

`game_event_proposals` (field 17) are multi-referee vote accumulations and are not needed.

---

## 2. Game States and Required Robot Behaviour

Rules sourced from the [SSL Rulebook](https://robocup-ssl.github.io/ssl-rules/sslrules.html).

| Command | Our robots must... | Key constraint |
|---|---|---|
| **HALT** | Immediately zero velocity. No movement. | 2-second grace period to brake. Highest priority. |
| **STOP** | Slow to ≤ 1.5 m/s, stay ≥ 0.5 m from ball. No ball contact. | Also ≥ 0.2 m from opponent defence area. |
| **TIMEOUT_YELLOW / BLUE** | Idle; effectively STOP behaviour. | Not our timeout: nothing forced, but safe to stop. |
| **PREPARE_KICKOFF (ours)** | All except kicker go to own half, outside centre circle. Kicker approaches ball at centre. Do not touch ball yet. | Centre circle radius = 0.5 m. |
| **PREPARE_KICKOFF (theirs)** | All robots to own half, outside centre circle. | Same zone constraint. |
| **NORMAL_START** (after kickoff / free kick) | Game live — pass to strategy tree. | Ball is now in play. |
| **FORCE_START** | Game live — pass to strategy tree. | Ball at current position; no placement needed. |
| **PREPARE_PENALTY (ours)** | Kicker: approach penalty mark, do not touch. Our other robots: ≥ 0.4 m behind penalty mark line. | Penalty mark: halfway between centre line and attacked goal in Utama's field model. |
| **PREPARE_PENALTY (theirs)** | Our goalkeeper: touch own goal line. All other our robots: ≥ 0.4 m behind the penalty mark (on our side). | Goalkeeper ID from `referee.{our_team}.goalkeeper`. |
| **DIRECT_FREE (ours)** | One robot (kicker) approaches ball. Others position freely. After NORMAL_START the kicker may shoot directly. | Ball must move ≥ 0.05 m to be in play. |
| **DIRECT_FREE (theirs)** | All our robots ≥ 0.5 m from ball. Full speed allowed (unlike STOP). | Same distance as STOP but no speed cap. |
| **BALL_PLACEMENT (ours)** | One robot moves the ball to `designated_position`. Other robots clear ≥ 0.5 m from ball. | If `can_place_ball` is False we cannot place — skip to STOP-like behaviour. |
| **BALL_PLACEMENT (theirs)** | All our robots stay ≥ 0.5 m from ball and from the `designated_position`. | Do not interfere with their placement robot. |

### "Ours vs. theirs" resolution

Each bilateral command (KICKOFF / PENALTY / DIRECT_FREE / BALL_PLACEMENT) comes in YELLOW and BLUE
variants. Resolved at **tick-time** inside each dispatcher node:

```python
# In _BallPlacementDispatch.update():
if self._is_yellow_command == self.blackboard.game.my_team_is_yellow:
    return self._ours.update()
else:
    return self._theirs.update()
```

This avoids any construction-time dependency on team colour.

### Using `next_command` for pre-positioning

During STOP, `next_command` reveals what restart is coming before it happens.
Example: if `next_command == DIRECT_FREE_BLUE` and we are blue, robots can already drift
toward their free-kick positions during the STOP phase, gaining a timing advantage.
This is an optimisation and does not affect compliance.

---

## 3. Architecture — Option B: Referee Priority Child (implemented)

The referee override layer sits as the first (highest-priority) child of a root `Selector`
inside every `AbstractStrategy`. This makes it:

- Visible in tree renders (DOT/PNG/SVG via `render()`).
- Automatically applied to every strategy — no per-strategy changes needed.
- Decoupled from `StrategyRunner` — all logic lives in the tree.

### Tree structure

```
Root [Selector, memory=False]
├── RefereeOverride [Selector, memory=False]   ← injected automatically by AbstractStrategy
│   ├── Halt [Sequence]
│   │   ├── CheckRefereeCommand(HALT)
│   │   └── HaltStep
│   ├── Stop [Sequence]
│   │   ├── CheckRefereeCommand(STOP)
│   │   └── StopStep
│   ├── Timeout [Sequence]
│   │   ├── CheckRefereeCommand(TIMEOUT_YELLOW | TIMEOUT_BLUE)
│   │   └── StopStep
│   ├── BallPlacementYellow [Sequence]
│   │   ├── CheckRefereeCommand(BALL_PLACEMENT_YELLOW)
│   │   └── _BallPlacementDispatch(is_yellow_command=True)
│   ├── BallPlacementBlue [Sequence]
│   │   ├── CheckRefereeCommand(BALL_PLACEMENT_BLUE)
│   │   └── _BallPlacementDispatch(is_yellow_command=False)
│   ├── KickoffYellow [Sequence]
│   ├── KickoffBlue [Sequence]
│   ├── PenaltyYellow [Sequence]
│   ├── PenaltyBlue [Sequence]
│   ├── DirectFreeYellow [Sequence]
│   └── DirectFreeBlue [Sequence]
│   [Each Sequence returns FAILURE if its condition doesn't match → Selector continues]
└── CoachRoot          ← returned by create_behaviour_tree() — unchanged
```

**Priority order**: HALT first (highest). NORMAL_START and FORCE_START have no override node —
the Selector falls through to the strategy tree ("game is live").

### Change to `AbstractStrategy`

`__init__` wraps the user's tree. Uses a lazy import to break the circular dependency:

```python
def __init__(self):
    # Lazy import to break circular dependency:
    # abstract_strategy → referee.tree → referee.conditions → abstract_behaviour
    #                                                        → strategy.common.__init__ → abstract_strategy
    from utama_core.strategy.referee.tree import build_referee_override_tree

    strategy_subtree = self.create_behaviour_tree()
    root = py_trees.composites.Selector(name="Root", memory=False)
    root.add_children([build_referee_override_tree(), strategy_subtree])
    self.behaviour_tree = py_trees.trees.BehaviourTree(root)
```

---

## 4. Folder Structure

```
utama_core/strategy/referee/
├── __init__.py        # exports CheckRefereeCommand, build_referee_override_tree
├── conditions.py      # CheckRefereeCommand — returns SUCCESS if command matches, else FAILURE
├── actions.py         # HaltStep, StopStep, BallPlacement{Ours,Theirs}Step,
│                      # PrepareKickoff{Ours,Theirs}Step, PreparePenalty{Ours,Theirs}Step,
│                      # DirectFree{Ours,Theirs}Step
└── tree.py            # build_referee_override_tree() + _*Dispatch routing nodes

utama_core/tests/referee/
├── __init__.py
├── referee_sim.py          # Visual RSim simulation cycling through all 11 commands
└── wandering_strategy.py   # Base strategy for simulation — robots patrol waypoints
```

---

## 5. Bugs Fixed During Integration

### Bug 1 — `RefereeRefiner.refine` called wrong methods on `GameFrame`

`refine(game, data)` was calling `game.current_frame` and `game.update_frame()` on a
`GameFrame` object (not a `Game`). This was a pre-existing bug that never triggered because
`RefereeMessageReceiver` was commented out.

**Fix**: Rewrote to use `dataclasses.replace(game_frame, referee=data)` directly.

### Bug 2 — `CurrentGameFrame` did not copy `referee` field

`CurrentGameFrame.__init__` was missing:
```python
object.__setattr__(self, "referee", game.referee)
```

This caused `AttributeError: 'CurrentGameFrame' object has no attribute 'referee'`.

### Bug 3 — Dispatcher nodes used `game.current_frame` instead of `game.my_team_is_yellow`

All four dispatcher nodes (`_BallPlacementDispatch`, `_KickoffDispatch`, `_PenaltyDispatch`,
`_DirectFreeDispatch`) used `self.blackboard.game.current_frame.my_team_is_yellow` — but
`Game` has no `current_frame` attribute (it has `current`). Fixed to use
`self.blackboard.game.my_team_is_yellow` directly via the `Game.my_team_is_yellow` property.

---

## 6. Data Flow

### Real mode (AutoReferee → WSL → robot)

```
AutoReferee (224.5.23.1:10003)
  → RefereeMessageReceiver (UDP multicast thread)
  → ref_buffer (deque maxlen=1)
  → strategy_runner._run_step reads ref_buffer
  → RefereeRefiner.refine(game_frame, referee_data)
  → game_frame.referee = RefereeData
  → game.referee (via Game.referee property)
  → CheckRefereeCommand reads game.referee.referee_command
```

Requires WSL `networkingMode=mirrored` in `.wslconfig` for multicast UDP to reach WSL.

### RSim mode (simulation)

```
ref_buffer.append(RefereeData)  ← pushed by CustomReferee.step() each tick (or externally, e.g. referee_sim.py)
  → strategy_runner._run_step reads ref_buffer
  → same pipeline as above
```

### RSim / grSim with CustomReferee (ball teleport)

When `custom_referee` is set and a simulator is active, `StrategyRunner` additionally
teleports the ball on the STOP transition edge if `designated_position` is set:

```
CustomReferee.step(game_frame, t)
  → RefereeData.designated_position = (0.0, 0.0)   (after a goal)
  → _prev_custom_ref_command != STOP, new command == STOP
  → sim_controller.teleport_ball(0.0, 0.0)          (RSim or grSim only)
  → _prev_custom_ref_command = STOP
```

This fires **once** on the transition edge and is silently skipped in Real mode.

---

## 7. Visualisation Simulation

Two scripts in `utama_core/tests/referee/` provide visual ways to verify referee behaviour in RSim.

### Automated command cycling (`referee_sim.py`)

```bash
pixi run python -m utama_core.tests.referee.referee_sim
```

- Uses `WanderingStrategy` as the base (robots continuously patrol waypoints)
- Cycles through all 11 referee commands every 5 seconds
- The `RefereeOverride` tree intercepts each command and you can watch robots reposition
- Command sequence starts with `NORMAL_START` so robots are visibly moving before the first override

To change timing, edit `SECS_PER_COMMAND` in `referee_sim.py`.

### Interactive GUI demo (`demo_referee_gui_rsim.py`)

```bash
pixi run python utama_core/tests/referee/demo_referee_gui_rsim.py
# RSim window opens; open http://localhost:8080 in a browser
```

- Starts a `CustomReferee` (default: `simulation` profile) with `enable_gui=True`
- `StrategyRunner` drives `referee.step()` on every tick — GUI receives live robot/ball positions
- Uses `WanderingStrategy` so robots visibly move and you can watch the `RefereeOverride` tree interrupt them
- Operator issues commands from the browser; the canvas shows robots repositioning in real time
- Edit `PROFILE`, `N_ROBOTS`, `MY_TEAM_IS_YELLOW`, `MY_TEAM_IS_RIGHT` constants at the top of the file to configure

---

## 8. Custom Referee Web GUI

The browser-based operator panel is started by passing `enable_gui=True` to `CustomReferee` and then running one of the demo scripts. There is no standalone `referee_gui.py` — the GUI is embedded inside `CustomReferee` itself.

### Starting the GUI

Use the provided demo script:

```bash
pixi run python demo_referee_gui_rsim.py
# RSim window opens; open http://localhost:8080 in a browser
```

The demo creates `CustomReferee(enable_gui=True, ...)` and passes it to `StrategyRunner` via the `referee=` argument. To change the profile, port, or robot counts, edit the module-level constants at the top of `demo_referee_gui_rsim.py`:

| Constant | Default | Description |
|---|---|---|
| `PROFILE` | `"human"` | Profile name or path to a YAML file passed to `load_profile()` |
| `PORT` | `8080` | HTTP port the GUI server listens on |
| `N_ROBOTS` | `3` | Number of robots per team |
| `MY_TEAM_IS_YELLOW` | `True` | Which team is ours |

### Using the GUI

The page is divided into four areas:

**Scoreboard** — live yellow / blue scores updated in real time.

**Status block** — four read-only fields:

| Field | What it shows |
|---|---|
| Command | Current referee command, colour-coded (red = HALT, orange = STOP, green = NORMAL / FORCE START, yellow/blue = team-specific commands) |
| Next | `next_command` — the command that will follow the current stoppage, if known |
| Stage | Current game stage (e.g. `NORMAL FIRST HALF`) and time remaining (mm:ss) |
| Designated | `designated_position` in metres — the ball placement target after a stoppage |

**Command buttons** — clicking a button immediately calls `CustomReferee.set_command()` on the server:

| Row | Buttons |
|---|---|
| Flow control | Halt · Stop · Normal Start · Force Start |
| Kickoffs | Kickoff Yellow · Kickoff Blue |
| Free kicks | Free Kick Yellow · Free Kick Blue |
| Penalties | Penalty Yellow · Penalty Blue |
| Ball placement | Ball Placement Yellow · Ball Placement Blue |

**Profile config panel** — read-only display of the active profile loaded at startup, split into six sections:

| Section | Fields shown |
|---|---|
| Field geometry | half length, half width, half goal width, defense length/width (half), centre circle radius — all in metres |
| Game settings | half duration (minutes), kickoff team, force-start-after-goal flag, stop duration (seconds) |
| Goal detection | enabled (ON/OFF), cooldown seconds |
| Out of bounds | enabled (ON/OFF), free-kick assigner method |
| Defense area | enabled (ON/OFF), max defenders, attacker infringement flag |
| Keep-out zone | enabled (ON/OFF), radius (metres), violation persistence (frames) |

Boolean fields are shown as green **ON** / red **OFF** pills. The panel title and browser tab title both show the profile name.

A small green dot at the bottom of the page indicates the SSE connection is live. If it turns grey the browser will reconnect automatically.

### Profiles

Built-in profiles live in `utama_core/custom_referee/profiles/`:

| Profile | Rules active | Auto-restart after goal |
|---|---|---|
| `human` | Goal detection only | No — operator controls the next stage |
| `simulation` | All four rules (goal, out-of-bounds, defense area, keep-out) | Yes — progresses automatically when restart criteria are met |

To customise, copy a YAML file, edit the values, then set `PROFILE` in `demo_referee_gui_rsim.py` to its path:

The YAML structure mirrors the dataclasses in `profile_loader.py`.
When running through `StrategyRunner`, geometry is always overridden from `full_field_dims`
at startup. For standalone use outside `StrategyRunner`, geometry can be passed to the constructor.

```yaml
profile_name: "my_profile"
rules:
  goal_detection:
    enabled: true
    cooldown_seconds: 1.0   # ignore further goals for this many seconds after one is scored
  out_of_bounds:
    enabled: true
    free_kick_assigner: "last_touch"  # only option currently
  defense_area:
    enabled: true
    max_defenders: 1                  # robots allowed inside own defense area
    attacker_infringement: true       # penalise attackers entering opponent defense area
  keep_out:
    enabled: true
    radius_meters: 0.5                # exclusion radius around ball during stoppages
    violation_persistence_frames: 30  # frames a robot must stay clear before violation clears
game:
  half_duration_seconds: 300.0
  kickoff_team: "yellow"
  force_start_after_goal: false  # true = auto FORCE START; false = wait for operator
  stop_duration_seconds: 3.0    # STOP hold time before auto FORCE START (if enabled)
```

### Architecture

```
CustomReferee (enable_gui=True)
│
├── _tick_loop (daemon thread, ~30 Hz)
│   └── CustomReferee.step() → stores latest RefereeData, broadcasts to SSE clients
│
├── GET /         → serves inline HTML + CSS + JS (no external resources)
├── GET /config   → returns profile config as JSON (fetched once on page load)
├── GET /events   → SSE stream; browser EventSource reconnects automatically
└── POST /command → JSON body {"command": "HALT"}, calls set_command(), returns 204
```

The server uses only `http.server.BaseHTTPRequestHandler` (stdlib). The browser page has no
framework dependency — state updates arrive via `EventSource`, config is fetched once with
`fetch('/config')`, and buttons call `fetch('/command')`.

### Typical operator workflow

1. Start the GUI with the desired profile.
2. The referee starts in **HALT**. Click **Stop** to begin a pre-match pause.
3. Click **Kickoff Yellow** (or Blue) to issue `PREPARE_KICKOFF_*`.
4. Click **Normal Start** to begin play — the stage timer starts counting down.
5. Use **Halt** / **Stop** between incidents; issue free kicks or penalties as needed.
6. The stage advances automatically (e.g. `NORMAL_FIRST_HALF` → `NORMAL_HALF_TIME`) according to the profile's `half_duration_seconds`.

### Button reference

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

#### Auto-detected vs manual

| Command category | Detection | Notes |
|---|---|---|
| Goal → kickoff | Auto (GoalRule) | Operator sets kickoff team before half starts |
| Out-of-bounds → free kick | Auto (OutOfBoundsRule) | `free_kick_assigner` in profile controls which team |
| Defense area → penalty | Auto (DefenseAreaRule, if enabled) | Penalty and ball placement are not exposed in the GUI |
| Ball placement | Auto (OutOfBoundsRule, if enabled) | Use `set_command()` in code if manual override is needed |

#### Typical sequences

Standard half start:
```
Halt → Stop → Kickoff Yellow → Normal Start
```

After goal (human profile — operator-controlled):
```
(Goal auto-detected) → Stop → (auto Force Start after stop_duration_seconds)
```

Manual free kick:
```
Halt  →  Stop  →  Free Kick Yellow  →  Normal Start
```

### Live field visualisation

The **Field** panel shows a top-down canvas updated in real time at ~30 Hz via SSE.

| Element | Colour | Notes |
|---|---|---|
| Field background | Green | Scaled to active referee geometry (set by StrategyRunner at startup) |
| Lines / circles | White | Boundary, centre line, centre circle, defence areas |
| Left goal | Yellow (translucent) | Yellow team's goal (negative x side) |
| Right goal | Blue (translucent) | Blue team's goal (positive x side) |
| Friendly robots | Yellow filled circle | ID label above; orientation line shows heading |
| Enemy robots | Blue filled circle | ID label above; orientation line shows heading |
| Ball | Orange filled circle | Minimum 4 px radius |
| Designated × | White cross | Ball placement target; hidden when `designated_position` is null |

> **Standalone mode**: the canvas shows an empty green field with the ball at (0, 0) because
> `_make_static_frame()` creates a frame with no robots.  Robots appear when driven by
> `demo_referee_gui_rsim.py` or another live game loop.

### Event log

The **Event Log** panel shows the 20 most recent events, newest first.

| Entry type | Colour | Trigger |
|---|---|---|
| Command change | Green | `d.command` differs from previous message |
| Score change | Yellow | `d.yellow_score` or `d.blue_score` differs from previous message |
| Status message | Grey (muted) | `d.status_message` is non-empty and differs from previous value |

---

## 9. Open Questions / Future Work

- **Active distance-keeping during STOP/free kicks**:
  Implemented for built-in referee nodes: STOP, opponent free kicks, and both ball-placement variants now actively clear illegal positions. Future work is mainly smarter pre-positioning, not basic compliance.

- **Ball placement precision**: `BallPlacementOursStep` uses `move()` which will stop the ball
  near (not exactly at) `designated_position`. The tolerance is ±0.15 m per the rules.
  Future work: dribble to position then release.

- **Kicker selection**: Currently lowest robot ID. Should prefer the robot closest to ball.

- **Using `next_command` for pre-positioning**: A future optimisation reads `next_command`
  during STOP and begins drifting to the correct position before the command changes.

- **`can_place_ball` fallback**: If `TeamInfo.can_place_ball` is False (too many placement
  failures), `BallPlacementOursStep` must fall back to STOP behaviour.

- **Penalty / ball-placement readiness tuning**: `simulation` now auto-progresses these restarts, but the exact readiness heuristics may still need iteration as we gather more simulator coverage.

- **End-to-end ball placement integration test**: The intended test scenario is:
  ball exits field → `STOP` → `BALL_PLACEMENT_YELLOW` → robot physically carries ball to
  `designated_position` → `DIRECT_FREE_YELLOW` → kicker drives to ball → `NORMAL_START`.
  This was attempted in `utama_core/tests/strategy_runner/test_referee_rsim.py` but deferred
  because `BallPlacementOursStep` cannot reliably carry the ball in RSim.  The robot drives
  to `ball.p` with the dribbler on, but the motion controller decelerates to a stop at the
  ball centre rather than capturing it, causing the robot to push the ball instead of carrying
  it.  Approaches tried: behind-ball offset (robot stops short), direct drive to ball with
  face-target orientation (hits ball side-on), proximity fallback for `has_ball` (still pushes).
  Root cause: the approach, dribbler-capture, and carry phases need a dedicated
  "get-behind-ball" skill with a slower final-approach speed before this can be tested
  end-to-end.  Additionally, `OutOfBoundsRule` currently issues `STOP → DIRECT_FREE` directly
  (no automatic ball placement step), so `BALL_PLACEMENT` must be injected manually via
  `set_command()` for this scenario.

- **Ball placement before free kick (SSL rule compliance)**: Per the official SSL rulebook,
  after a ball-out-of-bounds event the correct sequence is `STOP → BALL_PLACEMENT_* →
  DIRECT_FREE_* → NORMAL_START`, not `STOP → DIRECT_FREE_*` as `OutOfBoundsRule` currently
  produces.  `OutOfBoundsRule` should be updated to set `suggested_command=STOP` and
  `next_command=BALL_PLACEMENT_*` (with `designated_position` set to the infield restart
  spot), so the state machine progresses through ball placement before issuing the free kick.
  This requires `BallPlacementOursStep` to be working reliably first (see item above).

- **`BallPlacementOursStep` robot carry mechanics**: The current single-robot dribble
  approach does not work reliably — the robot pushes the ball rather than carrying it.
  Investigation needed.  Most competitive SSL teams use a two-robot "kissing" technique:
  one robot pushes from behind while a second robot lightly contacts the ball from the
  front to stabilise it, allowing the pair to transport it as a unit.  Alternatively,
  a single-robot approach with a dedicated "get-behind-ball" skill (slower final approach,
  approach vector aligned from behind relative to the target) may also be viable and is
  worth evaluating first before adding multi-robot coordination complexity.

- **GUI: suggested next action for human operators**: The current operator panel presents
  all available command buttons simultaneously, which is overwhelming for operators who
  are unfamiliar with SSL rules.  A future improvement would display a highlighted
  "suggested next step" banner or button based on the current referee command — for
  example, after a goal the suggestion would be "Kickoff [team]", after `PREPARE_KICKOFF`
  it would be "Normal Start (when robots are in position)".  This shifts the operator's
  job from deciding *what* to do next to simply deciding *when* it is safe to advance,
  reducing cognitive load and operator error during matches.

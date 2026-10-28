# Field logs: finding a moment and replaying it offline (design)

Status: design, 2026-10-28. Nothing here is built yet except what "Today" lists.

## Why

On the 2026-10-27 field day every bug was found the same way: a symptom someone saw ("it drives
sideways and back", "blue misses passes", "placement throws the ball away"), then a throwaway
script to find the moment in the log, then another to rebuild what the robot saw and decided at
that moment. The second step found every cause. Running the recorded camera packets through the
same refiners and tactic offline showed blue's receive point clamped to x 0.43, and the
attacker's hold point at (0.43, +0.07) wherever the ball was. Each script took 10-30 minutes
to write, and all of them were lost when `/tmp` was cleared. This doc designs one tool for the
whole loop, so a field symptom becomes a cause, and then a regression test, in minutes.

## Today

- Every real run records `replays/field/<time>_<mode>_<name>.log`, an SSL game log
  (`replay/game_log_writer.py`) with these records:
  - every camera packet, raw and per camera;
  - the radio both ways: commands (`RADIO_OUT`) and feedback, the ball sensor (`RADIO_IN`);
  - referee messages;
  - operator notes (`NOTE`, from the dashboard's Note box);
  - the run info: code version, the vision-to-radio mapping, the field.
- Beside the log is a `.npz` replay of the world the strategy saw.
- `analysis/robot_calibration.py` reads the log for one robot's numbers. `tools/plan_snapshot.py`
  draws one planner decision from an `.npz`, and `tools/moments.py` turns an `.npz` tick into a
  regression moment.

Missing:
- a timeline of what happened, to find the moment;
- a per-tick view that sets the raw camera, the refined world and the command side by side;
- rerunning the decision code on the recorded inputs.

## The tool: `tools/field_replay.py`

Three subcommands, each one step of the loop.

### 1. `index LOG`: find the moment

A timeline of the run, one line per event:
- notes;
- referee commands;
- each kick sent, with the ball's speed and direction after it;
- ball-sensor on/off edges per robot, with dropouts shorter than 0.5 s marked as such;
- possession changes, and the ball leaving the field;
- robots or the ball lost by vision for more than 0.2 s, and per-camera frame gaps;
- control-loop stalls: gaps in `RADIO_OUT` longer than 50 ms.

`--around T` prints only ±5 s. This replaces `ir.py`, `tl.py`, `passes.py` and `kicks.py` from that
night.

### 2. `window LOG --from A --to B [--robot yellow:1]`: look at it

Per tick:
- each camera's raw detection of the robot and the ball;
- the refined position, velocity and heading;
- the command sent, turned back into the world frame;
- the ball sensor, and the ball in the robot's frame (forward and left of its centre, which shows
  when the ball is on the dribbler);
- the referee command.

`--png` draws the window on the pitch: trails, command arrows, the ball, and each camera's
detections in its own colour, so a jump at the camera overlap is visible. These are pitch
pictures in the same style as `plan_snapshot`, not plots against time.

### 3. `rerun LOG --from A --to B --side yellow [--tactic module:Class]`: rerun the decision

1. Feed the window's recorded camera packets, at their recorded times, through a fresh refiner
   pipeline: position (with the fusion and Kalman filter), velocity, and robot info with the
   recorded ball-sensor feedback. Each refiner starts a few seconds before `A` to settle.
2. Tick the tactic, or the side's whole strategy from the run info, on each refined world, with
   the same control scheme.
3. **Fidelity check.** Compare the rerun's commands with the logged `RADIO_OUT`, tick by tick.
   They must agree, within float16 rounding, while the code is the version that ran (the log
   records which). If they don't, something the decision depends on wasn't recorded, which is a
   bug in the log or the tool, and the rerun says so before showing anything else.
4. **Counterfactual.** Run the same window with the working tree's code. It prints where its
   commands first differ from the recorded ones, and by how much. This is open loop: the robots
   still move as they really did. So it says what the fixed code would have decided at the
   moment it went wrong, not what would have happened next. That is exactly the check a fix
   needs, and why the result is a test and not a claim.
5. `--moment` writes the first diverging tick as a regression moment (`tools/moments.py`'s format)
   or a unit-test fixture: robot, ball and referee state, and the command it should have given.
   The fix's regression test then starts from the real field state.

What it needs from the code: every refiner and controller must read time from the frame, not the
wall clock. That holds for the refiners today. The real controller's dribbler bucket and the
kicker cooldown use their own timers, which the rerun replays from the log's timestamps. The
`sim_tac.py` / `bluew.py` prototypes from 2026-10-27 already do steps 1 and 2 for one tactic.

## The physical setup's own errors

A rerun reproduces what the robot believed, and that includes the setup's errors. Each error
below has one way to measure it. All of them become per-robot or per-camera numbers that the
live code and the rerun apply the same way.

| Error | What it does | How to measure it | Where the correction goes |
|---|---|---|---|
| Tag turned on the robot (yaw) | Vision's heading differs from the chassis's: the robot drives off its heading, and kicks go off by the same angle | A straight-forward command: the direction of travel against vision's heading. Calibration already drives forward and back, so this needs analysis only | Position refiner, per robot, before the Kalman filter |
| Tag off the robot's centre | The position is wrong by a fixed offset in the robot's frame. Turning on the spot looks like driving a small circle | The spin steps: fit a circle to the positions, whose radius and phase give the offset (already in the script) | Position refiner, per robot |
| Kicker turned on the robot | The ball leaves at an angle to the true heading. That night, yellow's kicks were about 7° off | Kick direction minus tag yaw (calibration's `direction_offset_deg`, new) | Kick aim: turn by the offset before kicking |
| Tag not flat, or the wrong height in ssl-vision | Position bias that grows away from the camera's centre, and differs between cameras | One robot at known marks (`vision_check --guided`); the two cameras' disagreement on the same robot in the overlap | The camera calibration first. What remains goes in a per-camera correction |
| Cameras disagree | A jump crossing the overlap; ball tracks that break | Disagreement map from every log with a robot in the overlap | Camera calibration; the fusion's weights |
| Latency per camera | Fused positions lag differently per camera | `t_capture` against receive time, per camera | Fusion |
| Robot dynamics | Overshoot sideways, turning 9% faster than commanded | Calibration (sideways about 1.8 m/s² against the 4 assumed; turn gain 1.09) | Controller limits and gains per robot; the sim |
| Ball sensor | Drops out with the ball still on | Calibration's dribble steps (`sensor_on`) | `RobotInfoRefiner`'s `contact_grace_s` (since 2026-10-28) |

These rows meet in one **robot profile**:
- one JSON per robot, written by `pixi run calibrate` and kept in the repo under
  `config/robot_profiles/`;
- it holds tag yaw and centre offset, kicker offset, kick speed, dribbler hold point, sensor
  reliability, turn gain and acceleration per axis;
- it is read by the position refiner, the controller, the kick aim, the sim (per-robot physics for
  realistic evaluation) and the rerun;
- a robot without a profile runs as today.

The calibration to-do list that fills it (driving per axis and diagonally, kicking, dribbling,
receiving, all robots at once, and where each number is used) is `roadmap.md` item 10.

The profile is what makes two robots of the same design behave the same in the code. Tonight's
"only one robot misses" was partly geometry, but robot-to-robot differences such as yellow's kick
offset will keep showing up the same way.

## Focused logs

The tool is quicker when the logs are short and labelled.
- One run per test, named for it: `--name pass_receive`. Today that is per example.
- A note at each change of set-up. With no dashboard, the calibration CLI writes its step names as
  notes itself: proposed, one line.
- Keep everything. A field log is small (2-3 MB a minute) and can't be re-recorded.

## Order of work

Each step is useful on its own and small enough for one session:

1. `index` and `window`, from the 2026-10-27 scripts. Gives back the speed of that night.
2. `rerun` with the fidelity check, for the examples' tactics, then for whole strategies.
3. Tag yaw and centre offset in `robot_calibration` (analysis only; the moves are already in
   the script), and the per-robot profile, applied in the position refiner.
4. Kick aim from the profile's kicker offset.
5. Camera disagreement map from logs.
6. The sim reads robot profiles. That changes evaluation results, so ask first, and batch it with
   the next forced rerun.

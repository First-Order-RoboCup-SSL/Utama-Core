# Real robots: setting up and checking vision and radio

The procedure for a test day with real robots: ssl-vision on its own Linux laptop sending vision
over the network, and our stack on a Windows laptop in WSL with the radio dongle. Each part is
checked on its own before anything drives a robot: vision with `tools/vision_check.py`, the radio
with teleop, then both together in the field check of `examples/shooting_drill.py`.

```
vision laptop (Linux)                      robot laptop (Windows + WSL)
  cameras -> ssl-vision  --multicast-->      tools/vision_check.py, then the runner
                         224.5.23.2:10006    radio dongle (USB, passed into WSL) -> robots
```

## Once per machine

### Vision laptop (Linux)

1. Build [ssl-vision](https://github.com/RoboCup-SSL/ssl-vision) (its README). For a FLIR
   camera install FLIR's **Spinnaker SDK** first: on the
   [download page](https://www.flir.com/products/spinnaker-sdk) take "Spinnaker SDK, 64-bit"
   for your Ubuntu (not ARM64 unless the laptop is ARM). Unpack it, install the libraries its
   README lists, then `sudo sh install_spinnaker.sh`: yes to the udev entry (add your user to
   `flirimaging`) and to USB-FS 1000 MB. Build ssl-vision with
   `cmake -B build -DUSE_SPINNAKER=true -DUSE_V4L=true && make -C build`.
2. Add our field as a preset: `git apply tools/ssl-vision-lab-field.patch` in the ssl-vision
   checkout (copy the file over), then rebuild. Field Configuration gets a **Lab Field** button
   (3 m x 2.25 m, goal 0.8 m, defense area 1 m x 0.5 m, two cameras). Do not use the Division B
   button with two cameras: it sets the camera count to 1.
3. USB webcams (the RER camera is one) need your user in `video`:
   `sudo usermod -aG video $USER`, then log in again.

### Robot laptop (Windows + WSL)

1. **USB into WSL:** in admin PowerShell, `winget install --exact dorssel.usbipd-win`. The
   `usbipd` you can run inside WSL is an unrelated Linux tool: always run it in PowerShell.
2. **Share the radio once, by its hardware ID** (stays the same whatever port it is in, unlike
   the bus ID): `usbipd list`, note the radio's `VID:PID`, then
   `usbipd bind --hardware-id <VID:PID>`.
3. **WSL on the real network:** in `%USERPROFILE%\.wslconfig`, under `[wsl2]`, add
   `networkingMode=mirrored`. Without it WSL sits on a private network that never sees the
   vision laptop's multicast.
4. **Let vision in:** in admin PowerShell, open UDP 10006 to WSL in the Hyper-V firewall:
   ```
   New-NetFirewallHyperVRule -Name SSLVision -DisplayName "SSL Vision multicast" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol UDP -LocalPorts 10006
   ```
   (`40E0AC32-...` is WSL's fixed ID in that firewall.) Then `wsl --shutdown`.
5. **Radio permission in WSL:** `sudo usermod -aG dialout $USER`, then `newgrp dialout` in the
   terminal you run from (or restart WSL).

## Each test day

Once the network, vision and radio below are set up, one command checks them all at the start of
a session:

```
pixi run field-check          # radios 0 to 7; --radios 0 1 for fewer
```

It listens to vision for 2 s (cameras and their packets per second, field size, ball, lids in
view), then finds the robots (`find-robots`, below): each one's radio, lid, share of its commands
answered (under 80%: range, interference or a flat battery) and whether its profile was
calibrated on that lid. Each line starts "ok" or "!!". It cannot see batteries: the robots do not
report them. The steps below are what to do when a line says "!!".

### 1. Network

Both laptops on the same network, by Ethernet to one switch or router if possible: many Wi-Fi
routers drop or throttle multicast. If the robot laptop has two networks (Wi-Fi for internet,
Ethernet to the switch), our receiver listens on the one Windows uses for internet: use one.

### 2. Vision

On the vision laptop: `./bin/vision -c 2` (the number of cameras) in the ssl-vision checkout.
For each camera under **Image Capture**: the capture module (**Spinnaker** for FLIR, **Video 4
Linux** for a USB webcam), then **Capture Control → Start**. Settings save on a clean exit or
with **Save Settings**, not on a crash.

- **USB webcam:** cam idx is the `/dev/video` number of its image; a camera shows up as two
  nodes and the second is metadata ("Inappropriate ioctl" means the wrong one). Use **MJPEG**:
  the RER camera's uncompressed YUYV is 5 fps at 1280x960, MJPEG 60-90 fps. Its controls:
  `v4l2-ctl -d /dev/video0 --list-ctrls`. Let auto white balance settle
  (`--set-ctrl=white_balance_automatic=1`), then lock it (`=0` and
  `--set-ctrl=white_balance_temperature=<what it picked>`): colours that drift break the colour
  calibration. Keep exposure under about 10 ms (`exposure_time_absolute=100`) so the ball is
  not smeared.
- **FLIR:** exposure and gain are in its Spinnaker node and change live.
- Field Configuration: **Lab Field → Apply**, then each camera's calibration, then colours per
  camera.
- `./bin/graphicalClient` on the vision laptop shows what ssl-vision sends, before any network.

On the robot laptop, in WSL:

```
pixi run python tools/vision_check.py
```

Expect each camera at about its frame rate and the field at 3000 x 2250 mm. Nothing at all
means the network or the firewall (steps 3-4 above, step 1 here); one camera missing means its
capture is not running.

After a calibration, check it against known spots: `tools/vision_check.py --guided ball`, then
`--guided "blue 2"` (any robot, facing +x). It asks for the ball or robot on the 9 points
(corners, edge midpoints, centre) and prints each camera's error there; the results land in
`replays/field/*_vision_truth_*.json` to compare with the next setup. The pattern points at the
cause: errors growing towards the edges are the distortion or focal length, the same shift
everywhere the camera's position, more at one end than the other its tilt or height. The ball
right but the robot not is the robot height in ssl-vision.

### 3. Radio

Plug the dongle in, then in admin PowerShell (leave it open: it re-attaches after a replug):

```
usbipd attach --wsl --hardware-id <VID:PID> --auto-attach
```

In WSL `ls /dev/ttyACM*` shows it; the code uses `PORT` in `utama_core/config/settings.py`
(`/dev/ttyACM0`). Then drive each robot by hand:

```
pixi run debug-robots
```

Number keys choose the robot by its **radio (firmware) ID**, WASD moves, Q/E turns, Shift is
fast (0.5 m/s, else 0.1), Space dribbles, B/C kick/chip, Esc stops every robot and quits. Hold
the robot off the floor for the first presses and check each key moves it the way it says.
Note each robot's radio ID and the pattern on top (its vision ID): the runner needs both.

One radio drives every robot, of either colour: robots are addressed by radio ID only, so the
radio IDs must differ across all robots on it.

### 4. Both together

Set the "Test-day configuration" block of `examples/shooting_drill.py`: `FIELD`,
`CAMERA_HALF` (`"full"` when the cameras cover the whole field), each robot's vision and radio
ID and working hardware. Then `pixi run python examples/shooting_drill.py`, open
http://localhost:8080 and go through the Live tab's field check
([custom_referee_gui.md](custom_referee_gui.md#field-check-camera-runs)): every row green or
amber, each robot under its own ID and colour, a ball in each corner inside the area in play,
and a robot pushed by hand across the cameras' overlap moving smoothly (a jump there means the
two calibrations disagree). Only then Force Start.

Each run records `replays/field/<time>_<mode>_<name>.log`, a game log of every vision packet,
the radio both ways, the referee and which code ran. Beside it is a `.npz` replay of the world
the strategy saw, written when the run closes. Use the Live tab's Note box (Enter writes it into
the log) to say what a stretch is: "ball placed by hand", "start dribble test", "ignore". Most of
a test day is setting up, and notes make the useful minutes easy to find later. The runner adds
a note whenever a robot's row in the Tactics panel changes (its tactic and step, or
`referee · ball placement · placer` while a restart drives it), so the log also says what each
robot was trying to do.

#### Running the demos

`pixi run demos` (`--mode rsim` to rehearse) starts the shooting drill behind one dashboard;
the Live tab's Demo panel switches to pass-receive, save-and-counter or score goal without
leaving the page. Halt, pick one, Run: the robots stop, the demo closes (its log saved, the
radio and vision released), the next one starts, its lids checked as when run alone, and the
page reloads by itself. A demo that fails to start (a robot not found) shows why under the
panel; pick again. The Strategy panel still switches tactics within a demo; Ctrl+C ends it.
Calibration (`pixi run calibrate`) and `planner_ab.py` stay their own commands.

### 5. Measuring a robot

**Which robot is which.** A robot is its radio ID: the radio board stays with its motors,
kicker and dribbler, and its profile goes by it. The lid with the pattern is a separate shell
and can be swapped. After switching robots on, or swapping a lid:

```
pixi run find-robots          # radios 0 to 7; --radios 0 1 4 for others
```

Each radio that answers turns about 70° on the spot and back (keep 15 cm clear round the
robots), and the pattern that turned with it is its lid. It prints radio, lid and whether the
profile was calibrated on that lid (a profile's tag offset is its lid's: calibrate again after a
swap), and the radios that did not answer and patterns no radio turned. The pairing is saved on
this machine (`replays/field/robots_found.json`). The robots an example uses are chosen by radio
(`RADIO_0`, `RADIO_1` in `shooting_drill.py`), one per colour, and no vision ID or colour is
written by hand: on real robots an example starts by watching vision for 0.5 s (nothing moves).
With each robot's saved lid in view it uses them; otherwise (never run here, a lid missing) it
finds those robots again first (`find_robots.confirm`). It cannot tell two robots in use that
swapped lids with each other, as both lids stay in view: run `find-robots` after such a swap.

One command measures a robot and writes its profile:

```
pixi run calibrate                    # finds the robots; the only one, or asks which
pixi run calibrate --radio 0          # the robot on radio 0
pixi run calibrate --radio 0 --ball-only   # the ball part again, no driving
pixi run calibrate --radio 0 --check       # the new profile against the committed one
```

Without both `--robot colour:id` and `--radio` it looks the robot up: with `--radio`, as the
examples do; with neither, by `find-robots`.

No dashboard: it runs in the terminal (`examples/calibrate_robot.py`). Once vision sees the robot
it asks you to put it near the middle of the camera's view facing +x, with 0.8 m clear ahead,
0.6 m to its left and a ball at rest in view; press Enter. It drives and turns for about 65 s.
Then the ball part (`examples/ball_trials.py`), where **the robot fetches the ball itself**:

- **Carry ladder**: it carries the ball 0.5 m forward, backward, left and right, and half a turn
  each way, at rising speeds (0.2 to 1 m/s, 0.5 to 4 rad/s), twice each (`--repeats`), from
  standing, then stands 0.5 s with it. A direction stops at the first speed that loses the ball
  (off the robot by 15 cm, or its ball sensor off 0.3 s). The carry runs towards the middle of
  the field, and the robot lets go and fetches the ball again before each trial.
- **Kicks**: 10 (`--kicks`) along the field. Every kick comes charged: the kicker holds its
  charge about 30 s and recharges in about 1 s, so a kick more than 30 s after the last one is
  preceded by a kick with no ball (the robot lets go of the ball first).

It asks only when the ball is out of view or within 25 cm of the edge, as after most kicks:
roll the ball back towards the middle and press Enter. Every step is printed as it starts; Ctrl+C
halts the robot. At the end it prints DONE, analyses the log (loop delay, speed gain and
acceleration per axis, vision noise at rest, kick speed; written beside the log as
`.calibration.json`) and **writes the robot's profile**, `utama_core/config/robots/radio_<radio>.json`,
which every example reads (`load_profile`). Nothing to copy by hand:

- `tag_forward_m`, `tag_left_m`: where its tag sits on it, from its turns on the spot. A lid's tag
  is a sticker, rarely centred; on real robots vision then reports the robot, not its tag.
- `kick_bend_deg_per_cm`, `kick_bias_deg`: how far its kicks leave the heading per cm the ball
  sat off-centre, and with it centred, fitted together. The heading is the tag's, so a tag stuck
  on turned (yellow 0's reads 3.5° clockwise) shows in the bias: no straight edge needed.
- `kick_speed`, `kick_spread_deg`: those kicks' median ball speed, and how far they scatter
  about bias and bend (the fit's rms).
- `carry_speed`, `carry_turn`: the carry ladder's slowest direction, the fastest speed it kept
  the ball every time; `move()` holds this robot to them (`RobotParams.CARRY_*` otherwise).

rsim kicks a robot given its profile as it measured (`real_kick.py`): at its kick speed, off its
heading by its bias and bend, scattered by its spread, and passes and shots aim against bend and
bias there too. So an example rehearsed in rsim (`--mode rsim`) kicks like the robot on the field.
Matches without profiles (the strategy evaluation) keep the sim's own kick.

**Kicker charge in play.** On real robots a robot that has not kicked for 25 s fires an empty
kick, when the ball is neither on it (ball sensor) nor within 30 cm (vision), and never in a HALT
(`kicker_charge.py`); so does each robot at its first chance after a HALT. 2026-10-28's kicks:
yellow 0 (radio 0) kicked about 2.2 m/s 1 s or 30 s after its last kick, 1.96 after longer; blue 0
(radio 1) recharges far slower than 1 s, a median 0.41 m/s within 15 s of its last kick and 1.29
after 15-30 s. Have the electronics team look at radio 1's charging.

What it could not measure stays as it was; hardware that breaks (`kicker`, `dribbler`, `ir`) is
set in the file by hand. `--mode rsim --yes` rehearses it in the simulator (its dribbler holds at
every speed) and writes nothing.

**Is the new profile better?** `git diff` shows what changed. `--check` tries it against the
committed one, on the field, same session: 4 blocks (`--blocks`) alternating the two, committed
first, so battery and floor fall on both alike. In each the robot carries the ball every way at
that profile's caps (`--repeats`, default 2) and kicks 5 times (`--kicks`) at the middle of the far
end, aimed against that profile's bend and bias. It prints per profile how often each carry kept
the ball and how far the kicks missed (mean of |miss| ± its standard error). Commit the file if
the new profile is no worse. Ten kicks a side tell apart only differences of a few degrees; more
`--kicks` for finer.

Calibrate every robot, twice per session (early and late): how far the two runs disagree is how
far to trust them. An older log is measured again with
`pixi run python tools/robot_calibration.py replays/field/<run>.log --tag yellow:0 0 -0.002`
(`--tag` per robot: its `tag_forward_m`, `tag_left_m`).


| Symptom | Cause |
|---|---|
| `usbipd` in WSL talks about `linux-tools` packages | It is the Linux tool; run `usbipd` in PowerShell |
| `usbipd list` shows "Unknown USB device (device descriptor request failed)", `0000:0002` | Windows cannot read the device: other port, no hub, other cable |
| No `/dev/video*` after attaching a webcam | `sudo modprobe uvcvideo` |
| ssl-vision: "unable to obtain instance of camera" / permission | User not in `video` (or `flirimaging`); `newgrp video` |
| ssl-vision: "Could not set format" on `/dev/video1` | cam idx points at the metadata node: use the other one |
| `tee: .../usbfs_memory_mb: No such file` | WSL loads USB support only once a device is attached; `/etc/modprobe.d/usbfs.conf` with `options usbcore usbfs_memory_mb=1000` sets it then |
| Whole image tinted one colour | White balance stuck at an extreme (2800 K): auto, then lock |
| `vision_check` sees nothing | Different networks, Wi-Fi multicast, `.wslconfig` not mirrored, or the firewall rule missing |
| Dashboard: "Address already in use" | Another dashboard (`dashboard_server.py`) holds 8080: stop it |

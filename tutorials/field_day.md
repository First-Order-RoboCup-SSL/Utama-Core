# A field day with real robots

This page takes you through a test day: from a laptop that sees nothing to robots playing a drill
on the field, and their logs saved for later. Do "Before the day" once per laptop. Do "On the day"
in order, every time. Each step checks one thing before the next step needs it.

You need two laptops:

- **The vision laptop** (Linux) runs the cameras and the vision program, which sends where every
  robot and the ball are over the network.
- **The robot laptop** (Windows with WSL) runs our code and talks to the robots through a USB
  radio dongle.

```
vision laptop (Linux)                         robot laptop (Windows + WSL)
  cameras -> ssl-vision or TagVision  --->     our code: checks, drills, strategies
             (224.5.23.2:10006)                radio dongle (USB) -> robots
```

## Two vision systems

You can run either; our code can't tell the difference.

- **ssl-vision**, the league's official program. It finds robots by the coloured dots on their
  lids. Its colour calibration takes time, and it drifts when the light changes.
- **TagVision**, our own (repo `First-Order-RoboCup-SSL/tagvision`). It finds robots by black
  and white ring markers you print, and calibrates from the field lines. It sends exactly what
  ssl-vision sends.

Pick one for the day. The rest of this page is the same for both.

## Before the day (once per laptop)

### The vision laptop

**ssl-vision:**

1. Build [ssl-vision](https://github.com/RoboCup-SSL/ssl-vision) (follow its README). For a FLIR
   camera, install FLIR's **Spinnaker SDK** first (see [real_robots.md](../docs/real_robots.md#field-day-reference)),
   then build with `cmake -B build -DUSE_SPINNAKER=true -DUSE_V4L=true && make -C build`.
2. Add our field: copy `tools/field/ssl-vision-lab-field.patch` into the ssl-vision folder,
   run `git apply ssl-vision-lab-field.patch` and build again. You get a **Lab Field** button.
3. For a USB webcam, add yourself to the `video` group: `sudo usermod -aG video $USER`, then log
   out and in.

**TagVision:** follow its `README.md` (one command, `make setup`), then its
`docs/setup-guide.md`, which covers cameras, mounting, printing markers and calibration. Try
its dry run first: it works with no cameras.

### The robot laptop

Run these in **PowerShell as administrator** unless it says WSL:

1. Install the tool that passes USB devices into WSL: `winget install --exact dorssel.usbipd-win`.
2. Plug in the radio dongle. Run `usbipd list` and note the dongle's `VID:PID`. Then
   `usbipd bind --hardware-id <VID:PID>`.
3. Put WSL on the real network: in `%USERPROFILE%\.wslconfig`, under `[wsl2]`, add the line
   `networkingMode=mirrored`.
4. Let vision through the firewall:
   ```
   New-NetFirewallHyperVRule -Name SSLVision -DisplayName "SSL Vision multicast" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol UDP -LocalPorts 10006
   ```
   Then `wsl --shutdown`, so steps 3 and 4 take effect.
5. In WSL: `sudo usermod -aG dialout $USER`, so you may use the radio. Restart WSL.

## On the day

### 1. Network

Connect both laptops to the same switch or router, **by cable** if you can. Wi-Fi, and phone
hotspots most of all, drop and bunch up the vision packets.

### 2. Vision

**ssl-vision**, on the vision laptop:

1. `./bin/vision -c 2` in the ssl-vision folder (2 is the number of cameras).
2. For each camera, under **Image Capture**: pick **Spinnaker** (FLIR) or **Video 4 Linux**
   (webcam), then **Capture Control → Start**.
3. **Field Configuration → Lab Field → Apply.** Then calibrate each camera, then its colours.
4. Save with **Save Settings**. A crash loses unsaved settings.

**TagVision**, on the vision laptop: `uv run tagvision run -c config/default.yaml`. Its guide
says how to calibrate each camera first, and `uv run tagvision listen` shows what it sends.

Then, on the robot laptop in WSL:

```
pixi run python tools/field/vision_check.py
```

You should see each camera at about its frame rate and the field at 3000 x 2250 mm. Nothing at
all: check the network and the firewall. One camera missing: its capture isn't running.

### 3. Radio

Plug in the dongle. In PowerShell as administrator, and leave the window open (it reconnects
the dongle if you unplug it):

```
usbipd attach --wsl --hardware-id <VID:PID> --auto-attach
```

In WSL, `ls /dev/ttyACM*` should show it. Now drive each robot by hand:

```
pixi run debug-robots
```

A number key picks a robot by its radio ID. WASD moves, Q/E turns, Space runs the dribbler, B
kicks, Esc stops every robot and quits. Hold the robot off the floor for the first presses and
check each key does what it says.

### 4. Check everything at once

Switch all robots on and put them on the field, with 15 cm clear around each. Then:

```
pixi run field-check
```

It listens to vision for 2 seconds, then turns each robot a little on the spot to see which lid
it wears. Every line starts with `ok` or `!!`. Fix each `!!` line before going on: the table at
the bottom of this page has the common causes.

### 5. Run a drill

```
pixi run demos
```

Open http://localhost:8080, go to the **Live** tab and go through its field check: every row
green or amber, each robot under its own ID and colour, the ball seen in every corner. Push a
robot by hand from one camera's view into the other's: it should move smoothly, not jump. Only
then press **Force Start**.

The **Demo** panel switches between the shooting drill, pass-receive, save-and-counter, score
goal and teleop. To rehearse at home, run `pixi run demos --mode rsim`: the same thing in the
simulator.

**Write notes as you go.** Type in the Live tab's **Note** box and press Enter: "ball placed by
hand", "start dribble test", "ignore this". Most of a test day is setting up; notes make the
useful minutes easy to find later.

### 6. Measure a robot

Do this when a robot is new, has been repaired, or wears a different lid:

```
pixi run calibrate --radio 0
```

It tells you where to put the robot and the ball, then drives, turns, carries the ball and
kicks it, by itself, for a few minutes. At the end it writes the robot's profile,
`utama_core/config/robots/radio_0.json`. Check it with `git diff`, and commit it if it looks
right. What each number means: [real_robots.md](../docs/real_robots.md#field-day-reference).

### 7. After the day

Every run wrote a log to `replays/field/`. **Never delete one.** To share them:

```
pixi run python tools/field/package_logs.py field --dry-run
pixi run python tools/field/package_logs.py field
```

Then write down what broke and what you measured. A coding agent can do the rest (trim the idle
minutes, register the session, measure from the logs): ask it to follow the `field-session` skill
(`.agents/skills/field-session/SKILL.md`).

## If something goes wrong

| What you see | What to do |
|---|---|
| `usbipd` in WSL talks about `linux-tools` | That is a different tool: run `usbipd` in PowerShell |
| `usbipd list` shows "Unknown USB device", `0000:0002` | Try another port or cable, without a hub |
| No `/dev/video*` after plugging in a webcam | `sudo modprobe uvcvideo` |
| ssl-vision: "unable to obtain instance of camera" | You are not in the `video` (or `flirimaging`) group: `newgrp video` |
| ssl-vision: "Could not set format" | Wrong camera number: a webcam shows up twice, use the other one |
| The whole image is one colour | White balance stuck: set it to auto, let it settle, then lock it |
| `vision_check` sees nothing | Different networks, Wi-Fi, `.wslconfig` not mirrored, or the firewall rule is missing |
| `field-check`: a robot answers under 80% | Range, interference or a flat battery |
| `field-check`: a lid differs from its profile | The lid was swapped: `pixi run calibrate` that robot again |
| Robots stop with "no vision packets" | Vision over Wi-Fi: use a cable |
| Dashboard: "Address already in use" | Another dashboard holds port 8080: stop it |

Everything in more depth (log format, robot profiles, camera settings): [real_robots.md](../docs/real_robots.md).

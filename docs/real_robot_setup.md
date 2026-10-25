# Real robots: setting up and checking vision and radio

The procedure for a test day with real robots: ssl-vision on its own Linux laptop sending vision
over the network, and our stack on a Windows laptop in WSL with the radio dongle. Each part is
checked on its own before anything drives a robot: vision with `tools/vision_check.py`, the radio
with teleop, then both together in the field check of `examples/real_attacker_defender.py`.

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

Set the "Test-day configuration" block of `examples/real_attacker_defender.py`: `FIELD`,
`CAMERA_HALF` (`"full"` when the cameras cover the whole field), each robot's vision and radio
ID and working hardware. Then `pixi run python examples/real_attacker_defender.py`, open
http://localhost:8080 and go through the Live tab's field check
([custom_referee_gui.md](custom_referee_gui.md#field-check-camera-runs)): every row green or
amber, each robot under its own ID and colour, a ball in each corner inside the area in play,
and a robot pushed by hand across the cameras' overlap moving smoothly (a jump there means the
two calibrations disagree). Only then Force Start.

Each run records `replays/field/<time>_<mode>_<name>.log`, a game log of every vision packet,
the radio both ways, the referee and which code ran. Beside it is a `.npz` replay of the world
the strategy saw, written when the run closes. Use the Live tab's Note box (Enter writes it into
the log) to say what a stretch is: "ball placed by hand", "start dribble test", "ignore". Most of
a test day is setting up, and notes make the useful minutes easy to find later.

## When something fails

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

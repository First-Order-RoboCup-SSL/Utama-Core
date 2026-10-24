# In-house vision: plan, experiments and findings

Status: plan, 2026-10-24. Nothing below is built unless it says so. Written after the
2026-10-22 test day, where vision took most of the day.

The goal is less time spent on vision each test day, and robot positions accurate to about
1 cm instead of 3–9 cm. This applies to **lab and test days only**: at competitions the league
runs ssl-vision with the standard colour patterns, and our stack only receives its packets.

## What went wrong on 2026-10-22

The setup: two cameras, ssl-vision on its own Linux laptop sending 224.5.23.2:10006 at 61 fps
per camera, our stack in WSL on the robot laptop (`docs/real_robot_setup.md`).

- **The cameras disagree.** The same robot was reported 13–20 cm apart by the two cameras
  early on; after the distortion was retuned in ssl-vision, 5 cm, then 85 mm later in the day
  (`replays/field/20261022_211142_real_score_goal.log`, constant 83–85 mm in every frame).
  Something moved between those checks, and nothing tells us when.
- **Against known spots** (`vision_check.py --guided ball`,
  `replays/field/20261022_212332_vision_truth_ball.json`): camera 0 saw 7 of 9 points, 33 mm
  mean error, 61 mm worst; camera 1 saw 5 of 9, 55 mm mean, 92 mm worst. Neither saw the ball
  at (1500, 0), right in front of the +x goal.
- **Camera height is unknown.** Nobody has a good measurement of it, and ssl-vision's
  calibration leans on it. A wrong robot height shifts each camera's robots along its own line
  of sight, in opposite directions for the two cameras.
- **Camera 1 never sent its calibration** in the geometry packet, so our camera weighting
  (`CameraCombiner`, below) ran with equal weights all day.
- **Colour calibration is slow and noisy at the edges and corners.** Yellow detection
  collapsed after a change to the robot tops (1.3 to 0.1 detections per packet), and one
  yellow robot came out as IDs 1, 2 and 3 at the same spot. Camera 0 detected no ball at all
  in one 2,010-packet run.
- **Stray orange objects became balls.** Camera 0 saw 1–4 fake balls in a line about 15 cm
  outside the +y touchline, each at confidence 1.00. Our stack picked a fake 22% of the time
  and switched between balls 107 times in 6 s: the flickering ball.
- **Focus and zoom can't be locked** on our cameras, so the lens changes whenever a camera is
  packed up, and every calibration starts from zero.

## What our stack already does about it

Built during the test day; it hides the cameras' disagreement but cannot remove it.

- `CameraCombiner` (`utama_core/data_processing/refiners/position.py`) weights each camera by
  how steeply it sees a robot (cos⁴ of the view angle) once every camera's position is known,
  and drops a detection more than `CAMERA_AGREEMENT` (0.1 m) from the best camera's.
- The ball is fused across cameras by the same rule, and each camera's ball is first moved by
  that camera's own error at the robot nearest it (`BALL_ROBOT_SHIFT_RANGE`), so a ball at the
  dribbler stays at the dribbler. 2.4% of frames in the log above still put the ball inside the
  robot: those where camera 1 sees the ball but loses the robot.
- `tools/vision_check.py` prints where each camera puts each robot and ball and the gap
  between them, and with `--guided` checks each camera against 9 known spots and saves the
  result as a baseline.

## Options considered

| | Time per test day | Build cost | Risk |
|---|---|---|---|
| 1. ssl-vision, more tuning and better fusion | stays high | none | what we have now |
| 2. Our own stack, with tags (and patterns) | low | weeks | most work, but ours to fix |
| 3. Modify ssl-vision's own code | medium | high: a large C++/Qt codebase and a fork to maintain | avoid |

Improving ssl-vision from the outside (our own calibration tool writing its settings) gets most
of what option 3 would, without a fork. So the real choice is 1 or 2, and two experiments
decide it (below).

## Decision, under working assumptions

The plan below assumes three things, each to be confirmed by an experiment:

1. **The cameras can't be locked** (focus and zoom change between sessions).
2. **Proper calibration alone doesn't remove the position error** (experiment 1).
3. **Tags are better than colour patterns** for robots (experiment 2; early indications say so).

Under those, **build our own stack, tag-only for robots, and recalibrate it automatically at
the start of every session**. Pattern mode is not rebuilt: when we want to practise with
patterns (as at competitions), we run ssl-vision. If an experiment contradicts an assumption,
revisit this section before building past the step it gates.

What the assumptions imply:

- Cameras that can't be locked: the lens changes each session, so calibration must run every
  session, in about a minute, with nothing done by hand, and keep checking during the session.
- Calibration that doesn't fix the error: don't rely on a correct camera model. Fit the
  image-to-field mapping each session from measurements, then correct what is left.
- Tags better than patterns: identify robots and their heading from tags, so robots need no
  colour at all. The ball is the only thing left that needs colour.

## Design

```
cameras -> capture -> [calibration each session] -> tag detector + ball detector -> SSL_WrapperPacket -> our stack (unchanged)
```

### 1. Calibration each session: one button

- About 20 A4 tags (AprilTag 36h11) on the floor; 3 of them on measured spots (the centre and
  two corners, lined up with the field lines) that fix the field's axes. The rest go anywhere:
  the solver finds where they are.
- A few tags on boxes at robot-top height (about 14 cm). A flat floor alone cannot tell a longer
  lens from a higher camera, which is today's camera-height problem; tags off the floor
  separate the two, and they calibrate exactly the plane the robot positions come from.
- Press calibrate: one bundle adjustment fits each camera's lens (focal length, image centre,
  distortion), its position and tilt, and the loose tags' positions, over hundreds of tag
  corners. Both cameras see shared tags, so they agree with each other by construction.
- Because the camera model alone isn't trusted (assumption 2), fit a smooth correction map from
  the residual error at the tags: one for robot-top height, one for ball height (about 21 mm).
- It ends by running the guided check itself and prints the error per camera.
- Expected accuracy: the field is about 3 m in roughly 1,300 pixels, 2–3 mm per pixel; tag
  corners are found to a fraction of a pixel and averaged over hundreds, so a few mm.

### 2. Watching for camera movement during the session

A few tags stay just outside the field lines all session. Every few seconds the tool
re-estimates each camera's position from them, corrects for a small move and raises an alarm
for a large one (a bumped or sagging mount). This answers "something moved between 52 mm and
85 mm" from the test day.

### 3. Robots: tags

- One tag per robot top; a config file maps tag ID to team and robot ID.
- `cv2.aruco` has AprilTag 36h11 built in; each detection gives position and heading from the
  tag's 4 corners.
- When a moving robot's tag is found but can't be decoded, tracking keeps its ID.
- Short exposure: motion blur is the main weakness of tags.

### 4. Ball: colour, but the easy kind

- Only inside the field mask that calibration gives, plus a margin: that alone would have
  removed the test day's fake balls.
- Flat-field correction from an empty-field frame (the median over a few seconds), so the ball
  looks the same colour in the corners as in the centre.
- Orange by hue, not brightness; blobs on a robot tag are ignored.

### 5. Output: standard SSL packets

The same `SSL_WrapperPacket` on the same multicast address, so `CameraCombiner`, the refiners
and everything above stay unchanged. While both run, send on a second port so the two can be
compared side by side on the same scene with `vision_check`.

### Practical

- Python and OpenCV; move to C++ only if measured latency forces it. Tag detection at 60 fps is
  fine with decimation.
- Runs on the vision laptop. FLIR cameras through PySpin (Spinnaker's Python API), webcams
  through OpenCV directly.
- Roughly 2–4 weeks part-time to step 4 of the order below. The calibration tool is the hardest
  part and the most valuable.

## Experiments

Run on recorded frames where possible, so they need no test-day time beyond the recording.

### Experiment 1: does proper calibration fix the position error?

- Checkerboard on A4, about 20 stills per camera, tilted, all over the image; OpenCV computes
  focal length and distortion.
- Tags or checkerboards on known spots for each camera's position and height; feed the result
  into ssl-vision.
- Rerun `vision_check.py --guided ball` and compare with the 2026-10-22 baseline above.
- **Down to about 1 cm:** calibration was the main problem. Stay on ssl-vision with our
  calibration tool (option 1), and assumption 2 is wrong.
- **Not down:** the error is elsewhere (colour, ssl-vision's simpler camera model), which
  points to our own stack.

### Experiment 2: are tags better than colour patterns?

- An AprilTag (8–10 cm) taped on one robot; record both cameras while pushing it slowly across
  the field, corners included.
- Offline, compare the tag detector with ssl-vision's pattern on the same frames: detection
  rate, position and heading jitter, ID swaps, corners against the centre, moving against still.
- Whether tags are big enough in pixels depends on the camera resolution; this answers it.

### Results

None yet. Record each result here with its date, the recording it used and the numbers.

## Order of work, each step with a pass mark

1. **Recorder:** synchronised frames from both cameras to disk. Needed for everything after,
   and it moves most of the work off test days.
2. **Calibration tool, offline on recorded frames.** Pass: guided error under 1 cm on both
   cameras.
3. **Live tag detector and SSL output.** Pass: beside ssl-vision, fewer dropouts and less
   jitter, no ID swaps, and corners as good as the centre.
4. **Ball detector.** Pass: no false balls on an empty field, and the ball tracked while a
   robot moves it.
5. Retire ssl-vision on lab days.

## Risks

- Tags blur on fast robots, and may be too few pixels at the far corners. Experiment 2's
  recordings show both early; the fixes are a shorter exposure or bigger tags.
- A lens that changes during a session (focus hunting) breaks the calibration mid-run; the
  movement watch (design 2) is meant to catch it.
- Lab-only: the team still needs ssl-vision and patterns working for competitions, so the
  patterns on the robots stay usable alongside the tags.

## Open questions

- Camera models, lenses and resolutions of both cameras (the RER webcam is 1280x960 MJPEG).
- Whether the cameras' focus and zoom are software controls (`v4l2-ctl`) that a startup script
  can set, even if they can't be locked physically.
- SSH access to the vision laptop, and which ssl-vision commit it runs.

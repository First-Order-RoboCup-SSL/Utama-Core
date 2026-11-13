# Real robots

The reference for real-robot work: what a field log holds and the tools that read it, and the
external software (grSim, the official GameController, AutoReferee, ssl-vision). The procedure for a
test day is the tutorial [tutorials/field_day.md](../tutorials/field_day.md).

## Field logs

Recorded on the physical pitch during real robot test days: vision frames, referee commands,
bidirectional radio packets, and session metadata.

---

### 1. File Location & Binary Format

- **Location**: `replays/field/<YYYYMMDD_HHMMSS>_real_<name>.log.gz`
- **Specification**: Follows the official RoboCup SSL Game Log format (`SSL_LOG_FILE`), stored compressed with standard `gzip`.
- **Streaming**: Python readers stream `.log.gz` directly without uncompressing:
  ```python
  from utama_core.replay.ssl_log import read_records

  for ts_ns, msg_type, payload in read_records("replays/field/<run>.log.gz"):
      ...
  ```

#### Record Types

| Type ID | Name | Contents |
|---|---|---|
| `4` | `VISION_2014` | Raw `SSL_WrapperPacket` from ssl-vision cameras (pixel positions, confidence, detection timestamps). |
| `3` | `REFEREE` | `Referee` state updates (commands, stage, score, team cards). |
| `100` | `RADIO_OUT` | 12-byte packed commands sent to robots over serial radio (forward, left, turn velocities, kick power, dribbler). |
| `101` | `RADIO_IN` | Feedback packets returned by robots (IR ball sensor trigger state). |
| `102` | `RUN_INFO` | Initial session metadata JSON: git commit SHA, dirty flag, vision-to-radio mapping, field bounds. |
| `103` | `NOTE` | Human operator notes entered from the dashboard during testing. |

---

### 2. Analysis & Calibration Tools

Every field log contains the ground-truth signals needed to evaluate robot hardware:

#### Robot Calibration (`tools/field/robot_calibration.py`)
Recovers physical dynamics and kicker geometry from a field test:
```bash
pixi run python tools/field/robot_calibration.py replays/field/<run>.log.gz
```
- **Loop Latency**: Sensing-to-action delay measured between commanded motion and camera-observed velocity (~61–62 ms).
- **Acceleration Limits**: Measured forward acceleration (~4.0–6.0 m/s²) vs. lateral acceleration (~2.7–3.6 m/s²).
- **Kick Characteristics**: Median ball launch velocity, directional offset bias (+2.7°), and bend caused by off-center dribbler contact (~4.2°/cm).
- **Dribbler Retention**: Sustained carry velocity and turn rate caps before the ball escapes (0.3 m/s, 0.6 rad/s).
- **Output**: Writes `<run>.calibration.json`.

#### Tag Alignment Calibration (`utama_core/analysis/tag_calibration.py`)
Measures optical tag placement relative to the chassis center:
```bash
pixi run python -m utama_core.analysis.tag_calibration replays/field/<run>.log.gz
```
- Fits circle trajectories during in-place spins to determine `(forward_m, left_m)` offsets.

#### Proximity & Close Calls (`utama_core/analysis/close_calls.py`)
Scans logs for incidents where robots came closer than 0.25 m:
```bash
pixi run python -m utama_core.analysis.close_calls replays/field/*real*.log.gz
```

#### Converting to Replay (`tools/realism/convert_ssl_log.py`)
Converts an SSL game log into our columnar `.npz` + `.pkl` replay format for replay visualization and scenario debugging:
```bash
pixi run python tools/realism/convert_ssl_log.py replays/field/<run>.log.gz --out replays/field/
```

---

### 3. Storage & Dataset Management

Raw field logs accumulate substantial disk space over a testing day. Three built-in tools manage log sizes:

#### 1. In-Place Compression
Raw `.log` files should be compressed to `.log.gz`:
```bash
pixi run package-field --compress-in-place
```
Reduces log sizes by **50%–60%** with zero data loss. All analysis tools support `.log.gz` out of the box.

#### 2. Trimming Idle Time Sectors (`tools/field/trim_idle_log.py`)
Field sessions frequently capture long periods of inactivity where the referee command is `HALT` or `STOP` (e.g. operators manually setting up the ball or discussing tests). Over **55% of field log duration is trailing dead time**:
```bash
# Preview reduction without modifying files
pixi run python tools/field/trim_idle_log.py replays/field/<run>.log.gz --dry-run

# Trim dead stretches and trailing halt in-place
pixi run python tools/field/trim_idle_log.py replays/field/<run>.log.gz --in-place
# Or batch trim all field logs:
pixi run trim-field
```
- Automatically removes idle pauses longer than 5 seconds while keeping 1.0 second safety padding around transitions.
- Fully strips trailing idle time recorded after drills end.
- Typically cuts file size by **65%–75%** while preserving every active drill frame.

#### 3. Packaging & Distributing (`tools/field/package_logs.py`)
Creates self-contained release archives of field logs and calibration results:
```bash
pixi run package-field
```
Bundles all `.log.gz` logs, `*.calibration.json`, and hardware profiles into `replays/field_logs_<timestamp>.tar.gz` for team sharing or dataset archival.

## External tools: grSim, GameController, AutoReferee, SSL Vision

None of this is needed for everyday work: rsim and the in-process
[`CustomReferee`](referee.md) run everything headless with no external process, and
behave the same across rsim, grSim and real mode. Set these up only to run against grSim, the
official competition software, or real robots.

### grSim

1. Go to the [grSim repo](https://github.com/RoboCup-SSL/grSim) and follow its
   [installation steps](https://github.com/RoboCup-SSL/grSim/blob/master/INSTALL.md).
2. Change the configuration to the values highlighted below:

   ![grsim_setup](../assets/images/grsim_setup.png)

3. To run it, execute `./bin/grSim` in the cloned repo.

### AutoReferee and the GameController

1. Make sure `grSim` is set up and can be called from a terminal.
2. `git clone` the [AutoReferee repo](https://github.com/TIGERs-Mannheim/AutoReferee) into a
   folder named `AutoReferee/` in the repository root.
3. Change `DIV_A` to `DIV_B` in `AutoReferee/config/moduli/moduli.xml`:

   ```xml
       <globalConfiguration>
           <environment>ROBOCUP</environment>
           <geometry>DIV_B</geometry>
       </globalConfiguration>
   ```

4. Get the latest [compiled game controller](https://github.com/RoboCup-SSL/ssl-game-controller/releases/),
   rename it to `ssl-game-controller`, and save it in a `ssl-game-controller/` directory in the
   repository root.

Both directories are gitignored.

#### Starting the external test environment

Only needed to test against the official referee software; the in-process `CustomReferee`
needs none of this. With grSim, the GameController and AutoReferee set up, start each in its
own terminal from the repository root:

```bash
grSim
cd ssl-game-controller && ./ssl-game-controller
cd AutoReferee && ./gradlew run
```

Then run your own strategy. The GameController's web UI is at http://localhost:8081/#/match
(this repo's dashboard is :8080).

### SSL Vision and the radio for real robots

The whole procedure (vision laptop, WSL networking and USB, radio, the checks before a robot
moves) is in [field_day.md](../tutorials/field_day.md).

---
name: field-session
description: Run a test day on the real robots, and process what it recorded - checking vision and radio, pairing robots with lids, calibrating a robot, then trimming, measuring and packaging the field logs. Use when the user is at the field, sets up real robots, sees dropouts or a robot misbehaving on the pitch, or brings back field logs. Not for simulated matches.
---

# A field session

Setup and the meaning of each check: `docs/real_robot_setup.md`. Log format and readers:
`docs/field_logs.md`. On the field, time is short: run the one command, read its "!!" lines,
fix only those.

## At the start

1. `pixi run field-check` (radios 0-7; `--radios 0 1` for fewer). It checks vision for 2 s
   (cameras, packets per second, field size, ball, lids), then finds the robots: each radio
   turns ~70° on the spot (keep 15 cm clear), the lid that turned is its, with the share of
   commands answered and whether its profile was calibrated on that lid. Every line is "ok" or
   "!!". It cannot see batteries or kicker charge.
2. On a "!!":
   - **Vision quiet or dropping**: check the network link before any code. Multicast over a
     2.4 GHz phone hotspot dropped frames on 2026-10-29. `pixi run python tools/vision_check.py
     --seconds 5` listens alone; `--guided ball` checks cameras against 9 known spots.
   - **Under 80% of commands answered**: range, interference or a flat battery.
   - **Lid swapped**: `pixi run find-robots`, and calibrate again (the profile's tag offset
     belongs to the lid).

## Measuring a robot

`pixi run calibrate --radio N` (65 s of driving, then the robot fetches a ball for the carry
ladder and kicks). It writes `config/robots/radio_<N>.json`; `--check` compares with the
committed profile, `--ball-only` redoes the ball part. Commit a new profile only when it
repeats. Known numbers: ~79 ms loop delay, sideways acceleration well below forward; carry
caps are 0.3 m/s and 0.6 rad/s on real robots (balls are lost turning faster).

## Things not to chase on the day

- Kick strength varies (battery, charge, many causes). The kicker is fixed power and holds its
  charge ~30 s; `team_controller/controllers/real/kicker_charge.py` keeps it up in play.
- Kick bend comes from the ball's offset on the dribbler (fitted per radio).
- The sim does not model a robot's shortfalls: robot capability is assumed league-level in sim,
  and real robots' gaps are compensated in the control layer, not in sim settings.

## After the session

1. **Keep the raw logs.** `replays/field/<YYYYMMDD_HHMMSS>_real_<name>.log.gz`. Never delete
   an untrimmed log or archive; ask before deleting anything under `replays/`.
2. **Trim dead time** to new files: `pixi run python tools/trim_idle_log.py
   replays/field/<run>.log.gz --dry-run`, then without `--dry-run` (`--in-place` only on copies).
3. **Register the session** in `tools/field_sessions.json`: its setup, and which measurements
   (`loop_delay`, `ball_roll`, `ball_noise`, `robot_noise`) it may be used for. Only registered
   sessions are read.
4. **Measure the world**: `pixi run python tools/field_physics.py [--json OUT]` prints each
   measurement per (day, radio) with its spread and whether the days agree. Stop rule: a number
   enters the sim (`tools/evaluation/reference/world.json`) only if it repeats across days, or
   its spread is deliberately used as the range. Changing `world.json` changes every match's
   environment: every stored result plays again, so batch it and ask first.
5. **One robot's numbers from a log**: `pixi run python tools/robot_calibration.py
   replays/field/<run>.log [--json OUT]`.
6. **Share**: `pixi run python tools/package_logs.py field [DIR] [--dry-run]` compresses the
   logs and bundles calibration and hardware config.
7. **Write down** what broke and what was measured (a dated note; open follow-ups listed).
   Bugs seen on the pitch go through the `debug-incident` skill with the field log as the
   replay (`docs/field_log_replay.md`).

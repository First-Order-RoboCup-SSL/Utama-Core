"""Draws rsim_tick.png for docs/sim_realism.md: one tick of rsim, where the seeded command-side noise and the
seeded vision noise sit between the strategy layer and the C++ physics.

pixi run python docs/img/sim_realism/make_rsim_tick.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagram_style import Diagram  # noqa: E402

# Every name a box shows; utama_core/tests/tools/test_diagram_names.py checks they still exist.
NAMES = [
    "StrategyRunner",
    "CustomReferee",
    "RSimController",
    "SSLStandardEnv",
    "SimNoise",
    "CommandDelay",
    "KickSpeedNoise",
    "KickAngleNoise",
    "MotionLimits",
    "RSimSSL",
    "RSimSubprocessWrapper",
    "set_physics_params",
    "_get_commands",
    "_frame_to_observations",
    "vision_rng",
    "utama_core/run/strategy_runner.py",
    "utama_core/custom_referee/custom_referee.py",
    "utama_core/team_controller/src/controllers/sim/rsim_controller.py",
    "utama_core/rsoccer_simulator/src/ssl/envs/standard_ssl.py",
    "utama_core/rsoccer_simulator/src/Utils/sim_noise.py",
    "utama_core/rsoccer_simulator/src/Simulators/rsim.py",
    "utama_core/rsoccer_simulator/src/Simulators/robosim/robosim_wrapper.py",
    "utama_core/rsoccer_simulator/src/Simulators/robosim/robosim_subprocess.py",
    "vendor/rSim/src/robosim",
]

d = Diagram(
    160,
    110,
    "One rsim tick: where the seeded noise goes",
    "Commands pass the command-side noise on the way in; observations pass the vision noise on the way out. "
    "The physics is the fork in vendor/rSim.",
    figsize=(14, 10),
)

# --- strategy layer (left) ------------------------------------------------------------------------
d.box(
    24,
    62,
    36,
    34,
    "StrategyRunner",
    "run/strategy_runner.py: _run_step\nstrategies, tactics, MotionController\n(strategy/, tactics/, motion_planning/)\n"
    "one RobotCommand per robot,\nsent through RSimController",
    kind="decision",
)
d.box(
    24, 22, 36, 14, "CustomReferee", "custom_referee/custom_referee.py\nstepped in process each tick", kind="decision"
)
d.arrow((18, 45), (18, 29), "frames", lx=-1.0, ha="right")
d.arrow((30, 29), (30, 45), "referee data", lx=1.0)

# --- command side (top) ---------------------------------------------------------------------------
d.box(
    74,
    84,
    50,
    22,
    "Command-side noise",
    "SSLStandardEnv, per tick, in order:\nCommandDelay: latency in whole ticks\n_get_commands: KickSpeedNoise, KickAngleNoise\n"
    "MotionLimits: top speed, acceleration, braking",
    kind="refiner",
)
d.arrow((42, 76), (49, 86))

# --- physics (top right) --------------------------------------------------------------------------
d.box(
    126,
    84,
    40,
    22,
    "rsim physics (C++)",
    "RSimSSL, RSimSubprocessWrapper\nrobosim_subprocess.py child process\nvendor/rSim/src/robosim",
    kind="source",
)
d.arrow((99, 86), (106, 86))

# --- vision side (right) --------------------------------------------------------------------------
d.box(
    126,
    36,
    40,
    24,
    "Vision-side noise",
    "_frame_to_observations, per tick:\ngaussian offsets on every detection\nvanishing: detections dropped\n"
    "SimNoise.vision_rng: a fresh seeded stream",
    kind="refiner",
)
d.arrow((126, 73), (126, 48), "frame", lx=1.0)
d.arrow((106, 36), (42, 52), "observations", lx=0.0, ly=2.0)

# --- per match, set at reset (bottom middle) ------------------------------------------------------
d.box(
    74,
    18,
    56,
    18,
    "Drawn once per match",
    "SimNoise.draw: each robot's latency and its kick\nspeed and angle offsets\n"
    "ball friction and substeps: set_physics_params,\nonce, when SSLStandardEnv is built",
    kind="refiner",
)

d.key(
    112,
    21,
    [
        ("source", "outside the process: the physics"),
        ("refiner", "noise on commands or observations"),
        ("decision", "strategy layer and referee"),
    ],
)
print(d.save(__file__, "rsim_tick.png"))

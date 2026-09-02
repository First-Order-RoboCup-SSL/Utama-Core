from utama_core.config.physical_constants import ROBOT_RADIUS


class trajsamplingconfig:
    ROBOT_RADIUS = ROBOT_RADIUS

    # Number of random intermediate targets tried when the direct trajectory
    # collides, per TIGERs 2024 champion paper section 2.2 ("Currently, we
    # use the random-based implementation with five targets").
    N_INTERMEDIATE_TARGETS = 5

    # How far (metres) an intermediate target is placed from the robot's
    # current position -- large enough to route around a robot-sized
    # obstacle plus margin, small enough that most candidates stay relevant
    # to the local situation rather than the whole field.
    INTERMEDIATE_TARGET_RADIUS = 1.0

    # Time (s) at which a two-segment candidate switches from the
    # intermediate-target sub-trajectory to a fresh trajectory toward the
    # real target -- paper section 2.2's t_intermediate = 200ms. Used as the
    # FIRST switch time tried, and the step between subsequent tries -- see
    # `_intermediate_targets`'s caller in `plan()`. Checked against TIGERs'
    # actual `SubPathCollisionChecker.findAcceptablePath`: they don't fix one
    # switch time per sampled direction, they slide it (`switchTime =
    # stepSizeOnSubPath + initialTimeOffset; ...; switchTime +=
    # stepSizeOnSubPath`) until a switch point is found where re-joining
    # toward the real target is collision-free, trying many switch times per
    # direction rather than one. Our first port fixed a single switch time
    # per direction, which is a materially weaker search -- a direction
    # whose only collision-free moment to turn back toward the target is,
    # say, 0.6s in rather than 0.2s in was simply never found.
    INTERMEDIATE_SWITCH_TIME = 0.2

    # How many switch times to try per intermediate-target direction before
    # giving up on that direction (mirrors their step loop bound implicitly
    # capped by `subTotalTime`; we cap explicitly since our first leg's
    # duration could in principle be long for a distant intermediate point).
    MAX_SWITCH_TIME_TRIES = 5

    # Collision-check time-stepping: paper section 2.3's adaptive step,
    # coarser far from any obstacle, capped finer near one.
    MAX_TIME_STEP = 0.1
    MIN_TIME_STEP = 0.02
    # A distance-to-step conversion factor: next step = clamp(closest
    # obstacle distance * this ratio, MIN_TIME_STEP, MAX_TIME_STEP). Chosen
    # so a robot at the robot's own top speed (~3 m/s) covers roughly
    # MIN_TIME_STEP worth of distance before the next check when right at
    # the safety margin boundary.
    STEP_DISTANCE_RATIO = 0.15

    # Dynamic safety margin: m = (min(v, v_max)/v_max)^2 * m_base -- paper
    # section 2.3 equation (4), same constants (v_max=3 m/s, m_base=0.2 m).
    MARGIN_V_MAX = 3.0
    MARGIN_BASE = 0.2

    # Opponent reachable-region lookahead cap (paper section 2.5's t_max).
    ENEMY_OBSTACLE_T_MAX = 0.5

    # How far ahead (s) collision-checking looks along a candidate
    # trajectory before accepting it outright -- checking the full
    # trajectory to a distant target is wasted effort once nothing relevant
    # can happen that far out (paper section 2.4: "caring about collisions
    # which happen one second in the future are just not worth to
    # consider").
    MAX_LOOKAHEAD_TIME = 1.5

    # Emergency-brake safety layer (paper section 2.6): a committed
    # trajectory is re-validated every tick (see `_try_reuse`), but between
    # one tick's "still safe" and the next tick's "now unsafe" there may not
    # be enough distance left to react by replanning alone -- two robots
    # can close distance faster than the planner's own per-tick obstacle
    # margin accounts for. Real hardware brakes harder than it accelerates
    # (paper: "our robots demonstrate a significantly faster braking
    # capability compared to acceleration"); this codebase has no separate
    # measured braking limit (`RobotParams` has one `MAX_ACCELERATION`), so
    # this multiplier stands in for that asymmetry as an explicit safety
    # margin rather than assuming symmetric accel/decel is enough.
    BRAKE_ACCELERATION_MULTIPLIER = 1.5

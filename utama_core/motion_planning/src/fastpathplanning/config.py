from utama_core.config.physical_constants import ROBOT_RADIUS


class fastpathplanningconfig:
    ROBOT_DIAMETER = 2 * ROBOT_RADIUS

    # how fat is the danger zone around obstacles, in multiples of robot diameter
    CLEARANCE_MULTIPLIER = 1.5
    OBSTACLE_CLEARANCE = ROBOT_DIAMETER * CLEARANCE_MULTIPLIER

    # How far outside the danger zone shold the waypoint be
    SUBGOAL_MULTIPLIER = 1.2
    SUBGOAL_DISTANCE = OBSTACLE_CLEARANCE * SUBGOAL_MULTIPLIER

    LOOK_AHEAD_RANGE = 3
    MAXRECURSION_LENGTH = 3
    PROJECTEDFRAMES = 20
    PROJECTION_DISTANCE = 1

    # Hysteresis margin for `check_segment`'s left/right detour choice, as a
    # fraction of `SUBGOAL_DISTANCE` (defined above, so this can't be set
    # before it — see the property below). Without this, the "closest
    # subgoal to target" heuristic re-decides from scratch every tick with no
    # memory of the previous choice: in a crowded, actively-moving obstacle
    # field (several robots' velocity-projected "ghost wall" segments — see
    # `_refresh_obstacle_cache` — all rotating slightly every tick from
    # ordinary station-keeping jitter), tiny per-tick shifts in which side
    # scores marginally closer flip the decision constantly, sending the
    # robot on a completely different detour every frame instead of
    # converging on any of them. Found live: a free-kick kicker in a crowded
    # defensive corner (5+ robots within ~1.5m) orbiting the ball forever,
    # confirmed via direct trace that the *target* was perfectly stable every
    # tick while the planner's returned waypoint swung wildly (e.g.
    # (-2.52,-1.90) one tick, (-3.78,-1.83) a few ticks later) — the same
    # class of instability `docs/roadmap.md` already flagged as a known,
    # unresolved `FastPathPlanning` convergence/local-minimum limitation.
    DETOUR_SWITCH_MARGIN_RATIO = 0.25

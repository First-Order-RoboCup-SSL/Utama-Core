"""RefereeGeometry: configurable field dimensions for the CustomReferee."""

import math
from dataclasses import dataclass

from utama_core.config.field_params import FieldBounds, FieldDimensions
from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.config.referee_constants import FREE_KICK_DEFENSE_AREA_DISTANCE


@dataclass(frozen=True)
class RefereeGeometry:
    """Immutable field geometry used by referee rule checkers.

    All measurements are in metres, using the standard SSL coordinate system
    (origin at centre, +x toward right goal, +y toward top of field).
    """

    half_length: float
    half_width: float
    half_goal_width: float
    half_defense_depth: float
    half_defense_width: float
    center_circle_radius: float
    goal_depth: float = 0.18  # depth of goal box behind the goal line (metres)

    @classmethod
    def from_field_dims(cls, field_dims: FieldDimensions) -> "RefereeGeometry":
        """Build geometry from a FieldDimensions instance.

        All goal/defense dimensions are taken from field_dims so that non-standard
        field sizes are fully supported.
        """
        return cls(
            half_length=field_dims.full_field_half_length,
            half_width=field_dims.full_field_half_width,
            half_goal_width=field_dims.half_goal_width,
            half_defense_depth=field_dims.half_defense_area_depth,
            half_defense_width=field_dims.half_defense_area_width,
            center_circle_radius=field_dims.center_circle_radius,
            goal_depth=field_dims.goal_depth,
        )

    # ------------------------------------------------------------------
    # Spatial query helpers
    # ------------------------------------------------------------------

    def is_in_field(self, x: float, y: float) -> bool:
        """True if (x, y) is within the playing field (including boundary)."""
        return abs(x) <= self.half_length and abs(y) <= self.half_width

    def is_in_left_goal(self, x: float, y: float) -> bool:
        """True if the ball has crossed the left goal line inside the goal."""
        return x < -self.half_length and abs(y) < self.half_goal_width

    def is_in_right_goal(self, x: float, y: float) -> bool:
        """True if the ball has crossed the right goal line inside the goal."""
        return x > self.half_length and abs(y) < self.half_goal_width

    def is_in_left_defense_area(self, x: float, y: float) -> bool:
        """True if (x, y) is inside the left defense area."""
        return x <= -self.half_length + 2 * self.half_defense_depth and abs(y) <= self.half_defense_width

    def is_in_right_defense_area(self, x: float, y: float) -> bool:
        """True if (x, y) is inside the right defense area."""
        return x >= self.half_length - 2 * self.half_defense_depth and abs(y) <= self.half_defense_width

    def distance_to_left_defense_area(self, x: float, y: float) -> float:
        """Distance from (x, y) to the nearest edge of the left defense area
        rectangle — 0.0 if already inside. Used by rules like
        `DefenseAreaStoppageRule` that need a standoff distance (rulebook
        §8.4.1: "0.2 meters distance to the opponent defense area"), not
        just an inside/outside check.
        """
        rect_max_x = -self.half_length + 2 * self.half_defense_depth
        dx = max(0.0, x - rect_max_x) if x > rect_max_x else 0.0
        dy = max(0.0, abs(y) - self.half_defense_width)
        return (dx * dx + dy * dy) ** 0.5

    def distance_to_right_defense_area(self, x: float, y: float) -> float:
        """Mirror of `distance_to_left_defense_area` for the right defense area."""
        rect_min_x = self.half_length - 2 * self.half_defense_depth
        dx = max(0.0, rect_min_x - x) if x < rect_min_x else 0.0
        dy = max(0.0, abs(y) - self.half_defense_width)
        return (dx * dx + dy * dy) ** 0.5

    # `legal_restart_position`'s clamp lands a point exactly `keep_dist` from
    # the raw defense-area edge -- precisely on `FastPathPlanner`'s own
    # obstacle-clearance line for that same edge (it uses the identical
    # `OPPONENT_DEFENSE_AREA_KEEP_DISTANCE` as its base margin). A target
    # sitting exactly on an obstacle line gets pushed further away by
    # `sanitize_target`'s own clearance ring before the delivering/kicking
    # robot ever gets a target to converge on -- but the referee state
    # machine's ball-placement-done / restart-legality checks still measure
    # against the *original*, un-pushed `designated_position`, which the
    # robot can now never actually reach. Traced live, 2026-09-05
    # (counter_flow_vs_tiki_taka_RK, BALL_PLACEMENT_YELLOW, 72.7s RESTART_STALL):
    # designated_position=(-3.25, 0.587) sat exactly on that obstacle line;
    # the carrier's carrot converged ~0.37m short of it and never closed the
    # last stretch. This buffer pushes the clamp result past the planner's
    # own clearance ring (see `fastpathplanningconfig.OBSTACLE_CLEARANCE`,
    # 0.27m at full clearance, shrinking to 0.8x = 0.216m in a crowded
    # scene) so `sanitize_target` is a no-op on an already-legal point
    # instead of displacing it further. Set just above the full (non-crowded)
    # 0.27m clearance rather than the crowded 0.216m floor, so the margin
    # holds regardless of how many robots happen to be nearby when this
    # position is chosen.
    _PLANNER_CLEARANCE_BUFFER_M = 0.28
    # The free-kick taker stands this far behind the ball (`DirectFreeOursStep.
    # _APPROACH_OFFSET`), towards the box whenever the kick points away from it, so
    # that spot has to clear the planner's ring too: a kick 0.54 m from the side edge
    # held its taker 1.3 m away (tournament_20260927_230330).
    _KICKER_APPROACH_M = ROBOT_RADIUS + 0.03

    def goal_kick_position(self, goal_x_sign: float, ball_y: float) -> tuple[float, float]:
        """SSL rulebook §6.2.1: a goal kick is placed "0.2 meters from the closest
        touch line and 1 meter from the goal line", in front of the goal on the
        `goal_x_sign` side and on the touch line nearer `ball_y`."""
        return (goal_x_sign * (self.half_length - 1.0), math.copysign(self.half_width - 0.2, ball_y))

    def legal_restart_position(self, x: float, y: float, keep_dist: float) -> tuple[float, float]:
        """Project (x, y) clear of BOTH defense areas (plus `keep_dist`), for
        use as a `DIRECT_FREE_*`/free-kick restart's `designated_position`.

        Any rule that derives a restart position directly from the ball's
        raw current position must run it through this first -- including a
        rule that first projects onto the field boundary
        (`OutOfBoundsRule`'s own boundary projection): that clamp alone is
        not sufficient, since the boundary offset is shallower than a
        defense area's depth and a ball going out near either goal line
        routinely projects to a point still inside it (found live,
        2026-09-04 -- `OutOfBoundsRule._nearest_infield_point` now chains
        into this method instead of assuming its own clamp was enough).
        Several rules exist specifically to fire *because* the
        ball is sitting inside a defense area (`KeeperHeldBallRule`, and
        `DefenseAreaRule`'s attacker-infringement branches) or can plausibly
        end up there (`ExcessiveDribblingRule`, `PushingRule`) -- an
        unprojected `designated_position` in that case is illegal the
        instant it's issued. `StrategyRunner`'s sim-mode shortcut (see its
        `_prev_custom_ref_command`-guarded STOP branch) teleports the ball
        straight to `designated_position` and force-starts play the same
        tick, so an illegal position there doesn't just look wrong on a
        scoreboard -- it lets the very defender/attacker that caused the
        violation instantly re-trigger it, deadlocking the match in a rapid
        STOP/FORCE_START churn. Confirmed live for `DefenseAreaRule`
        2026-09-04 (roadmap item 15/16): 6 churn cycles in under 2 seconds
        from exactly this gap. Only x needs clamping for either defense
        area (both are full-width rectangles spanning the goal line to
        `2*half_defense_depth` in) -- y is already legal by construction
        whenever a point clears the box on x alone. A point just past the box's
        side edge is within `clear_dist` of it too, so it is pushed out on x
        like one level with the box: a free kick left at the box corner
        (-4.25, -1.00) kept the kicker outside the planner's ring for the rest
        of the match (tournament_20260927_223257).
        """
        left_inner_x = -self.half_length + 2.0 * self.half_defense_depth
        right_inner_x = self.half_length - 2.0 * self.half_defense_depth
        # The rulebook's 1 m (§5.3.3) is wider than the planner's own margin.
        clear_dist = max(
            FREE_KICK_DEFENSE_AREA_DISTANCE,
            keep_dist + self._PLANNER_CLEARANCE_BUFFER_M + self._KICKER_APPROACH_M,
        )
        if abs(y) <= self.half_defense_width + clear_dist:
            if x <= left_inner_x + clear_dist:
                x = left_inner_x + clear_dist
            elif x >= right_inner_x - clear_dist:
                x = right_inner_x - clear_dist
        return (x, y)

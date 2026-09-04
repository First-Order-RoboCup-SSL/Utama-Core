"""RefereeGeometry: configurable field dimensions for the CustomReferee."""

from dataclasses import dataclass

from utama_core.config.field_params import FieldBounds, FieldDimensions


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

    def legal_restart_position(self, x: float, y: float, keep_dist: float) -> tuple[float, float]:
        """Project (x, y) clear of BOTH defense areas (plus `keep_dist`), for
        use as a `DIRECT_FREE_*`/free-kick restart's `designated_position`.

        Any rule that derives a restart position directly from the ball's
        raw current position (rather than an already-legal point, e.g.
        `OutOfBoundsRule`'s own boundary projection) must run it through
        this first. Several rules exist specifically to fire *because* the
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
        whenever a point clears the box on x alone.
        """
        left_inner_x = -self.half_length + 2.0 * self.half_defense_depth
        right_inner_x = self.half_length - 2.0 * self.half_defense_depth
        if abs(y) <= self.half_defense_width:
            if x <= left_inner_x + keep_dist:
                x = left_inner_x + keep_dist
            elif x >= right_inner_x - keep_dist:
                x = right_inner_x - keep_dist
        return (x, y)

from utama_core.config.field_params import STANDARD_FIELD_DIMS

BALL_KEEP_OUT_DISTANCE = 0.8
BALL_PLACEMENT_DONE_DISTANCE = 0.15
KICKER_READY_DISTANCE = 0.3
OPPONENT_DEFENSE_AREA_KEEP_DISTANCE = 0.25
# Standoff beyond ROBOT_RADIUS for an outfield defender's own defense-area
# edge target. A bare ROBOT_RADIUS margin (the robot's own footprint, zero
# tracking slack) let go_to_point's controller carry the robot into the box
# after "reaching" its target — measured via direct match trace: a defender
# approaching the box at speed (diagonal chase from a kickoff-ish spawn
# position) overshot its target by ~0.3m before decelerating, tripping the
# "too many defenders" foul even though the target itself was legal. 0.05m
# and 0.2m both measured as insufficient; 0.4m gives real headroom above
# the observed overshoot.
OWN_DEFENSE_AREA_STANDOFF_DISTANCE = 0.4
PENALTY_BEHIND_MARK_DISTANCE = 0.4
PENALTY_MARK_HALF_FIELD_RATIO = 0.5

PENALTY_LINE_Y_STEP_RATIO = 0.35 / STANDARD_FIELD_DIMS.full_field_half_width

KICKOFF_SUPPORT_POSITION_RATIOS_OWN_HALF = (
    (0.8 / STANDARD_FIELD_DIMS.full_field_half_length, 0.5 / STANDARD_FIELD_DIMS.full_field_half_width),
    (0.8 / STANDARD_FIELD_DIMS.full_field_half_length, -0.5 / STANDARD_FIELD_DIMS.full_field_half_width),
    (1.5 / STANDARD_FIELD_DIMS.full_field_half_length, 0.8 / STANDARD_FIELD_DIMS.full_field_half_width),
    (1.5 / STANDARD_FIELD_DIMS.full_field_half_length, -0.8 / STANDARD_FIELD_DIMS.full_field_half_width),
    (2.5 / STANDARD_FIELD_DIMS.full_field_half_length, 0.0),
)

KICKOFF_DEFENCE_POSITION_RATIOS_OWN_HALF = (
    (0.8 / STANDARD_FIELD_DIMS.full_field_half_length, 0.4 / STANDARD_FIELD_DIMS.full_field_half_width),
    (0.8 / STANDARD_FIELD_DIMS.full_field_half_length, -0.4 / STANDARD_FIELD_DIMS.full_field_half_width),
    (1.5 / STANDARD_FIELD_DIMS.full_field_half_length, 0.6 / STANDARD_FIELD_DIMS.full_field_half_width),
    (1.5 / STANDARD_FIELD_DIMS.full_field_half_length, -0.6 / STANDARD_FIELD_DIMS.full_field_half_width),
    (2.5 / STANDARD_FIELD_DIMS.full_field_half_length, 0.0),
    (1.5 / STANDARD_FIELD_DIMS.full_field_half_length, 0.0),
)

import math
from enum import Enum
from typing import NamedTuple

import numpy as np

from utama_core.config.field_params import FieldBounds, FieldDimensions
from utama_core.config.physical_constants import MAX_ROBOTS, ROBOT_RADIUS


class FormationEntry(NamedTuple):
    x: float
    y: float
    theta: float


# Starting positions for right team

#################### INSERT FORMATIONS HERE ######################
# Normalised right team formation that will be scaled,
# then mirrored for left.

# Anisotropic normalised scaling to field half-length, half-width
# e.g
# 0.75 = 3.375 (actual scale of x-coord) / 4.5 (field half-length)
# -0.06 = -0.18 (actual scale of y-coord) / 3.0 (field half-width)
##################################################################


class FormationType(Enum):
    START_ONE = "START_ONE"


FORMATIONS = {
    FormationType.START_ONE: [
        FormationEntry(0.9, 0.0, np.pi),
        FormationEntry(0.75, -0.2, np.pi),
        FormationEntry(0.75, 0.2, np.pi),
        FormationEntry(0.16, 0.0, np.pi),
        FormationEntry(0.16, 0.75, np.pi),
        FormationEntry(0.16, -0.75, np.pi),
    ],
}

################## END OF FORMATIONS ##########################


def _mirror(formation: list[FormationEntry], bounds: FieldBounds) -> list[FormationEntry]:
    cx, cy = bounds.center

    return [
        FormationEntry(
            2 * cx - entry.x,
            2 * cy - entry.y,
            (entry.theta + np.pi) % (2 * np.pi),
        )
        for entry in formation
    ]


def _scale(norm_formation, bounds: FieldBounds) -> list[FormationEntry]:
    x_min = bounds.top_left[0]
    x_max = bounds.bottom_right[0]
    y_max = bounds.top_left[1]
    y_min = bounds.bottom_right[1]

    L = (x_max - x_min) / 2
    W = (y_max - y_min) / 2

    cx, cy = bounds.center

    return [FormationEntry(cx + x * L, cy + y * W, theta) for x, y, theta in norm_formation]


def _validate_formation(
    formation: list[FormationEntry],
    bounds: FieldBounds,
) -> None:
    """
    Validate that the formation fits inside the bounding box and that robots do not overlap.
    The robot center may touch the edges.

    Raises ValueError if the formation cannot fit.

    Args:
        formation: List of (x, y, theta) tuples for robots.
        bounds: FieldBounds object with top_left and bottom_right coordinates.
        n_left: Expected number of robots on the left team.
        n_right: Expected number of robots on the right team.
    """
    # --- Bounding box edges (center can touch boundary) ---
    x_min = bounds.top_left[0]
    x_max = bounds.bottom_right[0]
    y_max = bounds.top_left[1]
    y_min = bounds.bottom_right[1]

    # --- Check bounds for each robot ---
    for i, (x, y, _) in enumerate(formation):
        if not (x_min <= x <= x_max):
            raise ValueError(f"Robot {i} x-position out of bounds: {x}, allowed: [{x_min}, {x_max}]")
        if not (y_min <= y <= y_max):
            raise ValueError(f"Robot {i} y-position out of bounds: {y}, allowed: [{y_min}, {y_max}]")

    # --- Check pairwise collisions ---
    n = len(formation)
    for i in range(n):
        x1, y1, _ = formation[i]
        for j in range(i + 1, n):
            x2, y2, _ = formation[j]
            dist = math.hypot(x1 - x2, y1 - y2)
            if dist < 2 * ROBOT_RADIUS:
                raise ValueError(
                    f"Could not fit all robots in provided FieldBounds/FieldDimensions. Robots {i} and {j} overlap (distance={dist:.3f})"
                )


# TODO: can consider a fitting algorithm that can optimise robot placement so that the chance of running out of space is reduced.


def get_formations(
    bounds: FieldBounds,
    n_left: int,
    n_right: int,
    formation_type: FormationType,
) -> tuple[list[FormationEntry], list[FormationEntry]]:
    """
    Returns the starting formations for both teams based on the provided field dimensions.
    The formations are defined as lists of FormationEntry objects, which contain the x and y coordinates

    Args:
        bounds: FieldBounds object defining the top-left and bottom-right corners of the field.
        n_left: Number of robots on the left team.
        n_right: Number of robots on the right team.
        formation_type: The type of formation to generate (e.g., START_ONE).

    Returns:
        tuple[list[FormationEntry], list[FormationEntry]]: A tuple containing two lists of FormationEntry objects.
                                                            left and right team formations respectively.
    """
    if n_left > MAX_ROBOTS or n_right > MAX_ROBOTS:
        raise ValueError(
            f"Number of robots per team cannot exceed {MAX_ROBOTS}. Got n_left={n_left}, n_right={n_right}."
        )
    if formation_type not in FORMATIONS:
        raise ValueError(f"Formation '{formation_type.value}' not found. Available: {list(FORMATIONS.keys())}")

    base = FORMATIONS[formation_type]
    if n_left > len(base) or n_right > len(base):
        raise ValueError(
            f"Formation '{formation_type.value}' only defines {len(base)} positions, but got n_left={n_left}, n_right={n_right}."
        )

    left = _mirror(_scale(base[:n_left], bounds), bounds)
    right = _scale(base[:n_right], bounds)

    _validate_formation(left, bounds)
    _validate_formation(right, bounds)

    return left, right

"""Checks that a field test is set up as configured, and the faults that must stop it.

Fed raw vision (every detection, before the roster and bounds filters), the vision
geometry packet and the radio's feedback rows, it answers the questions an operator
otherwise answers by walking the field: is every configured robot seen, under the ID
we expect, inside the part of the field in play? Is the camera sending? Is each robot
still answering on the radio? Any robot seen in play that we did not configure?

Each answer is a row with status "ok", "warn" or "fail". `StrategyRunner` shows the
rows on the dashboard's Live tab and, on real robots, forces HALT while any row
fails (see `halt_reason`), so a lost robot or a robot leaving the area stops play
whether or not anyone is watching the page.
"""

from __future__ import annotations

from typing import Iterable, Optional

from utama_core.config.field_params import FieldBounds, FieldDimensions
from utama_core.entities.data.raw_vision import RawVisionData

# A configured robot missing from vision this long fails: a robot we cannot see is
# driven on a stale position. On the 2026-10-27 field test (two cameras, 60-90 Hz) a
# robot's gaps were p99.9 50-100 ms; every longer one was our own loop stalling (see
# `StrategyRunner._tick_field_check`), not vision. 0.15 s until 2026-10-28, when a robot
# lost for 0.2 s halted a field run: the Kalman filter coasts it through such a gap.
VISION_LOST_S = 0.5
# No vision packet at all for this long fails (camera or network down). 0.15 s until 2026-10-28,
# raised with VISION_LOST_S.
CAMERA_SILENT_S = 0.5
# A gap between two ticks of the runner's loop longer than this is the loop stalling (it ticks
# every 6-17 ms on real robots).
LOOP_STALL_S = 0.1
# A robot this far outside the field bounds fails: the camera half is wrong, or the
# robot is leaving the area the planner keeps it in. Within it, a robot reaching
# over the line (a defender at the half-way line) is fine.
OUT_OF_BOUNDS_MARGIN_M = 0.3
# Vision geometry within this of the configured field matches (as the runner's own check).
GEOMETRY_TOLERANCE_M = 0.05

OK, WARN, FAIL = "ok", "warn", "fail"


def _row(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def _outside_by(bounds: FieldBounds, x: float, y: float) -> float:
    """How far (x, y) lies outside `bounds`, 0 inside."""
    (x1, y1), (x2, y2) = bounds.top_left, bounds.bottom_right
    dx = max(min(x1, x2) - x, 0.0, x - max(x1, x2))
    dy = max(min(y1, y2) - y, 0.0, y - max(y1, y2))
    return max(dx, dy)


class FieldCheck:
    """Tracks what vision and the radio last reported; `rows()` judges it.

    `expected` is every robot we drive, as (colour, vision ID) with colour "yellow"
    or "blue". Times are the runner's clock (`time.perf_counter()`), not vision's.
    """

    def __init__(self, expected: Iterable[tuple[str, int]], bounds: FieldBounds, field: FieldDimensions):
        self.expected = sorted(expected)
        self.bounds = bounds
        self.field = field
        self._robot_seen: dict[tuple[str, int], tuple[float, float, float]] = {}  # -> (time, x, y)
        self._ball_seen: Optional[tuple[float, float, float]] = None
        self._packet_times: list[float] = []
        self._geometry: Optional[tuple[float, float]] = None  # vision's (length, width), metres
        self._radio_answered: set[tuple[str, int]] = set()

    def observe_geometry(self, field_length_mm: float, field_width_mm: float) -> None:
        self._geometry = (field_length_mm / 1000.0, field_width_mm / 1000.0)

    def observe_vision(self, frames: Iterable[Optional[RawVisionData]], now: float) -> None:
        """Record this tick's new vision packets (None where a camera sent nothing)."""
        for frame in frames:
            if frame is None:
                continue
            self._packet_times.append(now)
            for colour, robots in (("yellow", frame.yellow_robots), ("blue", frame.blue_robots)):
                for r in robots:
                    self._robot_seen[(colour, r.id)] = (now, r.x, r.y)
            if frame.balls:
                b = max(frame.balls, key=lambda b: b.confidence)
                self._ball_seen = (now, b.x, b.y)
        # Keep one second of packet times, for the rate.
        while self._packet_times and now - self._packet_times[0] > 1.0:
            self._packet_times.pop(0)

    def rows(self, now: float, radio: Optional[list[dict]] = None) -> list[dict]:
        """One row per check. `radio` is the runner's robot feedback snapshot (real mode)."""
        rows = [self._camera_row(now), self._geometry_row()]
        radio_by_robot = {(r["team_color"], r["vision_id"]): r for r in radio or [] if "vision_id" in r}
        for colour, vision_id in self.expected:
            rows.append(self._robot_row(now, colour, vision_id))
            if radio is not None:
                rows.append(self._radio_row(colour, vision_id, radio_by_robot.get((colour, vision_id))))
        rows.extend(self._unexpected_rows(now))
        rows.append(self._ball_row(now))
        return rows

    def halt_reason(self, rows: list[dict]) -> Optional[str]:
        """The first failing row as one line, or None when nothing fails."""
        for row in rows:
            if row["status"] == FAIL:
                return f"{row['name']}: {row['detail']}"
        return None

    def _camera_row(self, now: float) -> dict:
        if not self._packet_times or now - self._packet_times[-1] > CAMERA_SILENT_S:
            return _row("Camera", FAIL, "no vision packets")
        return _row("Camera", OK, f"{len(self._packet_times)} packets/s")

    def _geometry_row(self) -> dict:
        want = (2 * self.field.full_field_half_length, 2 * self.field.full_field_half_width)
        if self._geometry is None:
            return _row("Field size", WARN, f"no geometry packet yet; configured {want[0]:.2f} x {want[1]:.2f} m")
        got = self._geometry
        if max(abs(got[0] - want[0]), abs(got[1] - want[1])) > GEOMETRY_TOLERANCE_M:
            return _row(
                "Field size", FAIL, f"vision {got[0]:.2f} x {got[1]:.2f} m, configured {want[0]:.2f} x {want[1]:.2f} m"
            )
        return _row("Field size", OK, f"{got[0]:.2f} x {got[1]:.2f} m")

    def _robot_row(self, now: float, colour: str, vision_id: int) -> dict:
        name = f"{colour} {vision_id}"
        seen = self._robot_seen.get((colour, vision_id))
        if seen is None:
            return _row(name, FAIL, "never seen by vision: check its pattern and the configured ID")
        t, x, y = seen
        if now - t > VISION_LOST_S:
            return _row(name, FAIL, f"lost by vision {now - t:.1f} s ago at ({x:.2f}, {y:.2f})")
        out = _outside_by(self.bounds, x, y)
        if out > OUT_OF_BOUNDS_MARGIN_M:
            return _row(name, FAIL, f"at ({x:.2f}, {y:.2f}), {out:.2f} m outside the area in play")
        return _row(name, OK, f"at ({x:.2f}, {y:.2f})")

    def _radio_row(self, colour: str, vision_id: int, feedback: Optional[dict]) -> dict:
        name = f"{colour} {vision_id} radio"
        key = (colour, vision_id)
        if feedback is not None and feedback["connected"]:
            self._radio_answered.add(key)
            return _row(name, OK, f"firmware {feedback['port_id']} answering")
        if key not in self._radio_answered:
            # Not a fault: the firmware may not send feedback at all.
            return _row(name, WARN, "no feedback yet")
        age = feedback["age_seconds"] if feedback is not None else None
        return _row(name, FAIL, "stopped answering" + (f" {age:.1f} s ago" if age is not None else ""))

    def _unexpected_rows(self, now: float) -> list[dict]:
        """Robots in play we do not drive: our planners cannot see them. Often a wrong ID."""
        expected = set(self.expected)
        rows = []
        for (colour, rid), (t, x, y) in sorted(self._robot_seen.items()):
            if (colour, rid) in expected or now - t > VISION_LOST_S or _outside_by(self.bounds, x, y) > 0:
                continue
            rows.append(_row(f"{colour} {rid}", WARN, f"in play at ({x:.2f}, {y:.2f}) but not configured"))
        return rows

    def _ball_row(self, now: float) -> dict:
        if self._ball_seen is None or now - self._ball_seen[0] > VISION_LOST_S:
            return _row("Ball", WARN, "not seen")
        _, x, y = self._ball_seen
        if _outside_by(self.bounds, x, y) > 0:
            return _row("Ball", WARN, f"at ({x:.2f}, {y:.2f}), outside the area in play")
        return _row("Ball", OK, f"at ({x:.2f}, {y:.2f})")

    def payload(self, rows: list[dict], auto_halt: Optional[str]) -> dict:
        """What the dashboard draws: the rows, the area in play, and the last auto-HALT."""
        (x1, y1), (x2, y2) = self.bounds.top_left, self.bounds.bottom_right
        return {
            "rows": rows,
            "bounds": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
            "margin": OUT_OF_BOUNDS_MARGIN_M,
            "auto_halt": auto_halt,
        }

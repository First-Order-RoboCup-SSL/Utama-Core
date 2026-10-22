"""The guided check of `tools/vision_check.py`: the spots it asks for and which ball it takes."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "vision_check", Path(__file__).resolve().parents[3] / "tools" / "vision_check.py"
)
vision_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vision_check)


def test_the_nine_spots_are_the_corners_edge_midpoints_and_centre():
    spots = vision_check.guide_points(3000, 2250)
    assert len(spots) == 9 and (0, 0) in spots
    assert {(-1500, 1125), (1500, 1125), (-1500, -1125), (1500, -1125)} <= set(spots)
    assert {(0, 1125), (0, -1125), (-1500, 0), (1500, 0)} <= set(spots)


def test_a_stray_ball_away_from_the_spot_is_left_out():
    # 2026-10-22: camera 0 also saw "a ball" 824 mm away from the real one.
    frames = [[(-365, -517), (1490, 1120)], [(1505, 1130)], [(-365, -517)]]
    assert vision_check.nearest_balls(frames, 1500, 1125) == [(1490, 1120, None), (1505, 1130, None)]
    assert vision_check.nearest_balls(frames, 1500, 1125, limit=vision_check.GUIDE_RADIUS) != []
    assert vision_check.mean_pose(vision_check.nearest_balls(frames, 1500, 1125)) == (1497.5, 1125, None)

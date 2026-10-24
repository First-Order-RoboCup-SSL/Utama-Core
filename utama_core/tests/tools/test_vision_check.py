"""The guided check of `tools/vision_check.py`: the spots it asks for and which ball it takes."""

import importlib.util
import select
import socket
import time
from pathlib import Path

from utama_core.team_controller.src.generated_code.ssl_vision_wrapper_pb2 import (
    SSL_WrapperPacket,
)

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


def _robot_at(x: float) -> bytes:
    packet = SSL_WrapperPacket()
    d = packet.detection
    d.frame_number, d.t_capture, d.t_sent, d.camera_id = 0, 0.0, 0.0, 0
    r = d.robots_blue.add()
    r.robot_id, r.x, r.y, r.confidence, r.pixel_x, r.pixel_y = 2, x, 0.0, 1.0, 0.0, 0.0
    return packet.SerializeToString()


def test_a_spot_is_measured_from_packets_after_enter_not_those_queued_while_walking():
    # Before, the guided check read the backlog first: the robot still at the last spot, or carried.
    receiver, sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM
    )
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(0.5)
    try:
        for _ in range(3):
            sender.sendto(_robot_at(-1500.0), receiver.getsockname())  # queued: the previous spot
        select.select([receiver], [], [], 1.0)
        time.sleep(0.05)  # all three delivered
        assert vision_check.drain(receiver) == 3
        sender.sendto(_robot_at(0.0), receiver.getsockname())
        heard = vision_check.listen(receiver, 0.3)
    finally:
        receiver.close()
        sender.close()
    assert [s[0] for s in heard["positions"][("blue 2", 0)]] == [0.0]

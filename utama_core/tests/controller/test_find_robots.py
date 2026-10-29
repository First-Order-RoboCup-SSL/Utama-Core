"""`find_robots` pairs each radio that answers with the lid that turns when it is told to turn,
and says when a robot wears a lid other than the one its profile was calibrated on."""

from __future__ import annotations

import json
import math

import pytest

from utama_core.entities.data.command import RobotResponse
from utama_core.team_controller.src.controllers.common.robot_profile import (
    RobotProfile,
    save_profile,
)
from utama_core.team_controller.src.controllers.real import find_robots as fr


class _Field:
    """Robots by radio, each wearing a lid, turning as commanded; a lid with no radio; a clock."""

    def __init__(self, lids: dict[int, fr.Lid], stray: fr.Lid = ("blue", 3)):
        self.lids = lids
        self.heading = {lid: 0.3 for lid in [*lids.values(), stray]}
        self.t = 0.0
        self._sent: dict[int, float] = {}

    # the radio
    def add_robot_commands(self, command, radio):
        self._sent[radio] = command.angular_vel

    def send_robot_commands(self):
        for radio, turn in self._sent.items():
            if radio in self.lids:
                self.heading[self.lids[radio]] += turn * fr.TICK_S

    def get_robots_responses(self):
        answers = [RobotResponse(radio, False) for radio in self._sent if radio in self.lids]
        self._sent = {}
        return answers

    # vision
    def read(self, _sock, into):
        for lid, h in self.heading.items():
            into.setdefault(lid, []).append(math.remainder(h, math.tau))

    # time
    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture
def field(monkeypatch):
    f = _Field({0: ("yellow", 0), 1: ("blue", 0)})
    monkeypatch.setattr(fr, "read_lids", f.read)
    return f


def test_each_answering_radio_is_paired_with_the_lid_that_turned_and_the_rest_reported(field):
    found = fr.find((0, 1, 2), controller=field, sock=object(), clock=field.clock, sleep=field.sleep)

    rows = {r["radio"]: r for r in found["robots"]}
    assert rows[0]["lid"] == ["yellow", 0] and rows[1]["lid"] == ["blue", 0]
    assert not rows[2]["answered"] and rows[2]["lid"] is None
    assert found["lids_seen"] == ["blue 0", "blue 3", "yellow 0"]
    # each robot turned back to about where it started
    assert field.heading[("yellow", 0)] == pytest.approx(0.3, abs=0.05)


def test_a_turn_smaller_than_turn_rad_is_not_the_probed_robot():
    before = {("yellow", 0): [0.0], ("blue", 0): [3.1]}
    after = {("yellow", 0): [fr.TURN_RAD - 0.01], ("blue", 0): [-3.1]}  # blue moved 0.08 rad across ±pi
    turned, how_far = fr.which_turned(before, after)
    assert turned == [] and how_far[("blue", 0)] == pytest.approx(2 * math.pi - 6.2)


def test_the_saved_pairing_reads_back_by_radio(tmp_path, field):
    path = fr.save(
        fr.find((0, 1, 2), controller=field, sock=object(), clock=field.clock, sleep=field.sleep),
        tmp_path / "found.json",
    )
    when, lids = fr.on_field(path)
    assert lids == {0: ("yellow", 0), 1: ("blue", 0)} and when is not None
    assert fr.on_field(tmp_path / "missing.json") == (None, {})


def test_the_report_says_when_a_robot_wears_another_lid_than_its_profiles(tmp_path, field):
    save_profile(0, RobotProfile(), {"lid": "yellow 0"}, tmp_path)
    save_profile(1, RobotProfile(), {"lid": "yellow 4"}, tmp_path)
    found = fr.find((0, 1, 2), controller=field, sock=object(), clock=field.clock, sleep=field.sleep)

    text = fr.report(found, tmp_path)

    assert "0      yellow 0   ok" in text
    assert "calibrated on lid yellow 4" in text
    assert "No answer from radio 2" in text and "no radio turned: blue 3" in text
    json.dumps(found)  # saveable as is

import dataclasses
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tools.evaluation import evaluate, match
from utama_core.bundle import archive, host, strategy_file, wire
from utama_core.config.enums import Mode
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.engine.match_log import MatchLog
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.data.vector import Vector2D, Vector3D
from utama_core.entities.game import Ball, Field, Game, GameFrame, GameHistory, Robot
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.motion_planning.src.common.control_schemes import get_control_scheme
from utama_core.replay.fingerprint import CodeGraph
from utama_core.strategy import catalog

STRATEGY = "pendulum"


def _robot(rid: int, friendly: bool, x: float, y: float) -> Robot:
    return Robot(rid, friendly, False, Vector2D(x, y), Vector2D(0.1, -0.2), Vector2D(0.0, 0.3), 0.1 + rid / 3)


def _frame(ts: float = 1.0) -> GameFrame:
    referee = RefereeData(
        source_identifier=None,
        time_sent=ts,
        time_received=ts,
        referee_command=RefereeCommand.NORMAL_START,
        referee_command_timestamp=0.0,
        stage=Stage.NORMAL_FIRST_HALF,
        stage_time_left=120.0,
        blue_team=TeamInfo("blue", score=2, yellow_card_times=[3]),
        yellow_team=TeamInfo("yellow", goalkeeper=0),
        designated_position=(0.5, -0.25),
        next_command=RefereeCommand.STOP,
    )
    return GameFrame(
        ts=ts,
        my_team_is_yellow=True,
        my_team_is_right=True,
        friendly_robots={i: _robot(i, True, 1.0 + i / 7, i / 3 - 0.5) for i in range(6)},
        enemy_robots={i: _robot(i, False, -1.0 - i / 7, 0.5 - i / 3) for i in range(6)},
        ball=Ball(Vector3D(0.2, 0.1, 0.0), Vector3D(1 / 3, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0)),
        referee=referee,
    )


def _field() -> Field:
    return host.decode_field(
        {
            "my_team_is_right": True,
            "dims": {
                "full_field_half_length": 4.5,
                "full_field_half_width": 3.0,
                "half_defense_area_depth": 0.5,
                "half_defense_area_width": 1.0,
                "half_goal_width": 0.5,
                "center_circle_radius": 0.5,
                "goal_depth": 0.18,
            },
            "top_left": [-4.5, 3.0],
            "bottom_right": [4.5, -3.0],
            "defense_area_rules": True,
        }
    )


def _over_the_wire(data):
    return json.loads(json.dumps(data))


def test_a_frame_comes_back_with_every_float_exact():
    frame = _frame()
    decoded = host.decode_frame(_over_the_wire(wire.encode_frame(frame)))
    assert [
        (r.id, r.is_friendly, r.has_ball, r.p, r.v, r.a, r.orientation) for r in decoded.friendly_robots.values()
    ] == [(r.id, r.is_friendly, r.has_ball, r.p, r.v, r.a, r.orientation) for r in frame.friendly_robots.values()]
    assert list(decoded.enemy_robots) == list(frame.enemy_robots)
    assert (decoded.ball.p, decoded.ball.v, decoded.ball.a) == (frame.ball.p, frame.ball.v, frame.ball.a)
    assert decoded.ts == frame.ts


def test_a_frame_comes_back_with_the_whole_referee():
    referee = host.decode_frame(_over_the_wire(wire.encode_frame(_frame()))).referee
    assert referee.referee_command == RefereeCommand.NORMAL_START
    assert referee.next_command == RefereeCommand.STOP
    assert referee.stage == Stage.NORMAL_FIRST_HALF
    assert referee.designated_position == (0.5, -0.25)
    assert (referee.blue_team.score, referee.blue_team.yellow_card_times) == (2, [3])


def test_a_frame_without_ball_or_referee_survives():
    frame = dataclasses.replace(_frame(), ball=None, referee=None)
    decoded = host.decode_frame(_over_the_wire(wire.encode_frame(frame)))
    assert decoded.ball is None and decoded.referee is None


def test_the_same_code_builds_the_same_bytes(tmp_path):
    first = strategy_file.build(STRATEGY, tmp_path / "a.strategy")
    second = strategy_file.build(STRATEGY, tmp_path / "b.strategy")
    assert first.read_bytes() == second.read_bytes()


def test_a_bundle_holds_its_code_and_no_tests(tmp_path):
    path = strategy_file.build(STRATEGY, tmp_path / "a.strategy")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
    assert {"utama_core/bundle/host.py", "utama_core/strategy/pendulum.py"} <= names
    assert not any(name.startswith("utama_core/tests/") for name in names)
    assert manifest["strategy"] == STRATEGY and manifest["format"] == strategy_file.FORMAT_VERSION


def test_a_strategy_that_does_not_exist_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nobody"):
        strategy_file.build("nobody", tmp_path / "a.strategy")


def test_a_frozen_file_is_named_after_its_commit_and_never_rebuilt(tmp_path):
    first = strategy_file.frozen(STRATEGY, "HEAD", tmp_path)
    assert first.name == f"pendulum@{strategy_file._commit_of('HEAD')[:8]}.strategy"
    first.write_bytes(b"kept")
    assert strategy_file.frozen(STRATEGY, "HEAD", tmp_path).read_bytes() == b"kept"


def test_an_annotated_tag_resolves_to_its_commit(tmp_path, monkeypatch):
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], cwd=tmp_path, check=True)
    subprocess.run([*git, "tag", "-a", "v1", "-m", "x"], cwd=tmp_path, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(strategy_file, "_REPO", tmp_path)
    assert strategy_file._commit_of("v1") == commit


def _rewritten(source: Path, output: Path, member: str, change) -> Path:
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(output, "w") as new:
        for name in old.namelist():
            data = old.read(name)
            new.writestr(name, change(data) if name == member else data)
    return output


def test_a_file_and_the_code_it_was_built_from_are_the_same_player(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "_CACHE", tmp_path / "cache")
    path = strategy_file.build(STRATEGY, tmp_path / "a.strategy")
    graph = CodeGraph()
    assert graph.strategy_fingerprint(path.as_posix()) == graph.strategy_fingerprint(STRATEGY)


def test_what_a_file_adds_around_the_code_is_not_part_of_the_player(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "_CACHE", tmp_path / "cache")
    path = strategy_file.build(STRATEGY, tmp_path / "a.strategy")
    other = _rewritten(path, tmp_path / "b.strategy", "utama_core/bundle/host.py", lambda d: d + b"\nX = 1\n")
    graph = CodeGraph()
    assert graph.strategy_fingerprint(other.as_posix()) == graph.strategy_fingerprint(path.as_posix())


@pytest.mark.parametrize(
    "member",
    [
        "utama_core/strategy/pendulum.py",
        "utama_core/skills/src/go_to_ball.py",
        "utama_core/motion_planning/src/common/control_schemes.py",
    ],
)
def test_a_file_of_different_code_is_a_different_player(tmp_path, monkeypatch, member):
    monkeypatch.setattr(archive, "_CACHE", tmp_path / "cache")
    path = strategy_file.build(STRATEGY, tmp_path / "a.strategy")
    other = _rewritten(path, tmp_path / "b.strategy", member, lambda d: d + b"\nX = 1\n")
    graph = CodeGraph()
    assert graph.strategy_fingerprint(other.as_posix()) != graph.strategy_fingerprint(path.as_posix())


def test_a_frozen_player_and_the_same_strategy_today_get_different_match_tags():
    assert match._short_name("replays/eval/frozen/pendulum@09493fd2.strategy") == "pendulum@09493fd2"
    assert match._short_name(STRATEGY) == "pendulum"


class _Commands:
    keeps_kickers_charged = False

    def __init__(self):
        self.sent = {}

    def add_robot_commands(self, command, robot_id):
        self.sent[robot_id] = command

    def send_robot_commands(self):
        pass


def _run(strategy) -> list[dict]:
    """Three ticks; the second and third depend on what the strategy kept from the first."""
    controller = _Commands()
    strategy.load_robot_controller(controller)
    strategy.load_motion_controller(get_control_scheme("fpp")(Mode.RSIM, None))
    game = Game(GameHistory(20), _frame(), field=_field())
    strategy.load_game(game)
    out = []
    for tick in range(3):
        game.add_game_frame(_frame(1.0 + (tick + 1) / 60))
        controller.sent = {}
        strategy.step()
        out.append(dict(controller.sent))
    return out


def test_a_bundled_strategy_sends_exactly_the_commands_of_the_same_strategy_in_process(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "_CACHE", tmp_path / "cache")
    bundled = strategy_file.BundledStrategy(strategy_file.build(STRATEGY, tmp_path / "a.strategy"))
    in_process = AbstractStrategy(catalog.builder(STRATEGY, (1, 2, 3, 4, 5)))
    assert _run(bundled) == _run(in_process)


def _setup_of(strategy, mode=Mode.REAL) -> dict:
    sent = []
    strategy._send = sent.append
    controller = get_control_scheme("fpp")(mode, None)
    controller.kick_bend, controller.kick_bias = {1: 0.5}, {2: -0.25}
    controller.robot_carry = {3: (0.3, 0.6)}
    controller.pose_ahead.latency_s = 0.079
    strategy.load_motion_controller(controller)
    return sent[0]


def test_real_robots_kick_calibration_reaches_the_files_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "_CACHE", tmp_path / "cache")
    strategy = strategy_file.BundledStrategy(strategy_file.build(STRATEGY, tmp_path / "a.strategy"))
    send = strategy._send
    setup = _setup_of(strategy)
    strategy._send = send  # for close()
    controller = host._controller(wire_round_trip(setup))
    assert controller.mode == Mode.REAL
    assert (controller.kick_bend, controller.kick_bias) == ({1: 0.5}, {2: -0.25})
    assert controller.robot_carry == {3: (0.3, 0.6)}
    assert controller.pose_ahead.latency_s == 0.079


def wire_round_trip(message: dict) -> dict:
    return json.loads(json.dumps(message))


class _KeepsCharged(_Commands):
    keeps_kickers_charged = True


def test_a_real_robots_kicker_charge_is_applied_to_a_files_commands(tmp_path, monkeypatch):
    """The charge step needs the real controller, so it stays with the runner: same commands as in process."""
    monkeypatch.setattr(archive, "_CACHE", tmp_path / "cache")
    monkeypatch.setattr(sys.modules[__name__], "_Commands", _KeepsCharged)
    bundled = strategy_file.BundledStrategy(strategy_file.build(STRATEGY, tmp_path / "a.strategy"))
    in_process = AbstractStrategy(catalog.builder(STRATEGY, (1, 2, 3, 4, 5)))
    assert _run(bundled) == _run(in_process)
    assert bundled._kicker_charge is not None


def test_a_file_can_play_with_the_robots_a_run_has(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "_CACHE", tmp_path / "cache")
    path = strategy_file.build(STRATEGY, tmp_path / "a.strategy")
    bundled = strategy_file.BundledStrategy(path, (1, 2))
    assert _run(bundled) == _run(AbstractStrategy(catalog.builder(STRATEGY, (1, 2))))


def _observed(strategy, path) -> dict:
    """Everything the runner reads from a strategy, one tick of each kind."""
    controller = _Commands()
    strategy.load_robot_controller(controller)
    strategy.load_motion_controller(get_control_scheme("fpp")(Mode.RSIM, None))
    log = MatchLog()
    kernel = strategy._kernel_strategy
    kernel.match_log = log
    game = Game(GameHistory(20), _frame(), field=_field())
    strategy.load_game(game)
    seen = {"commands": [], "committed": [], "partition": [], "names": [], "status": []}
    for tick in range(4):
        game.add_game_frame(_frame(1.0 + (tick + 1) / 60))
        # The runner's order: committed tactics are read on the new frame, before the step.
        seen["committed"].append(
            {tid: sorted(info["robots"]) for tid, info in kernel.slot_status(game).items() if info["committed"]}
        )
        controller.sent = {}
        strategy.step()
        seen["commands"].append(dict(controller.sent))
        partition = kernel.active_partition
        seen["partition"].append({tid: sorted(robots) for tid, robots in partition.items()})
        seen["names"].append({tid: type(kernel._tactics[tid]).__name__ for tid in partition})
        seen["status"].append(strategy.debug_status())
    log.to_jsonl(path)
    seen["log"] = Path(path).read_text()
    seen["keeper"] = kernel._pinned_robot_id if hasattr(kernel, "_pinned_robot_id") else None
    return seen


def test_the_runner_reads_the_same_from_a_bundled_strategy_as_from_the_same_strategy_in_process(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "_CACHE", tmp_path / "cache")
    bundled = strategy_file.BundledStrategy(strategy_file.build(STRATEGY, tmp_path / "a.strategy"))
    in_process = AbstractStrategy(catalog.builder(STRATEGY, (1, 2, 3, 4, 5)))
    seen_bundled = _observed(bundled, tmp_path / "bundled.jsonl")
    seen_in_process = _observed(in_process, tmp_path / "in_process.jsonl")
    assert seen_bundled["log"]  # the comparison below must not pass on two empty logs
    assert seen_bundled == seen_in_process


def test_a_strategy_file_is_named_like_any_strategy():
    names = evaluate.resolve_names(["split_shape", "replays/x@abc.strategy"])  # an old name works too
    assert names == [STRATEGY, "replays/x@abc.strategy"]


def test_a_bundle_whose_process_died_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "_CACHE", tmp_path / "cache")
    bundled = strategy_file.BundledStrategy(strategy_file.build(STRATEGY, tmp_path / "a.strategy"))
    bundled._process.kill()
    bundled._process.wait()
    with pytest.raises((RuntimeError, BrokenPipeError), match="exit code|Broken pipe"):
        bundled._send({"type": "tick"})
        bundled._reply()

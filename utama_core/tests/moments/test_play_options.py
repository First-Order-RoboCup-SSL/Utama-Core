"""Pure checks (no rsim) of `play`'s options in `tools/moments.py`: the seed parser, the events
list and the strategy resolver."""

import pytest

from tools import moments
from utama_core.entities.referee.referee_command import RefereeCommand

DT = 1 / moments.TICKS_PER_SECOND
STOP = RefereeCommand.STOP
THEIRS = RefereeCommand.DIRECT_FREE_BLUE  # the other team's restart, for us yellow
OURS = RefereeCommand.DIRECT_FREE_YELLOW


def _tick(i, command=None, ball=(0.0, 0.0), ours=None, ball_speed=None, foul=None):
    return moments.Tick(
        t=round(i * DT, 4),
        command=command,
        ball=ball,
        ours=ours or {},
        our_speeds={},
        our_foul=foul,
        ball_speed=ball_speed,
    )


def test_parse_seeds_takes_ranges_lists_and_both():
    assert moments.parse_seeds("0-4") == [0, 1, 2, 3, 4]
    assert moments.parse_seeds("0,2,3") == [0, 2, 3]
    assert moments.parse_seeds("0-1,3") == [0, 1, 3]
    assert moments.parse_seeds("2,2,1") == [2, 1]


@pytest.mark.parametrize("spec", ["4-1", "x", "1-", "", "0,,2", "-3"])
def test_parse_seeds_rejects_what_it_cannot_read(spec):
    with pytest.raises(ValueError):
        moments.parse_seeds(spec)


def test_events_lists_referee_changes_fouls_and_one_line_per_kick():
    ticks = [
        _tick(0, STOP, ours={1: (1.0, 0.0)}),
        _tick(1, STOP, ours={1: (1.0, 0.0)}),
        _tick(2, THEIRS, ours={1: (1.0, 0.0)}, foul=("keep_out", (3,))),
        _tick(3, THEIRS, ours={1: (1.0, 0.0)}, foul=("keep_out", (3,)), ball_speed=2.5),
        _tick(4, THEIRS, ours={1: (1.0, 0.0)}, foul=("keep_out", (3,)), ball_speed=2.4),
    ]
    lines = moments.events(ticks)
    assert any("referee: STOP -> DIRECT_FREE_BLUE" in line for line in lines)
    assert sum("foul: keep_out by robot(s) [3]" in line for line in lines) == 1
    kicks = [line for line in lines if "ball kicked" in line]
    assert len(kicks) == 1 and "2.50 m/s" in kicks[0]


def test_events_marks_entering_and_leaving_the_keep_out_circle_only_during_restarts_against_us():
    during_stop = [_tick(0, STOP, ours={2: (0.3, 0.0)}), _tick(1, STOP, ours={2: (0.3, 0.0)})]
    assert [line for line in moments.events(during_stop) if "keep-out" in line] == [
        "t=0.00s STOP: robot 2 enters the keep-out circle at 0.30 m"
    ]
    leaving = [_tick(0, STOP, ours={2: (0.3, 0.0)}), _tick(1, STOP, ours={2: (0.8, 0.0)})]
    assert any("robot 2 leaves the keep-out circle at 0.80 m" in line for line in moments.events(leaving))
    ours = [_tick(0, OURS, ours={2: (0.3, 0.0)}), _tick(1, OURS, ours={2: (0.3, 0.0)})]
    assert not any("keep-out" in line and "enters" in line for line in moments.events(ours))


def test_events_gives_each_robots_closest_approach_and_skips_ticks_without_a_ball():
    ticks = [
        _tick(0, ours={1: (1.0, 0.0)}),
        _tick(1, ours={1: (0.4, 0.0)}),
        _tick(2, ours={1: (0.7, 0.0)}),
        _tick(3, ball=None, ours={1: (0.0, 0.0)}),
    ]
    assert moments.events(ticks)[-1] == "robot 1 closest to the ball: 0.40 m at t=0.02s"


def test_events_counts_what_it_leaves_out_past_the_line_limit():
    ticks = [_tick(i, foul=("keep_out", (i % 2,))) for i in range(40)]
    lines = moments.events(ticks)
    assert len(lines) == moments.MAX_EVENT_LINES
    assert lines[-1] == "... 11 more"


def test_events_of_nothing_is_nothing():
    assert moments.events([]) == []


def test_factory_for_resolves_a_catalog_name_to_its_factory():
    from utama_core.strategy import kernel_strategy

    factory = moments.factory_for("build_split_shape_kernel_strategy")
    assert factory is kernel_strategy.build_split_shape_kernel_strategy


def test_factory_for_imports_a_factory_from_a_scratch_file(tmp_path):
    scratch = tmp_path / "mine.py"
    scratch.write_text(
        "from utama_core.strategy import kernel_strategy\n\n"
        "def build_mine_kernel_strategy(ids):\n"
        "    return kernel_strategy.build_split_shape_kernel_strategy(ids)\n"
    )
    name = f"{scratch}:build_mine_kernel_strategy"
    assert callable(moments.factory_for(name)(moments.OUTFIELD_ROBOT_IDS))
    assert moments._full_name(name) == name
    assert moments.label(name) == "mine:build_mine_kernel_strategy"


def test_factory_for_names_a_missing_file_a_missing_factory_and_an_unknown_strategy(tmp_path):
    with pytest.raises(moments.MomentError, match="no such file"):
        moments.factory_for(f"{tmp_path}/nope.py:build_x")
    scratch = tmp_path / "other.py"
    scratch.write_text("def something_else(ids):\n    return None\n")
    with pytest.raises(moments.MomentError, match="has no factory 'build_x'"):
        moments.factory_for(f"{scratch}:build_x")
    with pytest.raises(moments.MomentError, match="unknown strategy 'nope'"):
        moments.factory_for("build_nope_kernel_strategy")

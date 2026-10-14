"""`match.run_match` plays until the referee calls full time, with the sim-time
budget only as a cap. It used to step a fixed 600 s of sim time whatever the referee said."""

from __future__ import annotations

from types import SimpleNamespace

from tools.evaluation import match
from utama_core.entities.referee.stage import Stage


class _FakeRunner:
    """Stands in for `StrategyRunner`: the referee calls full time after `full_time_ticks`."""

    full_time_ticks = 50

    def __init__(self, *args, **kwargs):
        self.ticks = 0
        self.match_stats = None
        self.my = SimpleNamespace(game=SimpleNamespace(referee=self._referee(Stage.NORMAL_FIRST_HALF)))

    @staticmethod
    def _referee(stage):
        return SimpleNamespace(stage=stage, yellow_team=SimpleNamespace(score=2), blue_team=SimpleNamespace(score=1))

    def step_once(self):
        self.ticks += 1
        if self.ticks >= self.full_time_ticks:
            self.my.game.referee = self._referee(Stage.POST_GAME)
        _FakeRunner.last = self

    def close(self):
        pass


def test_a_match_ends_at_full_time_not_at_the_sim_time_cap(monkeypatch):
    monkeypatch.setattr(match, "StrategyRunner", _FakeRunner)

    result = match.run_match(
        "build_tiki_taka_kernel_strategy", "build_low_block_kernel_strategy", duration_seconds=1200.0
    )

    assert _FakeRunner.last.ticks == _FakeRunner.full_time_ticks
    assert (result.score_a, result.score_b) == (2, 1)


def test_the_cap_still_ends_a_match_that_never_reaches_full_time(monkeypatch):
    monkeypatch.setattr(_FakeRunner, "full_time_ticks", 10**9)
    monkeypatch.setattr(match, "StrategyRunner", _FakeRunner)

    match.run_match("build_tiki_taka_kernel_strategy", "build_low_block_kernel_strategy", duration_seconds=1.0)

    assert _FakeRunner.last.ticks == match.TICKS_PER_SECOND

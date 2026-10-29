"""`StrategyRunner.close` lets go of what the next runner in the same process needs
(`examples/demos.py`): the radio's serial port, after the robots' stop commands, and the vision
thread (stopped)."""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

from utama_core.config.enums import Mode
from utama_core.run.strategy_runner import StrategyRunner


def _runner(calls):
    runner = StrategyRunner.__new__(StrategyRunner)
    runner.__dict__.update(
        logger=logging.getLogger("test"),
        mode=Mode.REAL,
        profiler=None,
        match_log=None,
        match_stats=None,
        replay_writer=None,
        game_log=None,
        vision_stream=None,
        rsim_env=None,
        _serial_port=SimpleNamespace(close=lambda: calls.append("close serial")),
        _stop_event=threading.Event(),
        _stop_robots=lambda repeat: calls.append("stop robots"),
        _stop_fps_live=lambda: None,
    )
    return runner


def test_close_stops_the_robots_then_frees_the_radio_and_stops_vision():
    calls = []
    runner = _runner(calls)
    runner.close()
    assert calls == ["stop robots", "close serial"]
    assert runner._stop_event.is_set()


def test_stop_ends_the_run_loop_from_another_thread():
    runner = _runner([])
    runner.stop()
    assert runner._stop_event.is_set()

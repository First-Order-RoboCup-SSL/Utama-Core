"""`examples/demos.py`: plays the first demo, then each the page asks for, one runner at a time
(the last closed, and its dashboard routes dropped, before the next is built, so it can take the
radio); the settings set before each build; the field check between two runs of a demo; a demo
that fails to start waits for another pick; stopped otherwise (Ctrl+C), it ends."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
sys.path.insert(0, str(EXAMPLES))
try:
    import demos as ex
    import pass_receive
    import save_and_counter
    import score_goal
    import shooting_drill
    import teleop
finally:
    sys.path.remove(str(EXAMPLES))

from utama_core.dashboard.views.demo import DemoSwitcher


class _Server:
    def __init__(self, log):
        self.log = log

    def views(self):
        return "dashboard's own"

    def reset_views(self, views):
        self.log.append(f"reset to {views}")


class _Runner:
    def __init__(self, name, log, then=None):
        self.name, self.log, self.then, self.halted = name, log, then, True

    def stop(self):
        self.log.append(f"stop {self.name}")

    def run(self):
        self.log.append(f"run {self.name}")
        if self.then:
            self.then()  # what the page does while this demo plays
        self.log.append(f"closed {self.name}")


def _play(log, switcher, demos, first="a"):
    ex.play(
        "rsim",
        first,
        _Server(log),
        switcher,
        demos=demos,
        configure=lambda s: log.append(f"settings {s}"),
        check=lambda sw: log.append("check"),
    )


def test_each_demo_asked_for_is_built_after_the_last_closed_and_ctrl_c_ends_it():
    log, switcher = [], DemoSwitcher(["a", "b"], {"camera": ["full", "left"]})
    demos = {
        "a": lambda mode, server: _Runner("a", log, then=lambda: switcher.request("b", {"camera": "left"})),
        "b": lambda mode, server: _Runner("b", log),  # its run ends as Ctrl+C would
    }
    _play(log, switcher, demos)
    assert log == [
        "reset to dashboard's own",
        "settings {'camera': 'full'}",
        "run a",
        "stop a",
        "closed a",
        "reset to dashboard's own",
        "settings {'camera': 'left'}",
        "run b",
        "closed b",
    ]
    assert switcher.config()["running"] == "b" and switcher.config()["generation"] == 2


def test_the_field_check_runs_with_the_demo_closed_then_the_same_demo_again():
    log, switcher = [], DemoSwitcher(["a"], checks=True)
    runs = []

    def demo(mode, server):
        runs.append(1)
        return _Runner("a", log, then=switcher.request_check if len(runs) == 1 else None)

    _play(log, switcher, {"a": demo})
    assert [e for e in log if not e.startswith(("reset", "settings"))] == [
        "run a",
        "stop a",
        "closed a",
        "check",
        "run a",
        "closed a",
    ]


def test_a_demo_that_fails_to_start_waits_for_the_next_pick():
    log, switcher = [], DemoSwitcher(["a", "b"])

    def broken(mode, server):
        threading.Timer(0.05, switcher.request, args=("b",)).start()  # the page picks "b" while it waits
        raise SystemExit("radio 1 not found")

    _play(log, switcher, {"a": broken, "b": lambda mode, server: _Runner("b", log)})
    assert [e for e in log if e.startswith(("run", "closed"))] == ["run b", "closed b"]


def test_the_settings_set_every_examples_config_block():
    saved = {m: dict(vars(m)) for m in (shooting_drill, score_goal, pass_receive, save_and_counter, teleop)}
    try:
        ex.apply_settings({"first robot": "radio 1", "camera": "left", "rules": "on"})
        r0, r1 = shooting_drill.RADIO_0, shooting_drill.RADIO_1
        assert (shooting_drill.ATTACKER, shooting_drill.DEFENDER, score_goal.SHOOTER) == (r1, r0, r1)
        for m in (pass_receive, save_and_counter, teleop):
            assert (m.ROBOT_A, m.ROBOT_B) == (r1, r0)
        assert {m.CAMERA_HALF for m in (shooting_drill, score_goal, pass_receive, teleop)} == {"left"}
        assert shooting_drill._referee_profile().profile_name.endswith("(rules on)")
    finally:
        for m, attrs in saved.items():
            for k in ("ATTACKER", "DEFENDER", "SHOOTER", "ROBOT_A", "ROBOT_B", "CAMERA_HALF", "RULES"):
                if k in attrs:
                    setattr(m, k, attrs[k])


def test_the_defaults_are_the_examples_own_config():
    assert {k: v[0] for k, v in ex.SETTINGS.items()} == {
        "first robot": "radio 0" if shooting_drill.ATTACKER == shooting_drill.RADIO_0 else "radio 1",
        "camera": shooting_drill.CAMERA_HALF,
        "rules": shooting_drill.RULES,
    }

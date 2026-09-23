"""Hand-authored canonical bench scenarios.

These are the ~20 (4 to start) fixed anchors from roadmap item 14's bank v1
shape: scenarios that never move, don't depend on any match or replay, and
act as a stable reference when the harvested part of the bank turns over
between versions. Every scenario here has `ScenarioTrigger.HAND_AUTHORED`
and `anchor_tick=None` (see `bench_scenario.ScenarioProvenance`).

Coordinates follow the same plain pitch-frame metres as `scenario.py`
(friendly is always the +x/right side, matching `tournament.run_match`'s
`my_team_is_right=True` convention) — half length 4.5m, half width 3.0m
(`STANDARD_FIELD_DIMS`). Robot 0 is the goalkeeper on each side (pinned
separately by the kernel strategy, see `engine/strategy.py`); 1-5 are
outfield, matching `tournament_lib`'s `OUTFIELD_ROBOT_IDS`.

Run this module directly to print the static-screen result for every
scenario it defines (`pixi run python -m utama_core.replay.hand_authored_scenarios`).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.replay.bench_scenario import (
    BenchScenario,
    ScenarioFamily,
    ScenarioProvenance,
    ScenarioTrigger,
    static_screen,
)
from utama_core.replay.scenario import RobotState, Scenario

_HALF_LEN = 4.5
_HALF_WIDTH = 3.0


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _rs(id_: int, x: float, y: float, orientation: float = 0.0) -> RobotState:
    return RobotState(id=id_, x=x, y=y, orientation=orientation, vx=0.0, vy=0.0)


def _provenance(family: ScenarioFamily, perspective: str = "candidate_kicking") -> ScenarioProvenance:
    return ScenarioProvenance(
        source_run_id="hand_authored",
        evaluator_version=_git_revision(),
        trigger=ScenarioTrigger.HAND_AUTHORED,
        family=family,
        anchor_tick=None,
        source_replay=None,
        perspective=perspective,
    )


def _kickoff_center() -> BenchScenario:
    """Standard kickoff formation, candidate (friendly) taking the kick.
    Mirrors the default formation `StrategyRunner` constructs matches with."""
    friendly = (
        _rs(0, -_HALF_LEN + 0.3, 0.0),  # goalkeeper
        _rs(1, -0.5, 0.0, orientation=0.0),  # kicker, just behind center
        _rs(2, -1.5, 1.5),
        _rs(3, -1.5, -1.5),
        _rs(4, -3.0, 1.0),
        _rs(5, -3.0, -1.0),
    )
    enemy = (
        _rs(0, _HALF_LEN - 0.3, 0.0, orientation=3.14159),
        _rs(1, 1.5, 0.0, orientation=3.14159),
        _rs(2, 1.5, 1.5, orientation=3.14159),
        _rs(3, 1.5, -1.5, orientation=3.14159),
        _rs(4, 3.0, 1.0, orientation=3.14159),
        _rs(5, 3.0, -1.0, orientation=3.14159),
    )
    scenario = Scenario(
        sim_time=0.0,
        ball_x=0.0,
        ball_y=0.0,
        ball_vx=0.0,
        ball_vy=0.0,
        friendly_robots=friendly,
        enemy_robots=enemy,
        referee_command=RefereeCommand.NORMAL_START,
        source_replay=Path("hand_authored"),
        config_a_name=None,
        config_b_name=None,
        frame_ts=0.0,
    )
    return BenchScenario(
        scenario_id="kickoff_center_v1",
        scenario=scenario,
        provenance=_provenance(ScenarioFamily.KICKOFF),
    )


def _direct_free_defending_near_box() -> BenchScenario:
    """Enemy direct free kick just outside the candidate's own defense
    area — one of "the most important families" per the design discussion:
    defensive free kicks near the candidate's own box."""
    friendly = (
        _rs(0, -_HALF_LEN + 0.3, 0.0),
        _rs(1, -_HALF_LEN + 1.2, 0.6),  # wall/marker
        _rs(2, -_HALF_LEN + 1.2, -0.6),
        _rs(3, -1.0, 1.5),
        _rs(4, -0.5, -2.0),
        _rs(5, 1.0, 0.0),
    )
    enemy = (
        _rs(0, _HALF_LEN - 0.3, 0.0, orientation=3.14159),
        _rs(1, -_HALF_LEN + 2.2, 0.0, orientation=3.14159),  # kicker
        _rs(2, -1.5, 2.0, orientation=3.14159),
        _rs(3, -1.5, -2.0, orientation=3.14159),
        _rs(4, 0.0, 1.0, orientation=3.14159),
        _rs(5, 2.0, 0.5, orientation=3.14159),
    )
    scenario = Scenario(
        sim_time=0.0,
        ball_x=-_HALF_LEN + 2.2,
        ball_y=0.0,
        ball_vx=0.0,
        ball_vy=0.0,
        friendly_robots=friendly,
        enemy_robots=enemy,
        referee_command=RefereeCommand.DIRECT_FREE_BLUE,
        source_replay=Path("hand_authored"),
        config_a_name=None,
        config_b_name=None,
        frame_ts=0.0,
    )
    return BenchScenario(
        scenario_id="direct_free_defending_near_box_v1",
        scenario=scenario,
        provenance=_provenance(ScenarioFamily.DIRECT_FREE_DEFENDING, perspective="candidate_defending"),
    )


def _direct_free_attacking_near_box() -> BenchScenario:
    """Candidate's direct free kick just outside the enemy's defense area —
    the mirror image of the above; restarts are asymmetric so both
    perspectives are kept as separate scenarios (item 14)."""
    friendly = (
        _rs(0, -_HALF_LEN + 0.3, 0.0),
        _rs(1, _HALF_LEN - 2.2, 0.0),  # kicker
        _rs(2, 1.5, 2.0),
        _rs(3, 1.5, -2.0),
        _rs(4, 0.0, 1.0),
        _rs(5, -2.0, 0.5),
    )
    enemy = (
        _rs(0, _HALF_LEN - 0.3, 0.0, orientation=3.14159),
        _rs(1, _HALF_LEN - 1.2, 0.6, orientation=3.14159),  # wall/marker
        _rs(2, _HALF_LEN - 1.2, -0.6, orientation=3.14159),
        _rs(3, 1.0, 1.5, orientation=3.14159),
        _rs(4, 0.5, -2.0, orientation=3.14159),
        _rs(5, -1.0, 0.0, orientation=3.14159),
    )
    scenario = Scenario(
        sim_time=0.0,
        ball_x=_HALF_LEN - 2.2,
        ball_y=0.0,
        ball_vx=0.0,
        ball_vy=0.0,
        friendly_robots=friendly,
        enemy_robots=enemy,
        referee_command=RefereeCommand.DIRECT_FREE_YELLOW,
        source_replay=Path("hand_authored"),
        config_a_name=None,
        config_b_name=None,
        frame_ts=0.0,
    )
    return BenchScenario(
        scenario_id="direct_free_attacking_near_box_v1",
        scenario=scenario,
        provenance=_provenance(ScenarioFamily.DIRECT_FREE_ATTACKING, perspective="candidate_kicking"),
    )


def _open_play_3v2_counter() -> BenchScenario:
    """Candidate has a 3v2 numerical advantage on a counter-attack in the
    enemy half, ball loose just ahead of the lead attacker. Event-triggered
    family (numerical-advantage detector, item 14 sec 2), hand-authored here
    as a fixed anchor rather than harvested."""
    friendly = (
        _rs(0, -_HALF_LEN + 0.3, 0.0),
        _rs(1, 1.5, 0.0),  # lead attacker, on the ball
        _rs(2, 0.5, 1.8),
        _rs(3, 0.5, -1.8),
        _rs(4, -2.0, 0.0),
        _rs(5, -3.5, 1.0),
    )
    enemy = (
        _rs(0, _HALF_LEN - 0.3, 0.0, orientation=3.14159),
        _rs(1, 2.5, 0.5, orientation=3.14159),  # retreating defender
        _rs(2, 2.5, -0.5, orientation=3.14159),
        _rs(3, 3.5, 0.0, orientation=3.14159),
        _rs(4, 4.0, 1.5, orientation=3.14159),
        _rs(5, 4.0, -1.5, orientation=3.14159),
    )
    scenario = Scenario(
        sim_time=0.0,
        ball_x=1.7,
        ball_y=0.0,
        ball_vx=1.5,
        ball_vy=0.0,
        friendly_robots=friendly,
        enemy_robots=enemy,
        referee_command=RefereeCommand.FORCE_START,
        source_replay=Path("hand_authored"),
        config_a_name=None,
        config_b_name=None,
        frame_ts=0.0,
    )
    return BenchScenario(
        scenario_id="open_play_3v2_counter_v1",
        scenario=scenario,
        provenance=_provenance(ScenarioFamily.OPEN_PLAY_COUNTER, perspective="candidate_kicking"),
    )


def all_hand_authored_scenarios() -> tuple[BenchScenario, ...]:
    return (
        _kickoff_center(),
        _direct_free_defending_near_box(),
        _direct_free_attacking_near_box(),
        _open_play_3v2_counter(),
    )


if __name__ == "__main__":
    for bs in all_hand_authored_scenarios():
        result = static_screen(bs.scenario)
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {bs.scenario_id} ({bs.provenance.family.value})")
        for violation in result.violations:
            print(f"    - {violation}")

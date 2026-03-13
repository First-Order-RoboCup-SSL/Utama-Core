"""Visual RSim simulation for verifying referee command behaviour.

Run with:
    pixi run python utama_core/tests/referee/referee_sim.py

What it does:
  - Starts a 3v3 RSim with StartupStrategy as the base strategy.
  - The RefereeOverride tree (built automatically by AbstractStrategy) intercepts
    referee commands and overrides robot behaviour accordingly.
  - A scripted referee cycles through all referee commands every few seconds so
    you can watch how robots respond visually in the RSim window.

Command cycle (each held for SECS_PER_COMMAND seconds):
  1.  HALT                   → all robots stop immediately
  2.  STOP                   → all robots stop in place
  3.  TIMEOUT_YELLOW         → all robots idle
  4.  PREPARE_KICKOFF_YELLOW → robot 0 approaches centre, others fan out to own half
  5.  PREPARE_KICKOFF_BLUE   → all robots move to own-half defence positions
  6.  PREPARE_PENALTY_YELLOW → kicker at penalty mark, others line up behind
  7.  PREPARE_PENALTY_BLUE   → goalkeeper on goal line, others line up behind mark
  8.  DIRECT_FREE_YELLOW     → closest robot approaches ball
  9.  DIRECT_FREE_BLUE       → all robots stop
  10. BALL_PLACEMENT_YELLOW  → closest robot drives to designated position
  11. NORMAL_START           → pass-through: StartupStrategy runs freely
"""

import time

from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.team_info import TeamInfo
from utama_core.entities.referee.referee_command import RefereeCommand
from utama_core.entities.referee.stage import Stage
from utama_core.run import StrategyRunner
from utama_core.tests.referee.wandering_strategy import WanderingStrategy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECS_PER_COMMAND = 5.0  # seconds to hold each referee command before advancing
MY_TEAM_IS_YELLOW = True
MY_TEAM_IS_RIGHT = True
N_ROBOTS = 3  # robots per side

# Ball placement designated position (used for BALL_PLACEMENT_* commands)
DESIGNATED_POSITION = (1.0, 0.5)

# Goalkeeper robot ID used in PREPARE_PENALTY scenarios
GOALKEEPER_ID = 0

_COMMAND_SEQUENCE = [
    (RefereeCommand.NORMAL_START, "NORMAL_START — normal play (strategy free)"),
    (RefereeCommand.HALT, "HALT — all robots stop immediately"),
    (RefereeCommand.STOP, "STOP — all robots idle, keep ball distance"),
    (RefereeCommand.TIMEOUT_YELLOW, "TIMEOUT_YELLOW — team idle"),
    (RefereeCommand.PREPARE_KICKOFF_YELLOW, "PREPARE_KICKOFF_YELLOW — we kick off"),
    (RefereeCommand.PREPARE_KICKOFF_BLUE, "PREPARE_KICKOFF_BLUE — opponent kicks off"),
    (RefereeCommand.PREPARE_PENALTY_YELLOW, "PREPARE_PENALTY_YELLOW — our penalty"),
    (RefereeCommand.PREPARE_PENALTY_BLUE, "PREPARE_PENALTY_BLUE — opponent penalty"),
    (RefereeCommand.DIRECT_FREE_YELLOW, "DIRECT_FREE_YELLOW — our direct free kick"),
    (RefereeCommand.DIRECT_FREE_BLUE, "DIRECT_FREE_BLUE — opponent direct free kick"),
    (RefereeCommand.BALL_PLACEMENT_YELLOW, "BALL_PLACEMENT_YELLOW — we place the ball"),
]


# ---------------------------------------------------------------------------
# Scripted referee state machine
# ---------------------------------------------------------------------------


class _ScriptedReferee:
    """Cycles through _COMMAND_SEQUENCE, advancing every SECS_PER_COMMAND seconds."""

    def __init__(self):
        self._index = 0
        self._start = time.time()
        print("\n=== Referee Visualisation Simulation ===")
        print(f"Each command lasts {SECS_PER_COMMAND}s. Press Ctrl+C to stop.\n")
        self._print_current()

    def _print_current(self):
        cmd, desc = _COMMAND_SEQUENCE[self._index]
        print(f"  [{self._index + 1}/{len(_COMMAND_SEQUENCE)}] {desc}")

    def current_data(self) -> RefereeData:
        now = time.time()
        if now - self._start >= SECS_PER_COMMAND:
            self._index = (self._index + 1) % len(_COMMAND_SEQUENCE)
            self._start = now
            self._print_current()

        cmd, _ = _COMMAND_SEQUENCE[self._index]

        is_placement = cmd in (
            RefereeCommand.BALL_PLACEMENT_YELLOW,
            RefereeCommand.BALL_PLACEMENT_BLUE,
        )

        yellow_gk = GOALKEEPER_ID if MY_TEAM_IS_YELLOW else GOALKEEPER_ID
        blue_gk = GOALKEEPER_ID if not MY_TEAM_IS_YELLOW else GOALKEEPER_ID

        return RefereeData(
            source_identifier="scripted",
            time_sent=now,
            time_received=now,
            referee_command=cmd,
            referee_command_timestamp=now,
            stage=Stage.NORMAL_FIRST_HALF,
            stage_time_left=300.0,
            blue_team=_team_info(goalkeeper=blue_gk),
            yellow_team=_team_info(goalkeeper=yellow_gk),
            designated_position=DESIGNATED_POSITION if is_placement else None,
            blue_team_on_positive_half=False,
        )


def _team_info(goalkeeper: int = 0) -> TeamInfo:
    return TeamInfo(
        name="Demo",
        score=0,
        red_cards=0,
        yellow_card_times=[],
        yellow_cards=0,
        timeouts=2,
        timeout_time=300_000_000,
        goalkeeper=goalkeeper,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    scripted_referee = _ScriptedReferee()

    runner = StrategyRunner(
        strategy=WanderingStrategy(),
        my_team_is_yellow=MY_TEAM_IS_YELLOW,
        my_team_is_right=MY_TEAM_IS_RIGHT,
        mode="rsim",
        exp_friendly=N_ROBOTS,
        exp_enemy=N_ROBOTS,
        print_real_fps=True,
    )

    # Patch _run_step to push scripted RefereeData into ref_buffer before each
    # step. StrategyRunner._run_step now reads ref_buffer in RSim mode when
    # _frame_to_observations returns the standard 3-tuple.
    original_run_step = runner._run_step

    def _patched_run_step():
        runner.ref_buffer.append(scripted_referee.current_data())
        original_run_step()

    runner._run_step = _patched_run_step

    runner.run()


if __name__ == "__main__":
    main()

"""Classifies `RefereeCommand` transitions into the three reset severities.

This mapping is a judgment call, not something derivable purely from the
enum — documented here so the reasoning travels with the code instead of
living only in a design discussion.

Three tiers (see the "Tactics as Processes" design note for the full
argument):

1. Barrier reset — the previous tick's in-progress actions stop being
   meaningful for *everyone* at once. All tactics' `mem` resets and every
   `is_committed()` veto is overridden, unconditionally. This is not a targeted
   eviction of one stuck tactic; it's the game itself entering a new phase.
   Trigger: transitioning INTO a command that represents the start of a new
   phase of play — a kickoff/penalty/free-kick/ball-placement restart, or a
   goal. `NORMAL_START`/`FORCE_START` are the commands that end a barrier
   phase and hand control back to tactics with a clean slate; they are only
   a barrier-reset trigger when arriving directly from a barrier-tier
   command (see `classify_transition`), not on every occurrence (a
   long-running tactic issues no restart command mid-play, so if the
   command doesn't change there is nothing to classify).

2. Pause (SIGSTOP/CONT-style) — `HALT`/`STOP`. The game freezes and later
   resumes from the *same* state. `mem` and `is_committed()` must survive this
   untouched — a tactic mid-pass should pick up exactly where it left off
   once play resumes. The only thing that must change during a pause is that
   no tactic should be issuing live motion commands; that's a kernel-loop
   concern (skip ticking tactics while paused), not a `mem`/veto concern,
   and is handled by the `Strategy` loop itself, not by this module.

3. No reset — everything else (e.g. remaining on the same command tick over
   tick, or `TIMEOUT_*`, which pauses the clock but is not itself a restart
   of play). Deliberately conservative: if a command isn't recognized as a
   phase-starting restart, this module does not reset state for it.

`AbstractStrategy._REFEREE_STOPPAGE_COMMANDS` (the existing BT-path reset
list) resets on *every* stoppage command including plain `STOP`, once per
distinct `(command, timestamp)` token. That is a deliberately different,
more conservative choice than tier 2 here: this kernel's design explicitly
wants a plain pause to preserve `mem`, on the reasoning that a tactic mid
commitment should not lose its progress just because the referee paused
play without restarting it. This is a considered divergence from the
existing BT path, not an oversight — flagging it explicitly in case the two
mechanisms are ever compared or unified.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

from utama_core.entities.referee.referee_command import RefereeCommand

# Commands that represent the start of a new phase of play. Arriving at one
# of these (from a different previous command) means every tactic's
# in-progress action is moot: barrier reset.
_BARRIER_ENTRY_COMMANDS = frozenset(
    {
        RefereeCommand.PREPARE_KICKOFF_YELLOW,
        RefereeCommand.PREPARE_KICKOFF_BLUE,
        RefereeCommand.PREPARE_PENALTY_YELLOW,
        RefereeCommand.PREPARE_PENALTY_BLUE,
        RefereeCommand.DIRECT_FREE_YELLOW,
        RefereeCommand.DIRECT_FREE_BLUE,
        RefereeCommand.INDIRECT_FREE_YELLOW,  # deprecated, kept for completeness
        RefereeCommand.INDIRECT_FREE_BLUE,  # deprecated, kept for completeness
        RefereeCommand.BALL_PLACEMENT_YELLOW,
        RefereeCommand.BALL_PLACEMENT_BLUE,
        RefereeCommand.GOAL_YELLOW,  # deprecated, kept for completeness
        RefereeCommand.GOAL_BLUE,  # deprecated, kept for completeness
    }
)

# Commands that resume live play. Only treated as a barrier reset when the
# command immediately prior was itself a barrier-tier command (see
# docstring) — otherwise they are the normal "keep playing" tick-over-tick
# case and must not reset anything.
_RESUME_COMMANDS = frozenset({RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START})

# Commands that pause play without starting a new phase. `mem`/`is_committed()`
# must survive these untouched; only live command issuance should stop.
_PAUSE_COMMANDS = frozenset({RefereeCommand.HALT, RefereeCommand.STOP})


class ResetTier(Enum):
    NONE = auto()
    PAUSE = auto()
    BARRIER = auto()


def classify_transition(previous: Optional[RefereeCommand], current: RefereeCommand) -> ResetTier:
    """Classify a referee-command transition into a reset tier.

    `previous` is the command seen on the prior tick (`None` on the very
    first tick, treated as "no transition", i.e. `ResetTier.NONE` unless
    `current` is itself a barrier-entry command — starting a match already
    in a restart command is itself a fresh phase).
    """
    if current == previous:
        return ResetTier.NONE

    if current in _BARRIER_ENTRY_COMMANDS:
        return ResetTier.BARRIER

    if current in _RESUME_COMMANDS and previous in _BARRIER_ENTRY_COMMANDS:
        return ResetTier.BARRIER

    if current in _PAUSE_COMMANDS:
        return ResetTier.PAUSE

    return ResetTier.NONE


def is_paused(command: RefereeCommand) -> bool:
    """True while play is halted/stopped and no tactic should issue motion commands."""
    return command in _PAUSE_COMMANDS

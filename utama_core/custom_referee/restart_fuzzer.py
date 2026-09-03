"""RestartFuzzingReferee: opt-in referee subclass that injects scripted restarts.

Motivation (see `docs/STRATEGY_DEVELOPMENT.md` / `docs/custom_referee.md`):
rsim is deterministic, and the 231-pairing round-robin tournament always
starts from the same kickoff and only ever *reacts* to whatever restarts
naturally occur during a match — most of which never happen at all in a
given pairing/formation. Most stalls found in practice have been in referee
restart auto-advance paths (`GameStateMachine.step()`'s five `Auto-advance`
blocks in `state_machine.py`) that only get exercised when a real foul/goal
happens to occur. `RestartFuzzingReferee` exercises those code paths far
more often, in many more ball/robot geometries, by injecting additional
*legal* restarts at seeded-random sim times during otherwise-normal live
play.

This is deliberately a thin subclass, not a new referee: `CustomReferee`
already owns rule-checking, state transitions, and auto-advance; this class
only decides *when* and *what* to inject, then calls the same public
`set_command` API a human operator or scenario script would use (see
`utama_core/tests/strategy_runner/test_scenario_hooks.py`'s
`test_custom_referee_set_command_accepts_scripted_metadata`). Because
`set_command` inserts a real `STOP` before any set-piece command (see
`GameStateMachine.set_command`), an injected restart goes through exactly
the same `STOP -> queued restart -> NORMAL_START` auto-advance sequence a
naturally-detected foul/goal would, so a stall it exposes is indistinguishable
from — and just as real as — a naturally occurring one.

`StrategyRunner` never needs to know about this class: it only checks
`isinstance(self.referee, CustomReferee)` (see `strategy_runner.py`) before
calling `referee.step(...)`, `referee.attach_match_log(...)`, etc. — all of
which this subclass inherits or overrides compatibly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from utama_core.custom_referee.custom_referee import CustomReferee
from utama_core.custom_referee.profiles.profile_loader import load_profile
from utama_core.entities.data.referee import RefereeData
from utama_core.entities.game.game_frame import GameFrame
from utama_core.entities.referee.referee_command import RefereeCommand

# Commands that count as "live play" — the only state an injection may
# start from (mirrors OutOfBoundsRule/BallSpeedRule/DefenseAreaRule's own
# `_ACTIVE_PLAY_COMMANDS` gating, kept independently here since we do not own
# any of those rule files).
_LIVE_PLAY_COMMANDS = frozenset({RefereeCommand.NORMAL_START, RefereeCommand.FORCE_START})

# Rulebook margin (SSL §8.4.1: "keep at least 0.2 meters distance to the
# opponent defense area") reused from defense_area_stoppage_rule.py's
# `_MIN_DISTANCE` — not imported directly since that name is a private
# module constant of a file we don't own; duplicated as a literal with the
# same provenance noted here rather than reimplementing the geometry.
_DEFENSE_AREA_MARGIN = 0.2

# Delay (sim seconds) between the injected STOP and its FORCE_START
# follow-up for KIND_FORCE_START — long enough for `_all_robots_clear()`'s
# 0.5 m keep-out check (state_machine.py's `_BALL_CLEAR_DIST`) to plausibly
# be satisfied by the time FORCE_START lands, short enough to stay well
# under `_STOP_CLEAR_TIMEOUT_SECONDS` (15 s) so this follow-up is never
# mistaken for — or masks — the auto-advance stall detector it's meant to
# exercise coverage of.
_FORCE_START_STOP_DELAY_S = 2.0

# Field-line margin for a playable placement spot — mirrors
# OutOfBoundsRule._INFIELD_OFFSET (0.25 m), which OutOfBoundsRule itself
# picked "for a playable free-kick placement" against the same field-line
# geometry a fuzzed ball-placement target needs to respect.
_FIELD_LINE_MARGIN = 0.25

# Kinds of restart this fuzzer knows how to inject. Each is a self-contained
# scripted sequence built entirely from the public `set_command` API.
KIND_BALL_PLACEMENT_DIRECT_FREE = "ball_placement_direct_free"
KIND_FORCE_START = "force_start"
KIND_PREPARE_KICKOFF = "prepare_kickoff"

ALL_KINDS = (
    KIND_BALL_PLACEMENT_DIRECT_FREE,
    KIND_FORCE_START,
    KIND_PREPARE_KICKOFF,
)


@dataclass(frozen=True)
class Injection:
    """One record of a fuzzer-injected restart, kept on `injections` so a
    caller (validation script, test) can log/inspect what was injected."""

    sim_time: float
    kind: str
    team_is_yellow: Optional[bool]  # None for kinds with no team (none currently, kept for forward-compat)
    position: Optional[tuple[float, float]]  # designated_position, when the kind sets one


class RestartFuzzingReferee(CustomReferee):
    """`CustomReferee` subclass that periodically injects a scripted, legal
    restart during live play, on top of whatever the rule checkers detect
    naturally.

    Usage::

        referee = RestartFuzzingReferee.from_profile_name(
            "simulation", seed=1, interval_s=(8, 20), n_robots_yellow=6, n_robots_blue=6
        )
        runner = StrategyRunner(..., referee=referee)
        runner.run()
        for inj in referee.injections:
            print(inj)

    Args:
        profile: Same `RefereeProfile` `CustomReferee.__init__` takes.
        seed: RNG seed. Same seed (and same sequence of `step()` sim times)
            always produces the same injection schedule, kinds, teams, and
            positions — see `test_seed_determinism`.
        interval_s: `(min, max)` sim-seconds range between injections;
            each gap is drawn uniformly from this range using `seed`.
        kinds: Which injection kinds are enabled (subset of `ALL_KINDS`).
            Defaults to all three.
        n_robots_yellow, n_robots_blue: Passed through to `CustomReferee`.
    """

    def __init__(
        self,
        profile,
        seed: int,
        interval_s: tuple[float, float] = (8.0, 20.0),
        kinds: tuple[str, ...] = ALL_KINDS,
        n_robots_yellow: int = 3,
        n_robots_blue: int = 3,
    ) -> None:
        super().__init__(profile, n_robots_yellow=n_robots_yellow, n_robots_blue=n_robots_blue)
        if interval_s[0] <= 0 or interval_s[1] < interval_s[0]:
            raise ValueError(f"interval_s must be a positive, non-decreasing (min, max) pair, got {interval_s}")
        if not kinds:
            raise ValueError("kinds must be non-empty")
        unknown = set(kinds) - set(ALL_KINDS)
        if unknown:
            raise ValueError(f"unknown injection kind(s): {sorted(unknown)}")

        self._seed = seed
        self._interval_s = interval_s
        self._kinds = tuple(kinds)
        self._rng = random.Random(seed)

        # Scheduled sim-time of the next injection. Set lazily on the first
        # `step()` call once we know the clock's starting sim time (mirrors
        # how `CustomReferee.seed_clock()` defers timer alignment until the
        # first real game frame — we do the same here for the same reason:
        # rsim time and grsim/real wall-clock time start from very different
        # bases).
        self._next_injection_time: Optional[float] = None

        # Pending follow-up for the two-step STOP-then-FORCE_START sequence
        # (KIND_FORCE_START): the sim time at which to issue the follow-up
        # `set_command`, or None when no follow-up is pending. Deferred to a
        # later tick (rather than issued in the same `step()` call as the
        # STOP) so the STOP is a real, observable tick on its own — the same
        # shape a human GC operator or `strategy_runner.py`'s own
        # STOP-after-goal fast path uses — and so a stalled STOP->FORCE_START
        # transition is genuinely detectable rather than being collapsed
        # into a single atomic call.
        self._pending_force_start_at: Optional[float] = None

        # Pending follow-up for the KIND_BALL_PLACEMENT_DIRECT_FREE sequence:
        # `(placement_cmd, direct_free_cmd)` once the initial `set_command`
        # (queuing STOP -> placement_cmd) has been issued, or `None`. Consumed
        # the moment `step()` observes `referee_command == placement_cmd`
        # (i.e. GameStateMachine's own STOP auto-advance has actually flipped
        # into it), at which point `next_command` is overwritten to
        # `direct_free_cmd` so ball placement resumes into a real free kick
        # instead of straight to NORMAL_START. See `_inject`'s
        # KIND_BALL_PLACEMENT_DIRECT_FREE branch for why this two-step dance
        # is necessary rather than a single `set_command(..., next_command=)`
        # call.
        self._pending_direct_free: Optional[tuple[RefereeCommand, RefereeCommand]] = None

        self.injections: list[Injection] = []

    @classmethod
    def from_profile_name(
        cls,
        name: str,
        seed: int,
        interval_s: tuple[float, float] = (8.0, 20.0),
        kinds: tuple[str, ...] = ALL_KINDS,
        n_robots_yellow: int = 3,
        n_robots_blue: int = 3,
    ) -> "RestartFuzzingReferee":
        """Convenience constructor mirroring `CustomReferee.from_profile_name`."""
        profile = load_profile(name)
        return cls(
            profile,
            seed=seed,
            interval_s=interval_s,
            kinds=kinds,
            n_robots_yellow=n_robots_yellow,
            n_robots_blue=n_robots_blue,
        )

    # ------------------------------------------------------------------
    # Main loop interface
    # ------------------------------------------------------------------

    def step(self, game_frame: GameFrame, current_time: float) -> RefereeData:
        """Delegate to `CustomReferee.step()`, then decide whether to inject."""
        result = super().step(game_frame, current_time)

        # Fire any pending BALL_PLACEMENT_* -> DIRECT_FREE_* follow-up (see
        # `_pending_direct_free`) once the state machine has actually
        # transitioned into the queued placement command.
        if self._pending_direct_free is not None:
            placement_cmd, direct_free_cmd = self._pending_direct_free
            if result.referee_command == placement_cmd:
                self._pending_direct_free = None
                self._state.next_command = direct_free_cmd
                result = self._state._generate_referee_data(current_time)

        # Fire any pending STOP -> FORCE_START follow-up (see
        # `_pending_force_start_at`) before considering a brand new
        # injection, so the two never race within the same tick.
        if self._pending_force_start_at is not None and current_time >= self._pending_force_start_at:
            self._pending_force_start_at = None
            self.set_command(
                RefereeCommand.FORCE_START,
                timestamp=current_time,
                status_message="restart_fuzzer: injected FORCE_START",
            )
            # `set_command` mutates `self._state` directly but (unlike
            # `step()`) doesn't return a fresh `RefereeData` snapshot — ask
            # the state machine to regenerate one so this method's return
            # value reflects the FORCE_START we just issued, same as every
            # other caller of `CustomReferee.step()` expects a same-tick
            # snapshot. `_generate_referee_data` is a private read-only
            # accessor (no state mutation) on `GameStateMachine`, which we
            # don't own — used here rather than duplicating its trivial
            # dataclass-construction logic.
            result = self._state._generate_referee_data(current_time)

        if self._next_injection_time is None:
            self._schedule_next(current_time)
            return result

        if current_time < self._next_injection_time:
            return result

        if (
            not self._is_live_play(result)
            or self._pending_force_start_at is not None
            or self._pending_direct_free is not None
        ):
            # Not currently eligible (mid-restart, just after a goal / ball
            # not yet back in play, or a FORCE_START follow-up is already
            # pending) — don't burn the schedule; just wait for the next
            # tick and re-check. This means an injection can fire slightly
            # later than `_next_injection_time` if the game happens to be
            # mid-restart exactly then, but it will never fire *during* one.
            return result

        self._inject(game_frame, current_time)
        self._schedule_next(current_time)
        return result

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _schedule_next(self, current_time: float) -> None:
        gap = self._rng.uniform(*self._interval_s)
        self._next_injection_time = current_time + gap

    def _is_live_play(self, ref_data: RefereeData) -> bool:
        """True iff the game is in ordinary live play right now: command is
        NORMAL_START/FORCE_START (not mid-restart), and there is no
        just-scored-goal ball-placement queued (a goal always routes through
        `designated_position=(0.0, 0.0)` + a queued `BALL_PLACEMENT_*`/
        `PREPARE_KICKOFF_*` `next_command` per `GameStateMachine._handle_goal`
        — `next_command is None` during genuine live play, since it's only
        ever populated while a restart is queued/in-flight).
        """
        return ref_data.referee_command in _LIVE_PLAY_COMMANDS and ref_data.next_command is None

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def _inject(self, game_frame: GameFrame, current_time: float) -> None:
        kind = self._rng.choice(self._kinds)
        team_is_yellow = self._rng.choice([True, False])

        if kind == KIND_BALL_PLACEMENT_DIRECT_FREE:
            position = self._random_legal_position()
            placement_cmd = (
                RefereeCommand.BALL_PLACEMENT_YELLOW if team_is_yellow else RefereeCommand.BALL_PLACEMENT_BLUE
            )
            direct_free_cmd = RefereeCommand.DIRECT_FREE_YELLOW if team_is_yellow else RefereeCommand.DIRECT_FREE_BLUE
            # Chain BALL_PLACEMENT_* -> DIRECT_FREE_* -> NORMAL_START the same
            # way a real foul does (GameStateMachine._handle_foul routes a
            # foul's designated_position through ball placement first, then
            # `_post_ball_placement_command` resumes into the real restart
            # command once placement completes — see auto-advance 1 and 4 in
            # state_machine.py). `_post_ball_placement_command` has no public
            # setter, so it can't be reached through `set_command()` alone:
            # passing `next_command=direct_free_cmd` to `set_command()` would
            # silently overwrite the STOP-insertion's own `next_command =
            # placement_cmd` and skip ball placement entirely (verified via a
            # scratch rsim run: BALL_PLACEMENT_BLUE never appeared in the
            # observed command history when this was tried). Two separate
            # `set_command()` calls avoid that clobber: the first queues
            # BALL_PLACEMENT_* behind a STOP exactly like a real foul would;
            # once GameStateMachine's own auto-advance 1 flips STOP ->
            # BALL_PLACEMENT_* (next_command defaults to NORMAL_START at that
            # point per auto-advance 1's `self.next_command =
            # self._post_ball_placement_command or RefereeCommand.NORMAL_START`),
            # the second call (scheduled via `_pending_direct_free`, fired
            # from `step()` once BALL_PLACEMENT_* is actually observed)
            # overwrites `next_command` to the real DIRECT_FREE_* so auto-
            # advance 4 resumes into it instead of straight to NORMAL_START.
            self.set_command(
                placement_cmd,
                timestamp=current_time,
                designated_position=position,
                status_message="restart_fuzzer: injected ball placement",
            )
            self._pending_direct_free = (placement_cmd, direct_free_cmd)
        elif kind == KIND_FORCE_START:
            position = None
            # FORCE_START is not itself a member of `_NEEDS_STOP_FIRST`, so
            # `set_command` has no built-in way to queue it as a `next_command`
            # the way it does for BALL_PLACEMENT_*/PREPARE_KICKOFF_*/
            # DIRECT_FREE_*. Script the same two-step shape a human GC
            # operator (or `strategy_runner.py`'s own STOP-after-goal fast
            # path) uses instead: issue STOP now, then a genuine follow-up
            # `set_command(FORCE_START)` a few seconds later once robots have
            # had a chance to react to the STOP — scheduled via
            # `_pending_force_start_at` and fired from a later `step()` call
            # (see there) rather than in this same tick, so the STOP is a
            # real, independently observable transition and a stall in
            # either half of the sequence is genuinely detectable.
            self.set_command(
                RefereeCommand.STOP,
                timestamp=current_time,
                status_message="restart_fuzzer: injected STOP before FORCE_START",
            )
            self._pending_force_start_at = current_time + _FORCE_START_STOP_DELAY_S
        elif kind == KIND_PREPARE_KICKOFF:
            position = None
            kickoff_cmd = (
                RefereeCommand.PREPARE_KICKOFF_YELLOW if team_is_yellow else RefereeCommand.PREPARE_KICKOFF_BLUE
            )
            # Same metadata shape a real kickoff restart uses (no
            # designated_position — kickoff readiness is centre-circle
            # presence, not a placement target; see
            # GameStateMachine._kicker_in_centre_circle). set_command()
            # inserts STOP first since PREPARE_KICKOFF_* is in
            # `_NEEDS_STOP_FIRST`.
            self.set_command(
                kickoff_cmd,
                timestamp=current_time,
                status_message="restart_fuzzer: injected kickoff",
            )
        else:  # pragma: no cover - guarded by __init__'s kinds validation
            raise AssertionError(f"unhandled kind: {kind}")

        injection = Injection(
            sim_time=current_time,
            kind=kind,
            team_is_yellow=team_is_yellow,
            position=position,
        )
        self.injections.append(injection)

        if self._match_log is not None:
            self._match_log.trace(
                tick=self._match_log_tick,
                sim_time=current_time,
                key="restart_fuzzer_injection",
                value={
                    "kind": kind,
                    "team_is_yellow": team_is_yellow,
                    "position": list(position) if position is not None else None,
                },
            )

    def _random_legal_position(self) -> tuple[float, float]:
        """Draw a designated-placement position that is legal per the SSL
        rulebook: inside the field, outside both defense areas, and at
        least the rulebook margin from the field boundary lines.

        Reuses `RefereeGeometry`'s own inside-field/defense-area helpers
        (`is_in_field` is not used directly — sampling is done within a
        margin-shrunk field rectangle from the start, so every draw is
        legal by construction rather than rejection-sampled against
        `is_in_field`) plus `is_in_left/right_defense_area` to reject draws
        that land inside a defense area (rare, since defense areas sit at
        the ends of the field, but not impossible for a draw near the
        margin-shrunk rectangle's x-extremes).
        """
        geometry = self.geometry
        x_bound = geometry.half_length - _FIELD_LINE_MARGIN
        y_bound = geometry.half_width - _FIELD_LINE_MARGIN

        for _ in range(1000):
            x = self._rng.uniform(-x_bound, x_bound)
            y = self._rng.uniform(-y_bound, y_bound)
            if self._position_is_legal(x, y):
                return (x, y)
        # Practically unreachable (the margin-shrunk rectangle minus two
        # defense-area rectangles at its ends still covers most of the
        # field) — fall back to the centre spot, which is always legal on
        # any standard or shrunk field geometry.
        return (0.0, 0.0)

    def _position_is_legal(self, x: float, y: float) -> bool:
        """True iff (x, y) is inside the field with the field-line margin
        already baked into the caller's sampling rectangle, and at least
        `_DEFENSE_AREA_MARGIN` from both defense areas (rulebook §8.4.1's
        "0.2 meters distance to the opponent defense area", reused for
        *either* defense area — an own-defense-area placement is illegal
        for the same reason DefenseAreaRule keeps friendly robots out
        during live play). Uses `RefereeGeometry.distance_to_left/right_
        defense_area`, which is already 0.0 for a point inside the
        rectangle, so this rejects both "inside" and "too close" in one
        check per side.
        """
        geometry = self.geometry
        if geometry.distance_to_left_defense_area(x, y) < _DEFENSE_AREA_MARGIN:
            return False
        if geometry.distance_to_right_defense_area(x, y) < _DEFENSE_AREA_MARGIN:
            return False
        return True

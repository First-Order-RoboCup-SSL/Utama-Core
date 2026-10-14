"""Sim-only recovery from quirks of the native rsim engine, mixed into `StrategyRunner`
(`SimRecoveryMixin`): settling a teleported ball, holding a placement teleport off robots, and the
frozen-field watchdog. Each constant's rationale is in its comment below and on the method it serves."""

import math

from utama_core.config.enums import Mode
from utama_core.config.physical_constants import BALL_RADIUS, ROBOT_RADIUS
from utama_core.entities.referee.referee_command import RefereeCommand

# sim-only: `sim_controller.teleport_ball()` (used to instantly simulate a
# ball placement/free-kick reposition, since robots cannot physically
# retrieve an out-of-bounds ball in simulation) is meant to place the ball at
# rest at the designated position. Traced live in a tournament run
# (replays/tournament_20260905_002133/counter_flow_vs_zone_fluid_LK, 2026-09-05):
# the underlying native reset (`SSLWorld::replace()`) does correctly zero the
# ball's ODE body velocity at the instant of reset, but the very next physics
# step afterward produces a large spurious velocity (tens of m/s observed) —
# a contact-resolution artifact, not something teleport_ball's caller
# controls. That single spike is enough to send the ball rolling metres away
# from the placement target before it settles, so
# `GameStateMachine._ball_placement_done()` (gated purely on ball-to-target
# distance) never sees the ball close enough to auto-advance the restart —
# the match then sits in BALL_PLACEMENT_*/DIRECT_FREE_* for the rest of the
# match (RESTART_STALL in MatchStats.stall_events). Re-pinning the ball to
# the same target for a few ticks after every teleport absorbs that one-tick
# transient before it can accumulate into real displacement, without needing
# a fix in the vendored native engine.
_TELEPORT_SETTLE_TICKS = 3
# Ball speed (m/s) below which the settle window is allowed to close --
# i.e. the ball is considered genuinely at rest, not merely past the initial
# reset-artifact transient. The native engine's post-teleport spike (tens of
# m/s, see `_TELEPORT_SETTLE_TICKS`'s docstring) decays *through* a range of
# still-real, few-m/s speeds on its way to zero, and releasing the re-pin as
# soon as it first dips low is still enough for the ball to coast meters away
# over the following seconds before `_ball_placement_done()`'s distance gate
# can catch it -- a rolling SSL ball loses speed very slowly. Confirmed live,
# 2026-09-12: a teleport spike of ~13 m/s decayed to 0.83 m/s two ticks
# later (an earlier version of this constant released the window at any
# speed under 3.0 m/s, right where this case landed), and the ball then
# drifted, still slowing but never stopping, for the next ~14 seconds and
# ~2.5m before wedging against the field boundary in a corner -- the actual
# mechanism behind every "double-axis corner overshoot" RESTART_STALL traced
# this session (e.g. overload_flow_vs_tiki_taka's BALL_PLACEMENT_BLUE at
# t=24.2s), not a genuinely unreachable placement target. 0.05 m/s is
# comfortably above rsim's steady-state numerical noise floor (~1e-3 m/s
# once actually at rest, per _SIM_FROZEN_FIELD_VELOCITY_TOL_MPS's docstring)
# while still well below any speed that could carry the ball a visible
# distance before the next tick's re-pin.
_TELEPORT_SETTLE_SPEED_MPS = 0.05
# Cap on how many times the settle window can be extended by a detected
# spike, so a genuinely fast in-flight ball at the moment of a teleport
# (rather than the reset artifact) can't stall the re-pin loop indefinitely.
_TELEPORT_SETTLE_MAX_EXTENSIONS = 10
# sim-only: a BALL_PLACEMENT_* teleport is held back while any robot stands
# within this distance of the placement target. Teleporting the ball into a
# robot's body gets it shoved out by contact resolution faster than the
# settle re-pin can hold it: traced in
# tournament_20260924_082230/give_and_go_solo_vs_high_line_zone (t=37.1),
# where the ball landed 0.07m from a blue robot, was pushed off at ~1.5m/s
# and rolled into the goal net behind the line -- unreachable without
# entering the defense area, so DIRECT_FREE_YELLOW stalled for the rest of
# the match. The non-placing team clears the placement area during
# BALL_PLACEMENT_*, so the teleport fires as soon as the spot is free.
_PLACEMENT_TELEPORT_CLEARANCE_M = ROBOT_RADIUS + BALL_RADIUS + 0.02

# sim-only: recovery for a rare native-engine hang, distinct from the
# post-teleport velocity spike above. Traced live (tournament_20260905_083358,
# tiki_taka_vs_zone_fluid_Lk, 189s RESTART_STALL during DIRECT_FREE_BLUE):
# immediately after a mid-match `teleport_ball`-triggered `RSim.reset()`
# (from a BALL_PLACEMENT_BLUE placement), every robot AND the ball froze
# bit-for-bit (position std ~1e-13m, velocity exactly 0.0) for the rest of
# the match — the vendored native `robosim` subprocess kept returning valid
# JSON state each `step()` (no exception, no protocol desync), but the
# physics itself silently stopped advancing. Application-level command
# computation was verified correct and unrelated: the kicker's tactic, PID,
# and rotation math all produced real, correctly-directed, nonzero commands
# every tick throughout the stall — the freeze is entirely on the native
# engine's side, downstream of everything this codebase controls. Not
# reproducible live from any cold-teleported starting point (tried both
# mid-stall and pre-foul), so it can't be root-caused or fixed at the
# source — only detected and recovered from, the same trade-off already
# made for `_SIM_HALT_AUTO_RESUME_SECONDS` (no human GC in a sim tournament
# to intervene). If literally every robot and the ball report zero velocity
# for this long during live play (`current_command` neither paused nor mid-
# restart — those already legitimately hold everything still), force a
# fresh `teleport_ball` to the ball's own current position: `teleport_ball`
# always resets the *entire* frame (see `SSLBaseEnv.teleport_ball`'s
# docstring), not just the ball, so this is the cheapest way to re-issue a
# full-frame `RSim.reset()` and give the native engine another chance to
# resume advancing, without changing anyone's actual position.
_SIM_FROZEN_FIELD_AUTO_RECOVER_SECONDS = 8.0
# Exactly 0.0, not a small tolerance: ordinary rsim noise/refiner jitter
# never lands two consecutive ticks at bit-identical velocity, so this can
# only fire on the genuine native-engine freeze this exists to catch, never
# on a robot/ball that is merely moving slowly.
_SIM_FROZEN_FIELD_VELOCITY_TOL_MPS = 0.0


class SimRecoveryMixin:
    """`StrategyRunner` methods for the sim-only recovery above; they read and write runner state set in `__init__`."""

    def _teleport_ball_and_settle(self, x: float, y: float) -> None:
        """`sim_controller.teleport_ball(x, y)`, plus arming the re-pin window
        `_run_step` uses to counter the native engine's post-reset velocity
        spike (see `_TELEPORT_SETTLE_TICKS`'s module docstring).
        """
        self.sim_controller.teleport_ball(x, y)
        self._teleport_settle_target = (x, y)
        self._teleport_settle_ticks_left = _TELEPORT_SETTLE_TICKS
        self._teleport_settle_extensions_left = _TELEPORT_SETTLE_MAX_EXTENSIONS

    def _tick_pending_placement_teleport(self) -> None:
        """Fire the pending BALL_PLACEMENT_* teleport once no robot is within
        `_PLACEMENT_TELEPORT_CLEARANCE_M` of the target (see its docstring)."""
        if self._pending_placement_teleport is None:
            return
        x, y = self._pending_placement_teleport
        frame = self.my.current_game_frame
        robots = (*frame.friendly_robots.values(), *frame.enemy_robots.values())
        if any(math.hypot(r.p.x - x, r.p.y - y) < _PLACEMENT_TELEPORT_CLEARANCE_M for r in robots):
            return
        self._pending_placement_teleport = None
        self._teleport_ball_and_settle(x, y)

    def _tick_teleport_settle(self) -> None:
        """One tick of the re-pin window `_teleport_ball_and_settle` arms.

        Traced live in a tournament run (see `_TELEPORT_SETTLE_TICKS`'s
        docstring): the native rsim engine's post-teleport velocity spike
        doesn't decay with repeated resets — it simply waits for the first
        real physics step *without* a reset immediately before it, however
        many resets preceded it. So this doesn't just re-pin for a fixed
        number of ticks: it keeps extending the window (bounded by
        `_TELEPORT_SETTLE_MAX_EXTENSIONS`) for as long as the ball's most
        recent velocity still reads as unsettled, and only stops once a tick
        is observed with the window closed AND the ball has actually settled
        (`_TELEPORT_SETTLE_SPEED_MPS`'s docstring has the full rationale for
        why this needs to be a strict "at rest" check, not just "no longer
        looks like the initial reset spike").

        Reads the ball's velocity via `self.rsim_env.frame.ball` (the raw sim
        state, refreshed every physics `step()`), not `current_game_frame` —
        the latter is the *refined* frame `_step_game` builds once per tick,
        so at the point this runs (after the referee block, before
        `_step_game`) it is always one tick stale and would see last tick's
        already-corrected velocity instead of this tick's live spike.
        """
        if self.sim_controller is None or self._teleport_settle_target is None:
            return
        raw_ball = self.rsim_env.frame.ball if self.mode == Mode.RSIM and self.rsim_env is not None else None
        ball_speed = math.hypot(raw_ball.v_x, raw_ball.v_y) if raw_ball is not None else 0.0
        settled = ball_speed <= _TELEPORT_SETTLE_SPEED_MPS
        # The tick whose re-pin drove `_teleport_settle_ticks_left` to 0
        # still needs one more check here — its own `step()` (this tick,
        # further down in `_run_step`) is what actually produces or clears
        # the spike, and that only becomes observable next tick. So don't
        # drop `_teleport_settle_target` (and stop checking) until a tick is
        # seen with the window closed AND (settled OR out of extensions —
        # a ball that never actually settles must still terminate the loop
        # eventually rather than re-pin forever).
        if self._teleport_settle_ticks_left <= 0 and (settled or self._teleport_settle_extensions_left <= 0):
            self._teleport_settle_target = None
            return
        if not settled and self._teleport_settle_extensions_left > 0:
            self._teleport_settle_ticks_left += 1
            self._teleport_settle_extensions_left -= 1
        x, y = self._teleport_settle_target
        self.sim_controller.teleport_ball(x, y)
        self._teleport_settle_ticks_left = max(0, self._teleport_settle_ticks_left - 1)

    def _tick_frozen_field_watchdog(self) -> None:
        """Recover from the rare native-engine hang described in
        `_SIM_FROZEN_FIELD_AUTO_RECOVER_SECONDS`'s module docstring.

        Runs once per tick, after `_step_game` has refreshed
        `self.my.current_game_frame` for this tick. Only meaningful in sim
        (`Mode.HALT` has its own dedicated auto-resume timer above, and a
        real match has no `sim_controller` to recover through — `HALT` is
        additionally excluded on purpose, since it's the one live-play state
        where every robot legitimately holds zero velocity by design).
        """
        if self.sim_controller is None or self.mode != Mode.RSIM:
            return
        frame = self.my.current_game_frame
        if frame is None or frame.ball is None:
            self._frozen_field_since = None
            return
        referee = getattr(frame, "referee", None)
        current_command = getattr(referee, "referee_command", None) if referee is not None else None
        if current_command == RefereeCommand.HALT:
            self._frozen_field_since = None
            return

        tol = _SIM_FROZEN_FIELD_VELOCITY_TOL_MPS
        all_still = frame.ball.v.mag() <= tol and all(
            robot.v.mag() <= tol for robot in (*frame.friendly_robots.values(), *frame.enemy_robots.values())
        )
        if not all_still:
            self._frozen_field_since = None
            return

        if self._frozen_field_since is None:
            self._frozen_field_since = frame.ts
        elif frame.ts - self._frozen_field_since >= _SIM_FROZEN_FIELD_AUTO_RECOVER_SECONDS:
            self.logger.warning(
                "Frozen-field watchdog: every robot and the ball reported exactly zero "
                "velocity for %.1fs (native-engine hang, see _SIM_FROZEN_FIELD_AUTO_RECOVER_SECONDS) "
                "— forcing a full-frame reset via teleport_ball to recover.",
                frame.ts - self._frozen_field_since,
            )
            self.sim_controller.teleport_ball(frame.ball.p.x, frame.ball.p.y)
            self._frozen_field_since = None

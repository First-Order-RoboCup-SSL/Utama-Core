"""Switch-of-play attack tactic — deliberately relocate the ball to the weak
side before attacking, rather than combining on whichever side it already is.

New tactical logic, not a variation on an existing Core tactic. Every attack
tactic so far commits to goal (or to a lure) on whatever side of the field the
ball already happens to be: `PassAndShootTactic`/`GiveAndGoTactic` pass
between whoever is nearby, `LeadAndSupportTactic` drives straight at goal,
`DecoyOverloadTactic` drags a *single* marker sideways with a dribble. None of
them read the *global* left/right balance of the defense and reposition the
ball across the field's width in response. Real football does this
constantly: if the defense has collapsed toward the ball side, the correct
attacking option is often not to force something through a crowd but to move
the ball to the side with fewer defenders, where a well-timed run into space
turns into an easier chance than anything available on the strong side.

Three roles, an intentionally *directional* relay rather than the freeform
"whoever is closest becomes the passer" pattern the other tactics use:

- **carrier** — starts with (or collects) the ball centrally.
- **pivot** — the near-side outlet the carrier's first pass goes to; stands
  roughly central/behind the ball so the first pass is short and safe, not
  itself a scoring threat.
- **runner** — advances into the weak side (the flank with fewer/farther
  enemy robots, decided once per possession — see `_weak_side` below) while
  the carrier->pivot pass is happening, so that by the time the pivot relays
  the ball on, the runner is already arriving into open space rather than
  standing still waiting to be found.

Phases: `"assess"` (carrier gets/holds the ball, weak side gets picked, pivot
and runner move into their positions) -> `"switch"` (carrier passes to pivot)
-> `"relay"` (pivot one-touches/relays to runner, who by now should be well
into the weak-side space) -> `"finish"` (runner shoots, or the whole thing
times out back to `"assess"` if any leg stalls). This is a genuinely 3-robot,
multi-tick phase machine (not a single 2-robot pass) and the weak-side choice
is exactly the kind of live, continuously-recomputed sensor comparison
AGENTS.md's "Writing a Tactic" guidance (point 3) warns needs hysteresis: the
left/right defender balance can be nominally tied lots of ticks in a row, and
flip-flopping it would repeatedly redirect the runner mid-run.

Reuses `_pass_and_score`'s `_pass_exec`/`_score_goal` for the two legs of the
relay and the final shot (generic two-robot ball transfer / shot-taking, same
reuse rationale every other attack tactic here gives), and
`shared/pass_and_score_geometry` for the weak-side read and shot check. What's
new is only the weak-side decision, the pivot/runner positioning, and the
3-leg phase sequence layered on top.

Three-robot minimum in spirit (a carrier, a pivot to receive the switch, and
a runner to arrive in the vacated space); with exactly 2 assigned robots the
tactic degrades to a direct carrier->runner pass with no pivot leg (skips
"switch", goes straight to "relay") rather than inventing a role split no
concrete case has asked for. With 1 robot it is a plain dribble/shoot,
matching every other tactic's single-robot degradation here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utama_core.config.physical_constants import ROBOT_RADIUS
from utama_core.config.referee_constants import OWN_DEFENSE_AREA_STANDOFF_DISTANCE
from utama_core.engine.context import TickContext
from utama_core.engine.tactic import BaseTactic, RobotId, TacticId, TacticTag
from utama_core.entities.data.command import RobotCommand
from utama_core.entities.data.vector import Vector2D
from utama_core.entities.game import Game
from utama_core.shared.pass_and_score_geometry import (
    at_target,
    ball_line_receive_point,
    enemy_goal_line,
    enemy_positions,
    find_best_shot,
    has_ball,
    segment_blocked,
)
from utama_core.skills.src.go_to_ball import go_to_ball
from utama_core.skills.src.go_to_point import go_to_point
from utama_core.skills.src.utils.move_utils import move
from utama_core.tactics._pass_and_score import _pass_exec, _score_goal

# Guidance point 3: a naive "count enemies on each half of the field" read
# would flip which side counts as "weak" almost every tick under ordinary
# simulator position jitter whenever the split is close to even — and a flip
# mid-possession would yank the runner from one flank to the other, so it
# never arrives anywhere before being redirected. Require a real margin (not
# just a strictly-greater count) before switching sides, and once a side is
# picked for a possession it is not re-read again until the next possession
# starts (see `mem.weak_side`, set once in "assess" and left alone after).
_WEAK_SIDE_MARGIN = 1  # enemies — the more-open side must have at least this many fewer

# Guidance point 1: every phase this tactic drives is_committed() through
# needs a tick-budget so a stalled leg (pass never lands, runner never gets
# into a clean shot lane, etc.) resets back to "assess" instead of stranding
# these robots forever.
_PHASE_TIMEOUT_TICKS = 600  # ~10s at 60Hz, one full leg should never need this long
_LANE_BLOCKED_ABANDON_TICKS = 30  # ~0.5s at 60Hz — sustained-block bar, not one noisy tick

# "assess" resets `phase_ticks` to 0 on every tick it runs (see the phase
# dispatch below), so `_PHASE_TIMEOUT_TICKS` never bounds it at all -- unlike
# "switch"/"relay"/"finish", "assess" can run forever. In isolation that's
# fine (it's meant to hold until the carrier is ready), but the carrier's
# has_ball(visual=True) check below has no grace period: rsim's dribble
# physics can eject/reacquire the ball for a single tick with no tactic-level
# cause (same quirk _BALL_RECOVERY_RADIUS's comment documents for "relay"),
# and a bare check sends the carrier straight into go_to_ball on that one
# flicker, discarding its held position. Since `_pivot_target()` is a
# function of the carrier's own *live* position, every such flicker also
# drags the pivot's target along with the carrier's chase -- a non-convergent
# feedback loop, not just a wasted tick. Found live
# (high_line_zone_vs_high_press, COMMITTED_FROZEN, t=45.1s): the carrier
# visibly walked ~1m chasing the ball over the course of "assess", the pivot
# target sliding the same distance in lockstep, `_debounced_settled` never
# converging because the target itself never stopped moving. Same style grace
# period as `_pass_and_score.py`'s `_SETUP_BALL_LOSS_GRACE_TICKS`.
_CARRIER_BALL_LOSS_GRACE_TICKS = 10  # ~0.17s at 60Hz — matches _SETUP_BALL_LOSS_GRACE_TICKS

_RUNNER_DEPTH_FRACTION = 0.55  # how far up the weak flank the runner advances (fraction of half_length from centre)

# A receiving robot's readiness to start a pass leg needs both position AND
# velocity to have settled, not just position. `at_target`'s tolerance alone
# let the pivot/runner into "switch"/"relay" while still moving at real
# speed (observed: dist=0.249m — inside a 0.25m tolerance — but
# speed=0.287m/s, clearly still translating), which needlessly delayed
# convergence of the downstream `oriented_towards`/`intercept_point()` checks
# in `_pass_exec`. This alone was not sufficient to fix the "never reaches
# relay/finish" stall, though — see the carrier-orientation fix in the
# "assess" phase body below for the actual dominant root cause (the carrier
# facing the ball instead of the pivot, which pointed `intercept_point()`'s
# passer-orientation-based projection at an unrelated point on the field).
# Both fixes are real and load-bearing together; confirmed via a full
# two-team match trace reaching "finish" only once both were in place.
_ARRIVAL_SPEED_THRESHOLD = 0.1  # m/s — widened from 0.05: that tight a threshold flickered
# on ordinary station-keeping jitter (observed via trace: speed oscillating
# 0.03-0.07 m/s around a robot that had, for practical purposes, already
# arrived), which flapped `runner_ready` tick to tick in the "relay" phase
# below and never let the pass leg start. Root-caused as part of the
# `counter_press` "never scores" investigation.
_ARRIVAL_POSITION_TOLERANCE = 0.15  # metres — tighter than the original 0.25 for the same reason

# Widening `_ARRIVAL_SPEED_THRESHOLD` (above) was not sufficient on its own:
# a robot converged on a static target still never truly settles under this
# codebase's PID translation controller (`TwoDPID._calculate` in
# `motion_planning/src/pid/pid.py` — proportional-only near the target, no
# terminal deadband above its 3mm snap-to-zero), so speed keeps oscillating
# in a small band straddling the threshold indefinitely rather than
# converging below it. Live-traced on `shadow_switch_vs_zone_fluid`
# (tournament run 2026-09-04): runner sat within 0.05-0.13m of its relay
# target (well inside `_ARRIVAL_POSITION_TOLERANCE`) for the full 9.4s a
# "relay" phase was alive, but `_settled_at`'s speed check flipped True/False
# 51 times, the longest continuous True run only 0.53s (mean 71ms) — never
# stable long enough for the same-tick `_pass_exec` handoff below to fire.
# Same class of bug `_LANE_BLOCKED_ABANDON_TICKS` already exists to fix for
# lane-blocked detection (require a sustained signal, not one noisy tick);
# applied here via `SwitchOfPlayMem.settled_ticks` at both `_settled_at` call
# sites (pivot in "assess", runner in "relay").
#
# A hard reset-to-zero on any single non-settled tick (the original
# `_debounced_settled` behaviour) turned out to still under-fire even after
# widening `_ARRIVAL_SPEED_THRESHOLD`: live-traced (2026-09-05 zero-shots
# investigation, `counter_press`/`shadow_switch`/`switch_of_play` — see
# `docs/roadmap.md`) on `shadow_switch_vs_zone_fluid`, `runner_ready` flapped
# True/False 14 times in one 6.3s "relay" window with True-run lengths
# `[0.067, 0.067, 0.1, 0.067, 0.1, 0.017, 0.117, 0.017, 0.067, 0.083, 0.083,
# 0.1, 0.117, 0.083]`s — almost every run landing at or just under the 6-tick
# (0.1s) threshold, so a single stray miss tick near the end of an
# otherwise-converged streak wiped the counter back to 0 and restarted the
# wait. Root cause is the *hard reset*, not an insufficiently long window (a
# 231-match tournament run traced this same tactic reaching "finish" only
# once across an entire 65s match). Fixed by leaking the counter down by 1 on
# a miss instead of zeroing it, so an isolated single-tick dip inside a
# mostly-settled streak no longer discards all prior progress — the counter
# still resets effectively-to-zero over a handful of consecutive misses
# (never truly converging), so this doesn't mask a robot that is genuinely
# still moving, only tolerates the residual PID chatter documented above.
_SETTLE_DEBOUNCE_TICKS = 6  # ~0.1s at 60Hz — long enough to bridge the observed oscillation period

# Grace radius for treating the ball as still "held" by the relay source
# robot even on a tick where has_ball(visual=True) reads False. rsim has a
# known dribble-physics quirk where holding a ball for an extended period can
# eject it briefly with no tactic-level cause — see
# docs/strategies.md / project memory on rsim dribble issues. Without this,
# a single ejection tick during "relay" sent the source robot straight into
# go_to_ball, discarding its aim/hold state and restarting the leg. This is a
# mitigation for simulator noise, not a root-cause fix — the real fix belongs
# in rsim's dribbler physics, out of scope here.
#
# This radius is only a valid grace window if the source robot has actually
# held the ball at some point already this "relay" episode — gated via
# `SwitchOfPlayMem.source_had_ball` below. Without that gate, a source that
# approached to just outside contact range but never actually acquired the
# ball (has_ball never True) satisfied the bare distance check indefinitely,
# so this branch ran forever instead of falling through to go_to_ball: the
# robot stopped trying to acquire the ball and instead held its position
# facing the runner (this branch's "hold" command, oriented at the runner,
# not the ball), nudging the ball a few cm on each drift without ever
# gaining the dribbler contact needed for has_ball to go True. Live-traced on
# `shadow_switch_vs_zone_fluid` (tournament run 2026-09-04): source robot 3
# sat 0.11-0.13m from the ball for 3.5+ seconds, has_ball True on only 1 of
# 211 ticks in that window, the "hold" branch firing on every one of them —
# the user-visible "bumps into the ball and turns away, never grabs it" bug.
_BALL_RECOVERY_RADIUS = 0.3  # metres


def _weak_side(game: Game, prev_side: Optional[int]) -> int:
    """+1 / -1 for which half (by y) of the field currently has fewer enemy
    robots — the side a switch-of-play should target. See module docstring
    and `_WEAK_SIDE_MARGIN` for why this is hysteresis-gated rather than a
    raw comparison.
    """
    enemies = enemy_positions(game)
    if not enemies:
        return prev_side if prev_side is not None else 1

    top_count = sum(1 for p in enemies if p.y >= 0)
    bottom_count = len(enemies) - top_count
    # Fewer enemies on top => top is the weak side => +1 (top is +y here).
    if prev_side is not None:
        prev_count, other_count = (top_count, bottom_count) if prev_side > 0 else (bottom_count, top_count)
        if other_count < prev_count + _WEAK_SIDE_MARGIN:
            return prev_side

    if top_count + _WEAK_SIDE_MARGIN <= bottom_count:
        return 1
    if bottom_count + _WEAK_SIDE_MARGIN <= top_count:
        return -1
    return prev_side if prev_side is not None else 1


def _pivot_target(game: Game, carrier_pos: Vector2D) -> Vector2D:
    """A short, safe outlet near the carrier but pulled slightly back toward
    our own half, so the carrier->pivot pass is a low-risk square/back pass,
    not itself a step toward goal.

    `back_x`'s pullback is proportional to the *full* carrier-to-enemy-goal
    distance, uncapped. When the carrier is already deep in its own half
    (common right after a kickoff/restart, before the carrier has advanced
    the ball at all), this overshoots straight through the team's own goal
    line and into its own defense area -- found via direct match trace: the
    pivot held station inside its own box for an extended stretch, tripping
    the "too many defenders in own area" foul from the attacking side rather
    than `DefenseTactic`. Clamp the result to stay in front of (not inside)
    our own defense area, regardless of how far back the raw formula would
    place it.
    """
    goal_x, _goal_y1, _goal_y2 = enemy_goal_line(game)
    back_x = carrier_pos.x - 0.2 * (goal_x - carrier_pos.x)
    own_box_front_x = float(game.field.my_defense_area[1][0])
    # ROBOT_RADIUS + OWN_DEFENSE_AREA_STANDOFF_DISTANCE alone (the margin
    # `defend_parameter` uses) was measured as insufficient here: the pivot
    # approaches this target from open field at real speed, same overshoot
    # mechanism as `defend_parameter`'s own-box standoff bug, but unverified
    # for this different approach profile -- an extra ROBOT_RADIUS of
    # headroom on top confirmed clean via direct match trace.
    _pivot_standoff = ROBOT_RADIUS + OWN_DEFENSE_AREA_STANDOFF_DISTANCE + ROBOT_RADIUS
    if game.my_team_is_right:
        back_x = min(back_x, own_box_front_x - _pivot_standoff)
        behind_box = back_x <= own_box_front_x
    else:
        back_x = max(back_x, own_box_front_x + _pivot_standoff)
        behind_box = back_x >= own_box_front_x
    back_y = -carrier_pos.y * 0.3

    # The x-clamp above only guarantees this ENDPOINT sits outside the box --
    # not the straight-line approach to it. Whenever the target's x is behind
    # the box's own front edge (the box sits between it and the open field
    # any approaching robot -- carrier or pivot -- starts from), a `back_y`
    # that lands inside the box's own y-span means that straight approach
    # cuts through the box's near edge, regardless of which side the
    # approaching robot's CURRENT position happens to be on. Root-caused via
    # direct trajsample trace (high_line_zone_vs_high_press, COMMITTED_FROZEN,
    # two separate live matches): the planner rejected the direct path
    # against that exact edge every tick (margin oscillating 0.0-0.05 m) and
    # fell back to a fresh, near-randomly redirected two-segment detour on
    # almost every replan, so the pivot's real path wandered in a wide
    # non-convergent arc and never settled -- `"assess"` then waits forever
    # for a `pivot_ready` that can't happen. Push `back_y` just past the
    # box's own edge on whichever side it's already leaning, same standoff
    # used for the x-clamp, whenever the target sits behind the box at all
    # (not only when the x-clamp itself fired) -- the pivot's own current
    # position is not known here and may be on either side.
    if behind_box:
        box_half_width = float(game.field.my_defense_area[1][1])
        y_standoff = ROBOT_RADIUS + OWN_DEFENSE_AREA_STANDOFF_DISTANCE
        if -box_half_width - y_standoff <= back_y <= box_half_width + y_standoff:
            back_y = (box_half_width + y_standoff) if back_y >= 0 else (-box_half_width - y_standoff)
    return Vector2D(back_x, back_y)


def _runner_target(game: Game, weak_side: int) -> Vector2D:
    """Advance into the weak-side flank, ahead of the ball, so the runner is
    arriving into open space by the time the relay pass is ready rather than
    starting its run only after the ball gets there."""
    goal_x, _goal_y1, _goal_y2 = enemy_goal_line(game)
    half_width = game.field.half_width
    direction = (goal_x / abs(goal_x)) if goal_x != 0 else 1.0
    target_x = goal_x - direction * (game.field.half_length * (1.0 - _RUNNER_DEPTH_FRACTION))
    target_y = weak_side * (half_width - 0.6)
    return Vector2D(target_x, target_y)


def _settled_at(game: Game, robot_id: int, target: Vector2D) -> bool:
    """Position AND velocity both need to have converged — see
    `_ARRIVAL_SPEED_THRESHOLD`'s comment for why position alone isn't enough
    before handing off to `_pass_exec`'s tight orientation-tracking."""
    robot = game.friendly_robots[robot_id]
    if not at_target(game, robot_id, target, tolerance=_ARRIVAL_POSITION_TOLERANCE):
        return False
    speed = (robot.v.x**2 + robot.v.y**2) ** 0.5
    return speed <= _ARRIVAL_SPEED_THRESHOLD


def _debounced_settled(game: Game, robot_id: int, target: Vector2D, mem: SwitchOfPlayMem) -> bool:
    """`_settled_at`, debounced against `_SETTLE_DEBOUNCE_TICKS` consecutive
    ticks rather than one instantaneous read — see that constant's comment
    for why a single-tick check flaps forever on a robot that's converged in
    position but still has PID-driven residual velocity oscillating around
    `_ARRIVAL_SPEED_THRESHOLD`. Mutates `mem.settled_ticks` as a side effect
    (mirrors `mem.lane_blocked_ticks`'s existing consecutive-tick counter in
    this same file), so callers must use the returned `mem`."""
    if _settled_at(game, robot_id, target):
        mem.settled_ticks = min(mem.settled_ticks + 1, _SETTLE_DEBOUNCE_TICKS)
    else:
        mem.settled_ticks = max(mem.settled_ticks - 1, 0)
    return mem.settled_ticks >= _SETTLE_DEBOUNCE_TICKS


def _shot_open(game: Game, robot_id: int) -> bool:
    robot_pos = game.friendly_robots[robot_id].p
    goal_x, goal_y1, goal_y2 = enemy_goal_line(game)
    best_shot_y, gap = find_best_shot(robot_pos, list(game.enemy_robots.values()), goal_x, goal_y1, goal_y2)
    if best_shot_y is None or gap is None:
        return False
    return not segment_blocked(robot_pos, Vector2D(goal_x, best_shot_y), enemy_positions(game))


@dataclass
class SwitchOfPlayMem:
    phase: str = "assess"  # "assess" -> "switch" -> "relay" -> "finish" -> (goal_scored, loops to "assess")
    carrier_id: Optional[int] = None
    pivot_id: Optional[int] = None
    runner_id: Optional[int] = None
    weak_side: Optional[int] = None
    goal_scored: bool = False
    phase_ticks: int = 0  # ticks spent in the current phase; drives the timeout reset (guidance point 1)
    prev_best_shot_y: Optional[float] = None  # feeds _score_goal's switch-margin hysteresis; see _pass_and_score.py
    lane_blocked_ticks: int = 0  # consecutive ticks _pass_exec reported the lane blocked; feeds early phase timeout
    settled_ticks: int = 0  # consecutive ticks _settled_at read True; see _SETTLE_DEBOUNCE_TICKS
    source_had_ball: bool = (
        False  # True once has_ball(source_id) has read True this "relay" episode; see _BALL_RECOVERY_RADIUS
    )
    carrier_ball_loss_ticks: int = 0  # consecutive "assess" ticks has_ball(carrier, visual=True) read False


class SwitchOfPlayTactic(BaseTactic[SwitchOfPlayMem]):
    """3 (or 2, or 1) attackers: relocate the ball to the weak side via a
    carrier -> pivot -> runner relay before finishing. See module docstring
    for the full phase sequence and rationale.
    """

    tag = TacticTag.ATTACK

    def initial_mem(self) -> SwitchOfPlayMem:
        return SwitchOfPlayMem()

    def is_committed(self, game: Game, mem: SwitchOfPlayMem) -> bool:
        # Guidance point 1: is_committed() is a liveness contract. This
        # tactic's own tick() is responsible for actually walking every
        # committed phase back to "assess" on both success and timeout (see
        # the phase-timeout block in tick()) — this just reports the current
        # claim truthfully. "assess" is the only phase where robots aren't
        # mid-relay and reassignment is safe.
        if mem.carrier_id is None:
            return False
        return mem.phase != "assess"

    def suggest_next(self, game: Game, mem: SwitchOfPlayMem) -> Optional[TacticId]:
        """Purely advisory (see `Tactic.suggest_next`'s contract) — after a
        goal, `tick()` already loops `phase` back to "assess" and re-picks
        roles on its own (see the module docstring), so this isn't needed
        for correctness. It exists only for a `TacticGraph`-driven
        repertoire that wants a chance to try a different attacking pattern
        for the next possession rather than always re-running the same
        relay. No existing strategy calls this (nothing consulted
        `suggest_next` anywhere until `strategy/tactic_graph.py`), so this
        has no effect on any already-tuned `build_*_kernel_strategy` config.
        """
        if mem.goal_scored:
            return "givego"
        return None

    def highlights(self, mem: SwitchOfPlayMem) -> dict[RobotId, str]:
        highlights: dict[RobotId, str] = {}
        if mem.carrier_id is not None:
            highlights[mem.carrier_id] = "carrier"
        if mem.pivot_id is not None:
            highlights[mem.pivot_id] = "pivot"
        if mem.runner_id is not None:
            highlights[mem.runner_id] = "runner"
        return highlights

    def tick(
        self, game: Game, ctx: TickContext, robot_ids: tuple[RobotId, ...], mem: SwitchOfPlayMem
    ) -> tuple[dict[RobotId, RobotCommand], SwitchOfPlayMem]:
        if len(robot_ids) < 1:
            return {}, mem

        roles_valid = (
            mem.carrier_id in robot_ids
            and (mem.pivot_id is None or mem.pivot_id in robot_ids)
            and (mem.runner_id is None or mem.runner_id in robot_ids)
        )
        if mem.carrier_id is None or not roles_valid or mem.goal_scored:
            ordered = sorted(robot_ids, key=lambda rid: game.friendly_robots[rid].p.distance_to(game.ball.p.to_2d()))
            carrier = ordered[0]
            pivot = ordered[1] if len(ordered) > 1 else None
            runner = ordered[2] if len(ordered) > 2 else (ordered[1] if len(ordered) > 1 else None)
            mem = SwitchOfPlayMem(carrier_id=carrier, pivot_id=pivot, runner_id=runner)

        # Guidance point 1: a stalled leg must not strand these robots
        # forever. Reset phase_ticks whenever we're in "assess" (the
        # uncommitted phase — nothing to time out), and count up otherwise;
        # exceeding the budget drops back to "assess".
        #
        # Caught live in validation: the naive version of this reset (just
        # rebuild a fresh SwitchOfPlayMem with phase="assess") is not enough
        # on its own. "assess"'s own body immediately re-checks
        # `has_ball(carrier)` and advances straight back to "switch"/"relay"
        # in that *same* tick whenever the carrier still has the ball — which
        # it does whenever the actual stall is downstream (pivot/runner never
        # registering the catch), not upstream (carrier losing the ball). The
        # timeout fired, but its own escape hatch undid it before tick()
        # returned, so is_committed() never actually saw an uncommitted tick
        # and the tactic's robots stayed permanently unreassignable — exactly
        # the "one-way phase machine" failure mode this guidance point warns
        # about, just one level down: the reset existed, but the state it
        # reset *into* immediately re-armed the same commitment. Force the
        # ball to be dropped/re-approached on a timeout (not just phase reset
        # to "assess") so the auto-advance at the bottom of the "assess"
        # branch cannot re-fire until the carrier actually re-acquires it.
        timed_out = False
        if mem.phase == "assess":
            mem.phase_ticks = 0
            mem.lane_blocked_ticks = 0
        else:
            mem.phase_ticks += 1
            if mem.phase_ticks > _PHASE_TIMEOUT_TICKS:
                mem = SwitchOfPlayMem(carrier_id=mem.carrier_id, pivot_id=mem.pivot_id, runner_id=mem.runner_id)
                timed_out = True

        commands: dict[RobotId, RobotCommand] = {}
        carrier_id = mem.carrier_id
        pivot_id = mem.pivot_id
        runner_id = mem.runner_id
        two_robot_mode = pivot_id is None or runner_id == pivot_id

        # Every debugging session on this tactic (see docs/strategies.md's
        # "Known open bugs") has hand-written a one-off print of exactly this
        # tuple in a throwaway trace script. Recording it here instead means
        # any future investigation can get it back for free via
        # `debug_match.py --match-log <path>` + `load_jsonl`, with zero
        # runtime cost when match_log is unset (the default for every normal
        # run/test/tournament).
        if ctx.match_log is not None:
            ctx.match_log.trace_if_changed(
                tick=0,
                sim_time=getattr(game, "ts", 0.0),
                key="switch_of_play.phase",
                value={
                    "phase": mem.phase,
                    "carrier_id": carrier_id,
                    "pivot_id": pivot_id,
                    "runner_id": runner_id,
                },
            )
            # Where the weak-side run is heading — the one piece of this
            # tactic's decision that highlights() can't express as a label,
            # since it's a target point, not a robot to call out.
            if runner_id is not None:
                target = _runner_target(game, mem.weak_side if mem.weak_side is not None else 1)
                ctx.match_log.trace_if_changed(
                    tick=0,
                    sim_time=getattr(game, "ts", 0.0),
                    key="switch_of_play.runner_target",
                    value={"runner_id": runner_id, "x": target.x, "y": target.y},
                )

        if runner_id is None:
            # Solo allocation (a picker gave this slot exactly 1 robot): there
            # is no one to relay or pass to, so "switch"/"relay"/"finish" can
            # never be entered — every use of runner_id past "assess" assumes
            # it is set (that's what two_robot_mode's "collapse pivot into
            # runner" is for), and with a single robot there is no pivot to
            # collapse. The carrier just chases and holds the ball; nothing to
            # time out, since "assess" never advances past itself here.
            if not has_ball(game, carrier_id, visual=True):
                commands[carrier_id] = go_to_ball(
                    game=game, motion_controller=ctx.motion_controller, robot_id=carrier_id, ctx=ctx
                )
            else:
                carrier_pos = game.friendly_robots[carrier_id].p
                commands[carrier_id] = move(
                    game=game,
                    motion_controller=ctx.motion_controller,
                    robot_id=carrier_id,
                    target_coords=carrier_pos,
                    target_oren=carrier_pos.angle_to(game.ball.p.to_2d()),
                    dribbling=True,
                )
            return commands, mem

        if mem.phase == "assess":
            mem.weak_side = _weak_side(game, mem.weak_side)

            carrier_has_ball = has_ball(game, carrier_id, visual=True)
            if carrier_has_ball:
                mem.carrier_ball_loss_ticks = 0
            else:
                mem.carrier_ball_loss_ticks += 1

            if not carrier_has_ball and mem.carrier_ball_loss_ticks > _CARRIER_BALL_LOSS_GRACE_TICKS:
                commands[carrier_id] = go_to_ball(
                    game=game, motion_controller=ctx.motion_controller, robot_id=carrier_id, ctx=ctx
                )
            else:
                # Face the intended pass target (pivot, or runner in
                # two_robot_mode), not the ball. `go_to_point`'s default
                # orientation is `face_ball` (via its own `move()` call),
                # which only reflects wherever the carrier happened to be
                # facing when it picked up the ball -- unrelated to the
                # pivot's position. `_pass_exec`'s `intercept_point()`
                # (called once "switch" starts) projects the receive point
                # *along the carrier's current orientation* -- if that
                # orientation doesn't point roughly at the pivot, the
                # projected intercept point lands somewhere the pivot never
                # walks to, and the pass leg stalls forever. Root-caused via
                # direct match trace: pivot settled correctly at its planned
                # `_pivot_target()` position and stopped completely
                # (velocity zero), but `intercept_point()` kept computing a
                # target ~0.6m away in a completely different direction,
                # because the carrier was still facing the ball itself
                # (orientation ~pi, facing its own goal) rather than the
                # pivot it was about to pass to.
                receiver_id = pivot_id if not two_robot_mode else runner_id
                face_target = game.friendly_robots[receiver_id].p if receiver_id is not None else game.ball.p.to_2d()
                carrier_pos = game.friendly_robots[carrier_id].p
                commands[carrier_id] = move(
                    game=game,
                    motion_controller=ctx.motion_controller,
                    robot_id=carrier_id,
                    target_coords=carrier_pos,
                    target_oren=carrier_pos.angle_to(face_target),
                    dribbling=True,
                )

            if not two_robot_mode:
                pivot_target = _pivot_target(game, game.friendly_robots[carrier_id].p)
                commands[pivot_id] = go_to_point(
                    game=game, motion_controller=ctx.motion_controller, robot_id=pivot_id, target_coords=pivot_target
                )

            if runner_id is not None:
                runner_target = _runner_target(game, mem.weak_side if mem.weak_side is not None else 1)
                commands[runner_id] = go_to_point(
                    game=game, motion_controller=ctx.motion_controller, robot_id=runner_id, target_coords=runner_target
                )

            # Don't hand off to _pass_exec's carrier->pivot leg until the
            # pivot has actually arrived near its planned outlet position.
            # Caught live in validation: `_pass_exec`'s `intercept_point()` is
            # a function of the *receiver's live position* — if the pivot is
            # still crossing the field when "switch" starts, the intercept
            # point (and therefore the pivot's required facing angle) drifts
            # every tick right along with the pivot's own approach, so
            # `oriented_towards` never settles and the pass never completes.
            # `pass_and_shoot`'s `run_setup_phase` avoids exactly this by
            # gating its own pass leg on both robots already being at fixed
            # setup positions before calling the shared pass machinery — same
            # fix here, just for the pivot only (the carrier is already
            # stationary with the ball by this point).
            pivot_ready = two_robot_mode or _debounced_settled(
                game, pivot_id, _pivot_target(game, game.friendly_robots[carrier_id].p), mem
            )
            if not timed_out and has_ball(game, carrier_id, visual=True) and pivot_ready:
                mem.phase = "relay" if two_robot_mode else "switch"
                mem.settled_ticks = 0  # fresh debounce window for the next phase's own _debounced_settled use
                # two_robot_mode enters "relay" directly with source_id == carrier_id,
                # who has_ball is already confirmed True for (the `and` above) — start
                # source_had_ball true to match, not force a fresh false->true relatch.
                mem.source_had_ball = two_robot_mode
            return commands, mem

        if mem.phase == "switch":
            # Carrier -> pivot leg. Runner keeps advancing into the weak side
            # while this happens rather than waiting, so it is already in
            # position for the relay leg.
            leg_commands, leg_complete, lane_blocked = _pass_exec(game, ctx, carrier_id, pivot_id)
            commands.update(leg_commands)
            # An enemy settling onto the direct carrier-pivot line makes this
            # an easily-intercepted pass — force the existing phase timeout
            # early (after a short sustained block, not one noisy tick)
            # rather than waiting the full ~10s budget, which "assess"'s own
            # re-entry (fresh carrier/pivot/runner assignment) exists to
            # resolve anyway.
            mem.lane_blocked_ticks = mem.lane_blocked_ticks + 1 if lane_blocked else 0
            if mem.lane_blocked_ticks >= _LANE_BLOCKED_ABANDON_TICKS:
                mem.phase_ticks = max(mem.phase_ticks, _PHASE_TIMEOUT_TICKS + 1)
            if runner_id is not None and runner_id not in commands:
                runner_target = _runner_target(game, mem.weak_side if mem.weak_side is not None else 1)
                commands[runner_id] = go_to_point(
                    game=game, motion_controller=ctx.motion_controller, robot_id=runner_id, target_coords=runner_target
                )
            if leg_complete:
                mem.phase = "relay"
                # leg_complete == receiver_has_ball (see _pass_exec), i.e. the
                # pivot (this phase's "relay" source_id) has the ball right now.
                mem.source_had_ball = True
            return commands, mem

        if mem.phase == "relay":
            source_id = pivot_id if not two_robot_mode else carrier_id
            # Same fix as the "switch"->"assess" gate above, applied to the
            # runner: don't start _pass_exec's intercept-chasing aim/kick
            # logic until the runner is actually near its planned weak-side
            # target, or its live position drifting mid-approach would make
            # the intercept point (and required facing angle) drift right
            # along with it and never settle.
            runner_target = _runner_target(game, mem.weak_side if mem.weak_side is not None else 1)
            # A pass already rolling at the runner is met on its path (see `_pass_exec`),
            # which moves the runner off its spot; don't send it back there mid-pass.
            runner_ready = ball_line_receive_point(game, runner_id) is not None or _debounced_settled(
                game, runner_id, runner_target, mem
            )
            if ctx.match_log is not None:
                ctx.match_log.trace(
                    tick=0,
                    sim_time=getattr(game, "ts", 0.0),
                    key="switch_of_play.relay_gate",
                    value={"runner_ready": runner_ready},
                )
            if not runner_ready:
                commands[runner_id] = go_to_point(
                    game=game, motion_controller=ctx.motion_controller, robot_id=runner_id, target_coords=runner_target
                )
                # Source robot holds the ball (or keeps approaching it) while
                # the runner finishes arriving, rather than firing the pass
                # early — mirrors the pivot-arrival gate's "hold, don't chase
                # a moving target" fix.
                #
                # Face the runner while holding, not the ball (`go_to_point`'s
                # default `face_ball` orientation) — same bug and same fix as
                # the "assess" phase's carrier hold-command above:
                # `intercept_point()` projects the receive point along the
                # *source robot's current orientation*, so a source facing
                # the ball instead of the runner sends that projection
                # somewhere the runner never walks to, stalling "relay" the
                # same way "switch" stalled before that fix. Confirmed via
                # direct match trace: `src_oren` oscillating tick to tick
                # (holding a ball naturally reorients to face it, and the
                # ball's own tiny position jitter while held is enough to
                # spin that orientation around) with `intercept_pos` swinging
                # wildly in lockstep, `dst_at_intercept` never settling.
                source_pos = game.friendly_robots[source_id].p
                source_has_ball_now = has_ball(game, source_id, visual=True)
                mem.source_had_ball = mem.source_had_ball or source_has_ball_now
                if source_has_ball_now or (
                    mem.source_had_ball and source_pos.distance_to(game.ball.p.to_2d()) <= _BALL_RECOVERY_RADIUS
                ):
                    runner_pos = game.friendly_robots[runner_id].p
                    commands[source_id] = move(
                        game=game,
                        motion_controller=ctx.motion_controller,
                        robot_id=source_id,
                        target_coords=source_pos,
                        target_oren=source_pos.angle_to(runner_pos),
                        dribbling=True,
                    )
                else:
                    commands[source_id] = go_to_ball(
                        game=game, motion_controller=ctx.motion_controller, robot_id=source_id, ctx=ctx
                    )
                return commands, mem

            leg_commands, leg_complete, lane_blocked = _pass_exec(game, ctx, source_id, runner_id)
            commands.update(leg_commands)
            mem.lane_blocked_ticks = mem.lane_blocked_ticks + 1 if lane_blocked else 0
            if mem.lane_blocked_ticks >= _LANE_BLOCKED_ABANDON_TICKS:
                mem.phase_ticks = max(mem.phase_ticks, _PHASE_TIMEOUT_TICKS + 1)
            if leg_complete:
                mem.phase = "finish"
            return commands, mem

        # phase == "finish": runner shoots if it has an open lane, otherwise
        # holds/re-approaches the ball until one opens or the phase times out
        # back to "assess" (handled by the timeout block above).
        if has_ball(game, runner_id, visual=True) and _shot_open(game, runner_id):
            shot_cmd, scored, mem.prev_best_shot_y = _score_goal(game, ctx, runner_id, mem.prev_best_shot_y)
            commands[runner_id] = shot_cmd
            if scored:
                mem.goal_scored = True
                mem.phase = "assess"
        elif not has_ball(game, runner_id, visual=True):
            commands[runner_id] = go_to_ball(
                game=game, motion_controller=ctx.motion_controller, robot_id=runner_id, ctx=ctx
            )
        else:
            commands[runner_id] = go_to_point(
                game=game,
                motion_controller=ctx.motion_controller,
                robot_id=runner_id,
                target_coords=game.friendly_robots[runner_id].p,
                dribbling=True,
            )

        return commands, mem

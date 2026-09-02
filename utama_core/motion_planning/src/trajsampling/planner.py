"""Trajectory-sampling path planner, following TIGERs Mannheim's approach
(RoboCup SSL Division A champion 2021-2024; adopted by ER-Force in 2020 and
RobôCIn in 2023 -- see the 2024 champion paper, section 2). Unlike
FastPathPlanner's recursive geometric detour search over a static polyline
(fastpathplanning/planner.py), this searches directly in trajectory space:
generate a small number of full bang-bang (position+velocity) candidate
trajectories, collision-check each against motion-aware obstacles, and
return the first collision-free one.

Kept as a separate, opt-in `MotionController` (see
`controllers/trajsampling.py`) rather than replacing FastPathPlanner --
selectable via `control_scheme="trajsample"` in `get_control_scheme`.

Teammate priority (`_has_priority`, `_CommittedTrajectoryObstacle.owner_id`):
confirmed against TIGERs' actual published source (Sumatra,
github.com/TIGERs-Mannheim/Sumatra -- not just the paper's prose) after four
successive emergency-brake patches (persistence, missing brake layer,
own-radius accounting, closing-speed) all failed to stop two robots from
grazing each other in the mirror_swap 6v6 test, each stuck at ~0.176-0.179m
separation (collision threshold is 0.18m) with reached-count never
improving. Their `ObstacleGenerator.selectObstacleForOurBot` marks every
teammate obstacle `hasPriority(true)` unless the querying robot is
preferred by `PathFinderPrioMap` (default: higher bot ID wins -- see
`PathFinderPrioMap.mapByBotId`), and `MovingObstacleResultAcceptor.accept`
rejects a candidate outright when its first collision is against a
higher-priority obstacle, with NO braking-distance carve-out at all --
lower-priority robots simply aren't allowed to plan through a
higher-priority teammate's path. That's the actual fix for two peers
converging on each other: neither robot's own local "does my trajectory
look fine" check has a way to break the symmetry between two equally-valid
local decisions, so a strict, decided-in-advance ordering is what prevents
both robots from simultaneously concluding "I'm fine, the other one will
handle it." No amount of per-tick braking math (which only ever asks "is MY
trajectory currently safe", not "which of us should yield") fixes a
tie-breaking problem.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from dataclasses import replace as dataclasses_replace
from typing import List, Optional, Tuple

import numpy as np

from utama_core.config.referee_constants import OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
from utama_core.entities.game import Game
from utama_core.entities.game.field import FieldBounds
from utama_core.motion_planning.src.trajsampling import collision_numba as _cn
from utama_core.motion_planning.src.trajsampling.bang_bang import Trajectory2D
from utama_core.motion_planning.src.trajsampling.config import (
    trajsamplingconfig as config,
)
from utama_core.motion_planning.src.trajsampling.obstacles import (
    ConstantVelocityObstacle,
    EnemyRobotObstacle,
    OwnRobotObstacle,
    StaticSegmentObstacle,
    TimedObstacle,
    collect_static_obstacles,
    enemy_obstacle_from_robot,
)


@dataclass(frozen=True)
class PlanResult:
    trajectory: Trajectory2D
    has_collision: bool
    collision_time: Optional[float]  # None if has_collision is False
    # Seconds into `trajectory` that correspond to "now" -- 0.0 for a
    # freshly-computed trajectory, >0 when an already-committed trajectory
    # is being reused (see `plan()`'s early-out). Callers must evaluate
    # `trajectory.state_at(elapsed + lookahead)`, never `state_at(lookahead)`
    # directly, or a reused trajectory is silently re-read from its own t=0
    # every tick -- exactly the bug this field exists to prevent (see
    # `plan()`'s module-level docstring note and TIGERs' 2024 champion paper
    # section 2.6: "We take the next ... target position from the
    # trajectory ... every frame", which only makes sense against a
    # persisting trajectory read at increasing elapsed time).
    elapsed: float = 0.0
    # Distance (metres) from the robot's current position to the nearest
    # obstacle right now (at `elapsed`, not planned collision time) minus
    # that obstacle's own radius/margin baked in -- i.e. raw clearance, not
    # yet compared against any braking distance. `None` only when there are
    # no obstacles at all. Used by `TrajectorySamplingController`'s
    # emergency-brake check (paper section 2.6): per-tick revalidation here
    # only rejects an already-unsafe trajectory, it doesn't verify there's
    # still enough real-world distance left to brake before the *next*
    # tick's revalidation would catch a newly-closing obstacle -- that
    # requires knowing current clearance, not just "collision or not".
    nearest_obstacle_distance: Optional[float] = None
    # Rate (m/s) at which `nearest_obstacle_distance`'s gap is shrinking,
    # positive = closing, right now -- estimated by finite-differencing the
    # gap along this robot's own planned trajectory a short time step ahead,
    # so it captures the *combined* motion of both this robot and the
    # obstacle, not just this robot's own speed. Needed because `speed`
    # alone (this robot's speed against a static sqrt(2*a*d) threshold) is
    # blind to an obstacle that is itself closing: two robots each
    # individually "within their own braking distance" of a stationary wall
    # can still be closing on EACH OTHER faster than either one's own speed
    # suggests. Found live in the mirror_swap scenario: two robots slowly
    # converging (each ~0.2-0.6 m/s) held a roughly constant ~0.22m
    # separation for dozens of ticks -- comfortably inside each robot's own
    # braking distance individually -- while actually oscillating close
    # enough to the real 0.18m collision threshold to eventually breach it,
    # because neither robot's own-speed-only check ever saw the other
    # robot's contribution to the closing rate. `None` only when there are
    # no obstacles at all (mirrors `nearest_obstacle_distance`).
    closing_speed: Optional[float] = None


def _bangbang_to_row(bb) -> np.ndarray:
    return _cn.bangbang_row(bb.p0, bb.v0, bb.p1, bb.v_max, bb.a_max, bb.t1, bb.t2, bb.t_end, bb.sign, bb.v_cruise)


_ZERO_BB_ROW = _cn.bangbang_row(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _flatten_query_trajectory(trajectory) -> tuple:
    """Reduce either a `Trajectory2D` (direct path) or a `TwoSegmentTrajectory`
    (candidate) to the uniform (leg1, switch_t, leg2, duration) shape
    `first_collision_numba` expects -- a direct path is represented as a
    "two-segment" trajectory whose `switch_t` equals its own duration, so
    `t <= switch_t` always holds and `leg2` (a dummy zero-profile) is never
    evaluated. See `collision_numba._query_trajectory_state`.

    Also used by `_flatten_obstacles` for friendly-robot trajectory
    obstacles: `_commit` can store either shape too (whatever `plan()`
    picked), so the same uniform representation covers both the querying
    trajectory and a wrapped teammate's committed trajectory.
    """
    if isinstance(trajectory, TwoSegmentTrajectory):
        leg1, leg2 = trajectory.first_leg, trajectory.second_leg
        switch_t = trajectory.switch_t
    else:
        leg1 = trajectory
        leg2 = None
        switch_t = trajectory.duration

    leg1_row = _bangbang_to_row(leg1.along)
    leg1_args = (leg1_row, leg1.ux, leg1.uy, leg1.p0[0], leg1.p0[1])
    if leg2 is not None:
        leg2_row = _bangbang_to_row(leg2.along)
        leg2_args = (leg2_row, leg2.ux, leg2.uy, leg2.p0[0], leg2.p0[1])
    else:
        leg2_args = (_ZERO_BB_ROW, 0.0, 0.0, 0.0, 0.0)

    return leg1_args, switch_t, leg2_args, trajectory.duration


def _flatten_obstacle_rows(
    obstacles: List[TimedObstacle],
) -> Tuple[List[list], List[list], List[list], List[list]]:
    """Convert a list of `TimedObstacle`s into per-kind row lists (not yet
    stacked into numpy arrays) -- the shared building block for both
    `_flatten_obstacles` (a single `plan()` call's full obstacle set) and
    the per-tick shared-obstacle cache (see `TrajectorySamplingPlanner.
    _flatten_shared_obstacles`), so the two paths can't drift apart.
    """
    static_rows = []
    traj_rows = []
    cv_rows = []
    enemy_rows = []

    for obstacle in obstacles:
        if isinstance(obstacle, StaticSegmentObstacle):
            static_rows.append([obstacle.a[0], obstacle.a[1], obstacle.b[0], obstacle.b[1], obstacle.radius])
        elif isinstance(obstacle, ConstantVelocityObstacle):
            cv_rows.append([obstacle.p0[0], obstacle.p0[1], obstacle.v[0], obstacle.v[1], obstacle.radius])
        elif isinstance(obstacle, EnemyRobotObstacle):
            row = [
                obstacle.p0[0],
                obstacle.p0[1],
                obstacle.direction[0],
                obstacle.direction[1],
                obstacle.speed,
                obstacle.radius,
                obstacle.t_max,
            ]
            row.extend(_bangbang_to_row(obstacle._f_plus).tolist())
            row.extend(_bangbang_to_row(obstacle._f_minus).tolist())
            enemy_rows.append(row)
        elif isinstance(obstacle, (OwnRobotObstacle, _CommittedTrajectoryObstacle)):
            # Both wrap a committed trajectory -- either a plain
            # `Trajectory2D` or a `TwoSegmentTrajectory` (whatever `plan()`
            # last picked for that teammate, see `_commit`) -- and are
            # distinguished only by whether a `time_offset` applies (0.0 for
            # `OwnRobotObstacle`). Reuses `_flatten_query_trajectory`'s
            # uniform leg1/switch_t/leg2 representation.
            leg1_args, switch_t, leg2_args, duration = _flatten_query_trajectory(obstacle.trajectory)
            time_offset = getattr(obstacle, "time_offset", 0.0)
            row = leg1_args[0].tolist() + list(leg1_args[1:])
            row.append(switch_t)
            row += leg2_args[0].tolist() + list(leg2_args[1:])
            row += [obstacle.radius, time_offset, duration]
            traj_rows.append(row)
        else:
            raise TypeError(f"Unrecognized obstacle type for numba flattening: {type(obstacle)!r}")

    return static_rows, traj_rows, cv_rows, enemy_rows


def _stack_rows(
    rows: Tuple[List[list], List[list], List[list], List[list]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    static_rows, traj_rows, cv_rows, enemy_rows = rows
    static_arr = np.array(static_rows, dtype=np.float64) if static_rows else _cn.empty_row_array(_cn.STATIC_ROW_LEN)
    traj_arr = np.array(traj_rows, dtype=np.float64) if traj_rows else _cn.empty_row_array(_cn.TRAJ_ROW_LEN)
    cv_arr = np.array(cv_rows, dtype=np.float64) if cv_rows else _cn.empty_row_array(_cn.CV_ROW_LEN)
    enemy_arr = np.array(enemy_rows, dtype=np.float64) if enemy_rows else _cn.empty_row_array(_cn.ENEMY_ROW_LEN)
    return static_arr, traj_arr, cv_arr, enemy_arr


def _flatten_obstacles(
    obstacles: List[TimedObstacle],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert the heterogeneous per-tick obstacle list into the four flat
    float arrays `first_collision_numba` expects (one 2D array per obstacle
    kind, row layout defined in `collision_numba.py`). Done ONCE per
    `plan()` call -- all candidates checked within that call share the same
    obstacle snapshot, so this conversion cost is paid once, not once per
    candidate/timestep (see `_first_collision`'s docstring).
    """
    return _stack_rows(_flatten_obstacle_rows(obstacles))


def _has_priority(robot_id: int, other_id: int) -> bool:
    """True if `robot_id` outranks `other_id` and so may plan through its
    path -- `other_id` must yield instead. Mirrors TIGERs'
    `PathFinderPrioMap.mapByBotId`'s default ("higher botIds are
    preferred"): a plain, fixed ordering by ID, not dynamically computed
    from role or task. Good enough to break the symmetry that let two
    equally-"locally correct" robots both plan through each other; a
    role-aware ordering (e.g. attacker outranks support) is a refinement
    for later, not needed to fix the observed collision.
    """
    return robot_id > other_id


# How close (metres) the robot's actual position must stay to where a
# committed trajectory predicts it should be, before that trajectory is
# considered stale and a fresh one is computed. Compared against TIGERs'
# actual `TrajPath.relocate`, which always re-optimizes a fresh trajectory
# from the robot's real current state rather than tolerating drift from a
# frozen plan -- but a full port of `relocate()` was deferred as a bigger,
# riskier change (see git history/session notes) in favor of tightening
# this tolerance to force more frequent replans while keeping the
# `_try_reuse`/`elapsed`/`time_offset` machinery intact.
#
# 0.03 was tried first and made things measurably WORSE (collision_detected
# reverted to True, reached count collapsed to 1/12): forcing a full replan
# on almost every tick meant `_intermediate_targets`'s fresh random
# candidates got regenerated almost every tick too, since a full replan
# gets no benefit from `_last_intermediate_target`'s "try the previous
# winner first" stabilizer beyond what any two consecutive fresh replans
# would already share by chance -- reintroducing the exact tick-to-tick
# direction-flipping oscillation `_intermediate_targets`'s own docstring
# describes fixing earlier in this session. 0.08 is a middle ground: still
# noticeably tighter than the original 0.15 (more willing to replan on real
# drift), but loose enough that `_try_reuse` still succeeds across most
# consecutive ticks under normal tracking noise, preserving the stability
# benefit.
_TRAJECTORY_POSITION_TOLERANCE = 0.08


class TrajectorySamplingPlanner:
    def __init__(self, v_max: float, a_max: float):
        self.v_max = v_max
        self.a_max = a_max
        self.config = config
        # Per-robot currently-committed (trajectory, ts planned, target it
        # was planned for). Read by two consumers: (1) `plan()` itself, to
        # decide whether to keep executing it or replan from scratch (see
        # the early-out at the top of `plan()`); (2) other robots' obstacle
        # models via `_own_robot_obstacle` (`OwnRobotObstacle`/
        # `_CommittedTrajectoryObstacle`), so they route around where this
        # robot is actually headed rather than a constant-velocity guess.
        # Overwritten every time a robot is actually replanned; a robot that
        # stops being planned (e.g. switches to a non-MoveToSkill skill)
        # simply goes stale and the last trajectory's tail state is used
        # until `reset()` clears it (see `TrajectorySamplingController.reset`).
        self._committed: dict[int, Tuple[float, Trajectory2D, Tuple[float, float]]] = {}
        # Per-robot last accepted intermediate target, tried first among
        # candidates on the next full replan to bias reselection toward
        # stability (paper section 2.2: "the target from the previous
        # iteration is also added to the intermediate targets").
        self._last_intermediate_target: dict[int, Tuple[float, float]] = {}
        self._rng = random.Random(0)
        # Caches the tick-invariant portion of the obstacle set --
        # enemies, ball, and static field/defense-area geometry -- across
        # the multiple robots planned within one tick (one
        # `TrajectorySamplingPlanner` instance is shared by the whole team,
        # so `plan()` is called once per robot per tick with the SAME
        # `game`). These depend only on `game`/`field_bounds`/
        # `exempt_defense_area`, never on `self._committed` or which robot
        # is asking, unlike the per-robot teammate portion (see
        # `_own_robot_obstacle`) which MUST stay uncached -- it reads
        # `self._committed`, which `plan()` mutates as a side effect of
        # planning each robot, and later robots in the same tick are meant
        # to see earlier robots' freshly-committed trajectories, not a
        # stale start-of-tick snapshot. Keyed on `(game.ts,
        # exempt_defense_area)`: `game.ts` changes exactly once per tick
        # (see `Game.ts`/`Game.add_game_frame`), and `exempt_defense_area`
        # is included defensively in case a future caller (e.g. a
        # goalkeeper exemption) varies it per-robot within a tick, even
        # though no current caller does.
        self._shared_obstacle_cache_key: Optional[Tuple[float, bool]] = None
        self._shared_obstacles: List[TimedObstacle] = []
        # Row-list form (not yet stacked into numpy arrays) of
        # `_shared_obstacles`, cached alongside it under the same key --
        # `plan()` concatenates these row lists with the fresh per-robot
        # teammate rows BEFORE stacking into numpy arrays just once, rather
        # than re-flattening the shared portion from scratch or doing a
        # (more expensive, one extra copy) `np.concatenate` of two already-
        # stacked arrays every call.
        self._shared_obstacle_rows: Tuple[List[list], List[list], List[list], List[list]] = ([], [], [], [])

    def plan(
        self,
        game: Game,
        robot_id: int,
        target_pos: Tuple[float, float],
        field_bounds: FieldBounds,
        exempt_defense_area: bool = False,
    ) -> PlanResult:
        robot = game.friendly_robots[robot_id]
        p0 = (robot.p.x, robot.p.y)
        v0 = (robot.v.x, robot.v.y)

        # Teammate obstacles are rebuilt fresh every call (see
        # `_own_robot_obstacle`/`__init__`'s cache comment for why: they
        # read `self._committed`, which mutates as a side effect of planning
        # each robot, and later robots in this same tick must see earlier
        # ones' freshly-committed trajectories). Enemy/ball/static rows come
        # from the per-tick cache, shared across every robot planned this
        # tick. Concatenating row LISTS before stacking into numpy arrays
        # once (rather than stacking twice and `np.concatenate`-ing) avoids
        # an extra array copy.
        teammate_obstacles = [
            self._own_robot_obstacle(r, config.ROBOT_RADIUS, game.ts)
            for r in game.friendly_robots.values()
            if r.id != robot_id
        ]
        shared_obstacles = self._shared_obstacles_for_tick(game, field_bounds, exempt_defense_area)
        obstacles = teammate_obstacles + shared_obstacles

        teammate_rows = _flatten_obstacle_rows(teammate_obstacles)
        shared_static, shared_traj, shared_cv, shared_enemy = self._shared_obstacle_rows
        combined_rows = (
            teammate_rows[0] + shared_static,
            teammate_rows[1] + shared_traj,
            teammate_rows[2] + shared_cv,
            teammate_rows[3] + shared_enemy,
        )
        # Flattened once per `plan()` call (not per candidate/timestep) --
        # every candidate checked below shares this same obstacle snapshot.
        # See `_first_collision`'s docstring for why this matters.
        obstacle_arrays = _stack_rows(combined_rows)

        reused = self._try_reuse(robot_id, game.ts, p0, target_pos, obstacles, obstacle_arrays)
        if reused is not None:
            return self._with_current_clearance(reused, p0, obstacles)

        direct = Trajectory2D.compute(p0, v0, target_pos, self.v_max, self.a_max)
        collision_time = self._first_collision(direct, obstacle_arrays)
        if collision_time is None:
            self._commit(robot_id, game.ts, direct, target_pos)
            result = PlanResult(trajectory=direct, has_collision=False, collision_time=None)
            return self._with_current_clearance(result, p0, obstacles)

        # `best_fallback` ranks candidates by how long they survive before
        # colliding, EXCEPT a candidate whose collision is against a
        # higher-priority teammate is never eligible at all -- ranked as if
        # it collided instantly (`-inf`) regardless of its raw
        # `collision_time`. Mirrors TIGERs' `MovingObstacleResultAcceptor`:
        # a higher-priority obstacle's path is never an acceptable place to
        # end up, not even as the least-bad fallback, because the point of
        # priority is to force the LOWER-ranked robot to be the one that
        # yields -- letting it fall back to "the candidate that survives
        # longest before hitting the priority robot anyway" defeats that.
        def fallback_rank(t_col: float, blocked: bool) -> float:
            return -math.inf if blocked else t_col

        direct_blocked = self._blocked_by_priority_obstacle(robot_id, direct, obstacles, collision_time)
        best_fallback = PlanResult(trajectory=direct, has_collision=True, collision_time=collision_time)
        best_rank = fallback_rank(collision_time, direct_blocked)

        for candidate_target in self._intermediate_targets(robot_id, p0, target_pos):
            # Try every switch-time variant for this direction before moving
            # on to the next sampled direction -- see
            # `_two_segment_candidates`'s docstring for why a single fixed
            # switch time isn't enough.
            #
            # Tried splitting this into a once-per-direction shared-prefix
            # check plus a per-variant tail check, to avoid re-scanning each
            # variant's common `first_leg` prefix from t=0. Measured: no
            # speedup (107.9s vs 106.5s self-time on the same real-match
            # profile, i.e. a wash within noise) and `_first_collision`'s own
            # call count went UP (115k vs 77k), because most sampled
            # directions' first leg doesn't collide at all, so the
            # early-abandon path rarely fires -- the split then just adds a
            # second Python-level call per candidate without saving any
            # scanning. Reverted; see `profile_real_match_opt.log`. The real
            # fix for the per-timestep obstacle loop's cost turned out to be
            # `_first_collision`'s own numba rewrite (see its docstring),
            # not reducing candidate/call count.
            for candidate in self._two_segment_candidates(p0, v0, candidate_target, target_pos):
                t_col = self._first_collision(candidate, obstacle_arrays)
                if t_col is None:
                    self._commit(robot_id, game.ts, candidate, target_pos)
                    self._last_intermediate_target[robot_id] = candidate_target
                    result = PlanResult(trajectory=candidate, has_collision=False, collision_time=None)
                    return self._with_current_clearance(result, p0, obstacles)
                blocked = self._blocked_by_priority_obstacle(robot_id, candidate, obstacles, t_col)
                rank = fallback_rank(t_col, blocked)
                if rank > best_rank:
                    best_fallback = PlanResult(trajectory=candidate, has_collision=True, collision_time=t_col)
                    best_rank = rank

        # Nothing collision-free was found within the sample budget: return
        # whichever candidate survives longest before its first collision
        # (and isn't blocked by a higher-priority teammate -- see
        # `fallback_rank`), so the caller (`TrajectorySamplingController`'s
        # emergency-brake check -- see `PlanResult.nearest_obstacle_distance`)
        # can still let the robot make progress and brake in time rather
        # than the planner masking a bad situation by freezing outright. If
        # EVERY candidate is priority-blocked, `best_rank` stays `-inf` for
        # all of them and the direct trajectory (first considered) wins by
        # the tie -- correct: this robot has nowhere non-blocked to go and
        # must wait for the higher-priority teammate to clear, which the
        # controller's stop-if-imminent handling covers.
        self._commit(robot_id, game.ts, best_fallback.trajectory, target_pos)
        return self._with_current_clearance(best_fallback, p0, obstacles)

    @staticmethod
    def _with_current_clearance(
        result: PlanResult, p0: Tuple[float, float], obstacles: List[TimedObstacle]
    ) -> PlanResult:
        if not obstacles:
            return result
        # Same own-radius correction as `_first_collision` -- see its inline
        # comment. Without it the emergency-brake layer in
        # `TrajectorySamplingController` computes `max_safe_speed` from a
        # distance that's already short by one robot radius, i.e. it thinks
        # the robot has less real stopping room than it does, but the actual
        # bug this caused was the opposite direction: since this under-count
        # is silent (no threshold, just fed into a sqrt formula), it doesn't
        # trigger braking any earlier -- it just makes the reported
        # clearance meaningless as an actual surface-to-surface distance,
        # which is what let 0.148m of *reported* clearance correspond to
        # robots that were already inside the real 0.18m collision
        # threshold.
        nearest_obstacle = min(obstacles, key=lambda o: o.distance_at(result.elapsed, p0))
        nearest = nearest_obstacle.distance_at(result.elapsed, p0) - config.ROBOT_RADIUS

        # Closing speed: finite-difference the gap to THIS SAME obstacle a
        # short step further along the robot's own committed trajectory,
        # rather than re-minning across all obstacles at t+dt -- re-minning
        # could jump to a different obstacle between samples and produce a
        # meaningless rate. Advancing along `result.trajectory` (not just
        # holding the robot's own position fixed at `p0`) means both sides'
        # motion are captured: the robot's own velocity moves `point`,
        # `obstacle.distance_at`'s own `t` argument moves the obstacle.
        dt = config.MIN_TIME_STEP
        t1 = min(result.elapsed + dt, result.trajectory.duration)
        future_point, _ = result.trajectory.state_at(t1)
        future_gap = nearest_obstacle.distance_at(t1, future_point) - config.ROBOT_RADIUS
        actual_dt = t1 - result.elapsed
        closing_speed = (nearest - future_gap) / actual_dt if actual_dt > 1e-9 else 0.0

        return dataclasses_replace(result, nearest_obstacle_distance=nearest, closing_speed=closing_speed)

    def _try_reuse(
        self,
        robot_id: int,
        ts: float,
        p0: Tuple[float, float],
        target_pos: Tuple[float, float],
        obstacles: List[TimedObstacle],
        obstacle_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> Optional[PlanResult]:
        """Keep executing the already-committed trajectory, evaluated at its
        real elapsed time, instead of silently restarting it from t=0 every
        tick -- see `PlanResult.elapsed`'s docstring for why a fresh
        recompute every tick is wrong for anything but an instantaneous
        (zero-duration) plan. `None` return means "replan from scratch":
        target changed, the robot has drifted from where the plan expected
        it to be (a real disturbance, not just tracking noise -- see
        `_TRAJECTORY_POSITION_TOLERANCE`), the plan has already finished, or
        the committed trajectory is no longer collision-free against this
        tick's fresh obstacle positions (a moving obstacle can invalidate a
        previously-fine plan even though the robot itself tracked it
        perfectly).
        """
        committed = self._committed.get(robot_id)
        if committed is None:
            return None
        committed_ts, trajectory, committed_target = committed

        if committed_target != target_pos:
            return None

        elapsed = ts - committed_ts
        if elapsed < 0 or elapsed >= trajectory.duration:
            return None

        expected_p, _ = trajectory.state_at(elapsed)
        if math.hypot(p0[0] - expected_p[0], p0[1] - expected_p[1]) > _TRAJECTORY_POSITION_TOLERANCE:
            return None

        t_col = self._first_collision(trajectory, obstacle_arrays, start_t=elapsed)
        if t_col is not None:
            return None
        # A collision-free reuse can still be blocked by priority alone: a
        # higher-priority teammate may have replanned into a path that
        # merely *touches* this trajectory's margin without registering as
        # `_first_collision`'s harder threshold yet. Re-checking here (not
        # just relying on `plan()`'s fresh-replan branch, which this early
        # return skips) means a lower-priority robot notices and replans
        # away from a closing higher-priority teammate as soon as any
        # margin is crossed, not only once an actual collision would occur.
        for t in (elapsed, min(elapsed + config.MIN_TIME_STEP, trajectory.duration)):
            if self._blocked_by_priority_obstacle(robot_id, trajectory, obstacles, t):
                return None

        return PlanResult(trajectory=trajectory, has_collision=False, collision_time=None, elapsed=elapsed)

    def _two_segment_candidates(
        self,
        p0: Tuple[float, float],
        v0: Tuple[float, float],
        intermediate: Tuple[float, float],
        final_target: Tuple[float, float],
    ) -> List["TwoSegmentTrajectory"]:
        """All two-segment candidates for one intermediate-target direction,
        at increasing switch times (`INTERMEDIATE_SWITCH_TIME`,
        `2*INTERMEDIATE_SWITCH_TIME`, ... up to `MAX_SWITCH_TIME_TRIES`
        steps or the first leg's own duration, whichever is shorter).

        Ported from TIGERs' `SubPathCollisionChecker.findAcceptablePath`,
        which doesn't fix one switch time per direction -- it slides
        `switchTime` in `stepSizeOnSubPath` increments, re-appending a fresh
        leg to the real target at each one, and accepts the first
        collision-free switch point. A single fixed switch time (what this
        method's predecessor, `_build_two_segment_trajectory`, did) only
        finds a solution when 0.2s in happens to already be clear of
        whatever obstacle blocked the direct path -- a materially smaller
        search than trying several switch points per sampled direction.
        """
        first_leg = Trajectory2D.compute(p0, v0, intermediate, self.v_max, self.a_max)
        candidates = []
        for i in range(1, config.MAX_SWITCH_TIME_TRIES + 1):
            switch_t = config.INTERMEDIATE_SWITCH_TIME * i
            if switch_t >= first_leg.duration:
                break
            (sp, sv) = first_leg.state_at(switch_t)
            second_leg = Trajectory2D.compute(sp, sv, final_target, self.v_max, self.a_max)
            candidates.append(TwoSegmentTrajectory(first_leg=first_leg, switch_t=switch_t, second_leg=second_leg))
        if not candidates:
            # First leg is too short for even one switch-time step (e.g. the
            # intermediate target is very close) -- fall back to switching
            # at the leg's own end, the earliest point that's still valid.
            switch_t = first_leg.duration
            (sp, sv) = first_leg.state_at(switch_t)
            second_leg = Trajectory2D.compute(sp, sv, final_target, self.v_max, self.a_max)
            candidates.append(TwoSegmentTrajectory(first_leg=first_leg, switch_t=switch_t, second_leg=second_leg))
        return candidates

    def _intermediate_targets(
        self, robot_id: int, p0: Tuple[float, float], final_target: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        """The previous tick's winning intermediate target, tried first (and
        alone, if it's still collision-free -- see `plan()`'s early-out),
        followed by fresh random candidates sorted toward the final target.

        Randomizing which target wins every single tick -- even among
        "acceptable" candidates -- was tried first and caused a genuine
        stall: a fresh random point can be collision-free while pointing
        sideways or backward relative to real progress just as easily as one
        that helps, and with a completely independent random draw every
        tick, the winning candidate's direction changed tick to tick with no
        continuity, so net displacement over several seconds averaged out to
        near zero even though each individual tick's chosen trajectory
        looked reasonable in isolation (confirmed live in the mirror_swap
        6v6 scenario: commanded velocity direction flipped sign tick to tick
        for 200+ consecutive ticks once robots got close enough to be
        mutually obstacle-dense). Checking the previous winner FIRST, and
        only falling through to fresh candidates when it's no longer valid,
        is what actually gives this the stability the paper's "cache the
        previous target" note describes (section 2.2) -- appending it to a
        list that a better-angled random point could still out-sort was not
        enough on its own.
        """
        last = self._last_intermediate_target.get(robot_id)
        fresh: List[Tuple[float, float]] = []
        r = config.INTERMEDIATE_TARGET_RADIUS
        for _ in range(config.N_INTERMEDIATE_TARGETS):
            angle = self._rng.uniform(0, 2 * math.pi)
            fresh.append((p0[0] + r * math.cos(angle), p0[1] + r * math.sin(angle)))

        # Sort by angle between (p0 -> final_target) and (p0 -> candidate),
        # smallest first -- paper section 2.2: "This favors paths pointing
        # towards the target."
        fx, fy = final_target[0] - p0[0], final_target[1] - p0[1]
        final_angle = math.atan2(fy, fx)

        def angular_distance(t: Tuple[float, float]) -> float:
            cx, cy = t[0] - p0[0], t[1] - p0[1]
            if abs(cx) < 1e-9 and abs(cy) < 1e-9:
                return math.pi
            cand_angle = math.atan2(cy, cx)
            diff = abs(cand_angle - final_angle)
            return min(diff, 2 * math.pi - diff)

        fresh.sort(key=angular_distance)
        return ([last] if last is not None else []) + fresh

    def _first_collision(
        self,
        trajectory,
        obstacle_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        start_t: float = 0.0,
    ) -> Optional[float]:
        """Returns the absolute trajectory time of the first collision at or
        after `start_t` (not a time-since-`start_t` offset), or None if
        clear through `min(trajectory.duration, MAX_LOOKAHEAD_TIME)`.
        `start_t` lets `_try_reuse` revalidate only the remaining portion of
        an already-committed trajectory instead of re-checking time this
        robot has already safely passed through.

        `obstacle_arrays` is the `_flatten_obstacles(...)` output for the
        SAME obstacle snapshot across every candidate a single `plan()` call
        checks -- callers flatten once and reuse it, they don't call
        `_flatten_obstacles` per candidate.

        Delegates the actual timestep x obstacle scan to
        `collision_numba.first_collision_numba`, a native port of what used
        to be this method's own loop. Profiling (see `planner.py`'s module
        docstring and `/home/isaac/.claude/jobs/d50ccf8c/tmp/profile_real_
        match.log`) found this loop -- ~18 adaptive timesteps x ~20
        obstacles per call, tens of thousands of calls per match -- was the
        dominant cost of the whole planner, ~2.4x slower than
        FastPathPlanner in a real match. A microbenchmark
        (`numba_bench.py`/`numba_bench2.py`) confirmed decorating individual
        leaf functions (e.g. `BangBang1D.state_at`) with `@njit` and calling
        them one at a time from Python is actually SLOWER (Python<->native
        call-boundary overhead dominates for a function this small); moving
        the WHOLE scan loop into one native call, so that boundary is
        crossed once per `_first_collision` call instead of once per
        obstacle per timestep, is what actually pays off (~24x-143x on
        synthetic benchmarks of the same loop shape). `collision_numba.py`
        inlines each obstacle kind's exact distance formula from
        `obstacles.py`/`bang_bang.py` by hand -- there's no shared source
        with the Protocol-dispatched Python classes, so any change to those
        classes' math must be mirrored there too.

        No longer takes `robot_id`: it was never used inside the scan loop
        itself, only threaded through for `plan()`'s later
        `_blocked_by_priority_obstacle` call, which still takes the raw
        Python `obstacles` list (it needs `owner_id`, not present in the
        flattened arrays) and is unaffected by this rewrite.
        """
        leg1_args, switch_t, leg2_args, duration = _flatten_query_trajectory(trajectory)
        static_arr, traj_arr, cv_arr, enemy_arr = obstacle_arrays
        t_col = _cn.first_collision_numba(
            leg1_args[0],
            leg1_args[1],
            leg1_args[2],
            leg1_args[3],
            leg1_args[4],
            switch_t,
            leg2_args[0],
            leg2_args[1],
            leg2_args[2],
            leg2_args[3],
            leg2_args[4],
            duration,
            start_t,
            config.MAX_LOOKAHEAD_TIME,
            config.ROBOT_RADIUS,
            config.MARGIN_V_MAX,
            config.MARGIN_BASE,
            config.MAX_TIME_STEP,
            config.MIN_TIME_STEP,
            config.STEP_DISTANCE_RATIO,
            static_arr,
            traj_arr,
            cv_arr,
            enemy_arr,
        )
        return None if t_col < 0.0 else t_col

    def _blocked_by_priority_obstacle(
        self, robot_id: int, trajectory, obstacles: List[TimedObstacle], collision_time: float
    ) -> bool:
        """True if, at `collision_time`, this trajectory collides with a
        teammate that outranks `robot_id` (see `_has_priority`). Ported from
        TIGERs' `MovingObstacleResultAcceptor.accept`: a candidate whose
        first collision is against a higher-priority obstacle is rejected
        unconditionally there, with no braking-distance carve-out -- the
        lower-priority robot must find a different candidate (or, if none
        exists, is the one that ends up commanded to stop; see
        `TrajectorySamplingController`). Checked only at the specific
        `collision_time` `_first_collision` already found, not re-scanned,
        since that's the instant that actually matters for this decision.
        """
        point, _ = trajectory.state_at(collision_time)
        for obstacle in obstacles:
            owner_id = getattr(obstacle, "owner_id", None)
            if owner_id is None or not _has_priority(owner_id, robot_id):
                continue
            d = obstacle.distance_at(collision_time, point) - config.ROBOT_RADIUS
            if d < config.MARGIN_BASE:
                return True
        return False

    def _shared_obstacles_for_tick(
        self, game: Game, field_bounds: FieldBounds, exempt_defense_area: bool
    ) -> List[TimedObstacle]:
        """The tick-invariant portion of the full per-`plan()`-call obstacle
        set -- enemies, ball, static field/defense-area geometry -- cached
        across the several robots planned within one tick. See `__init__`'s
        comment on `_shared_obstacle_cache_key` for why only this portion is
        safe to cache (the teammate portion, built separately in `plan()`,
        must stay per-robot-per-call).
        """
        cache_key = (game.ts, exempt_defense_area)
        if cache_key == self._shared_obstacle_cache_key:
            return self._shared_obstacles

        obstacles: List[TimedObstacle] = []
        robot_radius = config.ROBOT_RADIUS

        for r in game.enemy_robots.values():
            obstacles.append(
                enemy_obstacle_from_robot(
                    p=(r.p.x, r.p.y),
                    v=(r.v.x, r.v.y),
                    radius=robot_radius,
                    v_max=self.v_max,
                    a_max=self.a_max,
                )
            )

        if game.ball is not None:
            obstacles.append(
                ConstantVelocityObstacle(
                    p0=(game.ball.p.x, game.ball.p.y), v=(game.ball.v.x, game.ball.v.y), radius=0.0215
                )
            )

        tl, br = np.array(field_bounds.top_left), np.array(field_bounds.bottom_right)
        tr = np.array([field_bounds.bottom_right[0], field_bounds.top_left[1]])
        bl = np.array([field_bounds.top_left[0], field_bounds.bottom_right[1]])
        static_segments = [(tl, tr), (tr, br), (br, bl), (bl, tl)]
        obstacles.extend(collect_static_obstacles(static_segments, margin=0.0))

        if not exempt_defense_area:
            corners = game.field.enemy_defense_area
            min_x = min(c[0] for c in corners) - OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
            max_x = max(c[0] for c in corners) + OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
            min_y = min(c[1] for c in corners) - OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
            max_y = max(c[1] for c in corners) + OPPONENT_DEFENSE_AREA_KEEP_DISTANCE
            c0, c1 = np.array([min_x, max_y]), np.array([max_x, max_y])
            c2, c3 = np.array([max_x, min_y]), np.array([min_x, min_y])
            defense_segments = [(c0, c1), (c1, c2), (c2, c3), (c3, c0)]
            obstacles.extend(collect_static_obstacles(defense_segments, margin=0.0))

        self._shared_obstacle_cache_key = cache_key
        self._shared_obstacles = obstacles
        self._shared_obstacle_rows = _flatten_obstacle_rows(obstacles)
        return obstacles

    def _own_robot_obstacle(self, robot, radius: float, current_ts: float):
        committed = self._committed.get(robot.id)
        if committed is not None:
            committed_ts, trajectory, _target = committed
            return _CommittedTrajectoryObstacle(
                trajectory=trajectory, radius=radius, time_offset=current_ts - committed_ts, owner_id=robot.id
            )
        return ConstantVelocityObstacle(p0=(robot.p.x, robot.p.y), v=(robot.v.x, robot.v.y), radius=radius)

    def _commit(self, robot_id: int, ts: float, trajectory, target_pos: Tuple[float, float]) -> None:
        self._committed[robot_id] = (ts, trajectory, target_pos)


@dataclass(frozen=True)
class _CommittedTrajectoryObstacle:
    """Wraps another friendly robot's already-planned trajectory as an
    obstacle, re-based by `time_offset` (the querying robot's current
    `game.ts` minus the tick the wrapped trajectory was originally committed
    on) so a query at the querying robot's own local `t` lands on the right
    point in the wrapped trajectory's *own* timeline. Needed because
    trajectories now persist across ticks via `_try_reuse` instead of being
    regenerated (and re-based to t=0) every tick -- two robots committed on
    different ticks otherwise have misaligned time origins.
    """

    trajectory: object
    radius: float
    time_offset: float
    # The friendly robot this trajectory belongs to -- `None` for obstacle
    # types that aren't a specific teammate (enemies, ball, static
    # geometry), which never carry priority over anyone (see
    # `_first_collision`'s use of this field).
    owner_id: Optional[int] = None

    def distance_at(self, t: float, point: Tuple[float, float]) -> float:
        query_t = max(0.0, min(t + self.time_offset, self.trajectory.duration))
        (ox, oy), _ = self.trajectory.state_at(query_t)
        return math.hypot(point[0] - ox, point[1] - oy) - self.radius


@dataclass(frozen=True)
class TwoSegmentTrajectory:
    """A trajectory to an intermediate target, switched at `switch_t` for a
    fresh trajectory toward the real final target -- paper section 2.2.
    """

    first_leg: Trajectory2D
    switch_t: float
    second_leg: Trajectory2D

    @property
    def duration(self) -> float:
        return self.switch_t + self.second_leg.duration

    def state_at(self, t: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        if t <= self.switch_t:
            return self.first_leg.state_at(t)
        return self.second_leg.state_at(t - self.switch_t)

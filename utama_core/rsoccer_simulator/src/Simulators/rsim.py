from typing import Dict, List

import numpy as np

from utama_core.rsoccer_simulator.src.Entities import Field, Frame, FrameSSL, FrameVSS
from utama_core.rsoccer_simulator.src.Simulators.robosim.robosim_wrapper import (
    RSimSubprocessWrapper,
)


class RSim:
    def __init__(
        self,
        field_type: int,
        n_robots_blue: int,
        n_robots_yellow: int,
        time_step_ms: int,
    ):
        self.n_robots_blue = n_robots_blue
        self.n_robots_yellow = n_robots_yellow

        self.simulator: RSimSubprocessWrapper = self._init_simulator(
            field_type=field_type,
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            time_step_ms=time_step_ms,
        )
        self.field = self.get_field_params()
        # `simulator.step()` already returns the post-step state over the same
        # pipe round-trip — `send_commands()` caches it here so `get_frame()`
        # can reuse it instead of issuing a second, separate `get_state()`
        # round-trip for the same information. Also required for correctness,
        # not just speed: `SSLWorld::getState()` computes robot/ball velocity
        # by finite-differencing against whatever the *previous* call
        # returned (see `vendor/rSim/FORK_NOTES.md`'s "get_state() is
        # stateful, not idempotent" note) — a fresh `get_state()` call issued
        # right after `step()` (with no simulation advancing in between)
        # silently zeroes the velocity fields instead of returning the real
        # post-step velocity.
        self._last_state = None

    def reset(self, frame: Frame):
        placement_pos = self._placement_dict_from_frame(frame)
        self.simulator.reset(
            placement_pos["ball_pos"],
            placement_pos["blue_robots_pos"],
            placement_pos["yellow_robots_pos"],
        )
        # `_last_state` is a cache of the *last `step()` call's* response — a
        # reset doesn't go through `step()`, so any previously cached state
        # is now stale (positions/velocities from before the reset) and must
        # not be reused. `get_frame()` falls back to a fresh `get_state()`
        # call whenever this is `None`.
        self._last_state = None

    def stop(self):
        self.simulator.close()
        del self.simulator

    def send_commands(self, commands):
        raise NotImplementedError

    def get_frame(self) -> Frame:
        raise NotImplementedError

    def get_field_params(self):
        return Field(**self.simulator.get_field_params())

    def _placement_dict_from_frame(self, frame: Frame):
        replacement_pos: Dict[str, np.ndarray] = {}

        ball_pos: List[float] = [
            frame.ball.x,
            frame.ball.y,
            frame.ball.v_x,
            frame.ball.v_y,
        ]
        replacement_pos["ball_pos"] = np.array(ball_pos)

        blue_pos: List[List[float]] = []
        for robot in frame.robots_blue.values():
            robot_pos: List[float] = [robot.x, robot.y, robot.theta]
            blue_pos.append(robot_pos)
        replacement_pos["blue_robots_pos"] = np.array(blue_pos)

        yellow_pos: List[List[float]] = []
        for robot in frame.robots_yellow.values():
            robot_pos: List[float] = [robot.x, robot.y, robot.theta]
            yellow_pos.append(robot_pos)
        replacement_pos["yellow_robots_pos"] = np.array(yellow_pos)

        return replacement_pos

    def _init_simulator(
        self,
        field_type,
        n_robots_blue,
        n_robots_yellow,
        ball_pos,
        blue_robots_pos,
        yellow_robots_pos,
        time_step_ms,
    ) -> RSimSubprocessWrapper:
        raise NotImplementedError


class RSimVSS(RSim):
    def send_commands(self, commands):
        sim_commands = np.zeros((self.n_robots_blue + self.n_robots_yellow, 2), dtype=np.float64)

        for cmd in commands:
            if cmd.yellow:
                rbt_id = self.n_robots_blue + cmd.id
            else:
                rbt_id = cmd.id
            sim_commands[rbt_id][0] = cmd.v_wheel0
            sim_commands[rbt_id][1] = cmd.v_wheel1
        self._last_state = self.simulator.step(sim_commands)

    def get_frame(self) -> FrameVSS:
        # Reuse the state `step()` already returned this tick instead of a
        # second, separate `get_state()` round-trip — see the `_last_state`
        # comment in `RSim.__init__`/`reset()` for why this is also a
        # correctness fix, not just avoiding redundant I/O. Fall back to a
        # real fetch when there's no cached step yet (first call after
        # construction, or right after a `reset()`).
        state = self._last_state if self._last_state is not None else self.simulator.get_state()
        # Update frame with new state
        frame = FrameVSS()
        frame.parse(state, self.n_robots_blue, self.n_robots_yellow)

        return frame

    def _init_simulator(
        self,
        field_type,
        n_robots_blue,
        n_robots_yellow,
        time_step_ms,
    ) -> RSimSubprocessWrapper:
        return RSimSubprocessWrapper(
            sim_type="VSS",
            n_blue=n_robots_blue,
            n_yellow=n_robots_yellow,
            field_type=field_type,
            time_step_ms=time_step_ms,
        )


class RSimSSL(RSim):
    def send_commands(self, commands):
        sim_cmds = np.zeros((self.n_robots_blue + self.n_robots_yellow, 8), dtype=np.float64)

        for cmd in commands:
            if cmd.yellow:
                rbt_id = self.n_robots_blue + cmd.id
            else:
                rbt_id = cmd.id
            if cmd.wheel_speed:
                sim_cmds[rbt_id][0] = cmd.wheel_speed
                sim_cmds[rbt_id][1] = cmd.v_wheel0
                sim_cmds[rbt_id][2] = cmd.v_wheel1
                sim_cmds[rbt_id][3] = cmd.v_wheel2
                sim_cmds[rbt_id][4] = cmd.v_wheel3
                sim_cmds[rbt_id][5] = cmd.kick_v_x
                sim_cmds[rbt_id][6] = cmd.kick_v_z
                sim_cmds[rbt_id][7] = cmd.dribbler
            else:
                sim_cmds[rbt_id][0] = cmd.wheel_speed
                sim_cmds[rbt_id][1] = cmd.v_x
                sim_cmds[rbt_id][2] = cmd.v_y
                sim_cmds[rbt_id][3] = cmd.v_theta
                sim_cmds[rbt_id][5] = cmd.kick_v_x
                sim_cmds[rbt_id][6] = cmd.kick_v_z
                sim_cmds[rbt_id][7] = cmd.dribbler

        self._last_state = self.simulator.step(sim_cmds)

    def get_frame(self) -> FrameSSL:
        # Reuse the state `step()` already returned this tick instead of a
        # second, separate `get_state()` round-trip — see the `_last_state`
        # comment in `RSim.__init__`/`reset()` for why this is also a
        # correctness fix, not just avoiding redundant I/O. Fall back to a
        # real fetch when there's no cached step yet (first call after
        # construction, or right after a `reset()`).
        state = self._last_state if self._last_state is not None else self.simulator.get_state()
        # Update frame with new state
        frame = FrameSSL()
        frame.parse(state, self.n_robots_blue, self.n_robots_yellow)

        return frame

    def _init_simulator(
        self,
        field_type,
        n_robots_blue,
        n_robots_yellow,
        time_step_ms,
    ) -> RSimSubprocessWrapper:
        return RSimSubprocessWrapper(
            sim_type="SSL",
            n_blue=n_robots_blue,
            n_yellow=n_robots_yellow,
            field_type=field_type,
            time_step_ms=time_step_ms,
        )

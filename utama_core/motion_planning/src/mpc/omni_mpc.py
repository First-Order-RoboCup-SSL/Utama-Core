"""
Fast Omnidirectional MPC for SSL Robots
SILENT VERSION (High Performance)
"""

import numpy as np
import cvxpy as cp
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time


@dataclass
class OmniMPCConfig:
    """Configuration for omnidirectional MPC"""
    # Prediction horizon
    T: int = 5          # Keep short for speed
    DT: float = 0.05    # 50ms timestep

    # Cost weights (AGGRESSIVE)
    Q_pos: float = 80.0   
    Q_vel: float = 200.0  
    R_accel: float = 0.01 
    R_jerk: float = 0.001 
    Q_obstacle: float = 50.0 

    # Robot limits 
    max_vel: float = 4.0    
    max_accel: float = 15.0 
    robot_radius: float = 0.09  

    # Safety margins
    safety_base: float = 0.15 
    safety_vel_coeff: float = 0.05 

    # Solver
    max_solve_time: float = 0.010  # Reduced timeout to prevent lag spikes
    verbose: bool = False 


class OmnidirectionalMPC:
    def __init__(self, config: OmniMPCConfig = None):
        self.config = config or OmniMPCConfig()
        self._build_dynamics()
        self.prev_states = None
        self.prev_controls = None
        self.solve_time = 0.0
        self.success = True
        
        # Only print ONCE at startup
        print(f"[OmniMPC] SILENT AGGRESSIVE SOLVER LOADED (T={self.config.T})")

    def _build_dynamics(self):
        dt = self.config.DT
        self.A = np.array([
            [1, 0, dt, 0], [0, 1, 0, dt],
            [0, 0, 1, 0],  [0, 0, 0, 1]
        ])
        self.B = np.array([
            [0, 0], [0, 0],
            [dt, 0], [0, dt]
        ])

    def solve(self, current_state, goal_pos, obstacles=None):
        start_time = time.time()
        T = self.config.T
        dt = self.config.DT
        
        X = cp.Variable((4, T+1))
        U = cp.Variable((2, T))
        
        cost = 0
        constraints = [X[:, 0] == current_state]

        for k in range(T):
            # Dynamics
            constraints += [X[:, k+1] == self.A @ X[:, k] + self.B @ U[:, k]]
            
            # Limits
            constraints += [cp.norm(X[2:4, k], 2) <= self.config.max_vel]
            constraints += [cp.norm(U[:, k], 2) <= self.config.max_accel]

            # Tracking Cost
            curr_dist = np.linalg.norm(np.array(goal_pos) - current_state[0:2])
            est_dist = max(0, curr_dist - (self.config.max_vel * 0.7) * (k * dt))
            max_safe_speed = np.sqrt(2 * self.config.max_accel * est_dist)
            target_speed = min(self.config.max_vel, max_safe_speed)
            
            if est_dist < 0.05: target_speed = 0.0
            
            if curr_dist > 0.001:
                dir_vec = (np.array(goal_pos) - current_state[0:2]) / curr_dist
                ref_vel = dir_vec * target_speed
            else:
                ref_vel = np.zeros(2)

            # Debug prints REMOVED for performance
            
            cost += self.config.Q_pos * cp.sum_squares(X[0:2, k] - goal_pos)
            cost += self.config.Q_vel * cp.sum_squares(X[2:4, k] - ref_vel)
            cost += self.config.R_accel * cp.sum_squares(U[:, k])

        cost += self.config.Q_pos * cp.sum_squares(X[0:2, T] - goal_pos)
        
        problem = cp.Problem(cp.Minimize(cost), constraints)

        if self.prev_states is not None:
            X.value = self.prev_states
            U.value = self.prev_controls

        try:
            # Switched to OSQP - often faster for small QPs, or stick to CLARABEL
            problem.solve(solver=cp.CLARABEL, verbose=False, time_limit=self.config.max_solve_time)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                self.prev_states = X.value
                self.prev_controls = U.value
                return U.value.T, X.value.T, {'success': True, 'solve_time': time.time() - start_time}
            else:
                return None, None, {'success': False, 'solve_time': time.time() - start_time}
        except Exception as e:
            return None, None, {'success': False, 'solve_time': time.time() - start_time}

    def get_control_velocities(self, current_state, goal_pos, obstacles=None):
        controls, trajectory, info = self.solve(current_state, goal_pos, obstacles)

        if controls is None:
            info['fallback'] = True
            return 0.0, 0.0, info

        # TURBO HACK: Lookahead 3 steps (or max T)
        lookahead_step = min(3, self.config.T)
        vx = trajectory[lookahead_step, 2]
        vy = trajectory[lookahead_step, 3]

        return vx, vy, info
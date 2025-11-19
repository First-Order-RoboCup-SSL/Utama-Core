"""
MPC Configuration for SSL Robot Local Planning

Adapted for differential drive robots with obstacle avoidance.
Author: Based on iterative linear MPC approach
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MPCConfig:
    """Configuration parameters for MPC controller"""

    # State and control dimensions
    NX: int = 5  # State: [x, y, theta, v, omega]
    NU: int = 2  # Control: [linear_accel, angular_accel]

    # Prediction horizon
    T: int = 25  # Number of steps (user requested long horizon)
    DT: float = 0.02  # Time step [s] - 50Hz internal rate (allows 60Hz control)

    # Cost function weights
    Q_pos: np.ndarray = None  # Position tracking cost
    Q_vel: np.ndarray = None  # Velocity tracking cost
    Q_terminal: np.ndarray = None  # Terminal state cost
    R: np.ndarray = None  # Control effort cost
    Rd: np.ndarray = None  # Control rate (smoothness) cost

    # Obstacle avoidance
    Q_obstacle: float = 10.0  # Obstacle avoidance weight
    safety_margin_base: float = 0.18  # Base safety radius [m] (2 * ROBOT_RADIUS)
    safety_margin_velocity_coeff: float = 0.1  # Velocity-dependent margin [s]
    obstacle_barrier_gain: float = 5.0  # Barrier function gain

    # Robot physical constraints
    robot_radius: float = 0.09  # Robot radius [m]
    max_velocity: float = 4.0  # Maximum linear velocity [m/s]
    min_velocity: float = -2.0  # Minimum linear velocity (reverse) [m/s]
    max_angular_velocity: float = 4.0  # Maximum angular velocity [rad/s]
    max_acceleration: float = 2.0  # Maximum linear acceleration [m/s²]
    max_angular_acceleration: float = 8.0  # Maximum angular acceleration [rad/s²]

    # Iterative MPC parameters
    max_iter: int = 3  # Maximum iterations for linearization
    du_threshold: float = 0.05  # Convergence threshold for control change

    # Goal tolerance
    goal_tolerance: float = 0.05  # Distance to goal considered "reached" [m]

    # Solver settings
    solver_max_time: float = 0.015  # Max solver time [s] - must be < 16ms for 60Hz
    solver_verbose: bool = False

    def __post_init__(self):
        """Initialize cost matrices with default values if not provided"""
        if self.Q_pos is None:
            # Position error cost: [x, y]
            self.Q_pos = np.diag([2.0, 2.0])

        if self.Q_vel is None:
            # Velocity tracking cost: [theta, v, omega]
            # Lower weight on theta, higher on velocities
            self.Q_vel = np.diag([0.5, 0.5, 0.3])

        if self.Q_terminal is None:
            # Higher terminal cost for position
            self.Q_terminal = np.diag([5.0, 5.0, 0.5, 0.5, 0.3])

        if self.R is None:
            # Control effort cost: [linear_accel, angular_accel]
            self.R = np.diag([0.01, 0.1])

        if self.Rd is None:
            # Control rate cost: [Δlinear_accel, Δangular_accel]
            self.Rd = np.diag([0.05, 0.5])

    def get_velocity_dependent_safety_radius(self, velocity: float) -> float:
        """
        Calculate safety radius based on current velocity.
        Higher speed = larger safety bubble to allow more reaction time.

        Args:
            velocity: Current linear velocity [m/s]

        Returns:
            Safety radius [m]
        """
        return self.safety_margin_base + self.safety_margin_velocity_coeff * abs(velocity)

    def prediction_horizon_time(self) -> float:
        """Total time covered by prediction horizon [s]"""
        return self.T * self.DT


# Default configuration for simulation
def get_default_sim_config() -> MPCConfig:
    """Get default MPC configuration for simulation"""
    return MPCConfig(
        T=25,
        DT=0.02,
        max_velocity=4.0,
        max_acceleration=2.0,
    )


# Configuration for real robots (more conservative)
def get_real_robot_config() -> MPCConfig:
    """Get MPC configuration for real robots"""
    return MPCConfig(
        T=15,  # Shorter horizon for faster computation
        DT=0.03,
        max_velocity=0.2,
        max_acceleration=0.3,
        max_angular_velocity=0.5,
        max_angular_acceleration=2.0,
        safety_margin_base=0.20,  # More conservative
    )

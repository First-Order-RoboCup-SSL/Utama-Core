"""
Performance monitoring for comparing PID vs MPC

Tracks metrics like collisions, solve times, path smoothness, etc.
"""

import time
from collections import defaultdict, deque
from typing import Dict, List

import numpy as np


class PerformanceMonitor:
    """Monitor and compare controller performance"""

    def __init__(self, window_size: int = 300):  # 5 seconds at 60Hz
        self.window_size = window_size

        # Metrics storage (last N frames)
        self.collision_distances = deque(maxlen=window_size)
        self.solve_times = deque(maxlen=window_size)
        self.velocities = deque(maxlen=window_size)
        self.accelerations = deque(maxlen=window_size)
        self.distance_to_goal = deque(maxlen=window_size)

        # Counters
        self.collision_count = 0
        self.near_collision_count = 0  # Within safety margin
        self.total_frames = 0

        # Timing
        self.start_time = time.time()

        # Last velocity for acceleration calculation
        self.last_vel = None

    def update(
        self,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: List,
        solve_time: float = None,
        min_safe_distance: float = 0.20,
    ):
        """Update metrics for current frame"""
        self.total_frames += 1

        # Distance to goal
        dist_to_goal = np.hypot(goal_pos[0] - robot_pos[0], goal_pos[1] - robot_pos[1])
        self.distance_to_goal.append(dist_to_goal)

        # Check collisions with obstacles
        min_dist_to_obstacle = float("inf")
        for obs in obstacles:
            obs_x, obs_y, _, _, obs_radius = obs
            dist = np.hypot(robot_pos[0] - obs_x, robot_pos[1] - obs_y)
            min_dist_to_obstacle = min(min_dist_to_obstacle, dist)

            # Collision check (robot radius + obstacle radius)
            if dist < (0.09 + obs_radius):
                self.collision_count += 1
            elif dist < min_safe_distance:
                self.near_collision_count += 1

        self.collision_distances.append(min_dist_to_obstacle)

        # Velocity
        speed = np.hypot(robot_vel[0], robot_vel[1])
        self.velocities.append(speed)

        # Acceleration (smoothness metric)
        if self.last_vel is not None:
            accel = np.hypot(robot_vel[0] - self.last_vel[0], robot_vel[1] - self.last_vel[1]) / (1 / 60)  # Assume 60Hz
            self.accelerations.append(accel)
        self.last_vel = robot_vel.copy()

        # Solve time
        if solve_time is not None:
            self.solve_times.append(solve_time)

    def get_stats(self) -> Dict:
        """Get summary statistics"""
        runtime = time.time() - self.start_time

        stats = {
            "runtime": runtime,
            "total_frames": self.total_frames,
            "fps": self.total_frames / runtime if runtime > 0 else 0,
            # Collision metrics
            "collision_count": self.collision_count,
            "near_collision_count": self.near_collision_count,
            "collision_rate": self.collision_count / max(1, self.total_frames),
            "min_distance": min(self.collision_distances) if self.collision_distances else float("inf"),
            "avg_distance": np.mean(self.collision_distances) if self.collision_distances else 0,
            # Performance metrics
            "avg_solve_time_ms": np.mean(self.solve_times) * 1000 if self.solve_times else 0,
            "max_solve_time_ms": np.max(self.solve_times) * 1000 if self.solve_times else 0,
            "solve_time_std_ms": np.std(self.solve_times) * 1000 if self.solve_times else 0,
            # Smoothness metrics
            "avg_speed": np.mean(self.velocities) if self.velocities else 0,
            "avg_acceleration": np.mean(self.accelerations) if self.accelerations else 0,
            "jerk": np.std(self.accelerations) if len(self.accelerations) > 1 else 0,
            # Goal tracking
            "avg_dist_to_goal": np.mean(self.distance_to_goal) if self.distance_to_goal else 0,
        }

        return stats

    def print_report(self, controller_name: str = "Controller"):
        """Print performance report"""
        stats = self.get_stats()

        print(f"\n{'='*60}")
        print(f"{controller_name} Performance Report")
        print(f"{'='*60}")
        print(f"Runtime: {stats['runtime']:.1f}s | Frames: {stats['total_frames']} | FPS: {stats['fps']:.1f}")
        print("\nCollisions:")
        print(f"  Total collisions: {stats['collision_count']}")
        print(f"  Near collisions: {stats['near_collision_count']}")
        print(f"  Collision rate: {stats['collision_rate']*100:.2f}%")
        print(f"  Min distance to obstacle: {stats['min_distance']:.3f}m")
        print(f"  Avg distance to obstacle: {stats['avg_distance']:.3f}m")
        print("\nPerformance:")
        print(f"  Avg solve time: {stats['avg_solve_time_ms']:.2f}ms")
        print(f"  Max solve time: {stats['max_solve_time_ms']:.2f}ms")
        print(f"  Solve time std: {stats['solve_time_std_ms']:.2f}ms")
        print("\nSmoothness:")
        print(f"  Avg speed: {stats['avg_speed']:.3f} m/s")
        print(f"  Avg acceleration: {stats['avg_acceleration']:.3f} m/s²")
        print(f"  Jerk (std of accel): {stats['jerk']:.3f}")
        print("\nGoal Tracking:")
        print(f"  Avg distance to goal: {stats['avg_dist_to_goal']:.3f}m")
        print(f"{'='*60}\n")

        return stats


# Global monitor instances
_pid_monitor = None
_mpc_monitor = None


def get_monitor(use_mpc: bool) -> PerformanceMonitor:
    """Get or create performance monitor"""
    global _pid_monitor, _mpc_monitor

    if use_mpc:
        if _mpc_monitor is None:
            _mpc_monitor = PerformanceMonitor()
        return _mpc_monitor
    else:
        if _pid_monitor is None:
            _pid_monitor = PerformanceMonitor()
        return _pid_monitor


def print_comparison():
    """Print comparison between PID and MPC"""
    global _pid_monitor, _mpc_monitor

    if _pid_monitor is not None:
        _pid_monitor.print_report("PID Controller")

    if _mpc_monitor is not None:
        _mpc_monitor.print_report("MPC Controller")

    if _pid_monitor is not None and _mpc_monitor is not None:
        pid_stats = _pid_monitor.get_stats()
        mpc_stats = _mpc_monitor.get_stats()

        print(f"\n{'='*60}")
        print("MPC vs PID Comparison")
        print(f"{'='*60}")

        print("\n🎯 Collision Reduction:")
        if pid_stats["collision_count"] > 0:
            reduction = (1 - mpc_stats["collision_count"] / pid_stats["collision_count"]) * 100
            print(f"  {reduction:+.1f}% fewer collisions with MPC")

        print("\n⚡ Speed:")
        speedup = pid_stats["avg_solve_time_ms"] / max(0.001, mpc_stats["avg_solve_time_ms"])
        print(f"  MPC is {speedup:.1f}x the solve time of PID")
        print("  (but MPC plans ahead, PID is reactive)")

        print("\n📊 Smoothness:")
        print(f"  MPC jerk: {mpc_stats['jerk']:.3f}")
        print(f"  PID jerk: {pid_stats['jerk']:.3f}")
        smoother = "MPC" if mpc_stats["jerk"] < pid_stats["jerk"] else "PID"
        print(f"  → {smoother} produces smoother motion")

        print(f"{'='*60}\n")

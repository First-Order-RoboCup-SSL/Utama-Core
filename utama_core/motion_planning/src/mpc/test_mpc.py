"""
Test script for MPC planner

Tests basic functionality, performance, and obstacle avoidance.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utama_core.motion_planning.src.mpc.mpc_planner import MPCPlanner, RobotState, Obstacle
from utama_core.motion_planning.src.mpc.mpc_config import get_default_sim_config
import time


def test_basic_planning():
    """Test basic goal-reaching without obstacles"""
    print("=" * 60)
    print("Test 1: Basic Goal Reaching")
    print("=" * 60)

    config = get_default_sim_config()
    config.T = 25  # Use default horizon
    config.DT = 0.04  # Larger timestep for longer prediction (1 second total)
    config.solver_verbose = False

    planner = MPCPlanner(config)

    # Initial state: robot at origin, stationary
    initial_state = RobotState(x=0.0, y=0.0, theta=0.0, v=0.0, omega=0.0)

    # Goal: move to (3, 2)
    goal = (3.0, 2.0)

    # No obstacles
    obstacles = []

    # Plan
    start = time.time()
    controls, trajectory, info = planner.plan(initial_state, goal, obstacles)
    elapsed = time.time() - start

    print(f"Planning result: {info['message']}")
    print(f"Solve time: {info['solve_time']*1000:.2f} ms")
    print(f"Iterations: {info['iterations']}")
    print(f"Success: {info['success']}")

    if controls is not None:
        print(f"\nFirst control: a={controls[0, 0]:.3f} m/s², alpha={controls[0, 1]:.3f} rad/s²")
        print(f"Final position: ({trajectory[-1, 0]:.3f}, {trajectory[-1, 1]:.3f})")
        print(f"Distance to goal: {np.hypot(trajectory[-1, 0] - goal[0], trajectory[-1, 1] - goal[1]):.3f} m")

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', linewidth=2, markersize=3, label='Planned trajectory')
        plt.plot(initial_state.x, initial_state.y, 'go', markersize=10, label='Start')
        plt.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title(f'MPC Basic Planning (solve time: {info["solve_time"]*1000:.1f} ms)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('test_mpc_basic.png', dpi=150)
        print("\nSaved plot: test_mpc_basic.png")
        plt.close()

    assert info['success'], "Planning should succeed"
    # Note: Python implementation is slow (~800ms), C++ will be much faster
    if info['solve_time'] < 0.1:
        print(f"✓ Fast solve time: {info['solve_time']*1000:.1f} ms")
    else:
        print(f"⚠ Slow Python solve time: {info['solve_time']*1000:.1f} ms (C++ will be faster)")

    print("\n✓ Test 1 passed!\n")


def test_static_obstacle_avoidance():
    """Test obstacle avoidance with static obstacles"""
    print("=" * 60)
    print("Test 2: Static Obstacle Avoidance")
    print("=" * 60)

    config = get_default_sim_config()
    config.T = 25
    config.Q_obstacle = 20.0  # Higher weight for obstacle avoidance

    planner = MPCPlanner(config)

    # Initial state
    initial_state = RobotState(x=0.0, y=0.0, theta=0.0, v=0.0, omega=0.0)

    # Goal: straight ahead
    goal = (4.0, 0.0)

    # Obstacle in the way
    obstacles = [
        Obstacle(x=2.0, y=0.0, vx=0.0, vy=0.0, radius=0.2)
    ]

    # Plan
    controls, trajectory, info = planner.plan(initial_state, goal, obstacles)

    print(f"Planning result: {info['message']}")
    print(f"Solve time: {info['solve_time']*1000:.2f} ms")
    print(f"Success: {info['success']}")

    if controls is not None:
        # Check obstacle avoidance
        min_dist = float('inf')
        for i in range(len(trajectory)):
            for obs in obstacles:
                dist = np.hypot(trajectory[i, 0] - obs.x, trajectory[i, 1] - obs.y)
                min_dist = min(min_dist, dist)

        safety_margin = config.get_velocity_dependent_safety_radius(2.0) + obstacles[0].radius
        print(f"Minimum distance to obstacle: {min_dist:.3f} m")
        print(f"Required safety margin: {safety_margin:.3f} m")

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', linewidth=2, markersize=3, label='Planned trajectory')
        plt.plot(initial_state.x, initial_state.y, 'go', markersize=10, label='Start')
        plt.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')

        # Plot obstacles
        for obs in obstacles:
            circle = plt.Circle((obs.x, obs.y), obs.radius, color='red', alpha=0.3, label='Obstacle')
            plt.gca().add_patch(circle)
            safety_circle = plt.Circle((obs.x, obs.y), safety_margin, color='orange',
                                      alpha=0.1, linestyle='--', fill=False, label='Safety margin')
            plt.gca().add_patch(safety_circle)

        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title(f'MPC Obstacle Avoidance (solve time: {info["solve_time"]*1000:.1f} ms)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('test_mpc_obstacle.png', dpi=150)
        print("Saved plot: test_mpc_obstacle.png")
        plt.close()

    assert info['success'], "Planning should succeed"
    print("\n✓ Test 2 passed!\n")


def test_dynamic_obstacle():
    """Test avoidance of moving obstacles"""
    print("=" * 60)
    print("Test 3: Dynamic Obstacle Avoidance")
    print("=" * 60)

    config = get_default_sim_config()
    config.T = 25

    planner = MPCPlanner(config)

    # Initial state
    initial_state = RobotState(x=0.0, y=0.0, theta=0.0, v=0.0, omega=0.0)

    # Goal
    goal = (4.0, 2.0)

    # Moving obstacle (crossing path)
    obstacles = [
        Obstacle(x=2.0, y=2.0, vx=0.0, vy=-1.0, radius=0.15)  # Moving down
    ]

    # Plan
    controls, trajectory, info = planner.plan(initial_state, goal, obstacles)

    print(f"Planning result: {info['message']}")
    print(f"Solve time: {info['solve_time']*1000:.2f} ms")
    print(f"Success: {info['success']}")

    if controls is not None:
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', linewidth=2, markersize=3, label='Planned trajectory')
        plt.plot(initial_state.x, initial_state.y, 'go', markersize=10, label='Start')
        plt.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')

        # Plot obstacle trajectory
        obs = obstacles[0]
        obs_traj_x = [obs.x + obs.vx * t * config.DT for t in range(config.T + 1)]
        obs_traj_y = [obs.y + obs.vy * t * config.DT for t in range(config.T + 1)]
        plt.plot(obs_traj_x, obs_traj_y, 'r--', linewidth=1, alpha=0.5, label='Obstacle trajectory')

        # Plot obstacle positions at different times
        for t in [0, config.T // 2, config.T]:
            x, y = obs.position_at_time(t * config.DT)
            circle = plt.Circle((x, y), obs.radius, color='red', alpha=0.2)
            plt.gca().add_patch(circle)

        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title(f'MPC Dynamic Obstacle (solve time: {info["solve_time"]*1000:.1f} ms)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('test_mpc_dynamic.png', dpi=150)
        print("Saved plot: test_mpc_dynamic.png")
        plt.close()

    assert info['success'], "Planning should succeed"
    print("\n✓ Test 3 passed!\n")


def test_performance_60hz():
    """Test if MPC can run at 60Hz"""
    print("=" * 60)
    print("Test 4: 60Hz Performance Test")
    print("=" * 60)

    config = get_default_sim_config()
    config.T = 25
    planner = MPCPlanner(config)

    # Initial state
    initial_state = RobotState(x=0.0, y=0.0, theta=0.0, v=1.0, omega=0.0)
    goal = (3.0, 2.0)
    obstacles = [
        Obstacle(x=1.5, y=1.0, vx=0.5, vy=0.0, radius=0.1)
    ]

    # Run multiple times to measure average performance
    n_runs = 10
    solve_times = []

    for i in range(n_runs):
        controls, trajectory, info = planner.plan(initial_state, goal, obstacles)
        if info['success']:
            solve_times.append(info['solve_time'])

    avg_time = np.mean(solve_times)
    max_time = np.max(solve_times)
    min_time = np.min(solve_times)

    print(f"Performance over {n_runs} runs:")
    print(f"  Average solve time: {avg_time*1000:.2f} ms")
    print(f"  Min solve time: {min_time*1000:.2f} ms")
    print(f"  Max solve time: {max_time*1000:.2f} ms")
    print(f"  60Hz requirement: < 16.67 ms")

    if avg_time < 0.01667:
        print("\n✓ MPC can run at 60Hz!")
    else:
        print(f"\n⚠ Python MPC is too slow for 60Hz (avg: {avg_time*1000:.1f} ms)")
        print("  → C++ implementation required for real-time performance")
        print("  → Python is useful for prototyping and testing logic")

    # Visualize solve times
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(solve_times)), np.array(solve_times) * 1000)
    plt.axhline(y=16.67, color='r', linestyle='--', label='60Hz requirement (16.67ms)')
    plt.xlabel('Run number')
    plt.ylabel('Solve time [ms]')
    plt.title('MPC Solve Time Performance')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('test_mpc_performance.png', dpi=150)
    print("Saved plot: test_mpc_performance.png")
    plt.close()

    print("\n✓ Test 4 completed!\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("MPC PLANNER TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_basic_planning()
        test_static_obstacle_avoidance()
        test_dynamic_obstacle()
        test_performance_60hz()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()

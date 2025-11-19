"""
Quick test to verify MPC integration works without full simulator
"""

import numpy as np

from utama_core.motion_planning.src.mpc.omni_mpc import (
    OmnidirectionalMPC,
    OmniMPCConfig,
)

print("=" * 60)
print("Testing Omnidirectional MPC")
print("=" * 60)

# Create MPC with default config (optimized for speed)
mpc = OmnidirectionalMPC()

# Test solve
current_state = np.array([0.0, 0.0, 0.0, 0.0])  # At origin, stationary
goal = (3.0, 2.0)

# Add some obstacles
obstacles = [
    (1.5, 1.0, 0.0, 0.0, 0.1),  # Static obstacle
    (2.0, 0.5, 0.5, 0.0, 0.1),  # Moving obstacle
]

print(f"\nCurrent state: {current_state}")
print(f"Goal: {goal}")
print(f"Obstacles: {len(obstacles)}")

# Solve
vx, vy, info = mpc.get_control_velocities(current_state, goal, obstacles)

print(f"\n{'='*60}")
print("Results:")
print(f"  Success: {info['success']}")
print(f"  Solve time: {info['solve_time']*1000:.2f} ms")
if info["success"]:
    print(f"  Velocity command: vx={vx:.3f} m/s, vy={vy:.3f} m/s")
    print(f"  Speed: {np.hypot(vx, vy):.3f} m/s")
    print(f"  Fallback used: {info.get('fallback', False)}")
print(f"{'='*60}\n")

if info["success"] and info["solve_time"] < 0.05:
    print("✓ MPC is working and fast enough for real-time!")
elif info["success"]:
    print(f"⚠ MPC works but is slow ({info['solve_time']*1000:.1f} ms)")
else:
    print("❌ MPC failed to solve")

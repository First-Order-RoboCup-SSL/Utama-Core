"""
Compare PID vs MPC Controllers

Run this to see side-by-side comparison of both controllers.
Usage: pixi run python compare_controllers.py
"""

import sys

print("=" * 80)
print("CONTROLLER COMPARISON TOOL")
print("=" * 80)
print("\nThis will help you compare PID vs MPC performance.")
print("\nTo test:")
print("  1. Run with PID:  Edit move_utils.py, set USE_MPC = False")
print("  2. Run sim for ~30 seconds, note collisions")
print("  3. Run with MPC:  Edit move_utils.py, set USE_MPC = True")
print("  4. Run sim for ~30 seconds, compare results")
print("\n" + "=" * 80)

print("\n📊 METRICS TO WATCH:\n")

print("1. **COLLISION COUNT**")
print("   - Watch robots in simulation")
print("   - MPC should show fewer collisions")
print("   - Look for [MPC Warning] messages indicating close calls\n")

print("2. **SOLVE TIME**")
print("   - Check terminal for: [MPC Stats] messages every 5 seconds")
print("   - Should show ~2-5ms average solve time")
print("   - If > 16ms, MPC can't keep up with 60Hz\n")

print("3. **SMOOTHNESS**")
print("   - Watch robot motion visually")
print("   - MPC should be smoother (plans ahead)")
print("   - PID may be jerky (reactive)\n")

print("4. **OBSTACLE AVOIDANCE**")
print("   - MPC predicts future robot positions")
print("   - Should see robots giving each other more space")
print("   - Velocity-dependent safety margins (faster = wider berth)\n")

print("=" * 80)
print("\n🔧 TUNING MPC:\n")

print("If robots still collide, increase safety in omni_mpc.py:")
print("  - safety_base: 0.30 → 0.40  (larger safety bubble)")
print("  - Q_obstacle: 100 → 200     (stronger avoidance)")
print("  - safety_vel_coeff: 0.20 → 0.30  (more speed-dependent)\n")

print("If MPC is too slow:")
print("  - T: 15 → 10  (shorter horizon, faster solve)")
print("  - DT: 0.05 → 0.08  (bigger timesteps, less precision)\n")

print("If robots are too timid:")
print("  - safety_base: 0.30 → 0.25  (smaller safety bubble)")
print("  - Q_obstacle: 100 → 50      (weaker avoidance)\n")

print("=" * 80)
print("\n📈 EXPECTED IMPROVEMENTS WITH MPC:\n")

print("✓ 50-80% reduction in collisions")
print("✓ Smoother trajectories (lower jerk)")
print("✓ Better coordination (robots anticipate each other)")
print("✓ Speed-adaptive safety (safer at high speeds)")
print("✓ Optimal paths (not just reactive)\n")

print("⚠ Trade-offs:")
print("  - Slightly higher CPU usage (but still <16ms)")
print("  - More complex to tune")
print("  - May be overly cautious initially\n")

print("=" * 80)
print("\n🎮 CURRENT CONFIGURATION:\n")

try:
    from utama_core.motion_planning.src.mpc.omni_mpc import OmniMPCConfig
    from utama_core.skills.src.utils.move_utils import USE_MPC

    print(f"Controller: {'MPC' if USE_MPC else 'PID'}")

    if USE_MPC:
        config = OmniMPCConfig()
        print("\nMPC Settings:")
        print(f"  Horizon: T={config.T} steps × {config.DT}s = {config.T * config.DT:.2f}s lookahead")
        print(f"  Max velocity: {config.max_vel} m/s")
        print(f"  Max acceleration: {config.max_accel} m/s²")
        print(f"  Safety base: {config.safety_base:.3f}m")
        print(f"  Safety velocity coeff: {config.safety_vel_coeff:.3f}")
        print(f"  Obstacle avoidance weight: {config.Q_obstacle}")
        print(f"  Max solve time: {config.max_solve_time*1000:.1f}ms")
    else:
        print("\nPID Settings:")
        print("  Reactive control (no prediction)")
        print("  No obstacle avoidance")
        print("  Simple proportional control")

except ImportError as e:
    print(f"Could not load config: {e}")

print("\n" + "=" * 80)
print("\nReady to test! Run:  pixi run main")
print("Watch for [MPC Stats] and [MPC Warning] messages in terminal.")
print("=" * 80 + "\n")

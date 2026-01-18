# MPC C++ Port - Development Notes

> **Last Updated:** 2026-01-12
> **Branch:** `feature/local-planner-mpc-cpp`
> **Status:** MPC C++ working, needs testing and cleanup before merge

---

## Project Overview

This is the **Imperial College London RoboCup SSL 2025** robot soccer control system. We ported the Python MPC (Model Predictive Control) to C++ for performance.

### The Problem
- Python MPC using CVXPY + CLARABEL solver ran at **~7-20 FPS** (10-50ms per solve)
- We needed **60+ FPS** for smooth robot control

### The Solution
- Created a **C++ heuristic solver** using nanobind + Eigen
- Achieved **350+ FPS** with **0.02ms latency**

---

## Key Architecture Decision

### Why Heuristic Instead of Full QP Solver?

**Python MPC** uses `cp.CLARABEL` (a conic solver) because of circular constraints:
```python
cp.norm(X[2:4, k], 2) <= self.config.max_vel  # Circle constraint
```

**OSQP** (quadratic solver) cannot handle circle constraints natively - it only understands squares/polygons.

**C++ Solution:** Instead of porting OSQP (which would require ~200+ lines of CSC sparse matrix math), we implemented a **heuristic solver** that:
1. Uses proportional control toward goal
2. Uses exponential repulsion forces for obstacle avoidance
3. Clamps acceleration and velocity

This naturally produces "overdamped" behavior and is actually **faster** than a full QP solver.

---

## File Locations

### Core MPC Files

| File | Purpose |
|------|---------|
| `utama_core/motion_planning/src/mpc/omni_mpc.py` | Python MPC (CLARABEL solver) - reference implementation |
| `utama_core/motion_planning/src/mpc_cpp/src/OmniMPC.cpp` | C++ heuristic solver |
| `utama_core/motion_planning/src/mpc_cpp/src/OmniMPC.hpp` | C++ header with MPCConfig struct |
| `utama_core/motion_planning/src/mpc_cpp/src/bindings.cpp` | nanobind Python bindings |
| `utama_core/motion_planning/src/mpc_cpp/CMakeLists.txt` | CMake build configuration |
| `utama_core/motion_planning/src/mpc_cpp/pyproject.toml` | Python package config for scikit-build-core |

### Controller Wrappers

| File | Purpose |
|------|---------|
| `utama_core/motion_planning/src/controllers/mpc_controller.py` | Python MPC wrapper |
| `utama_core/motion_planning/src/controllers/mpc_cpp_controller.py` | C++ MPC wrapper |
| `utama_core/motion_planning/src/controllers/pid_controller.py` | PID controller (base class) |
| `utama_core/motion_planning/src/controllers/dwa_controller.py` | DWA controller |

### Control Scheme Selection

| File | Purpose |
|------|---------|
| `utama_core/motion_planning/src/common/control_schemes.py` | `ControlScheme` enum for selecting controller |
| `utama_core/run/strategy_runner.py` | Uses `ControlScheme` enum |
| `main.py` | Entry point - set `control_scheme=ControlScheme.MPC_CPP` |

---

## How to Build the C++ Extension

```bash
cd utama_core/motion_planning/src/mpc_cpp

# Clean build artifacts
rm -rf build _skbuild dist *.egg-info

# Build and install (use --force-reinstall to ensure fresh build)
pixi run pip install . --force-reinstall --no-cache-dir --no-build-isolation

# Verify it works
pixi run python -c "import mpc_cpp_extension; print('OK')"
```

### Dependencies Added to pixi.toml
```toml
pip = ">=25.3,<26"
scikit-build-core = ">=0.11.6,<0.12"
cmake = ">=3.20"
ninja = "*"
cxx-compiler = "*"
eigen = "*"
nanobind = "*"
```

---

## How to Switch Control Schemes

In `main.py`:
```python
from utama_core.motion_planning.src.common.control_schemes import ControlScheme

runner = StrategyRunner(
    ...
    control_scheme=ControlScheme.MPC_CPP,  # C++ MPC (fast)
    # control_scheme=ControlScheme.MPC,    # Python MPC (slow but reference)
    # control_scheme=ControlScheme.DWA,    # DWA controller
    # control_scheme=ControlScheme.PID,    # Basic PID
)
```

---

## C++ Heuristic Solver Algorithm

Located in `OmniMPC.cpp`:

```cpp
// 1. Distance check - determine if arriving
bool is_arriving = dist_to_goal < 0.40;

// 2. Calculate target velocity (zero if arriving)
if (is_arriving || dist_to_goal < 0.15) {
    ref_vel.setZero();
} else {
    ref_vel = direction_to_goal.normalized() * max_vel;
}

// 3. Proportional control
double gain = is_arriving ? 2.0 : 4.0;
acc = (ref_vel - current_vel) * gain;

// 4. Obstacle avoidance (exponential repulsion)
for (obstacle : obstacles) {
    if (dist < safety_distance * 1.2) {
        double force = 50.0 * exp(violation * 10.0);
        repulsion += direction_away * force;
    }
}
acc += repulsion;

// 5. Clamp acceleration and velocity
// 6. Integrate to get next velocity
```

### Config Parameters (MPCConfig struct)
```cpp
int T = 5;              // Horizon (not used in heuristic)
double DT = 0.05;       // Time step
double max_vel = 2.0;   // Max velocity m/s
double max_accel = 3.0; // Max acceleration m/s^2
double Q_pos = 200.0;   // (not used in heuristic - leftover from QP)
double Q_vel = 20.0;    // (not used)
double R_accel = 0.5;   // (not used)
double robot_radius = 0.09;
double obstacle_buffer_ratio = 1.25;
double safety_vel_coeff = 0.15;
```

---

## Known Issues / TODO

### Cleanup Before Merge
- [ ] Delete backup MPC files: `omni_mpc [high-damping].py`, `[loud].py`, `[mid-damping].py`, `[over-damping].py`
- [ ] Delete or move test files: `compare_controllers.py`, `test_mpc_integration.py`, `test_mpc_basic.png`
- [ ] Review `performance_monitor.py` - keep or remove?
- [ ] Run full test suite with friend's test cases

### Simulator Issue (rc-robosim)
The `rc-robosim` package fails to build due to CMake version incompatibility:
```
CMake Error: Compatibility with CMake < 3.5 has been removed from CMake.
```
**Workaround:** The robosim environment uses an old pybind11 that's incompatible with newer CMake. For now, use `mode="grsim"` if this is an issue.

### Performance Verification
- MPC C++ achieves **350+ FPS** with **0.02ms latency**
- Robots decelerate smoothly when approaching targets
- Obstacle avoidance working via exponential repulsion

---

## Testing the C++ Extension

### Quick Verification
```bash
pixi run python -c "
import mpc_cpp_extension
import numpy as np

config = mpc_cpp_extension.MPCConfig()
mpc = mpc_cpp_extension.OmniMPC(config)

state = np.array([0.0, 0.0, 0.0, 0.0])  # x, y, vx, vy
goal = np.array([1.0, 0.0])
obstacles = []

vx, vy, success = mpc.get_control_velocities(state, goal, obstacles)
print(f'vx={vx:.3f}, vy={vy:.3f}, success={success}')
"
```

### Full Simulation Test
```bash
pixi run main
```
Look for output like:
```
[MPCCpp Stats] FPS: 350.0 | Latency: 0.02ms | R0: 2.00m/s | ...
```

---

## Git Commands

### View changes from main
```bash
git diff --stat main...HEAD
git log --oneline main..HEAD
```

### Commit and push
```bash
git add -A
git commit -m "your message"
git push origin feature/local-planner-mpc-cpp
```

---

## Contact / Context

- **Project:** RoboCup SSL 2025 - Imperial College London
- **Codebase:** Utama-Core
- **This branch:** Porting Python MPC to C++ for performance
- **Previous assistant:** Gemini (did initial C++ port and heuristic solver design)
- **Current session:** Claude (enum refactor, build debugging, documentation)

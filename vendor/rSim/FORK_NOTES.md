# This is a vendored fork of robocin/rSim

Upstream: https://github.com/robocin/rSim
Forked at commit: `b413932b7b4bbc0fc399f6a979e4cfcc4f56264f` (tag `v1.2`, the
same version this repo's `pixi.toml` pins via `rc-robosim>=1.2,<2` on PyPI).

## Why this is here

`Utama-Core` depends on `rc-robosim` as a compiled PyPI package (the
`robosim` feature environment in `pixi.toml`), not as source — normally
nothing in this repo needs rSim's own source at all. This copy exists
because we found and fixed two real bugs in rSim's native ball/dribbler
physics (see `docs/roadmap.md`'s "TODO — add grsim as a CI/tournament
environment" section, item 1, for the full writeup) and there was nowhere
else to keep a rebuildable, git-trackable copy of the fix.

The patch itself is also saved as a plain diff at
`docs/patches/rSim-dribbler-release.diff` — that's the reviewable artifact.
This directory is the actual buildable source it applies to, so whoever
picks up the open regression (see roadmap TODO) doesn't have to re-clone
upstream and re-apply the diff by hand first.

## What's patched here (already applied, not a diff you need to apply)

1. `src/robosim/sslworld.cpp` — `SSLWorld::setActions()`'s dribbler-off
   command was silently dropped (`if (rbtAction[7] > 0) setDribbler(true)`
   with no `else`); now calls `setDribbler(rbtAction[7] > 0)`
   unconditionally.
2. `src/robosim/sslrobot.cpp` — `Kicker::unholdBall()` only destroyed the
   ODE hinge joint, leaving the ball in resting contact with the kicker box
   (which ODE's own contact solver then re-clamped in place every physics
   substep). Now also repositions the ball just clear of the kicker's
   collision envelope and gives it a small outward velocity.
3. `CMakeLists.txt` — unrelated compiler-compatibility fix needed to build
   at all on this machine's GCC 13 (pybind11 2.6.2's vendored headers assume
   `<cstdint>`/`<cstddef>` are transitively included; forces them via
   `-include`). May not be needed on other toolchains.

## Known issue: NOT YET SAFE TO SHIP

Building this and installing it over `rc-robosim==1.2` in
`.pixi/envs/robosim` fixes the dribbler-release bug (confirmed via
reproduction) but currently regresses
`utama_core/tests/kernel/test_referee_override.py::test_their_kickoff_clears_our_robots_outside_center_circle`
(passes on stock 1.2, fails on this patch — a robot's path planner stalls
short of the keep-out radius near the field center). Not root-caused. See
`docs/roadmap.md` for the full investigation and current status. Do not
install this over the pinned `rc-robosim` version in `.pixi/envs/robosim`
until that regression is understood and fixed.

## Why this is plain tracked files, not a git submodule

Considered a submodule (would keep rSim's own history separate and make the
fork relationship more explicit) but decided against it for now: a submodule
needs a real pushed remote to point to (no `First-Order-RoboCup-SSL/rSim`
fork exists on GitHub), and submodules have real day-to-day friction — not
cloned by default, easy to leave uninitialized, need CI awareness — that
isn't worth paying while this fix is unshipped and has an open regression
(see below). This directory isn't referenced by `pixi.toml` or installed
anywhere; it's a parked fix, not a build dependency.

If this fix gets finished and actually adopted (regression resolved, and
`pixi.toml` switched from the PyPI `rc-robosim` dependency to building from
this fork), that's the point to revisit this — either push a real fork to
GitHub and convert this to a submodule, or use pixi's own git-source
dependency support to point directly at that fork instead of vendoring
source at all.

## How to rebuild

From the `Utama-Core` repo root, using the `robosim` pixi environment's own
Python (it needs `pip`/`scikit-build`/`cmake<3.27`/`ninja`/
`setuptools_scm>=6.2`/`wheel` bootstrapped in first — that env doesn't ship
with a build toolchain by default):

```bash
.pixi/envs/robosim/bin/python -m ensurepip --default-pip
.pixi/envs/robosim/bin/python -m pip install "scikit-build" "cmake<3.27" "ninja" "setuptools_scm>=6.2" "wheel"

export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
export SKBUILD_CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
export CMAKE_PREFIX_PATH="$(pwd)/.pixi/envs/robosim"
export CMAKE_LIBRARY_PATH="$(pwd)/.pixi/envs/robosim/lib"
export CMAKE_INCLUDE_PATH="$(pwd)/.pixi/envs/robosim/include"

cd vendor/rSim
/path/to/Utama-Core/.pixi/envs/robosim/bin/python -m pip wheel . --no-deps --no-build-isolation -w /tmp/wheelout

# Install (overwrites the pinned rc-robosim==1.2 — revert with:
#   .pixi/envs/robosim/bin/python -m pip install --force-reinstall --no-deps rc-robosim==1.2
/path/to/Utama-Core/.pixi/envs/robosim/bin/python -m pip install --force-reinstall --no-deps /tmp/wheelout/rc_robosim-*.whl
```

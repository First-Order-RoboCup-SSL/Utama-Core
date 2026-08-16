# This is a vendored fork of robocin/rSim

Upstream: https://github.com/robocin/rSim
Forked at commit: `b413932b7b4bbc0fc399f6a979e4cfcc4f56264f` (tag `v1.2`, the
same version this repo's `pixi.toml` pins via `rc-robosim>=1.2,<2` on PyPI).

## Why this is here

`Utama-Core` depends on `rc-robosim` as a compiled PyPI package (the
`robosim` feature environment in `pixi.toml`), not as source — normally
nothing in this repo needs rSim's own source at all. This copy exists
because we found and fixed real bugs in rSim's native ball/dribbler/kicker
physics (see `docs/roadmap.md`'s "TODO — add grsim as a CI/tournament
environment" section, item 1, for the dribbler-release writeup, and the
"scoreless-draw pattern" section for the kick-direction writeup) and there
was nowhere else to keep a rebuildable, git-trackable copy of the fixes.

The patches themselves are also saved as plain diffs at
`docs/patches/rSim-dribbler-release.diff` and
`docs/patches/rSim-kick-direction.diff` — those are the reviewable
artifacts. This directory is the actual buildable source they apply to, so
whoever picks up the open regression (see roadmap TODO) doesn't have to
re-clone upstream and re-apply the diffs by hand first.

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
3. `src/robosim/sslrobot.cpp` — `Kicker::kick()` had two compounding
   direction bugs (see `docs/patches/rSim-kick-direction.diff` and
   `docs/roadmap.md`'s "New blocker found, native-simulator-level" writeup
   for full detail):
   - It gated on `isTouchingBall()`, a razor-thin box check (~3cm forward,
     4cm lateral) meant for deciding whether to grab the ball into the
     dribbler hold — far stricter than any Python-side "do I have the
     ball" sensor (e.g. `has_ball(visual=True)`'s 0.15m radial capture
     distance). When `isTouchingBall()` happened to be false on the exact
     tick a kick command arrived (plausible any time the ball hasn't fully
     settled — receiving a fast pass, or mid-`turn_on_spot` pivot),
     `kick()` silently no-opped: the ball kept whatever pre-existing
     velocity it had, in a direction unrelated to the robot's facing
     angle. New `isNearKickerFace()` gate uses a radial distance from the
     chassis center (matching `has_ball()`'s own semantics) instead.
   - Even when the gate passed, the kick's ball-velocity formula only
     damped the ball's pre-kick velocity component *along* the kick
     direction (`kickerDampFactor=0.2`) but added the *tangential*
     (sideways) component back at full, undamped strength — directly
     bleeding any residual lateral ball velocity into the kick's resultant
     direction. Now damped by the same factor as the normal component.
   - Confirmed via direct reproduction (single-robot script, logging robot
     orientation and ball velocity immediately before/after a kick) that
     these two bugs, compounding, reproduce launch-direction deflections
     of tens of degrees under realistic conditions (a robot shooting
     shortly after receiving a moving pass) — closely matching a real
     match's observed ~31 degree kick-direction mismatch. Fixed; residual
     mismatch after the fix is under ~2 degrees in realistic scenarios
     (below 8 degrees in an adversarial synthetic worst case).
4. `CMakeLists.txt` — unrelated compiler-compatibility fix needed to build
   at all on this machine's GCC 13 (pybind11 2.6.2's vendored headers assume
   `<cstdint>`/`<cstddef>` are transitively included; forces them via
   `-include`). May not be needed on other toolchains.

## Known issue: test fragility, not a robot-behavior bug (root-caused 2026-08-16)

Building this and installing it over `rc-robosim==1.2` in
`.pixi/envs/robosim` fixes the dribbler-release bug and the kick-direction
bug (both confirmed via reproduction) but changes the outcome of
`utama_core/tests/kernel/test_referee_override.py::test_their_kickoff_clears_our_robots_outside_center_circle`
(passes on stock 1.2, fails on this patch — a robot's path planner stalls
short of the keep-out radius measured from the fixed field origin,
`dist_to_center≈0.244m` vs. the test's required `≥0.75m`). Isolated to the
dribbler-release fix specifically (reproduced identically with only that
patch applied, independent of the kick-direction patch).

**Root cause, fully traced:** not a bug in either patch, and not a bug in
the robot's actual restart behavior. `_clear_to_legal_positions`
(`utama_core/strategy/referee/actions.py`) correctly polices distance from
the *live ball position* (`game.ball.p`), not a fixed field-center point —
that's the real SSL rule and it's implemented correctly. The test's
assertion instead checks distance from the fixed origin `(0, 0)`, implicitly
assuming the ball stays parked at center throughout a kickoff-prepare
sequence. In this specific test's early ticks (`split_shape_runner`'s
default formation, before `PREPARE_KICKOFF_BLUE` is even injected), one
robot briefly touches the ball incidentally during normal split-shape play,
setting `dribblerOn=true`. Upstream v1.2 has a one-way-latch bug where
`setActions()` never calls `setDribbler(false)` (see "What's patched here"
item 1 above) — so on stock, that robot's dribbler is latched on
indefinitely afterward; on this patched build, `setDribbler(rbtAction[7] >
0)` correctly turns it back off next tick, exactly matching the SSL command
protocol's semantics (dribble is a per-tick on/off flag on every other
command field, not a one-way latch — this is objectively the correct fix).
That single-tick difference in `dribblerOn` state, occurring during real
ball-robot contact, sends the ODE contact solver down a different branch at
that exact contact event; contact-solver output is chaotically sensitive to
this kind of perturbation, so the ball's resulting trajectory diverges
substantially over the next ~40-80 ticks and it comes to rest 0.56m from
the true center instead of near it. The robot being tested (`robot 1`) is
in fact correctly positioned outside the *real* keep-out zone the whole
time — `dist(robot 1, actual ball position)` = 0.797m, comfortably above
the 0.75m threshold — it's only the test's `dist(robot 1, fixed origin)`
proxy that reads as a failure, because the proxy's assumption (ball stays
at center) doesn't hold in this particular scenario.

**Verification method:** built genuinely-pristine upstream v1.2 source
locally (docs/patches/rSim-dribbler-release.diff reverse-applied, plus only
the unrelated GCC-13 CMakeLists.txt compile-flag fix re-added to allow the
build to complete) with the *same local toolchain* as the patched build,
to rule out a PyPI-wheel-vs-local-build toolchain confound. The pristine
local build reproduces PyPI stock's ball trajectory bit-for-bit through the
early contact ticks and passes the test; only the dribbler-release-patched
build diverges, confirming the patch's source diff — not compiler/toolchain
differences — is what changes the outcome.

**Disposition (resolved 2026-08-16):** this was never a blocker for
installing the dribbler-release or kick-direction patches — the patches are
correct and the robot restart behavior they produce is correct. The test
itself was fragile (asserted against a fixed-origin proxy instead of the
actual ball position the real rule uses); fixed in
`utama_core/tests/kernel/test_referee_override.py` to assert
`dist(robot, game.ball.p) >= BALL_KEEP_OUT_DISTANCE - 0.05` instead of
`dist(robot, origin)`, the same way the sibling
`test_their_ball_placement_clears_our_robots_from_keep_out_zone` test
already does. Both patches are now built and installed over
`.pixi/envs/robosim`'s `rc-robosim==1.2` and verified: the full test suite
(638 tests) passes clean, including the fixed keep-out-zone test.

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
# setuptools-scm can't infer a version number from this directory (it's a
# subtree of Utama-Core's own git history, not its own tagged repo/tarball)
# — pin one explicitly or the metadata build step fails with
# "setuptools-scm was unable to detect version". Needed as of 2026-08-16;
# not mentioned in earlier revisions of this doc, so may depend on
# setuptools-scm version.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RC_ROBOSIM="1.2"

cd vendor/rSim
# If rebuilding after a previous build in this same checkout, clear the
# stale cache first -- scikit-build's incremental build did NOT reliably
# pick up source changes without this (confirmed directly: a rebuild with
# only `rm -rf /tmp/wheelout` produced a bit-identical .so despite real
# source edits; deleting _skbuild forced a genuine from-scratch recompile
# that did pick them up).
rm -rf _skbuild
/path/to/Utama-Core/.pixi/envs/robosim/bin/python -m pip wheel . --no-deps --no-build-isolation -w /tmp/wheelout

# Install (overwrites the pinned rc-robosim==1.2 — revert with:
#   .pixi/envs/robosim/bin/python -m pip install --force-reinstall --no-deps rc-robosim==1.2
/path/to/Utama-Core/.pixi/envs/robosim/bin/python -m pip install --force-reinstall --no-deps /tmp/wheelout/rc_robosim-*.whl
```

## Testing tip: `get_state()` is stateful, not idempotent

`SSLWorld::getState()` computes robot/ball velocity via finite difference
against whatever the *previous* `get_state()` call returned, then
overwrites that cache (see `src/robosim/sslworld.cpp`). Calling
`sim.get_state()` twice in a row without an intervening `sim.step(...)`
silently zeroes the velocity fields on the second call (positions are
still correct — only the finite-differenced velocity/`vtheta` fields are
affected). Any reproduction script needs to call `get_state()` exactly
once per tick and cache the result, or velocity readings will be
corrupted in a way that's easy to misread as "the ball/robot is at rest"
when it isn't.

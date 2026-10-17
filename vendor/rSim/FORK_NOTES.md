# This is a vendored fork of robocin/rSim

Upstream: https://github.com/robocin/rSim
Forked at commit: `b413932b7b4bbc0fc399f6a979e4cfcc4f56264f` (tag `v1.2`, the
same version this repo's `pixi.toml` pins via `rc-robosim>=1.2,<2` on PyPI).

This fork is the physics step of the rsim stack. The diagram of the whole tick, from
`StrategyRunner` through the seeded sim noise to this C++ code and back, is in
[docs/sim_realism.md](../../docs/sim_realism.md#how-one-rsim-tick-is-built).

## Why this is here

Since 2026-10-16 `pixi.toml` builds the `robosim` environment's `rc-robosim` from this
directory, not from PyPI (see "Installed from source by pixi" below). This copy first existed
because we found and fixed real bugs in rSim's native ball/dribbler/kicker
physics (see `docs/roadmap.md`'s "TODO — add grsim as a CI/tournament
environment" section, item 1, for the dribbler-release writeup, and the
"scoreless-draw pattern" section for the kick-direction writeup) and there
was nowhere else to keep a rebuildable, git-trackable copy of the fixes.

The patches themselves are also saved as plain diffs at
`patches/rSim-dribbler-release.diff` and
`patches/rSim-kick-direction.diff` — those are the reviewable
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
   direction bugs (see `patches/rSim-kick-direction.diff` and
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
5. Tunable physics for `SimNoise` (2026-10-16, `realism/physics-knobs`). Every default
   reproduces the old physics bit for bit (checked: a 30 s split_shape vs tiki_taka match's
   replay arrays hash the same as with the installed binary, `00c94df54af6231f`).
   - `SSL.set_physics_params(substeps, ball_slide_decel, ball_roll_decel, ball_slide_fraction)`
     (`robosim_py.cpp`, `SSLWorld::setPhysicsParams`). Kept across `reset()`; a new `SSL`
     object starts at the defaults (5, -1, -1, 5/7).
   - **Substeps** — physics steps per `step()`; was a hard-coded 5. At 5 the substep length is
     still computed as `timeStep * 0.2`, exactly as before (`timeStep / 5` rounds differently).
   - **Two-phase ball friction** (`SSLWorld::applyBallFriction`) — with `ball_roll_decel >= 0`
     the constant friction force on the ball (about 0.65 m/s^2) is replaced, while the ball is
     on the ground (z <= 1.5 r), by a direct speed decrement each substep: `ball_slide_decel`
     from a kick until the speed falls to `ball_slide_fraction` of the post-kick speed, then
     `ball_roll_decel`. Below 0.01 m/s the ball stops. The angular velocity is set to pure
     rolling each substep, so in this mode the ball carries no spin into collisions. A ball
     given speed by anything other than a kick (a teleport, a deflection) rolls at once.
     `-1` (default) keeps the old force. Real balls: slide about 0.5 s at 2-4 m/s^2, then roll
     at about 0.29 m/s^2 (per-match IQR 0.29-0.33).
   - **Kick direction** — `setActions` reads an optional 9th action field, an angle in radians
     the kick is rotated by from the robot's facing (`SSLRobot::kick(..., angle)`, counter-
     clockwise in rsim's own frame, which is the mirror of ours). An 8-field action, or 0,
     kicks exactly as before. Python sends 9 fields only on ticks where some kick has an angle.
   - `CMakeLists.txt` — `-march=native` is now opt-in (`-DRSIM_NATIVE_ARCH=ON`). It was added
     in #144 and changes floating-point results (FMA contraction): the same source built with
     it plays a different match from tick 9 (differences of 1e-8 that grow), and differently
     on another CPU. The installed `.pixi/envs/robosim` binary was built without it, so a
     plain build now matches it. (The default env's `robosim/_robosim.so`, used only with
     `UTAMA_USE_NATIVE_RSIM=1`, was copied from a native build, so native mode does not replay
     subprocess-mode matches exactly.)
   - **Installed 2026-10-16** into `.pixi/envs/robosim` (plain cmake Release build, no
     `-march=native`). Knobs off, a 30 s split_shape vs tiki_taka_plus match's replay hashes the
     same as with the binary it replaced (`7c27555e8249abc6`). The replaced binary and this one
     are kept in `~/dev/ssl/rsim-backup/` (outside the repo) to go back to.

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
locally (patches/rSim-dribbler-release.diff reverse-applied, plus only
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
isn't worth paying. Now that `pixi.toml` builds from this fork, a pushed fork with a
submodule or pixi's git-source dependency are the alternatives, if vendoring ever gets in
the way.

## Installed from source by pixi (2026-10-16)

`pixi.toml` installs this directory as the robosim environment's `rc-robosim` (a pypi path
dependency), so `pixi install` builds it with the machine's C++ compiler: no hand install, and
CI and every checkout run this fork. Needed for that: `setup.py` pins the version
(`1.2.0+utama`; setuptools_scm cannot read one from a vendored subtree) and `FindODE.cmake`
also looks in `$CONDA_PREFIX`, where the env's `libode` lives. After changing the C++
source, rebuild by deleting `.pixi/envs/robosim` and running `pixi install -e robosim` (a
cached build may otherwise be reused). Built this way (gcc 13, no `-march=native`), a 30 s split_shape vs
tiki_taka_plus match hashes the same as the hand-installed binary (`7c27555e8249abc6`).
From a fresh clone with empty pixi and uv caches (2026-10-17, gcc 13 from the system, as on CI's
ubuntu runner) `pixi install -e robosim` took 9.5 min, most of it downloads, and the built binary
steps 1800 ticks of 12 robots bit for bit like the installed one, with the knobs off and on.
Matches still differ across CPUs and compilers (see the `-march=native` note above), so
compare results from one machine.

The manual procedure below is kept for trying changes without touching the env.

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

## Building into a worktree without installing (for trying changes)

The rsim subprocess (`robosim_subprocess.py`, run by `.pixi/envs/robosim/bin/python`) inherits
the environment, so a build placed first on `PYTHONPATH` overrides the installed binary for one
process tree only, leaving the shared env alone:

```bash
cmake -S vendor/rSim -B build/rsim-py310 \
  -DPYTHON_EXECUTABLE=$(pwd)/.pixi/envs/robosim/bin/python -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build/rsim-py310 -j2
mkdir -p build/rsim-override/robosim
echo "from ._robosim import SSL, VSS" > build/rsim-override/robosim/__init__.py
ln -s $(pwd)/build/rsim-py310/_robosim.cpython-310-x86_64-linux-gnu.so build/rsim-override/robosim/
PYTHONPATH=$(pwd)/build/rsim-override pixi run pytest utama_core/tests/rsoccer_simulator --headless
```

Installing for everyone is then a copy of that `.so` over
`.pixi/envs/robosim/lib/python3.10/site-packages/robosim/_robosim.cpython-310-x86_64-linux-gnu.so`
(or the wheel procedure above), revertable with
`.pixi/envs/robosim/bin/python -m pip install --force-reinstall --no-deps rc-robosim==1.2`
(which restores upstream 1.2, without this fork's earlier fixes; keep a copy of the current
`.so` to go back to it instead). The match cache's code fingerprint includes the installed
binary, so installing one reruns every cached match; with the knobs off a non-native build
plays them identically.

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

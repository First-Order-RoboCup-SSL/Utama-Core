# Utama Core

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/First-Order-RoboCup-SSL/Utama-Core)

First Order Robotics' software stack for [RoboCup SSL](https://ssl.robocup.org/) (Small Size
League), where teams of six autonomous robots play football. It covers vision and referee
input, motion planning, robot control, an in-process referee, a fast simulator (rsim), and the
strategy layer that decides what every robot does.

## Quick start

1. Install [pixi](https://pixi.sh/latest/#installation)
   (`curl -fsSL https://pixi.sh/install.sh | sh`) and open a new terminal.
2. `pixi install` in the repository root, then `pixi run precommit-install` so every commit is
   linted.
3. `pixi run test` runs the test suite (add `--headless` when calling pytest directly).
4. `pixi run main` to watch a match in the simulator (Ctrl+C to stop).

**New here? Write your first strategy: [tutorials/first_strategy.md](tutorials/first_strategy.md).**

Everything runs in rsim with the in-process referee; grSim, the official GameController and
real robots are optional ([external setup](docs/real_robots.md#external-tools-grsim-gamecontroller-autoreferee-ssl-vision)).

## What are you working on?

| Task | Start here |
|---|---|
| Your first strategy | [tutorials/first_strategy.md](tutorials/first_strategy.md), then [first_tactic.md](tutorials/first_tactic.md) |
| Why did a robot do that? | [tutorials/debug_a_moment.md](tutorials/debug_a_moment.md) |
| Write a motion planner | [tutorials/first_planner.md](tutorials/first_planner.md) |
| Writing or changing a strategy, tactic or skill | [docs/strategy.md](docs/strategy.md) |
| Seeing what the robots did, and why | `pixi run dashboard` (every strategy's maps and replays); `tools/debug/find_bugs.py RUN_DIR` for a run's worst problems |
| Judging whether a strategy is better | [docs/strategy.md](docs/strategy.md), [docs/signals.md](docs/signals.md) |
| The referee | [docs/referee.md](docs/referee.md) |
| Motion planning | [docs/motion_planning.md](docs/motion_planning.md) |
| The simulator | [vendor/rSim/FORK_NOTES.md](vendor/rSim/FORK_NOTES.md) |
| A test day with real robots | [tutorials/field_day.md](tutorials/field_day.md) |
| Real robots, vision, radio | [utama_core/team_controller/README.md](utama_core/team_controller/README.md), [docs/real_robots.md, "External tools"](docs/real_robots.md#external-tools-grsim-gamecontroller-autoreferee-ssl-vision) |
| Which script does what | [docs/tools.md](docs/tools.md) |
| Everything else | [docs/README.md](docs/README.md), the index of every doc |

Coding agents: read [AGENTS.md](AGENTS.md) first (`CLAUDE.md` is a symlink to it).

## Layout

Everything lives under `utama_core/`:

- `strategy/`: the strategies (one module each, named after the strategy)
- `tactics/`, `skills/`, `geometry/`: reusable tactics, per-robot skills, and the geometry they share
- `engine/`: the tactic-kernel infrastructure (`Strategy`, `Tactic`, `TickContext`, `MatchLog`, referee overrides)
- `referee/`: the in-process referee (rules, state machine, restart positioning, profiles)
- `motion_planning/`: path planning and motion control
- `run/`: the main loop (`StrategyRunner`)
- `replay/`: replay files, the match-result cache and code fingerprints
- `analysis/`: offline analyses of tournament runs and replays (ball losses, chances, restarts, stalls)
- `simulator/`: the Python simulator environment over the rSim physics fork in `vendor/rSim/`
- `team_controller/`, `data_processing/`: vision, robot radio and referee input
- `dashboard/`: the browser dashboard
- `entities/`, `config/`, `geometry/`: data classes, settings and constants, geometry helpers
- `tests/`: all tests

Scripts live in `tools/` (grouped by who uses them: `debug/`, `field/`, `realism/`, `dev/`; the
strategy evaluation in `tools/evaluation/`) and `examples/`; see
[docs/tools.md](docs/tools.md).

## Field conventions

![field_guide](assets/images/field_guide.jpg)

- Distances in metres, velocities in metres per second.
- Angles in radians, normalised to [-pi, pi]; heading 0 faces the positive x-axis (left to right).
- The centre of the field is (0, 0). Unless stated otherwise, blue defends the left and yellow
  the right.

## System design

One tick of `StrategyRunner`, from vision, robot and referee data to robot commands:

![One tick of StrategyRunner](docs/img/pipeline/one_tick.png)

The same tick as a diagram, with every module: [docs/architecture.md](docs/architecture.md). How
vision, robot and referee data become one `Game`: [docs/architecture.md, "Pipeline method"](docs/architecture.md#pipeline-method).

## Contributing

Work on a branch and open a pull request into `main`. A pull request needs:

1. a `release:major`, `release:minor` or `release:patch` label: every merge to `main` releases
   a new version automatically;
2. passing CI (tests and lint; `pixi run test` and `pixi run lint` locally);
3. to be up to date with `main`;
4. every Copilot comment reviewed (not necessarily accepted);
5. an approval from an assigned reviewer.

Tips:

- Run `pixi install` again after adding a package (a folder with an `__init__.py`).
- In VS Code, if the run button picks the wrong environment, choose the interpreter by hand
  (`Ctrl + Shift + P` → `Select Interpreter`). Format on save with the `Black Formatter` extension.
- When a commit fails the pre-commit checks, the hook has often fixed the files already: stage
  them and commit again.
- A strategy branch (`strategy/*`) may change strategy code only
  (`tools/dev/check_strategy_branch.py`), so it can't change what it is scored against.
- `CLAUDE.md` is a symlink to `AGENTS.md`, and `.claude/skills` to `.agents/skills`: edit the
  targets. On native Windows without Developer
  Mode, fix it with `git config --global core.symlinks true && git checkout -- CLAUDE.md .claude/skills`.

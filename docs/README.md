# Docs

New to the repo: the [README](../README.md) first, then [first_strategy.md](first_strategy.md).
Docs that code comments cite but that were deleted are in git history.

- [first_strategy.md](first_strategy.md): start here. Look around the strategies, then write, watch and evaluate your own.
- [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md): the strategy layer and its evaluation: the tactic-kernel model, referee restarts, lessons from past bugs, the debugging tools.
- [tactic_model_design_decisions.md](tactic_model_design_decisions.md): why the strategy layer is shaped as it is, and what was deliberately not built.
- [signals.md](signals.md): every signal recorded about how a strategy plays, and the names of the pitch zones.
- [referee.md](referee.md): the in-process referee, how its commands reach robots, the operator panel, and its rule-by-rule design decisions.
- [motion_planning_comparison.md](motion_planning_comparison.md): the motion-planning benchmark and how to read it.
- [real_robot_setup.md](real_robot_setup.md): a test day with real robots, the field logs it records, and grSim / GameController / ssl-vision.
- [architecture.md](architecture.md): how the packages depend on each other, and how one tick turns vision, robot and referee data into commands.
- [tools.md](tools.md): every script and pixi task, by purpose.

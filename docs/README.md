# Docs

How the code works and why: look things up here. To learn by doing, start with the step-by-step
pages in [tutorials/](../tutorials/): [first_strategy.md](../tutorials/first_strategy.md) (new members
start there), [first_tactic.md](../tutorials/first_tactic.md) (its part 2),
[debug_a_moment.md](../tutorials/debug_a_moment.md) (from a wrong move in a replay to its cause and a
test) and [field_day.md](../tutorials/field_day.md) (a test day with real robots). Docs that
code comments cite but that were deleted are in git history.

- [strategy.md](strategy.md): the strategy layer and its evaluation: the tactic-kernel model, referee restarts, lessons from past bugs, the debugging tools.
- [strategy_design.md](strategy_design.md): why the strategy layer is shaped as it is, and what was deliberately not built.
- [signals.md](signals.md): every signal recorded about how a strategy plays, and the names of the pitch zones.
- [referee.md](referee.md): the in-process referee, how its commands reach robots, the operator panel, and its rule-by-rule design decisions.
- [motion_planning.md](motion_planning.md): the motion-planning benchmark and how to read it.
- [real_robots.md](real_robots.md): field logs and the tools that read them; grSim, the GameController and ssl-vision.
- [architecture.md](architecture.md): how the packages depend on each other, and how one tick turns vision, robot and referee data into commands.
- [tools.md](tools.md): every script and pixi task, by purpose.

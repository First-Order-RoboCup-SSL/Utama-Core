# Docs

Every doc in this folder, by kind. New to the repo: the [README](../README.md) first.

## How to

- [STRATEGY_DEVELOPMENT.md](STRATEGY_DEVELOPMENT.md): writing and evaluating strategies and tactics: the tactic-kernel model, referee restarts, lessons from past tactic bugs, round-robins, the scenario bench, observability tools. Read before any strategy-layer change.
- [custom_referee.md](custom_referee.md): the in-process `CustomReferee`: architecture, usage, profiles, and its known gaps.
- [custom_referee_gui.md](custom_referee_gui.md): the referee operator panel in the dashboard.
- [referee_integration.md](referee_integration.md): how referee commands reach robots (`RefereeOverride`) and what each command requires.
- [motion_planning_comparison.md](motion_planning_comparison.md): the motion-planning benchmark and how to read it.
- [tools.md](tools.md): every script and pixi task, by purpose.
- [setup_external.md](setup_external.md): grSim, the official GameController and AutoReferee, SSL Vision for real robots.
- [contributing.md](contributing.md): editor setup, commits, pull requests and releases.

## Design rationale

- [tactic_model_design_decisions.md](tactic_model_design_decisions.md): why the strategy layer (engine, tactics, strategies) is shaped the way it is, and what was deliberately not built.
- [scheduling_math_model.md](scheduling_math_model.md): a mathematical model of tactic scheduling and role allocation, a companion to the above.
- [custom_referee_design_decisions.md](custom_referee_design_decisions.md): rule-by-rule decisions from auditing `CustomReferee` against the SSL rulebook.
- [pipeline_method.md](pipeline_method.md): how vision, robot and referee data are refined into one `Game` state, and one tick from state to robot commands.

## Current results

- [strategies.md](strategies.md): the strategy catalog: every `build_*_kernel_strategy`, its status, and its latest round-robin results.
- [pitch_zones.md](pitch_zones.md): names for the parts of the pitch (thirds, wings, centre, the box, danger), with a diagram.
- [signals.md](signals.md): every signal recorded about how a strategy plays, grouped by question, with reference ranges and what an off value means.
- [signal_report.md](signal_report.md): figures of the main signals for every strategy in the latest full round-robin.

## Open work

- [roadmap.md](roadmap.md): larger open workstreams, and one-line pointers to finished ones.
- [testing_gaps.md](testing_gaps.md): kinds of testing gap that let bugs through, numbered (code cites them by number).
- [investigation_default_vs_lowblock_stalemate.md](investigation_default_vs_lowblock_stalemate.md): why `default` vs `low_block` stays 0-0; partly fixed.
- [investigation_ball_contact_orientation_divergence.md](investigation_ball_contact_orientation_divergence.md): a robot pinned against the ball rotates away from contact; localized, not fixed.

## Writing

- [blog/strategy-tactics-orchestration.md](blog/strategy-tactics-orchestration.md): Strategy = Tactics × Orchestration, a post on the architecture of the strategy layer.

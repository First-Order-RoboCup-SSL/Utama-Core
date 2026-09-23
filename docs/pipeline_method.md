# Pipeline method

![Dataflow Diagram](../assets/images/pipeline_new.drawio.png)

**Problem.** Combining new data with prior knowledge (e.g. a Kalman filter) kept bloating
`GameFrame` with new functions, and access to past/current/predicted state wasn't uniform.

**Solution.** `Game` holds our best estimate of the true current state and is augmented by a
pipeline of **refiners**: each takes the current game and one new datapoint and returns an
updated game. Later stages see earlier updates — e.g. the has-ball refiner uses IR for our
robots and the (already refined) positions for enemy robots. Friendly and enemy robots expose
the same fields, estimated from different sources.

- **Multiple cameras** — combined inside the position refiner.
- **Falling behind** — queues were replaced by one-slot buffers (`deque`, thread safe); the
  main loop runs at a fixed rate and takes the latest camera, robot and referee data once per
  frame.
- `GameFrame` is immutable; receivers (network) are separate from refiners, which makes testing
  easier; e-stop and current velocity are fields on the game.
- `GameHistory` (`utama_core/entities/game/game_history.py`) stores past frames.

The uniform `GameTimeline`/`Predictions` interface (pluggable per-property predictors) from
earlier drafts was never built; predictions are ad hoc per call site.

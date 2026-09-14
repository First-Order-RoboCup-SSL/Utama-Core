## Pipeline Method Explanation


### Problem: 
- When ingesting new data, we may want to combine this new data with our previous knowledge eg, Kalman filter. Currently this requires writing many new functions in the GameFrame object making it bloated.
- Interface for accessing past, current and future (predictions) is not uniform
- Uncertainty on strategies which we originally aimed to address with a Behaviour Tree;
  the strategy layer has since been rewritten kernel-native and no longer uses one
  (see `docs/STRATEGY_DEVELOPMENT.md`)
  
### System diagram
![Dataflow Diagram](../assets/images/pipeline_new.drawio.png)
### Solution
Solve this by maintaining the idea that the game represents everything we have about the "true current state" of the game but allow it to be passed through a pipeline in which it is repeatedly augmented by new data. 

The pipeline is composed of Refiner operators which take the current game state and a new datapoint, and return an updated game state.

Later pipeline stages use the game from earlier pipeline stages, so it sees earlier updates in the game. For example the has_ball refiner uses the IR data for our own robots and the positions in game for the enemy robots (and so the position refiner happens first in the pipeline). This illustrates the idea that we try to make everything as equal as possible for enemy and friendly robots (both have has_ball for example) but generate estimates using different data sources.  

We think this method resolves all of the concerns: 
 - How and where to combine multiple cameras? We have a camera combiner inside the position refiner which does this.

 - What if we get behind and need to drop frames? We've replaced the queues with 1 place buffers and we run the main loop at a fixed frame rate, taking the latest data available from every camera, the robot and the referee once per frame.
 -  Frame Limiting - Done through the concept of buffers
Estop - Done by a field in game
Current Vel cacls - Done by a field in game 
Prediction - handled by predictors
GameFrame gating - See diagram
Concurrency - deque is thread safe

### Key system points
- Main loop is now a fixed frequency
- GameFrame is immutable
- Separation of receivers (network) and refiners allow easier testing 
- Formalisation of Past, Present, Future games
- `GameHistory` (`utama_core/entities/game/game_history.py`) stores historical records

> The `GameTimeline` / `Predictions` interface described in earlier drafts of this
> design (uniform past/present/future access with pluggable per-property predictors)
> was never built. `GameHistory` and the `Refiner` pipeline above shipped; predictions
> are currently handled ad hoc per call site, not through a shared predictor interface.



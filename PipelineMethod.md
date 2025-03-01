## Pipeline Method Explanation

Idea: Solve the problem that we really need access to game data to make effective decisions about ingesting new sensor data, particularly with a view to the Kalman Filter later.
Solve this by maintaining the idea that the game represents everything we have about the "true current state" of the game but allow it to be passed through a pipeline in which
it is repeatedly augmented by new data. The pipeline is composed of Refiner operators which take the current game state and a new datapoint, and return an updated game state.
We compose these in such an order that later stages can use updates written into the game object by previous stages. For example the has_ball refiner uses the IR data for our own robots and the positions in game for the enemy robots (and so the position refiner happens first in the pipeline). This illustrates the idea that we try to make everything as equal as possible for enemy and friendly robots (both have has_ball for example) but generate estimates using different data. 

We think this method resolves all of the concerns: 
 -> How and where to combine multiple cameras? We have a camera combiner inside the position refiner which does this.
 -> What if we get behind and need to drop frames? We've replaced the queues with 1 place buffers and we run the main loop at a fixed frame rate, taking the latest data available from every camera, the robot and the referee once per frame. \Frame Limiting - Done through the concept of buffers
Estop - Done by a field in game
Current Vel cacls - Done by a field in game 
Prediction - handled by predictors
Game gating - See diagram
Concurrency - deque is thread safe

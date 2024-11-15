# entities

Contains the entity objects used to store game state and game data.

## data

This folder contains `namedtuple` declarations that are used across the repository.

- `vision.py` declares the namedtuples output from our vision systems (grSim or SSLVision)
- TODO: `instruction.py` declares the namedtuples compiled and sent to the physical robots


## game

- `game.py` contains all records of the various frames in this game
- `field.py` contains the pre-defined positions of critical field objects, stored as `Shapely` objects for easy spatial analysis
- `ball.py`, `robot.py` are not currently in use, but can be implemented to create persistent objects to store relevant info about the ball or each robot

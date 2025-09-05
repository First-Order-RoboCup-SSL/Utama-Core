# entities

Contains the entity objects used to store game state and game data.

## data

This folder contains `namedtuple` declarations that are used across the repository.

- `vision.py` declares the namedtuples output from our vision systems (grSim or SSLVision)
- `command.py` declares the namedtuples sent and received from the physical robots

## game

- `game.py` contains all records of the various frames in this game
- `field.py` contains the pre-defined positions of critical field objects, stored as `Shapely` objects for easy spatial analysis
- `ball.py`, `robot.py` are not currently in use, but can be implemented to create persistent objects to store relevant info about the ball or each robot
- `team_info.py`: **TeamInfo Class** represents the information about a team, including the team's name, score, red and yellow cards, timeouts, and goalie.
  - **Attributes**:
    - `name`: The team's name.
    - `score`: The number of goals scored by the team.
    - `red_cards`: The number of red cards issued to the team.
    - `yellow_cards`: The total number of yellow cards ever issued to the team.
    - `timeouts`: The number of timeouts this team can still call.
    - `timeout_time`: The number of microseconds of timeout this team can use.
  - **Methods**:
    - `parse_referee_packet(packet)`: Parses the referee packet and updates the team information.

## referee

This folder contains classes related to the referee system and game state management.

### referee_command.py

- **RefereeCommand Class**: Represents a referee command with an ID and name.
  - **Attributes**:
    - `command_id`: The ID of the command.
    - `name`: The name of the command.
  - **Methods**:
    - `from_id(command_id: int)`: Creates a `Command` instance from the given command ID.

### stage.py

- **Stage Class**: Represents a game stage with an ID and name.
  - **Attributes**:
    - `stage_id`: The ID of the stage.
    - `name`: The name of the stage.
  - **Methods**:
    - `from_id(stage_id: int)`: Creates a `Stage` instance from the given stage ID.

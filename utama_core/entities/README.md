# Entities

Data containers that describe everything the rest of the code base reasons about: robot and ball pose, field geometry, referee state and the packets exchanged with hardware.

## Folder overview

### `data/`
- `command.py`, `referee.py`, `vision.py`, `raw_vision.py`: strongly-typed packet descriptions for low-level IO (robot commands, referee messages, SSL-Vision feed).
- `vector.py`, `object.py`: small utility types used to expose consistent vector maths and object identifiers across the stack.

### `game/`
- `game_frame.py`, `current_game_frame.py`, `game_history.py`: immutable snapshots of the match and helpers to track history/current context.
- `game.py`: lightweight façade that exposes the current snapshot while preserving history.
- `field.py`: canonical field geometry (Shapely primitives) used by planners and strategies.
- `ball.py`, `robot.py`, `team_info.py`, `proximity_lookup.py`: value objects that capture per-entity state and provide fast distance queries.

### `referee/`
(WORK IN PROGRESS)
- `referee_command.py`, `stage.py`: enums/lookup helpers that map integer IDs from the referee box into descriptive objects shared by the strategy layer.

> These modules are intentionally light on behaviour—other packages consume them as immutable records or simple helpers.

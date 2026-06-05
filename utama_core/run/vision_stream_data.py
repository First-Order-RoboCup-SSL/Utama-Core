import random

VS_KICK_THRESHOLD = 0.5  # m/s — ball speed above this triggers kick commentary

VS_TEAM_NAMES = [
    "Team 67",
    "Team 21",
]

VS_IDLE_LINES = [
    "The robots are thinking...",
    "Calculating optimal trajectory",
    "Both sides plotting their next move",
    "The crowd holds its breath",
    "Pure silicon determination out there",
    "No human reflexes required",
    "Running at full clock speed",
    "Algorithms at war",
    "404: defence not found",
    "This is peak robot football",
]

VS_KICK_LINES = [
    "What a strike!",
    "They've let it fly!",
    "Big boot from the robot!",
    "The ball is moving!",
    "Powerful kick!",
    "Sending it downfield!",
]

VS_GOAL_LINES_YELLOW = [
    "GOAL! Yellow draws blood!",
    "Yellow scores! Unbelievable!",
    "The yellow machine delivers!",
    "Yellow puts it in the net!",
]

VS_GOAL_LINES_BLUE = [
    "GOAL! Blue strikes back!",
    "Blue finds the net!",
    "Brilliant from the blue side!",
    "Blue pulls one back!",
]

VS_FOOTBALLER_NAMES = [
    "Martin",
    "Joel",
    "Fred",
    "Louis",
]


def assign_team_names() -> tuple[str, str]:
    pool = list(VS_TEAM_NAMES)
    random.shuffle(pool)
    return pool[0], pool[1]


def get_robot_name(
    robot_names: dict[tuple[bool, int], str],
    is_friendly: bool,
    robot_id: int,
) -> str:
    key = (is_friendly, robot_id)
    if key not in robot_names:
        used = set(robot_names.values())
        pool = [n for n in VS_FOOTBALLER_NAMES if n not in used]
        if not pool:
            pool = VS_FOOTBALLER_NAMES
        robot_names[key] = random.choice(pool)
    return robot_names[key]

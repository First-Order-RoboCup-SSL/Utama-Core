from settings import GRAVITY, HEIGHT_COM, ROBOT_RADIUS


def max_acceleration() -> float:
    return GRAVITY * ROBOT_RADIUS / HEIGHT_COM

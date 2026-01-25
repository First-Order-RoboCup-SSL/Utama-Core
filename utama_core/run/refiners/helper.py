import pandas as pd
try:
    from utama_core.entities.data.vision import VisionRobotData
    from utama_core.entities.game import GameFrame, Robot
except ModuleNotFoundError:
    sys.path.append("../utama_core/entities/")
    from data.vision import VisionRobotData
    from game import GameFrame, Robot
# For running analytics from Jupyter notebook

@staticmethod
def filter_robot(
    filter,
    raw_data,
) -> VisionRobotData:
    
    # class VisionRobotData: id: int; x: float; y: float; orientation: float
    data, ts = raw_data[0], raw_data[1]
    x_f, y_f = filter.step(data.x, data.y, ts)

    return VisionRobotData(id=data.id, x=x_f, y=y_f, orientation=data.orientation)

@staticmethod
def no_filter(
    filter,
    raw_data,
) -> VisionRobotData:

    return raw_data[0]


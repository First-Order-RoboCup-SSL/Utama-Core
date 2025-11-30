from utama_core.run.refiners.filters import FIR_filter

import pandas as pd
from os.path import join

DATA_PATH = "vision_data"
NOISY = join(DATA_PATH, "noisy-raw.csv")
CLEAN = join(DATA_PATH, "clean-raw.csv")

NO_ROBOTS = 6
FILTERS = [FIR_filter() for _ in range(NO_ROBOTS)]


def format_data(filename: str, id: int = 0) -> pd.DataFrame:  # for a single robot
    data = pd.read_csv(filename)
    return data.loc[data["id"] == id, ["x", "y", "orientation"]]


NOISY_FORMATTED = format_data(NOISY)
CLEAN_FORMATTED = format_data(CLEAN)


def sum_of_squares(
    data_1: pd.DataFrame,
    data_2: pd.DataFrame,
    attr: str
    ) -> float:
    return data_1[attr].combine(
            other=data_2[attr],
            func=lambda predicted, real: (predicted - real) ** 2
        ).sum()


BASELINE_X  = sum_of_squares(NOISY_FORMATTED, CLEAN_FORMATTED, "x")
BASELINE_Y  = sum_of_squares(NOISY_FORMATTED, CLEAN_FORMATTED, "y")
BASELINE_TH = sum_of_squares(NOISY_FORMATTED, CLEAN_FORMATTED, "orientation")


# Test that the least squares errors for all 3 parameters is below baseline.
# Do this for all 6 robots.
def test_filter_reduces_error():
    filtered = pd.DataFrame(columns=["x","y","orientation"])
    
    for (_, entry) in NOISY_FORMATTED.iterrows():
        (x_f, y_f, th_f) = FILTERS[0].step([entry["x"], entry["y"], entry["orientation"]])
        new_row = pd.DataFrame(
            {
                "x": x_f,
                "y": y_f,
                "orientation": th_f
            },
            index=[0]
        )
        pd.concat([filtered, new_row], ignore_index=True)
    
    error_x = sum_of_squares(CLEAN_FORMATTED, filtered, "x")
    error_y = sum_of_squares(CLEAN_FORMATTED, filtered, "y")
    error_th = sum_of_squares(CLEAN_FORMATTED, filtered, "orientation")
    
    assert error_x < BASELINE_X
    assert error_y < BASELINE_Y
    assert error_th < BASELINE_TH
    
# Goals:
# 1. Write unit tests for the filters
#    To completely isolate the filters, we need to instantiate one here.
#    Then, we check the mean squares error in VisionData at the end.
#    We have clean raw and noisy raw.
#    yellow_robots=list(
#                 map(FIR_filter.filter_robot,
#                 self.fir_filters_yellow,
#                 sorted(combined_vision_data.yellow_robots, key=lambda r: r.id))
#                 )
# 2. Modify the visualisation to match what we have now.
# 3. How to do live testing?
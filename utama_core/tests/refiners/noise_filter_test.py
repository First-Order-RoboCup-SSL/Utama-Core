from utama_core.run.refiners.filters import FIR_filter

import pandas as pd
from os.path import join

DATA_PATH = "vision_data"  # Tests are run from UTAMA-CORE
NOISY = join(DATA_PATH, "noisy-grsim-raw.csv")
CLEAN = join(DATA_PATH, "clean-raw.csv")

ID_COL, X_COL, Y_COL, TH_COL = "id", "x", "y", "orientation"
COLS = [X_COL, Y_COL, TH_COL]
COLS_ALL = COLS + [ID_COL]


def format_data(filename: str) -> pd.DataFrame:  # for all 6 robots
    data = pd.read_csv(filename)
    return data[COLS_ALL]


NOISY_FORMATTED = format_data(NOISY)
CLEAN_FORMATTED = format_data(CLEAN)


class RobotFilterTest:
    def __init__(self, id: int, fs=60, taps=None, window_len=5, cutoff=None):
        self.id = id
        
        self._filter = FIR_filter(fs, taps, window_len, cutoff)
        self._clean = CLEAN_FORMATTED[CLEAN_FORMATTED[ID_COL]==self.id].drop(ID_COL, axis=1).reset_index(drop=True)
        self._noisy = NOISY_FORMATTED[NOISY_FORMATTED[ID_COL]==self.id].drop(ID_COL, axis=1).reset_index(drop=True)
        
        self.baseline_x  = RobotFilterTest._mean_squared_error(
            self._clean,
            self._noisy,
            X_COL
            )
        self.baseline_y  = RobotFilterTest._mean_squared_error(
            self._clean,
            self._noisy,
            Y_COL
            )
        # self.baseline_th = RobotFilterTest._mean_squared_error(
        #     self._clean,
        #     self._noisy,
        #     TH_COL
        #     )
        
        self._filtered = pd.DataFrame(columns=COLS)
        
        for (_, entry) in self._noisy.iterrows():
            (x_f, y_f, th_f) = self._filter.step([entry[X_COL], entry[Y_COL], entry[TH_COL]])
            
            new_row = pd.DataFrame(
                {
                    X_COL: x_f,
                    Y_COL: y_f,
                    TH_COL: th_f
                },
                index=[0]
            )
            self._filtered = pd.concat([self._filtered, new_row], ignore_index=True)
        
        self._filtered = self._filtered.reset_index(drop=True)
        
        self.error_x  = RobotFilterTest._mean_squared_error(
            self._clean,
            self._filtered,
            X_COL
            )
        self.error_y  = RobotFilterTest._mean_squared_error(
            self._clean,
            self._filtered,
            Y_COL
            )
        # self.error_th = RobotFilterTest._mean_squared_error(
        #     self._clean,
        #     self._filtered,
        #     TH_COL
        #     )
        
    @staticmethod
    def _mean_squared_error(
        true_data: pd.DataFrame,
        actual_data: pd.DataFrame,
        param: str
        ) -> float:
        return true_data[param].combine(
                other=actual_data[param],
                func=lambda t, a: (t - a) ** 2
            ).mean()


NO_ROBOTS = 6
ROBOTTESTS = [RobotFilterTest(id) for id in range(NO_ROBOTS)]


# # Sum of squares error for all 3 params is below baseline.
# def test_test():
#     assert ROBOTTESTS[2].error_x > ROBOTTESTS[2].baseline_x

def test_filter_reduces_x_error():
    for robot in ROBOTTESTS:
        assert robot.error_x < robot.baseline_x


def test_filter_reduces_y_error():
    for robot in ROBOTTESTS:
        assert robot.error_y < robot.baseline_y


# def test_filter_reduces_orientation_error():
#     for robot in ROBOTTESTS:
#         assert True
#         # assert robot.error_th < robot.baseline_th


# Goals:
# 1. Write unit tests for the filters
#    To completely isolate the filters, we need to instantiate one here.
#    Then, we check the mean squares error in VisionData at the end.
# 2. Modify the visualisation to match what we have now.
# 3. How to do live testing?
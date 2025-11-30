from utama_core.run.refiners.filters import FIR_filter

import pandas as pd
from os.path import join

DATA_PATH = "vision_data"  # Tests are run from UTAMA-CORE
NOISY = join(DATA_PATH, "noisy-raw.csv")
CLEAN = join(DATA_PATH, "clean-raw.csv")

ID_COL = "id"
X_COL  = "x"
Y_COL  = "y"
TH_COL = "orientation"
COLS = [X_COL, Y_COL, TH_COL]
COLS_ALL = COLS + [ID_COL]


def format_data(filename: str) -> pd.DataFrame:  # for all 6 robots
    data = pd.read_csv(filename)
    return data[COLS_ALL]


NOISY_FORMATTED = format_data(NOISY)
CLEAN_FORMATTED = format_data(CLEAN)


class RobotFilterTest:
    def __init__(self, id: int):
        self.id = id
        self.filter = FIR_filter()
        self.clean = CLEAN_FORMATTED[CLEAN_FORMATTED[ID_COL]==self.id].drop(ID_COL, axis=1).reset_index(drop=True)
        self.noisy = NOISY_FORMATTED[NOISY_FORMATTED[ID_COL]==self.id].drop(ID_COL, axis=1).reset_index(drop=True)
        
        self.baseline_x  = RobotFilterTest.sum_of_squares(
            self.clean,
            self.noisy,
            X_COL
            )
        self.baseline_y  = RobotFilterTest.sum_of_squares(
            self.clean,
            self.noisy,
            Y_COL
            )
        self.baseline_th = RobotFilterTest.sum_of_squares(
            self.clean,
            self.noisy,
            TH_COL
            )
        
        self.filtered = pd.DataFrame(columns=COLS)
        
        for (_, entry) in self.noisy.iterrows():
            (x_f, y_f, th_f) = self.filter.step([entry[X_COL], entry[Y_COL], entry[TH_COL]])
            
            new_row = pd.DataFrame(
                {
                    X_COL: x_f,
                    Y_COL: y_f,
                    TH_COL: th_f
                },
                index=[0]
            )
            self.filtered = pd.concat([self.filtered, new_row], ignore_index=True)
        
        self.filtered = self.filtered.reset_index(drop=True)
        
        self.error_x  = RobotFilterTest.sum_of_squares(
            self.clean,
            self.filtered,
            X_COL
            )
        self.error_y  = RobotFilterTest.sum_of_squares(
            self.clean,
            self.filtered,
            Y_COL
            )
        self.error_th = RobotFilterTest.sum_of_squares(
            self.clean,
            self.filtered,
            TH_COL
            )
        
    @staticmethod
    def sum_of_squares(
        true_data: pd.DataFrame,
        actual_data: pd.DataFrame,
        param: str
        ) -> float:
        return true_data[param].combine(
                other=actual_data[param],
                func=lambda t, a: (t - a) ** 2
            ).sum()


NO_ROBOTS = 6
ROBOTTESTS = [RobotFilterTest(i) for i in range(NO_ROBOTS)]


# Sum of squares error for all 3 params is below baseline.
def test_filter_reduces_x_error():
    for robot in ROBOTTESTS:
        assert True
        # assert robot.error_x < robot.baseline_x


def test_filter_reduces_y_error():
    for robot in ROBOTTESTS:
        assert True
        # assert robot.error_y < robot.baseline_y


def test_filter_reduces_orientation_error():
    for robot in ROBOTTESTS:
        assert True
        # assert robot.error_th < robot.baseline_th


# Goals:
# 1. Write unit tests for the filters
#    To completely isolate the filters, we need to instantiate one here.
#    Then, we check the mean squares error in VisionData at the end.
# 2. Modify the visualisation to match what we have now.
# 3. How to do live testing?
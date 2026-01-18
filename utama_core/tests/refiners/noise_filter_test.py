from utama_core.run.refiners.filters import FIR_filter

import pandas as pd
import numpy as np
from os.path import join
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = "utama_core/tests/refiners/datasets"  # Tests are run from UTAMA-CORE
NOISY = join(DATA_PATH, "noisy.csv")
CLEAN = join(DATA_PATH, "clean.csv")

ID_COL, X_COL, Y_COL, TH_COL = "id", "x", "y", "orientation"
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
        
        self._filter = FIR_filter()
        self._clean = RobotFilterTest.extract_robot(CLEAN_FORMATTED, self.id)
        self._noisy = RobotFilterTest.extract_robot(NOISY_FORMATTED, self.id)
        
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
        
        self.baseline_v  = RobotFilterTest._mean_squared_error_vec(
            self._clean,
            self._noisy
            )
        
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
        
        self.error_v  = RobotFilterTest._mean_squared_error_vec(
            self._clean,
            self._filtered
            )
        
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
    
    @staticmethod
    def vectorify(x: float, y: float) -> np.ndarray:
        return np.array((x, y))
    
    @staticmethod
    def _mean_squared_error_vec(
        true_data: pd.DataFrame,
        actual_data: pd.DataFrame
        ) -> float:
        
        true_vec = true_data[X_COL].combine(
            other=true_data[Y_COL],
            func=RobotFilterTest.vectorify
        )
        
        actual_vec = actual_data[X_COL].combine(
            other=actual_data[Y_COL],
            func=RobotFilterTest.vectorify
        )
    
        return (np.linalg.norm(true_vec - actual_vec) ** 2).mean()
    
    @staticmethod
    def extract_robot(data: pd.DataFrame, id: int) -> pd.DataFrame:
        return data[data[ID_COL]==id].drop(ID_COL, axis=1).reset_index(drop=True)


NO_ROBOTS = 6
ROBOTTESTS = [RobotFilterTest(id) for id in range(NO_ROBOTS)]


# Sum of squares error for all 3 params is below baseline.
def test_filter_reduces_x_error():
    for robot in ROBOTTESTS:
        assert robot.error_x < robot.baseline_x


def test_filter_reduces_y_error():
    for robot in ROBOTTESTS:
        assert robot.error_y < robot.baseline_y


def test_filter_reduces_v_error():
    for robot in ROBOTTESTS:
        assert robot.error_v < robot.baseline_v
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

V_COL = "vector"
ID_COL, X_COL, Y_COL, TH_COL = "id", "x", "y", "orientation"
COLS = [X_COL, Y_COL, TH_COL]
COLS_ALL = COLS + [ID_COL]
LIMITS = 2000
X_AXES = np.arange(LIMITS)
ROBOTS_NUM = 6

def format_data(filename: str, id: bool = True) -> pd.DataFrame:  # for all 6 robots
    data = pd.read_csv(filename)
    if id:
        return data[COLS_ALL]
    return data[COLS].iloc[:LIMITS]

def extract_robot(data: pd.DataFrame, id: int) -> pd.DataFrame:
    return data[data[ID_COL]==id].iloc[:LIMITS].drop(ID_COL, axis=1).reset_index(drop=True)

def get_current_ax(axs: np.ndarray, id: int) -> plt.Axes:
    return axs[id//3][id%3]

def percentage_diff(
    old: float,
    new: float
) -> float:
    return round((new-old) / old * 100, 1)

def diff_squared(t: float, a: float) -> float:
    return (t - a) ** 2

def mean_squared_error(
    true_data: pd.DataFrame,
    actual_data: pd.DataFrame,
    param: str,
    id: Union[int, None] = None
    ) -> float:
    if id is not None:
        true_data = extract_robot(true_data, id)
        actual_data = extract_robot(actual_data, id)
        
    return true_data[param].combine(
            other=actual_data[param],
            func=diff_squared
        ).mean()

def vectorify(x: float, y: float) -> np.ndarray:
    return np.array((x, y))

def mean_squared_error_vec(
    true_data: pd.DataFrame,
    actual_data: pd.DataFrame,
    id: Union[int, None] = None
    ) -> float:
    if id is not None:
        true_data = extract_robot(true_data, id)
        actual_data = extract_robot(actual_data, id)
    
    true_vec = true_data[X_COL].combine(
        other=true_data[Y_COL],
        func=vectorify
    )
    
    actual_vec = actual_data[X_COL].combine(
        other=actual_data[Y_COL],
        func=vectorify
    )
    
    return (np.linalg.norm(true_vec - actual_vec) ** 2).mean()

def behavior_plot(robots_num, baselines, clean_data, filtered_data):
    fig, axs = plt.subplots(2,3)
    fig.set_size_inches(12,8)
    fig.suptitle("% change in MSE (Kalman)")

    baseline_xs, baseline_ys, baseline_vs= baselines

    for id in range(robots_num):
        baseline_x, baseline_y, baseline_v = baseline_xs[id], baseline_ys[id], baseline_vs[id]

        error_x = mean_squared_error(
            clean_data,
            filtered_data,
            X_COL,
            id
        )
        
        error_y = mean_squared_error(
            clean_data,
            filtered_data,
            Y_COL,
            id
        )
        
        error_v = mean_squared_error_vec(
            clean_data,
            filtered_data,
            id
        )
        
        delta_x = percentage_diff(baseline_x, error_x)
        delta_y = percentage_diff(baseline_y, error_y)
        delta_v = percentage_diff(baseline_v, error_v)
        
        current_ax = get_current_ax(axs, id)
        current_ax.set_title(f"Robot {id}")
        
        deltas = [delta_x, delta_y, delta_v]
        colours = ['red' if d > 0 else 'green' for d in deltas]
        bars = current_ax.bar([X_COL, Y_COL, V_COL], deltas, color=colours)
        current_ax.bar_label(bars, label_type='center')

def visualized_plot(robots_num, clean_data, noisy_data, kalman_filtered_data, col = X_COL):
    fig, axs = plt.subplots(2,3)
    fig.set_size_inches(12,8)
    fig.suptitle(col + " coordinates against time before and after filtering")

    for id in range(robots_num):
        clean = extract_robot(clean_data, id)
        noisy = extract_robot(noisy_data, id)
        noisy_k = extract_robot(kalman_filtered_data, id)
        
        current_ax = get_current_ax(axs, id)
        current_ax.set_title(f"Robot {id}")
        current_ax.plot(X_AXES, noisy_k[col], "#6bdb79")
        # current_ax.plot(X_AXES, clean_k[X_COL], "#84c6de")
        # current_ax.plot(X_AXES, clean_f[X_COL], "b")
        current_ax.plot(X_AXES, noisy[col], "r")
        current_ax.plot(X_AXES, clean[col], "k")
        
        current_ax.legend(("Kalman (noisy)", "Noisy", "Clean"))
        # current_ax.legend(("Kalman (noisy)", "Kalman (clean)", "FIR (noisy)", "FIR (clean)", "Noisy", "Clean"))
    
def plots(seed: int):
    directory = str(seed) + "/"
    clean = directory + "clean_dwa.csv"
    noisy = directory +"noisy_dwa.csv"
    kalman = directory + "kalman_filtered_dwa.csv"
    clean_fd = format_data(clean)
    noisy_fd = format_data(noisy)
    kalman_fd = format_data(kalman)

    baseline_xs = [mean_squared_error(clean_fd, noisy_fd, X_COL, id) for id in range(ROBOTS_NUM)]
    baseline_ys = [mean_squared_error(clean_fd, noisy_fd, Y_COL, id) for id in range(ROBOTS_NUM)]
    baseline_vs = [mean_squared_error_vec(clean_fd, noisy_fd, id) for id in range(ROBOTS_NUM)]
    baselines = [baseline_xs, baseline_ys, baseline_vs]
    behavior_plot(ROBOTS_NUM, baselines, clean_fd, kalman_fd)
    visualized_plot(ROBOTS_NUM, clean_fd, noisy_fd, kalman_fd)

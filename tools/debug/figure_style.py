"""The colours and matplotlib style every figure tool shares (pitch_maps, explain, match_timeline),
so the figures read as one set. Importing it sets the style."""

import matplotlib.pyplot as plt

INK, INK_2, INK_3, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
FOR, AGAINST = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
# Categorical slots in fixed order (validated adjacent-pair for stacks); never cycled.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_3,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 10,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK,
        "font.size": 9,
    }
)

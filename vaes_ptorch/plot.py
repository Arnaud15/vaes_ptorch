from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_points_series(xl: List[np.ndarray]):
    with plt.rc_context({"axes.spines.right": False, "axes.spines.top": False}):
        plt.gca().xaxis.grid(True, linestyle="--", alpha=0.5)
        plt.gca().yaxis.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(np.arange(-2, 2.01, 0.5))
        plt.yticks(np.arange(-2, 2.01, 0.5))
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        for x in xl:
            plt.scatter(x[:2000, 0], x[:2000, 1], alpha=0.1)
        plt.show()

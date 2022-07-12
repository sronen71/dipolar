import numpy as np
from matplotlib import pyplot as plt


def plot_contour(x, y, v):
    fig, ax = plt.subplots(1, 1)
    vt = np.transpose(v)
    plt.contourf(x, y, vt)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar()
    plt.show()

import numpy as np
from matplotlib import pyplot as plt


def plot_one(psi):
    mid_z = psi.shape[2] // 2
    plt.imshow(np.abs(psi[:, :, mid_z] ** 2))
    plt.show()


def plot_image(ax, xmin, xmax, out, aho, limx):
    unit = 2 * limx * aho * 1e6 / out.shape[0]
    mid_z = out.shape[2] // 2
    width = xmax - xmin
    ax.imshow(
        abs(out[xmin:xmax, xmin:xmax, mid_z]) ** 2,
        extent=np.array([0, width, 0, width]) * unit,
    )


def plot_table(wavefunctions, lattice_spacings, lattice_depths, aho, lattice_type, limx, tag):
    # rows : lattice spacings
    # cols: lattice depth

    fig, axs = plt.subplots(
        nrows=wavefunctions.shape[0], ncols=wavefunctions.shape[1], figsize=(50, 50)
    )
    xmin_plot = np.where(wavefunctions > 1e-1 * wavefunctions.max())[2].min()
    xmax_plot = np.where(wavefunctions > 1e-1 * wavefunctions.max())[2].max()
    for i in range(wavefunctions.shape[0]):  # spacings
        for j in range(wavefunctions.shape[1]):  # depths

            lspace = lattice_spacings[i] * 1e6 * aho
            ldepth = lattice_depths[j]
            out = abs(wavefunctions[i, j]) ** 2
            ax = axs[i, j]
            plot_image(ax, xmin_plot, xmax_plot, out, aho, limx)
            if j == 0:
                ax.set_ylabel(f"{lspace:1.2f} um", fontsize=70)
            else:
                ax.set_yticks([])
            if i == 0:
                ax.set_title(rf"{ldepth:1.2f} $\hbar \omega$", fontsize=70)

            if i == (wavefunctions.shape[0] - 1):
                ax.set_xlabel("x (um)", fontsize=70)
                ax.tick_params(axis="x", labelsize=50)
            else:
                ax.set_xticks([])
    plt.savefig(f"figs/phases_{lattice_type}_{tag}.png", bbox_inches="tight")

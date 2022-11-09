import matplotlib
import numpy as np
from matplotlib import pyplot as plt

# matplotlib.use("WebAgg")


def plot_one(psi):
    fig, axs = plt.subplots(1, 2)
    mid_z = psi.shape[2] // 2
    mid_y = psi.shape[1] // 2
    axs[0].imshow(np.abs(psi[:, :, mid_z] ** 2))
    axs[1].imshow(np.abs(psi[:, mid_y, :].transpose() ** 2))
    plt.show()


def plot_image(ax, xmin, xmax, out, limx):
    unit = 2 * limx / out.shape[0]
    mid_z = out.shape[2] // 2
    width = xmax - xmin
    ax.imshow(
        abs(out[xmin:xmax, xmin:xmax, mid_z]) ** 2,
        extent=np.array([0, width, 0, width]) * unit,
    )


def plot_table(
    wavefunctions, lattice_spacings, lattice_depths, aho, lattice_type, limx, tag, energies
):
    # rows : lattice spacings
    # cols: lattice depth

    fig, axs = plt.subplots(
        nrows=wavefunctions.shape[1], ncols=wavefunctions.shape[0], figsize=(50, 50)
    )
    xmin_plot = np.where(wavefunctions > 1e-1 * wavefunctions.max())[2].min()
    xmax_plot = np.where(wavefunctions > 1e-1 * wavefunctions.max())[2].max()
    for i in range(wavefunctions.shape[0]):  # spacings
        for j in range(wavefunctions.shape[1]):  # depths
            energy = energies[i, j]
            lspace = lattice_spacings[i]
            ldepth = lattice_depths[j]
            out = abs(wavefunctions[i, j]) ** 2
            ax = axs[j, i]
            plot_image(ax, xmin_plot, xmax_plot, out, limx)
            title = ""
            if j == 0:
                title = f"{lspace:1.2f} $a_{{ho}}$\n"
            title += f"E={energy:.4f}"
            ax.set_title(title, fontsize=70)
            if i == 0:
                ax.set_ylabel(rf"{ldepth:1.2f} $\hbar \omega$", fontsize=70)
                ax.tick_params(axis="y", labelsize=50)
            else:
                ax.set_yticks([])

            if j == (wavefunctions.shape[1] - 1):
                ax.set_xlabel("x ($a_{{ho}}$)", fontsize=70)
                ax.tick_params(axis="x", labelsize=50)
            else:
                ax.set_xticks([])
    plt.savefig(f"figs/phases_{lattice_type}_{tag}.png", bbox_inches="tight")

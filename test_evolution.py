import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation

import constants
import dipolar
from perlin_noise import perlin_noise
from system_parameters import SystemParameters

matplotlib.use("Agg")


def plot_psit(psit, outfile, time_unit=1):
    print("Animation")
    fig, axs = plt.subplots(1, 1)
    fig.tight_layout()

    ims = axs.imshow(abs(psit[0, :, :]) ** 2)
    # cmap = plt.get_cmap("viridis")
    # plt.set_cmap(cmap)
    time = axs.annotate(0, xy=(1, 8), xytext=(1, 8))
    time.set_color("white")

    def update_figure(frame, *fargs):
        ims.set_array(abs(psit[frame, :, :]) ** 2)
        t = frame * time_unit
        time.set_text(f"t={t:.2f}")

        return ims, time

    frames = psit.shape[0]

    ani = animation.FuncAnimation(fig, update_figure, frames=frames, interval=50)
    ani.save(outfile)


system_parameters = SystemParameters()

nx = 176  #
nz = 64
limx = 15  # [aho]
limz = 15  # [aho]
Bcutoff = 15  # [aho]
omegas = np.array([1, 1, 2.0])  # trap frequencies in units of omega
lattice_constant = 4  # in units of aho
start_lattice_depth = 0.1  # in units of hbar*omegar
lattice_type = "triangular"  # "triangular"
potential = dipolar.Potential(
    omegas,
    lattice_constant,
    start_lattice_depth,
    lattice_type,
)
grid = dipolar.Grid(nx, nx, nz, limx, limx, limz)
dbec = dipolar.DBEC(
    system_parameters.scattering_length / system_parameters.aho,
    system_parameters.dipole_length / system_parameters.aho,
    system_parameters.num_atoms,
    potential,
    grid,
    Bcutoff,
)
sigmas = np.array([2, 2, 2]) / np.sqrt(2)
noise = perlin_noise(nx, nx, nz, scale=8)  # make sure nx,nz are divisable by scale
psi0 = dipolar.gaussian_psi(dbec.grid, sigmas) * (1 + 0.4 * noise)

energy_opt, psi_opt = dbec.optimize(psi0)  # find ground state

dt = 4e-3
n = 500  #
substeps = 10

lattice_depth = 0.0
potential = dipolar.Potential(
    omegas,
    lattice_constant,
    lattice_depth,
    lattice_type,
)
dbec = dipolar.DBEC(
    system_parameters.scattering_length / system_parameters.aho,
    system_parameters.dipole_length / system_parameters.aho,
    system_parameters.num_atoms,
    potential,
    grid,
    Bcutoff,
)
print("Evolve...")

times, psit, energies, densities = dbec.evolve(
    torch.tensor(psi_opt, dtype=torch.complex128), dt=dt, n=n, substeps=substeps
)
del dbec
mid = psit.shape[-1] // 2
time_unit = dt * substeps
ss = system_parameters.scattering_length / constants.a0
dl = system_parameters.dipole_length / constants.a0
name = f"{int(system_parameters.num_atoms)}_{ss:.1f}_{dl:.1f}_{lattice_type}_{lattice_constant}_{start_lattice_depth}.mp4"
plot_psit(psit[..., mid], "animation/" + name, time_unit=time_unit)
DE = energies[-1] - energies[0]
print("Energy abs/relative error", DE, DE / energies[0])
print("Density error ", densities[-1] - 1)

import numpy as np
import torch

import constants
import dipolar
import test_dipolar
from visualize import plot_contour


def main():
    mass = 162 * constants.dalton  # 162Dy mass in kg
    omega_x = 2 * np.pi * 125  # Trap radial frequency, in rad/s
    aho = np.sqrt(constants.hbar / mass / omega_x)  # oscillator length in meters
    omegas = np.array([1, 1, 2])  # trap frequencies in units of omega_x

    num_atoms = 150 * 1e3  # Number of atoms
    scattering_length = 86 * constants.a0
    dipole_length = 130 * constants.a0

    lattice_constant = 1  # in units of aho
    lattice_depth = 1  # in units of hbar*omegar_r
    lattice_type = None
    potential = dipolar.Potential(
        omegas,
        lattice_constant,
        lattice_depth,
        lattice_type,
        lattice_shift=[0.0, 0.0],
    )
    nx = 128
    nz = 128
    mid = nz // 2
    limx = 10  # [aho]
    limz = 10  # [aho]
    Bcutoff = 16  # [aho]
    grid = dipolar.Grid(nx, nx, nz, limx, limx, limz)
    # potential = potential_func.lattice(grid.x, grid.y)
    # plot_contour(grid.x1, grid.y1, potential)

    dbec = dipolar.DBEC(
        scattering_length / aho, dipole_length / aho, num_atoms, potential, grid, Bcutoff
    )
    sigmas = np.array([1, 1, 1]) / np.sqrt(2)
    psi1 = dipolar.gaussian_psi(grid, sigmas)
    optimized_energy, psi_opt = dbec.optimize(psi1)
    plot_contour(grid.x1, grid.y1, abs(psi_opt[:, :, mid]))


if __name__ == "__main__":
    main()

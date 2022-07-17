import logging
import os
import pickle
import warnings

import numpy as np
import torch

import constants
import dipolar
from perlin_noise import perlin_noise
from visualize import plot_table

# warnings.filterwarnings("ignore")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("logs/log.log"), logging.StreamHandler()],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    mass = 162 * constants.dalton  # 162Dy mass in kg
    omega_x = 2 * np.pi * 125  # Trap radial frequency, in rad/s
    aho = np.sqrt(constants.hbar / mass / omega_x)  # oscillator length in meters
    omegas = np.array([1, 1, 2])  # trap frequencies in units of omega_x

    num_atoms = 100 * 1e3  # Number of atoms
    scattering_length = 85 * constants.a0
    dipole_length = 131 * constants.a0

    lattice_constant = 1  # in units of aho
    lattice_depth = 0  # in units of hbar*omegar_r
    lattice_type = "square"
    potential = dipolar.Potential(
        omegas,
        lattice_constant,
        lattice_depth,
        lattice_type,
        lattice_shift=[0.0, 0.0],
    )
    nx = 128
    nz = 96
    limx = 12  # [aho]
    limz = 12  # [aho]
    Bcutoff = 12  # [aho]
    logging.info("START RUN")
    logging.info(
        f"num_atoms {num_atoms} omega_x 2*pi*{omega_x/2/np.pi:.2f} "
        f"omegas {omegas[0]} {omegas[1]} {omegas[2]}"
    )
    logging.info(
        f"mass {mass/constants.dalton} scattering_length {scattering_length/constants.a0} "
        f"dipole_length {dipole_length/constants.a0}  "
    )
    logging.info(f"nx {nx} nz {nz} limx {limx} limz {limz} Bcutoff {Bcutoff}")
    grid = dipolar.Grid(nx, nx, nz, limx, limx, limz)
    # gpotential = potential_func.lattice(grid.x, grid.y)
    # plot_contour(grid.x1, grid.y1, gpotential)

    precision = "float64"  # change that to get x5 speed on newer nvidia GPUs,
    # at slight cost of precision (relative energy error ~1e-6)

    dbec = dipolar.DBEC(
        scattering_length / aho,
        dipole_length / aho,
        num_atoms,
        potential,
        grid,
        Bcutoff,
        precision=precision,
    )
    sigmas = np.array([2, 2, 2]) / np.sqrt(2)
    torch.manual_seed(1)
    num_spacings = 6
    num_depths = 5
    lattice_spacings = np.linspace(0.4, 2.4, num_spacings) * 1e-6 / aho  # convert [1e-6 m] -> [aho]
    lattice_depths = np.linspace(0.4, 4.0, num_depths)  # [hbar omega]
    wavefunctions = np.empty((num_spacings, num_depths) + (nx, nx, nz))
    energies = np.empty((num_spacings, num_depths))
    for lattice_type in ["square", "triangular"]:
        for i, lat_spacing in enumerate(lattice_spacings):
            for j, lat_depth in enumerate(lattice_depths):
                logging.info(
                    f"{lattice_type} [{i},{j}]: depth {lat_depth:1.2f} "
                    f"spacings {lat_spacing:1.2f}",
                )
                if lat_depth == 0 and j > 0:
                    wavefunctions[i, j, :, :, :] = wavefunctions[0, 0, :, :, :]
                else:
                    potential.lattice_depth = lat_depth
                    potential.lattice_constant = lat_spacing
                    potential.lattice_type = lattice_type
                    noise = perlin_noise(
                        nx, nx, nz, scale=8
                    )  # make sure nx,nz are divisable by scale
                    psi1 = dipolar.gaussian_psi(grid, sigmas) * (1 + 0.2 * noise)
                    energy, psi_opt = dbec.optimize(psi1)
                    # plot_contour(grid.x1, grid.y1, abs(psi_opt[:, :, mid]))
                    energies[i, j] = energy
                    wavefunctions[i, j, :, :, :] = psi_opt
        results = {"wavefunctions": wavefunctions, "energies": energies}
        pickle.dump(results, open("results/results.pkl", "wb"))
        plot_table(wavefunctions, lattice_spacings, lattice_depths, aho, lattice_type, limx)


if __name__ == "__main__":
    main()

import logging
import os
import pickle

import numpy as np
import torch

import constants
import dipolar
from perlin_noise import perlin_noise
from visualize import plot_one, plot_table

EXCITATIONS = False
SCAN = True


def scan(dbec, aho, spacing_linspace=[1.0, 5.0, 6], depths_linspace=[0.1, 4.0, 6]):
    # spacing_linspace: [start,end,num] in units of aho
    # dephts_linspace: [start,end,num] in units of hbar omega
    sigmas = np.array([2, 2, 2]) / np.sqrt(2)

    num_spacings = spacing_linspace[2]
    num_depths = depths_linspace[2]
    lattice_spacings = np.linspace(*spacing_linspace)  # aho
    lattice_depths = np.linspace(*depths_linspace)  # [hbar omega]
    nx = dbec.grid.nx
    nz = dbec.grid.nz

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
                    dbec.potential.lattice_depth = lat_depth
                    dbec.potential.lattice_constant = lat_spacing
                    dbec.potential.lattice_type = lattice_type
                    noise = perlin_noise(
                        nx, nx, nz, scale=8
                    )  # make sure nx,nz are divisable by scale
                    # noise = np.random.rand(nx, nx, nz)
                    psi1 = dipolar.gaussian_psi(dbec.grid, sigmas) * (1 + 0.4 * noise)
                    energy, psi_opt = dbec.optimize(psi1)
                    energies[i, j] = energy
                    wavefunctions[i, j, :, :, :] = psi_opt
        results = {"wavefunctions": wavefunctions, "energies": energies}
        tag = (
            f"s{dbec.scattering_length*aho/constants.a0:3.1f}_"
            f"d{dbec.dipole_length*aho/constants.a0:3.1f}_"
            f"n{int(dbec.num_atoms):d}"
        )
        pickle.dump(results, open(f"results/results_{tag}.pkl", "wb"))
        limx = dbec.grid.limx
        plot_table(wavefunctions, lattice_spacings, lattice_depths, aho, lattice_type, limx, tag)


def main():
    torch.manual_seed(3)
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("logs/log.log"), logging.StreamHandler()],
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mass = 162 * constants.dalton  # 162Dy mass in kg
    omega_x = 2 * np.pi * 125  # Trap radial frequency, in rad/s
    aho = np.sqrt(constants.hbar / mass / omega_x)  # oscillator length in meters
    omegas = np.array([1.0, 1.0, 2.0])  # trap frequencies in units of omega_x
    num_atoms = 800 * 1e3  # Number of atoms 100
    scattering_length = 89 * constants.a0
    dipole_length = 131 * constants.a0

    lattice_constant = 1  # in units of aho
    lattice_depth = 0.0  # in units of hbar*omegar_r
    lattice_type = "square"
    potential = dipolar.Potential(
        omegas,
        lattice_constant,
        lattice_depth,
        lattice_type,
        lattice_shift=[0.0, 0.0],
    )
    nx = 128  #
    nz = 96
    limx = 15  # [aho]
    limz = 15  # [aho]
    Bcutoff = 15  # [aho]
    logging.info("START RUN")
    logging.info(
        f"num_atoms {num_atoms} omega_x 2*pi*{omega_x/2/np.pi:.2f} "
        f"omegas {omegas[0]} {omegas[1]} {omegas[2]}"
    )
    logging.info(
        f"mass {(mass/constants.dalton):.2f} "
        f"scattering_length {(scattering_length/constants.a0):.2f} "
        f"dipole_length {(dipole_length/constants.a0):.2f}  "
    )
    logging.info(f"nx {nx} nz {nz} limx {limx} limz {limz} Bcutoff {Bcutoff}")
    grid = dipolar.Grid(nx, nx, nz, limx, limx, limz)

    precision = "float64"
    # "float64"for full precision
    # "float32" to get x5 speed on newer nvidia GPUs,
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

    if SCAN:

        scan(
            dbec, aho, spacing_linspace=[1, 5, 6], depths_linspace=[0.1, 4, 6]
        )  # Uncomment this to create subplot tables
    else:
        sigmas = np.array([2, 2, 2]) / np.sqrt(2)
        noise = perlin_noise(nx, nx, nz, scale=8)  # make sure nx,nz are divisable by scale
        psi1 = dipolar.gaussian_psi(dbec.grid, sigmas) * (1 + 0.4 * noise)
        energy, psi_opt = dbec.optimize(psi1)

        print("Energy", energy)
        plot_one(psi_opt)

        if EXCITATIONS:
            print("Calc excitations...")
            if precision == "float64":
                eigenvalues, eigenvectors = dbec.calc_excitations_slepc(
                    psi_opt, k=20, bk=300, maxiter=20000, tol=1e-6
                )

                logging.info(eigenvalues)


if __name__ == "__main__":
    main()

import logging
import os
import pickle

import numpy as np
import torch
from numpy.random import default_rng

import constants
import dipolar
from perlin_noise import perlin_noise
from system_parameters import SystemParameters
from visualize import plot_one, plot_table

EXCITATIONS = False
SCAN = True
# SYMMETRY = "square"
# SYMMETRY = None
SYMMETRY = "triangular"

rng = default_rng(12345)


def scan(dbec, aho, spacing_linspace=[1.0, 10.0, 2], depths_linspace=[0.1, 4.0, 2]):
    # spacing_linspace: [start,end,num] in units of aho
    # dephts_linspace: [start,end,num] in units of hbar omega

    num_spacings = spacing_linspace[2]
    num_depths = depths_linspace[2]
    lattice_spacings = np.linspace(*spacing_linspace)  # aho
    lattice_depths = np.linspace(*depths_linspace)  # [hbar omega]
    nx = dbec.grid.nx
    nz = dbec.grid.nz

    wavefunctions = np.empty((num_spacings, num_depths) + (nx, nx, nz))
    energies = np.empty((num_spacings, num_depths))
    # lattice_types=["square","triangular"]
    lattice_types = ["square"]
    ntrials = 50
    for lattice_type in lattice_types:
        for i, lat_spacing in enumerate(lattice_spacings):
            for j, lat_depth in enumerate(lattice_depths):
                min_energy = 1e10
                for k in range(ntrials):
                    sz1 = rng.uniform(5, 10)
                    sz2 = rng.uniform(5, 10)
                    sigmas = np.array([sz1, sz2, 2]) / np.sqrt(2)
                    logging.info(
                        f"trial {k} "
                        f"{lattice_type} [{i},{j}]: depth {lat_depth:1.2f} "
                        f"spacings {lat_spacing:1.2f}",
                    )
                    dbec.potential.lattice_depth = lat_depth
                    dbec.potential.lattice_constant = lat_spacing
                    dbec.potential.lattice_type = lattice_type
                    scale = rng.choice([2, 4, 8])
                    perlin = perlin_noise(
                        nx, nx, nz, scale=scale
                    )  # make sure nx,nz are divisable by scale
                    noise = rng.random((nx, nx, nz))
                    psi1 = dipolar.gaussian_psi(dbec.grid, sigmas) * (
                        1 + 0.4 * perlin + 0.1 * noise
                    )
                    energy, psi_opt = dbec.optimize(psi1)
                    if energy < min_energy:
                        min_energy = energy
                        energies[i, j] = energy
                        wavefunctions[i, j, :, :, :] = psi_opt
                    print("best energy", min_energy)
        results = {"wavefunctions": wavefunctions, "energies": energies}
        tag = (
            f"s{dbec.scattering_length*aho/constants.a0:3.1f}_"
            f"d{dbec.dipole_length*aho/constants.a0:3.1f}_"
            f"n{int(dbec.num_atoms):d}"
        )
        if SYMMETRY:
            tag = tag + f"_sym_{SYMMETRY}"

        pickle.dump(results, open(f"results/results_{tag}.pkl", "wb"))
        limx = dbec.grid.limx
        plot_table(
            wavefunctions, lattice_spacings, lattice_depths, aho, lattice_type, limx, tag, energies
        )


def main():
    torch.manual_seed(1)
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("logs/log.log"), logging.StreamHandler()],
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    params = SystemParameters()
    mass = params.mass
    omega = params.omega
    aho = params.aho
    omegas = np.array([1.0, 1.0, 2.0])  # trap frequencies in units of omega
    # omegas = np.array([1, 1, 1])
    num_atoms = params.num_atoms
    scattering_length = params.scattering_length
    dipole_length = params.dipole_length

    lattice_constant = 3  # in units of aho
    lattice_depth = 2.5  # in units of hbar*omegar
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
    limx = 14  # [aho]
    limz = limx  # [aho]
    Bcutoff = limx  # [aho]
    logging.info("START RUN")
    logging.info(
        f"num_atoms {num_atoms} omega 2*pi*{omega/2/np.pi:.2f} "
        f"omegas {omegas[0]} {omegas[1]} {omegas[2]}"
    )
    logging.info(
        f"mass {(mass/constants.dalton):.2f} "
        f"scattering_length {(scattering_length/constants.a0):.2f} "
        f"dipole_length {(dipole_length/constants.a0):.2f}  "
        f"aho in bohr {aho/constants.a0}"
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
        symmetry=SYMMETRY,
    )
    # print(dipole_length / aho * num_atoms)
    if SCAN:

        scan(
            dbec, aho, spacing_linspace=[1.5, 3, 5], depths_linspace=[0, 2.5, 6]  # 5,6
        )  # Uncomment this to create subplot tables
    else:
        sigmas = np.array([7, 7, 1]) / np.sqrt(2)
        noise = perlin_noise(nx, nx, nz, scale=8)  # make sure nx,nz are divisable by scale
        psi1 = dipolar.gaussian_psi(dbec.grid, sigmas) * (1 + 0.4 * noise)
        energy, psi_opt = dbec.optimize(psi1)

        print("Energy", energy)
        file_name = (
            f"Single1_s{dbec.scattering_length*aho/constants.a0:3.1f}_"
            f"d{dbec.dipole_length*aho/constants.a0:3.1f}_"
            f"n{int(dbec.num_atoms):d}"
        )
        np.save(file_name, psi_opt)
        plot_one(psi_opt)

        if EXCITATIONS:
            print("Calc excitations...")
            if precision == "float64":
                eigenvalues, eigenvectors = dbec.calc_excitations_slepc(
                    psi_opt, k=500, bk=250, maxiter=20000, tol=1e-6
                )

                logging.info(eigenvalues)


if __name__ == "__main__":
    main()

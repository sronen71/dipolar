import numpy as np
import torch

import dipolar

# Insturctions;
# Install pytest with pip install pytest.py
# type "pytest" at the command line.
# It automatically finds all files in the current directoy
# with functions named "test_*" and runs these functions


def dipolar_f(x):

    if x == 1:
        return 0
    f = (
        1
        + 2 * x**2
        - 3 * x**2 * np.emath.arctanh(np.emath.sqrt(1 - x**2)) / np.emath.sqrt(1 - x**2)
    ) / (1 - x**2)

    return f


def analytic_energies(dbec, sigmas):
    """
    from https://arxiv.org/pdf/1605.04964.pdf
    Bisset et al.
    Ground-state phase diagram of a dipolar condensate with quantum fluctuations
    Bisset's sigma are related to ours by:
    sigma_Bisset=sqrt(8)*sigma_here

    NOTE: Assume radial symmetry: sigmas[0]=sigmas[1],  omegas[0]=omegas[1]
    """

    kinetic_energy = 1 / 8 * np.sum((1 / sigmas) ** 2)
    potential_energy = 1 / 2 * np.sum((sigmas) ** 2 * dbec.potential.omegas**2)
    scattering_energy = (
        dbec.num_atoms * dbec.scattering_length / (4 * np.sqrt(np.pi) * np.prod(sigmas))
    )
    dipolar_energy = (
        -dbec.num_atoms
        * dbec.dipole_length
        * dipolar_f(sigmas[0] / sigmas[2])
        / (4 * np.sqrt(np.pi) * np.prod(sigmas))
    )
    lhy_energy = (
        dbec.num_atoms ** (3 / 2)
        * 128
        * dbec.glhy
        / (
            25
            * np.sqrt(5)
            * np.pi ** (9 / 4)
            * 8 ** (9 / 4)
            * sigmas[0] ** 3
            * sigmas[2] ** (3 / 2)
        )
    )
    return kinetic_energy, potential_energy, scattering_energy, dipolar_energy, lhy_energy


def test_gaussian_normalization():
    nx = 256
    nz = 256
    limx = 12  # [aho]
    limz = 12  # [aho]
    grid = dipolar.Grid(nx, nx, nz, limx, limx, limz)

    sigmas = np.array([1, 1, 1])  # in units of aho
    psi1 = dipolar.gaussian_psi(grid, sigmas)
    # plot_contour(grid.x1, grid.y1, psi1[:,:,mid])
    dvol = grid.dx * grid.dy * grid.dz

    norm = np.sum(abs(psi1) ** 2) * dvol

    assert abs(norm - 1) < 2.0e-14


def test_square_lattice():
    omegas = np.array([1, 1, 1])  # trap frequencies in units of omega_x
    lattice_constant = 1  # in units of aho
    lattice_depth = 1  # in units of hbar*omegar_r
    potential_func = dipolar.Potential(
        omegas,
        lattice_constant,
        lattice_depth,
        lattice_type="square",
        lattice_shift=[0.0, 0.0],
    )
    potential1 = potential_func.lattice(0, 0)
    assert potential1 == -0.5
    potential2 = potential_func.lattice(1, 1)
    assert potential2 == -0.5
    potential3 = potential_func.lattice(0, 1)
    assert potential3 == -0.5
    potential4 = potential_func.lattice(0.5, 0.5)
    assert potential4 == 0.5


def test_triangular_lattice():
    omegas = np.array([1, 1, 1])  # trap frequencies in units of omega_x
    lattice_constant = 1  # in units of aho
    lattice_depth = 1  # in units of hbar*omegar_r
    potential_func = dipolar.Potential(
        omegas,
        lattice_constant,
        lattice_depth,
        lattice_type="triangular",
        lattice_shift=[0.0, 0.0],
    )
    potential1 = potential_func.lattice(0, 0)
    assert potential1 == -0.75
    potential2 = potential_func.lattice(1, 1 / np.sqrt(3))
    assert potential2 == -0.75
    potential3 = potential_func.lattice(0, 2 / np.sqrt(3))
    assert potential3 == -0.75


def test_energy():

    omegas = np.array([1, 1, 2])  # trap frequencies in units of omega_x
    lattice_constant = 0  # in units of aho
    lattice_depth = 0  # in units of hbar*omegar_r
    potential_func = dipolar.Potential(
        omegas,
        lattice_constant,
        lattice_depth,
        lattice_type=None,
        lattice_shift=[0.0, 0.0],
    )

    num_atoms = 1000
    scattering_length_aho = 0.01  # [aho]
    dipole_length_aho = 0.005  # [aho]
    nx = 128
    nz = 128
    limx = 12  # [aho]
    limz = 12  # [aho]
    Bcutoff = 12  # [aho]
    grid = dipolar.Grid(nx, nx, nz, limx, limx, limz)

    dbec = dipolar.DBEC(
        scattering_length_aho, dipole_length_aho, num_atoms, potential_func, grid, Bcutoff
    )
    sigmas_list = [
        np.array([1, 1, 1 / np.sqrt(omegas[2])]) / np.sqrt(2),  # ground state
        np.array([1, 1, 1]),
        np.array([1, 1, 1 / 3]),
        np.array([1, 1, 2]),
        np.array([2, 2, 1]),
    ]  # in units of aho, ground state

    for sigmas in sigmas_list:
        psi1 = dipolar.gaussian_psi(grid, sigmas)
        (
            total_energy,
            kinetic_energy,
            potential_energy,
            scattering_energy,
            dipolar_energy,
            lhy_energy,
        ) = dbec.calculate_energy(torch.as_tensor(psi1), return_all=True)

        (
            analytic_kinetic_energy,
            analytic_potential_energy,
            analytic_scattering_energy,
            analytic_dipolar_energy,
            analytic_lhy_energy,
        ) = analytic_energies(dbec, sigmas)

        # print("ED", dipolar_energy, analytic_dipolar_energy)
        err = 1e-6
        assert abs(kinetic_energy / analytic_kinetic_energy - 1) < err
        assert abs(potential_energy / analytic_potential_energy - 1) < err
        assert abs(scattering_energy / analytic_scattering_energy - 1) < err
        assert abs(lhy_energy / analytic_lhy_energy - 1) < err
        if abs(analytic_dipolar_energy) > 1:
            assert abs(dipolar_energy / analytic_dipolar_energy - 1) < err
        else:
            assert abs(dipolar_energy - analytic_dipolar_energy) < err

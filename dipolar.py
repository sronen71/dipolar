import logging
import sys
import warnings

import numpy as np
import slepc4py
import torch
import torch.utils.dlpack as dlpack
import torchvision.transforms.functional as TF
from numpy.fft import fftfreq
from torch.optim import LBFGS
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

slepc4py.init(sys.argv)  # it needs to come before petsc import to get options correctly
from petsc4py import PETSc

from utils import solve_eigensystem

options = PETSc.Options()
options.setValue("vec_type", "cuda")
options.setValue("mat_type", "aijcusparse")
options.setValue("eps_view", None)
# options.setValue("eps_monitor_conv", None)
options.setValue("eps_monitor", None)
options.setValue("eps_conv_abs", None)
warnings.filterwarnings(
    "ignore", message="Casting complex values to real discards the imaginary part"
)

if torch.cuda.is_available:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def rotate(psi, angle):
    # from matplotlib import pyplot as plt

    psi1 = torch.transpose(psi, 0, 2)
    # psi1 = torch.transpose(psi1, 1, 2)
    psi1 = TF.rotate(psi1, angle, interpolation=InterpolationMode.BILINEAR)
    # psi1 = torch.transpose(psi1, 1, 2)
    psi1 = torch.transpose(psi1, 0, 2)
    # psi11 = psi.detach().cpu().numpy()
    # psi12 = psi1.detach().cpu().numpy()
    # plt.figure()
    # plt.imshow(psi11[:, :, 48])
    # plt.figure()
    # plt.imshow(psi12[:, :, 48])
    # plt.show()
    return psi1


class Basis:
    def __init__(self, dbec, psi):
        self.dbec = dbec
        n = dbec.grid.nx * dbec.grid.ny * dbec.grid.nz
        # self.bdg = self.dbec.bdg_op(psi)
        self.basis = self.dbec.basis_op(psi)
        self.psi = psi
        tmpx = torch.zeros(n, dtype=psi.dtype)
        tmpy = torch.zeros(n, dtype=psi.dtype)
        self.x_cache = PETSc.Vec().createWithDLPack(dlpack.to_dlpack(tmpx.clone()))
        self.y_cache = PETSc.Vec().createWithDLPack(dlpack.to_dlpack(tmpy.clone()))
        self.shadow = PETSc.Mat().createAIJ([n, n])
        self.shadow.setFromOptions()

    def mult(self, A, x, y):
        x.attachDLPackInfo(self.x_cache)
        x_tensor = torch.from_dlpack(x.toDLPack())
        y.attachDLPackInfo(self.y_cache)
        y_tensor = torch.from_dlpack(y.toDLPack())
        # y_tensor[...] = self.bdg(x_tensor)
        y_tensor[...] = self.basis(x_tensor)

    def createVecs(self, A, side=None):
        return self.shadow.createVecs(side=side)


class Potential:
    def __init__(
        self,
        omegas,
        lattice_constant,
        lattice_depth,
        lattice_type="square",
        lattice_shift=[0, 0],
    ):

        """
        Potential energy with scaled units hbar=m=omega=1

        Args:
        omegas: 3-tupple [rad/sec]
        lattice_constant: lattice constant [m]
        lattice_depth: lattice depth [J]
        lattice_type: square or triangular
        lattice_shift: 2-tuple of (x,y) shift in lattice constant units
        """

        self.omegas = omegas
        self.lattice_constant = lattice_constant
        self.lattice_depth = lattice_depth
        self.lattice_type = lattice_type
        self.lattice_shift = lattice_shift

    def trap(self, x, y, z):

        vtrap = 0.5 * (
            self.omegas[0] ** 2 * x**2
            + self.omegas[1] ** 2 * y**2
            + self.omegas[2] ** 2 * z**2
        )
        return vtrap

    def lattice(self, x, y):
        if self.lattice_type is None:
            return np.zeros(x.shape)
        k_lat = 2 * np.pi / self.lattice_constant
        xs = (x + self.lattice_constant * self.lattice_shift[0]) * k_lat
        ys = (y + self.lattice_constant * self.lattice_shift[1]) * k_lat
        vlattice = 0
        if self.lattice_type == "square":
            vlattice = -self.lattice_depth / 4 * (np.cos(xs) + np.cos(ys))
        elif self.lattice_type == "triangular":
            sq3 = np.sqrt(3)
            vlattice = (
                -self.lattice_depth
                / 4
                * (np.cos(xs) + np.cos(-xs / 2 + sq3 / 2 * ys) + np.cos(-xs / 2 - sq3 / 2 * ys))
            )
        else:
            logging.error("Unknown type")
        return vlattice

    def __call__(self, x, y, z):
        """
        Args:
            x: x position
            y: y position
            z: z position
        Returns:
            float: potential energy [hbar*omega]
        """
        vtrap = self.trap(x, y, z)
        if self.lattice_type:
            vtrap += self.lattice(x, y)
        return vtrap


def gaussian_psi(grid, sigmas):
    # sigmas is 3-tupple
    # gaussian_psi**2 is normalized to 1, with std of sigmas.
    sigmas = np.array(sigmas)
    return (
        np.power(2 * np.pi, -3 / 4)
        * np.power(np.prod(sigmas), -1 / 2)
        * np.exp(
            -(
                grid.x**2 / (4 * sigmas[0] ** 2)
                + grid.y**2 / (4 * sigmas[1] ** 2)
                + grid.z**2 / (4 * sigmas[2] ** 2)
            )
        )
    )


class DBEC:
    def __init__(
        self,
        scattering_length,
        dipole_length,
        num_atoms,
        potential,
        grid,
        Bcutoff,
        precision="float32",
        symmetry=None,
    ):
        """
        Dipolar BEC
        Units are scaled: hbar=mass=omega=0

        Args:
        scattering_length: [aho] a_s in the literature
        dipole_length: [aho] a_dd in the literature
        num_atoms: number of atoms
        potential_func: potential object
        grid: grid object [aho]
        Bcutoff: cutoff for dipolar interaction [aho]
        """
        if precision == "float32":
            self.precision = torch.float32
        else:
            self.precision = torch.float64

        self.scattering_length = scattering_length
        self.num_atoms = num_atoms
        self.g = 4 * np.pi * scattering_length
        self.dipole_length = dipole_length
        self.gdd = 3 * dipole_length
        if scattering_length == 0 and dipole_length == 0:
            self.edd = 0
        else:
            self.edd = self.dipole_length / self.scattering_length
        self.potential = potential
        self.grid = grid
        self.potential_grid = torch.as_tensor(
            self.potential(self.grid.x, self.grid.y, self.grid.z), dtype=self.precision
        )  # also update in optimize if potential changed
        self.Bcutoff = Bcutoff
        self.k2, self.vdk = self._make_kspace_operators()
        self.glhy = (
            32
            / 3
            / np.sqrt(np.pi)
            * self.g
            * scattering_length ** (3 / 2)
            * (1 + 3 / 2 * self.edd**2)
        )
        self.symmetry = symmetry
        # self.grad = None

    def _make_kspace_operators(self):
        """
        return dipole insteraction operator vdk and kinetic energy operator k2, in momentum space.
        """
        np.seterr(invalid="ignore")  # ignore 0/0 divison
        k2 = self.grid.kx**2 + self.grid.ky**2 + self.grid.kz**2
        vdk = self.gdd * 4 * np.pi / 3 * (3 * self.grid.kz**2 / k2 - 1)
        xq = self.Bcutoff * np.sqrt(k2)
        spherical_cut = 1 + 3 * (xq * np.cos(xq) - np.sin(xq)) / xq**3
        vdk *= spherical_cut
        vdk[0, 0, 0] = 0

        return (
            torch.as_tensor(k2, dtype=self.precision),
            torch.as_tensor(vdk, dtype=self.precision),
        )

    def symmetry_loss(self, psi):
        psi_norm = self.normalize(psi)
        if self.symmetry == "triangular":
            diff = psi_norm - rotate(psi_norm, 60)
            loss_sym = torch.linalg.vector_norm(diff)
        else:
            loss_sym = 0
        return loss_sym

    def sym(self, psi):
        if self.symmetry == "square":
            psi_sym = psi + (
                torch.rot90(psi, 1, (0, 1))
                + torch.rot90(psi, 2, (0, 1))
                + torch.rot90(psi, 3, (0, 1))
            )
            psi_sym = psi_sym / 4
        # elif self.symmetry == "triangular":
        # psi_sym = psi + rotate(psi, 60) + rotate(psi, -60)
        # psi_sym = psi_sym + torch.rot90(psi_sym, 2, (0, 1))
        # psi_sym = psi_sym / 6
        else:
            psi_sym = psi
        return psi_sym

    def normalize(self, psi):
        norm = torch.linalg.vector_norm(psi * np.sqrt(self.grid.dvol))
        return psi / norm

    def normalize_sym(self, psi):
        psi_sym = self.sym(psi)
        norm = torch.linalg.vector_norm(psi_sym * np.sqrt(self.grid.dvol))
        return psi_sym / norm

    def kinetic_energy(self, psi):
        return (
            torch.sum(self.k2 / 2 * torch.abs(torch.fft.fftn(psi, norm="ortho")) ** 2)
            * self.grid.dvol
        )

    def potential_energy(self, psi2):
        return torch.sum(self.potential_grid * psi2) * self.grid.dvol

    def scattering_energy(self, psi2):
        if self.scattering_length == 0:
            scattering_energy = 0
        else:
            scattering_energy = (
                self.num_atoms * 0.5 * torch.sum(self.g * psi2**2) * self.grid.dvol
            )
        return scattering_energy

    def dipolar_energy(self, psi2):

        if self.dipole_length == 0:
            dipolar_energy = 0
        else:
            dipolar_energy = (
                self.num_atoms
                * 0.5
                * (torch.sum(self.vdk * torch.abs(torch.fft.fftn(psi2, norm="ortho")) ** 2))
                * self.grid.dvol
            )
        return dipolar_energy

    def lhy_energy(self, psi2):
        return (
            (2 / 5)
            * self.num_atoms ** (3 / 2)
            * self.glhy
            * torch.sum(psi2 ** (5 / 2))
            * self.grid.dvol
        )

    def calculate_mu(self, psi, return_all=False):
        """

        Args:
            psi : [nx,ny,nz], assumed to be normalized to 1

        Return: chemical potential mu per atom, in hbar*omega units
        """
        psi_norm = self.normalize_sym(psi)
        psi2 = torch.abs(psi_norm) ** 2
        kinetic_energy = self.kinetic_energy(psi_norm)
        potential_energy = self.potential_energy(psi2)
        scattering_energy = self.scattering_energy(psi2)
        dipolar_energy = self.dipolar_energy(psi2)
        lhy_energy = self.lhy_energy(psi2)
        mu = (
            kinetic_energy
            + potential_energy
            + 2 * scattering_energy
            + 2 * dipolar_energy
            + 5 / 2 * lhy_energy
        )

        return mu

    def calculate_energy(self, psi, return_all=False):
        """

        Args:
            psi : [nx,ny,nz], assumed to be normalized to 1

        Return: energy per atom, in hbar*omega units
        """
        psi_norm = self.normalize_sym(psi)
        psi2 = torch.abs(psi_norm) ** 2
        kinetic_energy = self.kinetic_energy(psi_norm)
        potential_energy = self.potential_energy(psi2)
        scattering_energy = self.scattering_energy(psi2)
        dipolar_energy = self.dipolar_energy(psi2)
        lhy_energy = self.lhy_energy(psi2)
        energy = kinetic_energy + potential_energy + scattering_energy + dipolar_energy + lhy_energy
        if return_all:
            return (
                energy,
                kinetic_energy,
                potential_energy,
                scattering_energy,
                dipolar_energy,
                lhy_energy,
            )
        else:
            return energy

    def h0_func(self, f):
        return torch.fft.ifftn(self.k2 / 2 * torch.fft.fftn(f)) + self.potential_grid * f

    def c_op(self, psi):
        psi2 = torch.abs(psi**2)
        dpot = torch.fft.ifftn(torch.fft.fftn(psi2) * self.vdk)

        def c_func(f):
            return (
                self.num_atoms
                * (self.g * psi2 + self.num_atoms ** (1 / 2) * self.glhy * psi2 ** (3 / 2) + dpot)
                * f
            )

        return c_func

    def x_op(self, psi):
        def x_func(f):
            psi2 = torch.abs(psi**2)
            return self.num_atoms * (
                self.g * psi2 * f
                + 3 / 2 * self.num_atoms ** (1 / 2) * self.glhy * psi2 ** (3 / 2) * f
                + torch.fft.ifftn(torch.fft.fftn(torch.conj(psi) * f) * self.vdk) * psi
            )

        return x_func

    def basis_op(self, psi):  # BdG operator, eq. (29a) in Ronen et al. 2006.
        # psi should be normalized
        # psi is numpy input
        psi = torch.as_tensor(psi, dtype=self.precision)
        mu = self.calculate_mu(psi)
        c_func = self.c_op(psi)

        def gp_func(x):
            f = torch.as_tensor(x, dtype=self.precision)
            f = f.reshape(self.grid.nx, self.grid.ny, self.grid.nz)
            y = self.h0_func(f) - mu * f + c_func(f)
            return y.flatten().to(x.dtype)

        return gp_func

    def calc_excitations_slepc(self, psi, k=4, bk=4, maxiter=10, tol=1e-3, v0=None):
        if self.precision != torch.float64:
            print("Must use double precision for spectrum")
            return [], []
        options.setValue("eps_max_it", maxiter)
        options.setValue("eps_tol", tol)
        # assume ground state psi already normalized
        n = psi.size
        # calculate initial basis of GP operator
        context = Basis(self, torch.as_tensor(psi, dtype=self.precision))
        A = PETSc.Mat().createPython([n, n], context)
        if v0 is None:
            v0 = psi
        w, v = solve_eigensystem(A, bk, x0=v0, problem_type="HEP")
        # Calculate BdG matrix elements in basis
        # bk = bk - 1
        # w = w[1:]
        # v = v[1:]
        B = PETSc.Mat().create()
        B.setSizes([2 * bk, 2 * bk])
        B.setFromOptions()
        B.setUp()
        x_op = self.x_op(torch.as_tensor(psi, dtype=self.precision))
        vt = []
        for i in range(bk):
            vi = torch.as_tensor(v[i]).reshape(self.grid.nx, self.grid.ny, self.grid.nz)
            vt.append(vi)
        for i in range(bk):
            xi = x_op(vt[i])
            for j in range(bk):
                d = 0

                xji = torch.sum(torch.conj(vt[j]) * xi).item()

                check_real = np.real(xji) - np.real_if_close(xji)
                if check_real != 0:
                    print("C", i, j, xji)
                    exit()
                xji = np.real(xji)
                if j == i:
                    d = w[i]
                B[j, i] = xji + d
                B[j + bk, i + bk] = -xji - d
                B[j, i + bk] = xji
                B[j + bk, i] = -xji

        B.assemble()
        w, v = solve_eigensystem(B, k)
        return w, v

    def optimize(self, psi):
        self.potential_grid = torch.as_tensor(
            self.potential(self.grid.x, self.grid.y, self.grid.z), dtype=self.precision
        )
        self.k2, self.vdk = self._make_kspace_operators()
        psi = torch.as_tensor(psi, dtype=self.precision)
        psi = self.normalize_sym(psi)
        best_psi = psi
        min_energy = self.calculate_energy(psi).item()
        calls = 0
        iterations = 50
        psi.requires_grad_()
        tol = 1e-12
        optimizer = LBFGS(
            [psi],
            history_size=15,
            max_iter=100,
            tolerance_grad=tol,
            tolerance_change=tol,
            line_search_fn="strong_wolfe",
        )

        def closure():
            nonlocal calls
            calls += 1
            optimizer.zero_grad()
            loss = self.calculate_energy(psi) + 0.1 * self.symmetry_loss(psi)
            loss.backward()
            return loss

        with tqdm(range(1, iterations + 1)) as pbar:
            for i in pbar:
                with torch.no_grad():
                    psi.copy_(self.sym(psi))
                optimizer.step(closure)
                G = psi.grad.abs().max().item()

                with torch.no_grad():
                    # norm_psi = self.normalize_sym(torch.real(psi))
                    norm_psi = self.normalize_sym(psi)
                    energy = self.calculate_energy(psi).item()
                    loss_sym = self.symmetry_loss(psi)
                    mu = self.calculate_mu(psi).item()
                    GP = self.h0_func(norm_psi) + self.c_op(norm_psi)(norm_psi) - mu * norm_psi
                    GP = GP.abs().max().item()
                    dE = energy - min_energy
                if energy < min_energy:
                    min_energy = energy
                    best_psi = psi
                iterations_info = (
                    f"Iteration {i} Calls {calls} Energy {energy:.8f} mu {mu:.8f} "
                    f"dE {dE:.1e} Grad_max {G:.1e} GPmax {GP:.1e} Symmtery loss {loss_sym:.8f}"
                )
                pbar.set_description(iterations_info)
                logging.info(iterations_info)
                if abs(dE) < tol or dE > 0:
                    break

        # psi1 = norm_psi.detach().cpu().numpy()
        # plt.figure()
        # plt.imshow(psi1[:, :, 48])
        # plt.show()
        # psi = torch.as_tensor(psi)
        energy = self.calculate_energy(best_psi).detach().item()
        psi_opt = self.normalize_sym(best_psi)  # explicitly symmetrize in the end
        psi_opt = psi_opt.detach().cpu().numpy()
        return energy, psi_opt

    def mean_field_potential(self, psi):
        external_potential = self.potential_grid
        contact_potential = self.num_atoms * self.g * abs(psi) ** 2
        dipole_potential = self.num_atoms * torch.fft.ifftn(
            self.vdk * torch.fft.fftn(abs(psi) ** 2)
        )
        lhy_potential = self.num_atoms ** (3 / 2) * self.glhy * abs(psi) ** 3
        return external_potential + contact_potential + dipole_potential + lhy_potential

    def one_time_step(self, psi, dt):
        """
        Helper function for evolve
        :param psi: takes psi0 on the gpu, evolves it in time by dt
        :param dt: the amount of time to evolve by in units of 1/omega
        :return: psi_dt = exp(-i H dt) psi0
        """

        # do dt/2 evolution in real space
        psi = torch.exp(-1.0j * self.mean_field_potential(psi) * dt / 2) * psi
        # do dt evolution in momentum space
        psi = torch.fft.ifftn(torch.exp(-1.0j * self.k2 / 2 * dt) * torch.fft.fftn(psi))
        # do another dt/2 evolution in real space
        psi = torch.exp(-1.0j * self.mean_field_potential(psi) * dt / 2) * psi

        return psi

    def evolve(self, psi0, n, dt, substeps=10, imaginary_time=False):
        """
        evolve for n steps with step dt
        :param psi0: the initial state, should be torch
        :param list_of_times: a list of times, starting at t=0
        :param substeps: record every this number of steps
        :param imaginary_time: if time is imaginary, evolve in imaginary time.
        :return: a list of times t and a list of psi(t) recorded every substeps steps
        """

        psi = psi0.clone()

        # generate list for saving time slices
        psit = np.empty((n,) + psi.shape, dtype=np.complex128)
        psit[0, :, :, :] = psi.detach().cpu().numpy()
        energies = np.empty(n)
        densities = np.empty(n)
        energy = self.calculate_energy(psi)
        density = torch.sum(torch.abs(psi**2)) * self.grid.dvol
        energies[0] = energy
        densities[0] = density
        recording_times = np.empty(n)
        recording_index = 1
        if imaginary_time:
            dt = -1.0j * dt
        for i in tqdm(range(1, n * substeps)):
            psi = self.one_time_step(psi, dt)
            if imaginary_time:
                psi = self.normalize_sym(psi)
            if i % substeps == 0:
                recording_times[recording_index] = i * dt
                energy = self.calculate_energy(psi)
                density = torch.sum(torch.abs(psi**2)) * self.grid.dvol
                psit[recording_index, :, :, :] = psi.detach().cpu().numpy()
                energies[recording_index] = energy.item()
                densities[recording_index] = density.item()
                recording_index += 1
        return recording_times, psit, energies, densities


class Grid:
    def __init__(self, nx, ny, nz, limx, limy, limz):
        x1 = np.linspace(-limx, limx, nx)
        y1 = np.linspace(-limy, limy, ny)
        z1 = np.linspace(-limz, limz, nz)
        x, y, z = np.meshgrid(x1, y1, z1, indexing="ij")
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.x = x
        self.y = y
        self.z = z
        self.dx = self.x1[1] - self.x1[0]
        self.dy = self.y1[1] - self.y1[0]
        self.dz = self.z1[1] - self.z1[0]
        kx = 2 * np.pi * fftfreq(nx) / self.dx
        kx = kx.reshape(nx, 1, 1)
        self.kx = kx
        ky = 2 * np.pi * fftfreq(ny) / self.dy
        ky = ky.reshape(1, ny, 1)
        self.ky = ky

        kz = 2 * np.pi * fftfreq(nz) / self.dz
        kz = kz.reshape(1, 1, -1)
        self.kz = kz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.limx = limx
        self.limy = limy
        self.limz = limz
        self.dvol = self.dx * self.dy * self.dz

import logging

import numpy as np
import torch
from numpy.fft import fftfreq
from torch.optim import LBFGS
from tqdm import trange

if torch.cuda.is_available:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


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
        Potential energy with scaled units hbar=m=omega_x=1

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
            float: potential energy [hbar*omega_x]
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
        precision="float64",
    ):
        """
        Dipolar BEC
        Units are scaled: hbar=mass=omega_x=0

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

        # Disable this weighting for now, it breaks Parseval's theorem
        #  (unitarity of fourier transform):
        # if k2.shape[0] // 2 == 0:  # for even size need double weight
        # for highest positive frequency
        #    k2[k2.shape[0] // 2, :, :] *= 2
        # if k2.shape[1] // 2 == 0:
        #    k2[:, k2.shape[1] // 2, :] *= 2
        # if k2.shape[2] // 2 == 0:
        #    k2[:, :, k2.shape[2] // 2] *= 2

        return (
            torch.as_tensor(k2, dtype=self.precision),
            torch.as_tensor(vdk, dtype=self.precision),
        )

    def normalize(self, psi):
        dvol = self.grid.dx * self.grid.dy * self.grid.dz
        norm = torch.sqrt(torch.sum(torch.abs(psi) ** 2) * dvol)
        return psi / norm

    def calculate_energy(self, psi, return_all=False):
        """

        Args:
            psi : [nx,ny,nz], assumed to be normalized to 1

        Return: energy per atom, in hbar*omega_x units
        """
        psi_norm = self.normalize(psi)
        psi2 = torch.abs(psi_norm) ** 2
        dvol = self.grid.dx * self.grid.dy * self.grid.dz
        kinetic_energy = (
            torch.sum(self.k2 / 2 * torch.abs(torch.fft.fftn(psi_norm, norm="ortho")) ** 2) * dvol
        )

        potential_energy = torch.sum(self.potential_grid * psi2) * dvol
        if self.scattering_length == 0:
            scattering_energy = 0
        else:
            scattering_energy = self.num_atoms * 0.5 * torch.sum(self.g * psi2**2) * dvol
        if self.dipole_length == 0:
            dipolar_energy = 0
            lhy_energy = 0
        else:
            dipolar_energy = (
                self.num_atoms
                * 0.5
                * (torch.sum(self.vdk * torch.abs(torch.fft.fftn(psi2, norm="ortho")) ** 2))
                * dvol
            )

            lhy_energy = (
                self.num_atoms ** (3 / 2) * (2 / 5) * self.glhy * torch.sum(psi2 ** (5 / 2)) * dvol
            )

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

    def optimize(self, psi):

        self.potential_grid = torch.as_tensor(
            self.potential(self.grid.x, self.grid.y, self.grid.z), dtype=self.precision
        )
        self.k2, self.vdk = self._make_kspace_operators()
        psi = torch.as_tensor(psi, dtype=self.precision)
        psi = self.normalize(psi)
        min_energy = self.calculate_energy(psi).item()
        psi.requires_grad_()
        optimizer = LBFGS([psi], history_size=15, max_iter=100, line_search_fn="strong_wolfe")
        calls = 0
        iterations = 50
        with trange(iterations) as pbar:
            for i in pbar:

                def closure():
                    nonlocal calls
                    calls += 1
                    optimizer.zero_grad()
                    energy1 = self.calculate_energy(psi)
                    energy1.backward()
                    return energy1

                optimizer.step(closure)
                # with torch.no_grad():

                energy = self.calculate_energy(psi).item()
                diff = energy - min_energy

                if energy < min_energy:
                    min_energy = energy
                iterations_info = f"Iteration {i} Calls {calls} Energy {energy:.8f} diff {diff:.1e}"
                pbar.set_description(iterations_info)
                if abs(diff) < 1e-8:
                    break
        logging.info(iterations_info)

        psi_opt = self.normalize(torch.abs(psi))
        energy = self.calculate_energy(psi).detach().cpu().numpy()
        psi_opt = psi_opt.detach().cpu().numpy()
        return energy, psi_opt


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

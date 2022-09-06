import numpy as np

import constants


class SystemParameters:
    def __init__(self):
        self.mass = 162 * constants.dalton  # 164Dy mass in kg
        self.omega = 2 * np.pi * 125  # Trap frequency, in rad/s
        self.num_atoms = 100 * 1e3  # Number of atoms
        self.aho = np.sqrt(constants.hbar / self.mass / self.omega)  # oscillator length in meters
        self.scattering_length = 89 * constants.a0
        self.dipole_length = 131 * constants.a0

import numpy as np
from pystackreg import StackReg  # pip install pystackreg
from scipy import ndimage

from visualize import plot_one


def similarity(ref3_den, mov3_den):
    ref2_den = np.sum(ref3_den, axis=2)
    mov2_den = np.sum(mov3_den, axis=2)
    sr = StackReg(StackReg.RIGID_BODY)
    sr.register(ref2_den, mov2_den)
    out = np.zeros(mov3_den.shape)
    for i in range(mov3_den.shape[2]):
        out[:, :, i] = sr.transform(mov3_den[:, :, i])
    similarity = np.sum(ref3_den * out) / (np.linalg.norm(ref3_den[::]) * np.linalg.norm(out[::]))
    return similarity  # (similarity in range 0-1)


if __name__ == "__main__":
    ref3 = np.load("Single1_s86.0_d130.0_n100000.npy")
    mov3 = np.zeros(ref3.shape)
    for i in range(mov3.shape[2]):
        mov3[:, :, i] = ndimage.rotate(ref3[:, :, i], 15, reshape=False)  # rotate 10 degrees
    # plot_one(ref3)
    # plot_one(mov3)
    ref3_density = np.abs(ref3) ** 2
    mov3_density = np.abs(mov3) ** 2
    ref3_density = ref3_density / np.sum(ref3_density)
    mov3_density = mov3_density / np.sum(mov3_density)

    sim = similarity(ref3_density, mov3_density)
    print(sim)

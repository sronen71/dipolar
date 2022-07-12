import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftn, ifft, ifftn
from scipy.optimize import fmin_cg, minimize

# constants
# Conversions and Constants
a0 = 0.529e-10  # Bohr radius in meters
hbar = 1.054e-34
m = 162 * 1.67e-27  # 164Dy mass in kg
omega = 2 * np.pi * 125  # Trap frequency, in rad/s
ntot = 100 * 1e3  # Number of atoms
aho = np.sqrt(hbar / m / omega)  # oscillator length in meters

# Systems Parameters
# initial scattering length
ai = 85 * a0
ai /= aho

g = 4 * np.pi * ntot * ai

# initial dipole length
add = 131 * a0
add /= aho

gdd = 3 * ntot * add

# LHY
glhy = 32 / 3 * g * ai**1.5 / np.pi**0.5 * (1 + 3 / 2 * add**2 / ai**2) * ntot**0.5
# glhy = 0 #turn off, for now

# schrodinger equation energy
# variational guess
# create a 3d grid
xgridn = 130
xgridmax = 12
xgridmin = -12

xgrid = np.linspace(xgridmin, xgridmax, xgridn)
dx = xgrid[1] - xgrid[0]

kx = 2 * np.pi * np.fft.fftfreq(len(xgrid)) / dx
kx = kx.reshape(kx.size, 1, 1)

ygridn = 130
ygridmax = 12
ygridmin = -12

ygrid = np.linspace(ygridmin, ygridmax, ygridn)
dy = ygrid[1] - ygrid[0]

ky = 2 * np.pi * np.fft.fftfreq(len(ygrid)) / dy
ky = ky.reshape(1, ky.size, 1)

zgridn = 20
zgridmax = 10
zgridmin = -10

zgrid = np.linspace(zgridmin, zgridmax, zgridn)
dz = zgrid[1] - zgrid[0]

kz = 2 * np.pi * np.fft.fftfreq(len(zgrid)) / dz
kz = kz.reshape(1, 1, kz.size)

x, y, z = np.meshgrid(xgrid, ygrid, zgrid, indexing="ij")

vtrap = 1 / 2 * (x**2 + y**2 + 4 * z**2)
# v=x*0
# add in lattice potential?
kx_lat = 2 * np.pi / 0.8e-6 * aho
ky_lat = 2 * np.pi / 0.8e-6 * aho
lat_amp = -3e-1
v = vtrap + lat_amp * (np.cos(kx_lat * x) + np.cos(ky_lat * y))

# triangular lattice? use kx_lat
# v = vtrap + lat_amp * (np.cos(kx_lat * x) + np.cos(kx_lat * (-x / 2 + 3 ** 0.5 / 2 * y)) + np.cos(
#    kx_lat * (-x / 2 - 3 ** 0.5 / 2 * y)))


def psi_w(x, y, z, w):
    return np.pi**-0.75 * w**-1.5 * np.exp(-(x**2 + y**2 + z**2) / 2 / w**2)


def psi_2d(x, y, z, wp, wz):
    return (8 / np.pi**1.5 / wp**2 / wz) ** 0.5 * np.exp(
        -2 * (x**2 + y**2) / wp**2 - 2 * z**2 / wz**2
    )


def make_Vdk(kx, ky, kz, B):
    with np.errstate(divide="ignore", invalid="ignore"):
        k2 = kx**2 + ky**2 + kz**2
        Vdk = gdd * 4 * np.pi * (kz**2 / k2 - 1 / 3)
        Vdk[k2 == 0] = 0
        xq = B * np.sqrt(k2)
        spherical_cut = 1 + 3 * (xq * np.cos(xq) - np.sin(xq)) / xq**3
        Vdk = Vdk * spherical_cut
        Vdk[k2 == 0] = 0
    return Vdk


Vdk = make_Vdk(kx, ky, kz, 12)


def psi_energy(psi, v, dx, dy, dz):
    if not psi.ndim == 3:
        psi = psi.reshape((xgrid.size, ygrid.size, zgrid.size))
    psi /= np.sum(abs(psi) ** 2 * dx * dy * dz) ** 0.5
    Ek = np.real(ifftn((kx**2 + ky**2 + kz**2) / 2 * fftn(psi)))
    # Ek_x = ifft(kx ** 2 / 2 * fft(psi,axis = 0))
    # Ek_xy = ifft(ky ** 2 / 2 * fft(Ek_x,axis=1))
    # Ek = ifft(kz ** 2 / 2 * fft(Ek_xy,axis=2))
    V = v * psi
    Es = g * psi**3 / 2
    Edd = np.real(ifftn(Vdk * fftn(abs(psi) ** 2)) / 2) * psi
    Elhy = 0.4 * glhy * psi * (psi**2) ** 1.5
    grad = Ek + V + Es + Edd + Elhy
    energy = np.sum(psi * grad * dx * dy * dz)
    return energy


w_list = np.linspace(0.5, 2, 20)
E_analytic = 3 * (1 + w_list**4) / 4 / w_list**2
E_test = np.zeros_like(E_analytic)
for w_i, w in enumerate(w_list):
    E_test[w_i] = psi_energy(psi_w(x, y, z, w), v, dx, dy, dz)


def psi_gradient(psi, v, dx, dy, dz):
    if not psi.ndim == 3:
        psi = psi.reshape((xgrid.size, ygrid.size, zgrid.size))
    psi /= np.sum(abs(psi) ** 2 * dx * dy * dz) ** 0.5
    Ek = np.real(ifftn((kx**2 + ky**2 + kz**2) / 2 * fftn(psi)))
    V = v * psi
    Es = g * psi**3
    Edd = np.real(ifftn(Vdk * fftn(abs(psi) ** 2))) * psi
    Elhy = glhy * psi * (psi * psi) ** 1.5
    grad = Ek + V + Es + Edd + Elhy
    # project grad onto psi!
    proj = np.sum(grad * psi * dx * dy * dz)
    grad -= proj * psi
    return grad.flatten()


def normalize(f, dx):
    return f / np.sum(abs(f) ** 2 * dx) ** 0.5


# add in lattice potential?
# lat_spacings = np.linspace(0.8,1.6,7)
# lat_depths = np.linspace(0,0.4,8)

lat_spacings = np.array([1])
lat_depths = np.array([0])


results = np.zeros((lat_spacings.size, lat_depths.size) + x.shape)

for i, lat_depth in enumerate(lat_depths):
    for j, lat_spacing in enumerate(lat_spacings):
        kx_lat = 2 * np.pi / (lat_spacing * 1e-6) * aho
        lat_amp = -lat_depth
        # v = vtrap + lat_amp * (np.cos(kx_lat * x) + np.cos(kx_lat * (-x / 2 + 3 ** 0.5 / 2 * y)) + np.cos(
        #    kx_lat * (-x / 2 - 3 ** 0.5 / 2 * y)))
        v = vtrap + lat_amp * (np.cos(kx_lat * x) + np.cos(kx_lat * y))

        fmin = lambda psi: psi_energy(psi, v, dx, dy, dz)
        fprime = lambda psi: psi_gradient(psi, v, dx, dy, dz)
        x0 = psi_w(x, y, z, 2) * (1 + 0.4 * np.random.rand(*x.shape))

        out = fmin_cg(fmin, x0, fprime=fprime)
        out = normalize(out, dx)
        out = out.reshape((xgrid.size, ygrid.size, zgrid.size))

        results[j, i, :, :, :] = out

plt.clf()
fig, axs = plt.subplots(results.shape[0], results.shape[1], figsize=(50, 50))

xmin_plot = np.where(results > 1e-1 * results.max())[2].min()
xmax_plot = np.where(results > 1e-1 * results.max())[2].max()
ymin_plot = xmin_plot  # np.where(results > 1e-1 * results.max())[3].min()
ymax_plot = xmax_plot  # np.where(results > 1e-1 * results.max())[3].max()

for i in range(results.shape[0]):
    for j in range(results.shape[1]):
        # find where density is
        out = results[i, j]

        kx_lat = 2 * np.pi / (lat_spacings[i] * 1e-6) * aho
        lat_amp = -lat_depths[j]
        v_lat = lat_amp * (
            np.cos(kx_lat * x)
            + np.cos(kx_lat * (-x / 2 + 3**0.5 / 2 * y))
            + np.cos(kx_lat * (-x / 2 - 3**0.5 / 2 * y))
        )

        axs[i, j].imshow(
            abs(out[xmin_plot:xmax_plot, ymin_plot:ymax_plot, zgridn // 2]) ** 2,
            origin="upper",
            extent=[
                x[xmin_plot, 0, 0] * aho * 1e6,
                x[xmax_plot, 0, 0] * aho * 1e6,
                y[0, ymin_plot, 0] * aho * 1e6,
                y[0, ymax_plot, 0] * aho * 1e6,
            ],
            cmap="viridis",
        )  # cmocean.cm.dense)
        # axs[i,j].contour(v_lat[xmin_plot:xmax_plot, ymin_plot:ymax_plot, zgridn // 2], colors=('k',),alpha = 0.1,
        #                 extent = [x[xmin_plot,0,0],x[xmax_plot,0,0],y[0,ymin_plot,0],y[0,ymax_plot,0]],levels=2)

        if j == 0:
            axs[i, j].set_ylabel("%1.2f um" % (lat_spacings[i] * aho * 1e6), fontsize=70)
        else:
            axs[i, j].set_yticks([])

        if i == 0:
            axs[i, j].set_title("%1.2f" % (lat_depths[j]) + r"$\hbar \omega$", fontsize=70)

        if i == (results.shape[0] - 1):
            axs[i, j].set_xlabel("x (um)", fontsize=70)
        else:
            axs[i, j].set_xticks([])


plt.savefig("phase_dia_square_lat", bbox_inches="tight")

import torch


def interp(t):
    return 1 - 3 * t**2 + 2 * t**3


def perlin_noise(nx, ny, nz, scale=8):
    # 3D perlin noise
    # nx,ny,nz should be divisable by scale
    width_x = nx // scale
    width_y = ny // scale
    width_z = nz // scale
    gx, gy, gz = torch.randn(3, width_x + 1, width_y + 1, width_z + 1, 1, 1, 1)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None, None]
    ys = torch.linspace(0, 1, scale + 1)[None, :-1, None]
    zs = torch.linspace(0, 1, scale + 1)[None, None, :-1]

    wx = interp(xs)
    wy = interp(ys)
    wz = interp(zs)

    dots = 0
    dots += (
        wx * wy * wz * (gx[:-1, :-1, :-1] * xs + gy[:-1, :-1, :-1] * ys + gz[:-1, :-1, :-1] * zs)
    )
    dots += (
        (1 - wx)
        * wy
        * wz
        * (-gx[1:, :-1, :-1] * (1 - xs) + gy[1:, :-1, :-1] * ys + gz[1:, :-1, :-1] * zs)
    )
    dots += (
        wx
        * (1 - wy)
        * wz
        * (gx[:-1, 1:, :-1] * xs - gy[:-1, 1:, :-1] * (1 - ys) + gz[:-1, 1:, :-1] * zs)
    )
    dots += (
        (1 - wx)
        * (1 - wy)
        * wz
        * (-gx[1:, 1:, :-1] * (1 - xs) - gy[1:, 1:, :-1] * (1 - ys) + gz[1:, 1:, :-1] * zs)
    )

    dots += (
        wx
        * wy
        * (1 - wz)
        * (gx[:-1, :-1, :1] * xs + gy[:-1, :-1, :1] * ys - gz[:-1, :-1, :1] * (1 - zs))
    )
    dots += (
        (1 - wx)
        * wy
        * (1 - wz)
        * (-gx[1:, :-1, :1] * (1 - xs) + gy[1:, :-1, :1] * ys - gz[1:, :-1, :1] * (1 - zs))
    )
    dots += (
        wx
        * (1 - wy)
        * (1 - wz)
        * (gx[:-1, 1:, :1] * xs - gy[:-1, 1:, :1] * (1 - ys) + gz[:-1, 1:, :1] * (1 - zs))
    )
    dots += (
        (1 - wx)
        * (1 - wy)
        * (1 - wz)
        * (-gx[1:, 1:, :1] * (1 - xs) - gy[1:, 1:, :1] * (1 - ys) + gz[1:, 1:, :1] * (1 - zs))
    )

    return (
        dots.permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .view(width_x * scale, width_y * scale, width_z * scale)
        .cpu()
        .numpy()
    )

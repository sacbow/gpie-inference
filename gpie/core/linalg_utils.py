import math
from typing import Optional

from .backend import np  # callable backend selector
from .types import ArrayLike
from .rng_utils import get_rng


# ------------------------------------------------------------
# Precision utilities
# ------------------------------------------------------------
def reduce_precision_to_scalar(precision_array):
    xp = np()
    arr = xp.asarray(precision_array, dtype=xp.float32)

    if xp.any(arr <= 0):
        raise ValueError("Precision values must be positive.")

    inv_var = 1.0 / arr
    harmonic_mean = 1.0 / xp.mean(
        inv_var,
        axis=tuple(range(1, arr.ndim)),
    )
    return harmonic_mean


# ------------------------------------------------------------
# Random array generators
# RNG is NumPy, arrays are moved to backend
# ------------------------------------------------------------
def random_normal_array(
    shape,
    dtype=None,
    rng=None,
):
    rng = get_rng() if rng is None else rng
    xp = np()
    dtype = xp.complex128 if dtype is None else dtype

    kind = xp.dtype(dtype).kind

    if kind == "c":
        real = rng.normal(size=shape)
        imag = rng.normal(size=shape)
        out = (real + 1j * imag) / math.sqrt(2.0)
        return xp.asarray(out, dtype=dtype)

    elif kind == "f":
        out = rng.normal(size=shape)
        return xp.asarray(out, dtype=dtype)

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def sparse_complex_array(
    shape,
    sparsity: float,
    dtype=None,
    rng=None,
):
    rng = get_rng() if rng is None else rng
    xp = np()
    dtype = xp.complex128 if dtype is None else dtype

    shape = (shape,) if isinstance(shape, int) else shape
    N = int(math.prod(shape))
    num_nonzero = int(sparsity * N)

    out = xp.zeros(N, dtype=dtype)

    idx = rng.choice(N, size=num_nonzero, replace=False)
    real = rng.normal(scale=math.sqrt(0.5), size=num_nonzero)
    imag = rng.normal(scale=math.sqrt(0.5), size=num_nonzero)

    out[idx] = xp.asarray(real + 1j * imag, dtype=dtype)
    return out.reshape(shape)


def random_unitary_matrix(
    n: int,
    dtype=None,
    rng=None,
):
    rng = get_rng() if rng is None else rng
    xp = np()
    dtype = xp.complex64 if dtype is None else dtype

    A = random_normal_array((n, n), dtype=dtype, rng=rng)
    U, _, _ = xp.linalg.svd(A)
    return U


def random_binary_mask(
    shape,
    subsampling_rate: float = 0.5,
    rng=None,
):
    rng = get_rng() if rng is None else rng
    xp = np()

    shape = (shape,) if isinstance(shape, int) else shape
    total = int(math.prod(shape))
    num = int(total * subsampling_rate)

    idx = rng.choice(total, size=num, replace=False)
    mask = xp.zeros(total, dtype=bool)
    mask[idx] = True
    return mask.reshape(shape)


def random_phase_mask(
    shape,
    dtype=None,
    rng=None,
):
    rng = get_rng() if rng is None else rng
    xp = np()
    dtype = xp.complex64 if dtype is None else dtype

    theta = rng.uniform(0.0, 2.0 * math.pi, size=shape)
    return xp.exp(1j * xp.asarray(theta)).astype(dtype)


def masked_random_array(
    support: ArrayLike,
    dtype=None,
    rng=None,
):
    rng = get_rng() if rng is None else rng
    xp = np()
    dtype = xp.complex64 if dtype is None else dtype

    support = xp.asarray(support, dtype=bool)
    full = random_normal_array(support.shape, dtype=dtype, rng=rng)
    return xp.where(support, full, 0)


# ------------------------------------------------------------
# Geometry masks (backend-safe)
# ------------------------------------------------------------
def circular_aperture(shape, radius, center=None):
    xp = np()
    H, W = shape

    if not (0.0 < radius < 0.5):
        raise ValueError("radius must be between 0 and 0.5")

    min_dim = min(H, W)
    abs_radius = radius * min_dim

    cy_pix = H // 2
    cx_pix = W // 2

    if center is not None:
        dx = center[0] * min_dim
        dy = -center[1] * min_dim
        cx_pix = int(round(W // 2 + dx))
        cy_pix = int(round(H // 2 + dy))

    if not (0 <= cy_pix < H and 0 <= cx_pix < W):
        raise ValueError("center out of bounds")

    yy, xx = xp.ogrid[:H, :W]
    dist2 = (yy - cy_pix) ** 2 + (xx - cx_pix) ** 2
    return dist2 <= abs_radius ** 2



def square_aperture(shape, radius, center=None):
    xp = np()
    H, W = shape

    if not (0.0 < radius < 0.5):
        raise ValueError("radius must be between 0 and 0.5")

    min_dim = min(H, W)
    half = int(radius * min_dim)

    cy_pix = H // 2
    cx_pix = W // 2

    if center is not None:
        dx = center[0] * min_dim
        dy = -center[1] * min_dim
        cx_pix = int(round(W // 2 + dx))
        cy_pix = int(round(H // 2 + dy))

    y0, y1 = cy_pix - half, cy_pix + half
    x0, x1 = cx_pix - half, cx_pix + half

    if y0 < 0 or y1 >= H or x0 < 0 or x1 >= W:
        raise ValueError("square aperture goes out of bounds")

    mask = xp.zeros((H, W), dtype=bool)
    mask[y0:y1 + 1, x0:x1 + 1] = True
    return mask



def angular_spectrum_phase_mask(
    shape,
    wavelength,
    distance,
    dx,
    dy=None,
    dtype=None,
):
    xp = np()
    dtype = xp.complex128 if dtype is None else dtype
    H, W = shape
    dy = dx if dy is None else dy

    # Frequency grids (backend FFT)
    fx = xp.fft.fftfreq(W, d=dx)
    fy = xp.fft.fftfreq(H, d=dy)
    FX, FY = xp.meshgrid(fx, fy)

    k = 1.0 / wavelength  # spatial frequency

    # Clip evanescent components (negative inside sqrt)
    root = xp.maximum(0.0, k * k - FX * FX - FY * FY)
    phase = 2j * xp.pi * distance * xp.sqrt(root)

    return xp.exp(phase).astype(dtype, copy=False)



# ------------------------------------------------------------
# Scatter add
# ------------------------------------------------------------
def scatter_add(a, indices, values):
    xp = np()
    xp.add.at(a, indices, values)

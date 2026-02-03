import pytest
import numpy as np
import importlib.util

from gpie.core import backend
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import (
    reduce_precision_to_scalar,
    random_normal_array,
    sparse_complex_array,
    random_unitary_matrix,
    random_binary_mask,
    random_phase_mask,
    circular_aperture,
    square_aperture,
    masked_random_array,
    angular_spectrum_phase_mask,
    scatter_add
)
import warnings

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_reduce_precision_to_scalar_valid_and_invalid(xp):
    backend.set_backend(xp)

    # --- arra-precision, one batch ---
    arr_scalar = xp.array([[1.0, 2.0, 4.0]])
    scalar = reduce_precision_to_scalar(arr_scalar)
    expected_scalar = xp.array([[1.0 / xp.mean(1.0 / arr_scalar)]])
    assert xp.allclose(scalar, expected_scalar)

    with pytest.raises(ValueError):
        reduce_precision_to_scalar(xp.array([1.0, -1.0]))

    # --- Batched case ---
    arr_batch = xp.array([
        [1.0, 2.0, 4.0],    # harmonic mean ≈ 1.714
        [1.0, 1.0, 1.0],    # harmonic mean = 1.0
        [2.0, 2.0, 2.0]     # harmonic mean = 2.0
    ])
    reduced = reduce_precision_to_scalar(arr_batch)

    expected = xp.array([
        1.0 / xp.mean(1.0 / arr_batch[0]),
        1.0 / xp.mean(1.0 / arr_batch[1]),
        1.0 / xp.mean(1.0 / arr_batch[2]),
    ])
    assert reduced.shape == (3,)
    assert xp.allclose(reduced, expected, atol=1e-6)


    # --- Batched case with invalid input ---
    with pytest.raises(ValueError):
        arr_invalid = xp.array([
            [1.0, 2.0, -1.0],
            [1.0, 1.0, 1.0]
        ])
        reduce_precision_to_scalar(arr_invalid)


@pytest.mark.parametrize("xp", backend_libs)
def test_random_normal_array_real_and_complex_and_invalid(xp):
    backend.set_backend(xp)
    rng = get_rng(0)
    c = random_normal_array((3,), dtype=xp.complex128, rng=rng)
    assert xp.iscomplexobj(c)

    r = random_normal_array((3,), dtype=xp.float32, rng=rng)
    assert r.dtype == xp.float32

    with pytest.raises(ValueError):
        random_normal_array((2,), dtype=xp.int32)


@pytest.mark.parametrize("xp", backend_libs)
def test_sparse_complex_array(xp):
    backend.set_backend(xp)
    arr = sparse_complex_array((4,), sparsity=0.5)
    assert arr.shape == (4,)
    assert xp.iscomplexobj(arr)
    zeros = xp.sum(arr == 0)
    assert 0 < zeros < arr.size


@pytest.mark.parametrize("xp", backend_libs)
def test_random_unitary_matrix(xp):
    backend.set_backend(xp)
    U = random_unitary_matrix(4, dtype=xp.complex128)
    assert U.shape == (4, 4)
    I = U.conj().T @ U
    assert xp.allclose(I, xp.eye(4), atol=1e-12)


@pytest.mark.parametrize("xp", backend_libs)
def test_random_binary_mask(xp):
    backend.set_backend(xp)
    mask = random_binary_mask((8,), subsampling_rate=0.25)
    assert mask.shape == (8,)
    assert mask.dtype == bool
    assert 0 < mask.sum() < 8

    mask2 = random_binary_mask(4)
    assert mask2.shape == (4,)


@pytest.mark.parametrize("xp", backend_libs)
def test_random_phase_mask(xp):
    backend.set_backend(xp)
    mask = random_phase_mask((4, 4))
    assert mask.shape == (4, 4)
    assert xp.allclose(xp.abs(mask), 1.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_circular_aperture_valid_and_invalid(xp):
    backend.set_backend(xp)
    mask = circular_aperture((10, 10), radius=0.3)
    assert mask.shape == (10, 10)
    assert mask.dtype == bool

    with pytest.raises(ValueError):
        circular_aperture((10, 10), radius=0.6)

    with pytest.raises(ValueError):
        circular_aperture((10, 10), radius=0.3, center=(2.0, 2.0))


@pytest.mark.parametrize("xp", backend_libs)
def test_square_aperture_valid_and_invalid(xp):
    backend.set_backend(xp)
    mask = square_aperture((10, 10), radius=0.3)
    assert mask.shape == (10, 10)
    assert mask.dtype == bool

    with pytest.raises(ValueError):
        square_aperture((10, 10), radius=0.6)

    with pytest.raises(ValueError):
        square_aperture((5, 5), radius=0.49, center=(0.4, 0.4))


@pytest.mark.parametrize("xp", backend_libs)
def test_masked_random_array(xp):
    backend.set_backend(xp)
    support = xp.array([[True, False], [False, True]])
    arr = masked_random_array(support, dtype=xp.complex128)
    assert arr.shape == support.shape
    assert arr[0, 1] == 0


@pytest.mark.parametrize("xp", backend_libs)
def test_angular_spectrum_phase_mask(xp):
    backend.set_backend(xp)
    mask = angular_spectrum_phase_mask((8, 8), wavelength=500e-9, distance=0.01, dx=1e-6)
    assert mask.shape == (8, 8)
    assert xp.iscomplexobj(mask)
    assert xp.allclose(xp.abs(mask), 1.0, atol=1e-12)

@pytest.mark.parametrize("xp", backend_libs)
def test_scatter_add_accumulates(xp):
    backend.set_backend(xp)

    # 1D case with overlapping indices
    a = xp.zeros((6,), dtype=xp.float32)
    idx = (xp.array([1, 0, 1]),)  # tupleで渡す
    vals = xp.array([1.0, 1.0, 1.0], dtype=xp.float32)

    scatter_add(a, idx, vals)

    expected = xp.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=xp.float32)
    assert xp.allclose(a, expected)

    # 2D case with overlapping indices
    b = xp.zeros((3, 3), dtype=xp.float32)
    rows = xp.array([0, 1, 1, 2])
    cols = xp.array([0, 1, 1, 2])
    vals2 = xp.array([1.0, 2.0, 3.0, 4.0], dtype=xp.float32)

    scatter_add(b, (rows, cols), vals2)

    expected2 = xp.array([
        [1.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 4.0]
    ], dtype=xp.float32)

    assert xp.allclose(b, expected2)

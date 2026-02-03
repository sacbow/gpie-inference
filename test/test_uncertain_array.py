import pytest
import numpy as np
import importlib.util
import warnings

from gpie.core import backend
from gpie.core.uncertain_array import UncertainArray
from gpie.core.types import PrecisionMode

# Optional CuPy
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_astype_complex_to_real_with_warning(xp):
    backend.set_backend(xp)
    # complex UA with nonzero imaginary part
    data = xp.array([[1+1j, 2+2j]], dtype=xp.complex64)
    ua = UncertainArray(data, dtype=xp.complex64, precision=2.0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ua_real = ua.astype(xp.float32)
        # warning should be raised
        assert any("discard imaginary part" in str(wi.message) for wi in w)

    # dtype changed to real
    assert ua_real.is_real()
    # precision doubled
    assert np.allclose(ua_real.precision(raw=False), 4.0)



@pytest.mark.parametrize("xp", backend_libs)
def test_real_property_from_complex(xp):
    backend.set_backend(xp)
    data = xp.array([[1+2j, 3+4j]], dtype=xp.complex64)
    ua = UncertainArray(data, dtype=xp.complex64, precision=5.0)

    ua_real = ua.real
    # dtype is real counterpart
    assert ua_real.is_real()
    # precision doubled
    assert np.allclose(ua_real.precision(raw=False), 10.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_init_scalar_and_array_precision_vectorized(xp):
    backend.set_backend(xp)
    ua_scalar = UncertainArray.zeros(event_shape=(4, 4), batch_size=3, precision=2.0, scalar_precision=True)
    ua_array = UncertainArray.zeros(event_shape=(4, 4), batch_size=3, precision=2.0, scalar_precision=False)

    assert ua_scalar.batch_size == 3
    assert ua_scalar.event_shape == (4, 4)
    assert ua_scalar.precision_mode == PrecisionMode.SCALAR
    assert np.allclose(ua_scalar.precision(), 2.0)

    assert ua_array.precision_mode == PrecisionMode.ARRAY
    assert ua_array.precision().shape == (3, 4, 4)


@pytest.mark.parametrize("xp", backend_libs)
def test_mul_and_div_vectorized(xp):
    backend.set_backend(xp)
    ua1 = UncertainArray.random(event_shape=(2, 2), batch_size=5, precision=1.0)
    ua2 = UncertainArray.random(event_shape=(2, 2), batch_size=5, precision=2.0)
    ua_mul = ua1 * ua2
    ua_recovered = ua_mul / ua2

    assert np.allclose(ua1.data, ua_recovered.data, atol=1e-5)


@pytest.mark.parametrize("xp", backend_libs)
def test_damp_with_extremes(xp):
    backend.set_backend(xp)
    ua1 = UncertainArray.random(event_shape=(2, 2), batch_size=5, precision=1.0)
    ua2 = UncertainArray.random(event_shape=(2, 2), batch_size=5, precision=10.0)

    ua_0 = ua1.damp_with(ua2, alpha=0.0)
    ua_1 = ua1.damp_with(ua2, alpha=1.0)

    assert np.allclose(ua_0.data, ua1.data)
    assert np.allclose(ua_1.data, ua2.data)


@pytest.mark.parametrize("xp", backend_libs)
def test_product_reduce_over_batch_preserves_precision_mode(xp):
    backend.set_backend(xp)

    # Case 1: scalar precision mode
    ua_scalar = UncertainArray.random(event_shape=(4, 4), batch_size=10, precision=2.0, scalar_precision=True)
    scalar_precision =  ua_scalar.precision(raw = True)
    assert scalar_precision.shape == (10,1,1)

    reduced_scalar = ua_scalar.product_reduce_over_batch()

    assert reduced_scalar.event_shape == (4, 4)
    assert xp.allclose(
        reduced_scalar.precision(),
        xp.sum(ua_scalar.precision(), axis=0),
        atol=1e-5
    )
    # Precision mode should remain scalar
    assert reduced_scalar.precision_mode == ua_scalar.precision_mode
    #reduced_precision = reduced_scalar.precision(True)
    #assert reduced_precision.shape == (1,1,1)

    # Case 2: array precision mode
    ua_array = UncertainArray.random(event_shape=(4, 4), batch_size=10, precision=2.0, scalar_precision=False)
    reduced_array = ua_array.product_reduce_over_batch()

    assert reduced_array.event_shape == (4, 4)
    assert xp.allclose(
        reduced_array.precision(),
        xp.sum(ua_array.precision(), axis=0),
        atol=1e-5
    )
    # Precision mode should remain array
    assert reduced_array.precision_mode == ua_array.precision_mode


@pytest.mark.parametrize("xp", backend_libs)
def test_to_backend_roundtrip(xp):
    import numpy as np

    if not has_cupy:
        pytest.skip("CuPy not installed")
    else:
        import cupy as cp

    backend.set_backend(np)
    ua = UncertainArray.zeros(event_shape=(2, 2), dtype=np.complex64, precision=1.0)

    backend.set_backend(cp)
    ua.to_backend()
    assert isinstance(ua.data, cp.ndarray)
    assert ua.dtype == cp.complex64



from gpie.core import fft
import itertools

try:
    import pyfftw
    has_pyfftw = True
except ImportError:
    has_pyfftw = False


@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("fft_backend", ["numpy", "fftw"] if has_pyfftw else ["numpy"])
def test_fft2_ifft2_centered_reconstruction(xp, fft_backend):
    backend.set_backend(xp)
    if fft_backend == "fftw" and xp.__name__ != "numpy":
        pytest.skip("FFTW backend requires NumPy")

    fft.set_fft_backend(fft_backend)

    ua = UncertainArray.random(
        event_shape=(32, 32),
        batch_size=4,
        dtype=xp.complex64,
        scalar_precision=False,
    )

    ua_hat = ua.fft2_centered()
    assert ua_hat.precision_mode == PrecisionMode.SCALAR

    ua_rec = ua_hat.ifft2_centered()
    assert ua_rec.precision_mode == PrecisionMode.SCALAR

    assert np.allclose(ua.data, ua_rec.data, atol=1e-5), f"FFT->IFFT failed for {fft_backend}, {xp.__name__}"

@pytest.mark.parametrize("xp", backend_libs)
def test_fork_basic_and_error(xp):
    backend.set_backend(xp)
    ua = UncertainArray.zeros(event_shape=(4, 4), batch_size=1, precision=2.0)
    ua4 = ua.fork(batch_size=4)
    assert ua4.batch_size == 4
    assert ua4.event_shape == (4, 4)
    # All copies should match original
    for i in range(4):
        assert np.allclose(ua4.data[i], ua.data[0])
        assert np.allclose(ua4.precision()[i], ua.precision()[0])
    # Error if batch_size != 1
    ua_multi = UncertainArray.zeros(event_shape=(4, 4), batch_size=2)
    with pytest.raises(ValueError):
        _ = ua_multi.fork(batch_size=3)


@pytest.mark.parametrize("xp", backend_libs)
def test_zero_pad_data_and_precision(xp):
    backend.set_backend(xp)
    ua = UncertainArray.zeros(event_shape=(4, 4), batch_size=1, precision=5.0)
    ua_padded = ua.zero_pad(((1, 1), (2, 2)))
    assert ua_padded.event_shape == (6, 8)
    # Original data region should remain zero, padded region also zero
    assert xp.allclose(ua_padded.data, 0.0)
    # Precision in pad region must be zero
    center_prec = ua_padded.precision()[0, 1:-1, 2:-2]
    pad_prec = ua_padded.precision()[0]
    assert xp.allclose(center_prec, 5.0)
    assert xp.all(pad_prec >= 0)
    assert xp.allclose(pad_prec[:, :2], 1e8)  # left pad zero


@pytest.mark.parametrize("xp", backend_libs)
def test_getitem_basic_and_error(xp):
    backend.set_backend(xp)
    ua = UncertainArray.random(event_shape=(8, 8), batch_size=1, precision=3.0)
    sub = ua[2:6, 2:6]
    assert sub.event_shape == (4, 4)
    assert sub.batch_size == 1
    # Error if batch_size != 1
    ua_multi = UncertainArray.random(event_shape=(8, 8), batch_size=2)
    with pytest.raises(ValueError):
        _ = ua_multi[2:6, 2:6]


@pytest.mark.parametrize("xp", backend_libs)
def test_extract_patches_basic_and_error(xp):
    backend.set_backend(xp)
    ua = UncertainArray.random(event_shape=(8, 8), batch_size=1, precision=1.0)
    patches = ua.extract_patches([
        (slice(0, 4), slice(0, 4)),
        (slice(4, 8), slice(4, 8))
    ])
    assert patches.batch_size == 2
    assert patches.event_shape == (4, 4)
    # Check that extracted patches match original UA data
    assert xp.allclose(patches.data[0], ua.data[0, 0:4, 0:4])
    assert xp.allclose(patches.data[1], ua.data[0, 4:8, 4:8])
    # Error if batch_size != 1
    ua_multi = UncertainArray.random(event_shape=(8, 8), batch_size=2)
    with pytest.raises(ValueError):
        _ = ua_multi.extract_patches([(slice(0, 4), slice(0, 4))])


@pytest.mark.parametrize("xp", backend_libs)
def test_extract_block_basic(xp):
    backend.set_backend(xp)
    ua = UncertainArray.random(event_shape=(3, 3), batch_size=5, precision=2.0)

    block = slice(1, 4)  # size=3
    sub = ua.extract_block(block)

    assert sub.batch_size == 3
    assert sub.event_shape == ua.event_shape
    assert sub.dtype == ua.dtype
    assert sub.precision_mode == ua.precision_mode

    # Check data and precision consistency
    assert xp.allclose(sub.data, ua.data[1:4])
    assert xp.allclose(sub.precision(raw=True), ua.precision(raw=True)[1:4])


@pytest.mark.parametrize("xp", backend_libs)
def test_extract_block_invalid(xp):
    backend.set_backend(xp)
    ua = UncertainArray.random(event_shape=(2, 2), batch_size=4, precision=1.0)

    with pytest.raises(ValueError):
        ua.extract_block(slice(-1, 2))

    with pytest.raises(ValueError):
        ua.extract_block(slice(2, 10))

    with pytest.raises(ValueError):
        ua.extract_block(slice(3, 2))  # empty or reversed slice


@pytest.mark.parametrize("xp", backend_libs)
def test_insert_block_basic(xp):
    backend.set_backend(xp)
    ua = UncertainArray.random(event_shape=(3, 3), batch_size=5, precision=2.0)

    # extract block
    block = slice(1, 4)  # size=3
    sub = ua.extract_block(block)

    # modify sub to ensure actual change
    sub.data = sub.data + 10.0

    # insert into a fresh UA
    ua2 = ua  # in-place update of original
    ua2.insert_block(block, sub)

    # correctness check
    assert xp.allclose(ua2.data[1:4], sub.data)
    assert xp.allclose(
        ua2.precision(raw=True)[1:4],
        sub.precision(raw=True)
    )

    # unchanged regions
    assert xp.allclose(ua2.data[0], ua.data[0])
    assert xp.allclose(ua2.data[4], ua.data[4])


@pytest.mark.parametrize("xp", backend_libs)
def test_insert_block_mismatch_errors(xp):
    backend.set_backend(xp)
    ua = UncertainArray.random(event_shape=(3, 3), batch_size=5, precision=2.0)
    sub = UncertainArray.random(event_shape=(3, 3), batch_size=2, precision=2.0)

    # block size mismatch
    with pytest.raises(ValueError):
        ua.insert_block(slice(1, 4), sub)  # block size=3, sub.batch=2

    # event shape mismatch
    sub_wrong_shape = UncertainArray.random(event_shape=(2, 3), batch_size=3)
    with pytest.raises(ValueError):
        ua.insert_block(slice(1, 4), sub_wrong_shape)

    # dtype mismatch
    sub_wrong_dtype = UncertainArray.random(
        event_shape=(3, 3), batch_size=3, dtype=xp.float32
    )
    with pytest.raises(TypeError):
        ua.insert_block(slice(1, 4), sub_wrong_dtype)

    # precision mode mismatch
    sub_wrong_prec = UncertainArray.random(
        event_shape=(3, 3), batch_size=3, precision=1.0, scalar_precision=False
    )
    with pytest.raises(ValueError):
        ua.insert_block(slice(1, 4), sub_wrong_prec)


@pytest.mark.parametrize("xp", backend_libs)
def test_ua_copy_is_deep(xp):
    backend.set_backend(xp)

    ua = UncertainArray.zeros((2, 2), batch_size=2, dtype=xp.complex64, precision=1.0)
    ua.data[...] = 3.0

    ua_copy = ua.copy()

    # Modify original
    ua.data[...] = 5.0
    ua.precision(raw=True)[...] = 7.0

    # Copy must remain unchanged
    assert xp.allclose(ua_copy.data, 3.0)
    assert xp.allclose(ua_copy.precision(raw=False), 1.0)


@pytest.mark.skipif(not has_cupy, reason="CuPy required")
def test_to_backend_moves_precision_array():
    import numpy as np
    import cupy as cp
    from gpie.core import backend
    from gpie.core.uncertain_array import UncertainArray

    backend.set_backend(np)

    ua = UncertainArray.zeros(
        event_shape=(2, 2),
        batch_size=3,
        dtype=np.complex64,
        precision=2.0,
        scalar_precision=False, 
    )

    # sanity check
    assert isinstance(ua.data, np.ndarray)
    assert isinstance(ua.precision(raw=True), np.ndarray)

    backend.set_backend(cp)
    ua.to_backend()

    assert isinstance(ua.data, cp.ndarray)
    assert isinstance(ua.precision(raw=True), cp.ndarray)


@pytest.mark.skipif(not has_cupy, reason="CuPy required")
def test_precision_numpy_input_is_moved_to_cupy_backend():
    import numpy as np
    import cupy as cp
    from gpie.core import backend
    from gpie.core.uncertain_array import UncertainArray

    backend.set_backend(cp)

    # Mean data is on GPU
    data = cp.zeros((2, 2, 2), dtype=cp.complex64)

    # Precision is given as NumPy array
    precision_np = np.ones((2, 2, 2), dtype=np.float32)

    ua = UncertainArray(
        data,
        dtype=cp.complex64,
        precision=precision_np,  # ndarray → array-precision mode
        batched=True,
    )

    # Data must stay on GPU
    assert isinstance(ua.data, cp.ndarray)

    # Precision must be moved to GPU as well
    prec = ua.precision(raw=True)
    assert isinstance(prec, cp.ndarray)

    # Precision mode should be ARRAY
    from gpie.core.types import PrecisionMode
    assert ua.precision_mode == PrecisionMode.ARRAY


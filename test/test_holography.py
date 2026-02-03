import importlib.util
import pytest
import numpy as np

from gpie import model, SupportPrior, ifft2, AmplitudeMeasurement, mse
from gpie.core.linalg_utils import circular_aperture, masked_random_array
from gpie.core.rng_utils import get_rng
from gpie.core.backend import set_backend

# ------------------------------------------------------------
# Optional CuPy support
# ------------------------------------------------------------
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp


# ------------------------------------------------------------
# Holography model (DSL definition)
# ------------------------------------------------------------

@model
def holography_model(var, ref_wave, support, batch_size, dtype=np.complex64):
    """
    Holography model with explicit batch_size handling.

    Args:
        var: Noise variance
        ref_wave: ndarray of shape (B, H, W)
        support: boolean ndarray of shape (H, W)
        batch_size: int
        dtype: complex dtype
    """
    set_backend(np)
    
    H, W = support.shape

    obj = ~SupportPrior(
        support=support,
        event_shape=(H, W),
        batch_size=batch_size,
        label="obj",
        dtype=dtype,
    )

    AmplitudeMeasurement(var=var) << ifft2(ref_wave + obj)
    return


# ------------------------------------------------------------
# Utility: build a test graph (CPU-side constants)
# ------------------------------------------------------------

def build_holography_graph(
    batch_size,
    shape=(64, 64),
    noise=1e-4,
    seed=0,
):
    rng = get_rng(seed)
    H, W = shape

    support_x = circular_aperture(shape, radius=0.1, center=(-0.2, -0.2))
    support_y = circular_aperture(shape, radius=0.1, center=(0.2, 0.2))

    # Reference wave (CPU-side generation)
    ref_single = masked_random_array(
        support_x,
        dtype=np.complex128,
        rng=rng,
    )

    # Batched reference wave
    ref_wave = np.stack([ref_single] * batch_size, axis=0)

    g = holography_model(
        var=noise,
        ref_wave=ref_wave,
        support=support_y,
        batch_size=batch_size,
        dtype=np.complex64,
    )

    g.set_init_rng(get_rng(seed + 1))
    g.generate_sample(rng=get_rng(seed + 2), update_observed=True)

    return g


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_holography_single_batch_convergence(device):
    """
    Basic sanity check: batch_size = 1 holography reconstruction converges.
    """
    if device == "cuda" and not has_cupy:
        pytest.skip("CuPy is not available; skipping CUDA test.")

    g = build_holography_graph(
        batch_size=1,
        seed=10,
    )

    true_obj = g.get_wave("obj").get_sample()

    g.run(
        n_iter=50,
        device=device,
        verbose=False,
    )

    recon = g.get_wave("obj").compute_belief().data
    err = mse(recon, true_obj)

    assert err < 1e-3
    assert recon.shape == true_obj.shape


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_holography_batchwise_parallel_convergence(device):
    """
    Sanity check: batch_size > 1 converges under parallel scheduling.
    """
    if device == "cuda" and not has_cupy:
        pytest.skip("CuPy is not available; skipping CUDA test.")

    g = build_holography_graph(
        batch_size=2,
        seed=20,
    )

    true_obj = g.get_wave("obj").get_sample()

    g.run(
        n_iter=50,
        schedule="parallel",
        device=device,
        verbose=False,
    )

    recon = g.get_wave("obj").compute_belief().data
    err = mse(recon, true_obj)

    assert err < 1e-3
    assert recon.shape == true_obj.shape


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("schedule", ["parallel", "sequential"])
def test_holography_batchwise_parallel_sequential(device, schedule):
    """
    Sanity check: batch_size > 1 converges under both
    parallel and sequential scheduling.
    """
    if device == "cuda" and not has_cupy:
        pytest.skip("CuPy is not available; skipping CUDA test.")

    g = build_holography_graph(
        batch_size=2,
        seed=30,
    )

    true_obj = g.get_wave("obj").get_sample()

    g.run(
        n_iter=50,
        schedule=schedule,
        device=device,
        verbose=False,
    )

    recon = g.get_wave("obj").compute_belief().data
    err = mse(recon, true_obj)

    assert err < 1e-3
    assert recon.shape == true_obj.shape

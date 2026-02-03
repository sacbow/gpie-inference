import importlib.util
import pytest
import numpy as np

from gpie import model, fft2, mse
from gpie import SparsePrior, GaussianMeasurement
from gpie.core.linalg_utils import random_binary_mask
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
# Graph builder
# ------------------------------------------------------------

def build_fft_cs_graph(
    *,
    event_shape,
    batch_size,
    rho,
    var,
    mask,
    seed_init=11,
    seed_sample=123,
):
    """
    Build and initialize an FFT-based compressive sensing graph.
    All constants are generated on CPU.
    """

    @model
    def fft_cs_model(rho, shape, var, batch_size):
        x = ~SparsePrior(
            rho=rho,
            event_shape=shape,
            batch_size=batch_size,
            label="x",
            dtype=np.complex64,
        )
        GaussianMeasurement(var=var, with_mask=True) << fft2(x)

    g = fft_cs_model(
        rho=rho,
        shape=event_shape,
        var=var,
        batch_size=batch_size,
    )

    g.set_init_rng(get_rng(seed=seed_init))
    g.generate_sample(
        rng=get_rng(seed=seed_sample),
        update_observed=True,
        mask=mask,
    )

    return g


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fft_compressive_sensing_batch2_parallel_vs_sequential(device):
    """
    FFT-based compressive sensing with batch_size=2 should yield
    equivalent reconstructions for parallel and sequential schedules.
    """
    set_backend(np)
    if device == "cuda" and not has_cupy:
        pytest.skip("CuPy is not available; skipping CUDA test.")

    # ----------------------
    # Parameters
    # ----------------------
    event_shape = (128, 128)
    batch_size = 2
    rho = 0.1
    var = 1e-4
    subsample_ratio = 0.3
    n_iter = 50

    rng = get_rng(seed=42)

    # Mask is generated on CPU
    mask = random_binary_mask(
        (batch_size, *event_shape),
        subsampling_rate=subsample_ratio,
        rng=rng,
    )

    # ----------------------
    # Reference graph (parallel)
    # ----------------------
    g_parallel = build_fft_cs_graph(
        event_shape=event_shape,
        batch_size=batch_size,
        rho=rho,
        var=var,
        mask=mask,
    )

    true_x = g_parallel.get_wave("x").get_sample()

    
    if device == "cuda":
        true_x_monitor = cp.asarray(true_x)
    else:
        true_x_monitor = true_x

    mse_parallel = []

    def monitor_parallel(graph, t):
        est = graph.get_wave("x").compute_belief().data
        err = mse(est, true_x_monitor)
        mse_parallel.append(float(err))

    g_parallel.run(
        n_iter=n_iter,
        schedule="parallel",
        device=device,
        callback=monitor_parallel,
        verbose=False,
    )

    est_parallel = g_parallel.get_wave("x").compute_belief().data.copy()

    # ----------------------
    # Sequential graph
    # ----------------------
    g_sequential = build_fft_cs_graph(
        event_shape=event_shape,
        batch_size=batch_size,
        rho=rho,
        var=var,
        mask=mask,
    )

    mse_sequential = []

    def monitor_sequential(graph, t):
        est = graph.get_wave("x").compute_belief().data
        err = mse(est, true_x_monitor)
        mse_sequential.append(float(err))

    g_sequential.run(
        n_iter=n_iter,
        schedule="sequential",
        block_size=1,
        device=device,
        callback=monitor_sequential,
        verbose=False,
    )

    est_sequential = g_sequential.get_wave("x").compute_belief().data

    # ----------------------
    # Assertions
    # ----------------------
    assert len(mse_parallel) == n_iter
    assert len(mse_sequential) == n_iter

    assert mse_parallel[-1] < 1e-4, (
        f"Parallel schedule failed to converge: "
        f"MSE={mse_parallel[-1]:.2e}"
    )

    assert mse_sequential[-1] < 1e-4, (
        f"Sequential schedule failed to converge: "
        f"MSE={mse_sequential[-1]:.2e}"
    )

    assert np.allclose(
        est_parallel,
        est_sequential,
        atol=1e-4,
    ), "Parallel and sequential estimates differ"

import importlib.util
import pytest
import numpy as np

from gpie import model, observe, mse
from gpie import SparsePrior, GaussianMeasurement, UnitaryMatrixPropagator
from gpie.core.linalg_utils import random_unitary_matrix, random_binary_mask
from gpie.core.rng_utils import get_rng
from gpie.core.backend import set_backend


# ------------------------------------------------------------
# Optional CuPy support
# ------------------------------------------------------------
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compressive_sensing_mse_decreases(device):
    """
    Integration test for compressive sensing.

    This test verifies that:
      1. EP-based inference runs correctly on CPU and GPU backends.
      2. Reconstruction MSE decreases over iterations.
      3. Final reconstruction error is sufficiently small.

    Backend switching is controlled exclusively via Graph.run(device=...).
    """
    set_backend(np)
    if device == "cuda" and not has_cupy:
        pytest.skip("CuPy is not available; skipping CUDA test.")

    # ------------------------------------------------------------
    # 1. Problem setup
    # ------------------------------------------------------------
    n = 512
    rho = 0.1
    var = 1e-4
    mask_ratio = 0.3

    # ------------------------------------------------------------
    # 2. Random operators and mask (CPU-side RNG)
    # ------------------------------------------------------------
    rng = get_rng(seed=12)

    U = random_unitary_matrix(
        n,
        rng=rng,
        dtype=np.complex64,
    )

    mask = random_binary_mask(
        n,
        subsampling_rate=mask_ratio,
        rng=rng,
    )

    # ------------------------------------------------------------
    # 3. Model definition
    # ------------------------------------------------------------
    @model
    def compressive_sensing():
        x = ~SparsePrior(
            rho=rho,
            event_shape=(n,),
            label="x",
            dtype=np.complex64,
        )

        GaussianMeasurement(
            var=var,
            with_mask=True,
        ) << (UnitaryMatrixPropagator(U) @ x)

    g = compressive_sensing()

    # ------------------------------------------------------------
    # 4. Initialization and data generation
    # ------------------------------------------------------------
    g.set_init_rng(get_rng(seed=11))

    g.generate_sample(
        rng=get_rng(seed=42),
        mask=mask,
    )

    # Ground truth (sampled latent variable)
    true_x = g.get_wave("x").get_sample()

    # Ensure true_x is on the same backend as inference
    if device == "cuda":
        import cupy as cp
        true_x_device = cp.asarray(true_x)
    else:
        true_x_device = true_x

    # ------------------------------------------------------------
    # 5. Inference with MSE monitoring
    # ------------------------------------------------------------
    mse_list = []

    def monitor(graph, t):
        est = graph.get_wave("x").compute_belief().data
        err = mse(est, true_x_device)
        mse_list.append(err)

    g.run(
        n_iter=15,
        device=device,
        callback=monitor,
    )

    # ------------------------------------------------------------
    # 6. Assertions
    # ------------------------------------------------------------
    assert len(mse_list) > 1, "Monitor callback was not called."

    # MSE should decrease
    assert mse_list[0] > mse_list[-1], (
        "MSE did not decrease during inference."
    )

    # Final reconstruction quality
    final_est = g.get_wave("x").compute_belief().data
    final_mse = mse(final_est, true_x)

    assert final_mse < 1e-3, (
        f"Final MSE too high: {final_mse}"
    )

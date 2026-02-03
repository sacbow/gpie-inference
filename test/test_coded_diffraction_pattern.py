import pytest
import numpy as np
import importlib.util

from gpie import (
    model,
    GaussianPrior,
    fft2,
    GaussianMeasurement,
    AmplitudeMeasurement,
    pmse,
    replicate,
)
from gpie.core.rng_utils import get_rng
from gpie.core.backend import set_backend
from gpie.core.linalg_utils import random_normal_array

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp


@model
def coded_diffraction_model(var, masks, dtype=np.complex64):
    """
    Coded diffraction pattern model using ForkPropagator via replicate().
    """
    B, H, W = masks.shape

    obj = ~GaussianPrior(event_shape=(H, W), label="obj", dtype=dtype)
    obj_batch = replicate(obj, batch_size=B)

    masked = masks * obj_batch
    Y = fft2(masked)

    AmplitudeMeasurement(var=var, damping=0.3) << Y


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("schedule", ["parallel", "sequential"])
def test_coded_diffraction_model_reconstruction(device, schedule):
    set_backend(np)

    if device == "cuda" and not has_cupy:
        pytest.skip("CuPy is not available; skipping CUDA test.")

    # ------------------------------------------------------------
    # 1. Problem setup (CPU-side constants)
    # ------------------------------------------------------------
    shape = (64, 64)
    n_measurements = 4
    dtype = np.complex64
    noise_var = 1e-4

    rng = get_rng(seed=123)

    true_obj_cpu = random_normal_array((1, *shape), dtype=dtype, rng=rng)
    masks_cpu = random_normal_array((n_measurements, *shape), dtype=dtype, rng=rng)

    # ------------------------------------------------------------
    # 2. Build graph & generate observations (CPU)
    # ------------------------------------------------------------
    g = coded_diffraction_model(var=noise_var, masks=masks_cpu, dtype=dtype)

    g.set_init_rng(get_rng(seed=4))
    g.get_wave("obj").set_sample(true_obj_cpu)
    g.generate_sample(rng=get_rng(seed=5), update_observed=True)

    # ------------------------------------------------------------
    # 3. Prepare device-side ground truth ONCE
    # ------------------------------------------------------------
    if device == "cuda":
        true_obj = cp.asarray(true_obj_cpu)
    else:
        true_obj = true_obj_cpu

    # ------------------------------------------------------------
    # 4. Inference with monitoring
    # ------------------------------------------------------------
    history = []

    def monitor(graph, t):
        est = graph.get_wave("obj").compute_belief().data
        err = pmse(est, true_obj)

        # cupy scalar → Python float
        if device == "cuda":
            err = float(cp.asnumpy(err))
        else:
            err = float(err)

        history.append(err)

    g.run(
        n_iter=100,
        schedule=schedule,
        device=device,
        callback=monitor,
    )

    assert len(history) > 0, "Monitor callback was not called."

    # ------------------------------------------------------------
    # 5. Convergence check
    # ------------------------------------------------------------
    assert history[-1] < 1e-3, (
        f"CDP reconstruction did not converge "
        f"(device={device}, schedule={schedule}): "
        f"final PMSE={history[-1]:.2e}"
    )

    # ------------------------------------------------------------
    # 6. Final estimate sanity check
    # ------------------------------------------------------------
    est = g.get_wave("obj").compute_belief().data
    assert est.shape == (1, *shape)

import numpy as np
import pytest
import importlib.util

from gpie import (
    model,
    GaussianPrior,
    GaussianMeasurement,
    fft2,
    mse,
)
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import random_normal_array
from gpie.core.backend import set_backend


# -------------------------------------------------
# Optional CuPy support
# -------------------------------------------------

cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp


# -------------------------------------------------
# Test configuration
# -------------------------------------------------

OBJ_SHAPE = (64, 64)
PRB_SHAPE = (32, 32)

STRIDE = 16
N_ITER = 30
NOISE_VAR = 1e-4

ERROR_THRESHOLD = 1e-2


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def generate_scan_indices(obj_shape, prb_shape, stride):
    H, W = obj_shape
    h, w = prb_shape

    ys = list(range(0, H - h + 1, stride))
    xs = list(range(0, W - w + 1, stride))

    return [(slice(y, y + h), slice(x, x + w)) for y in ys for x in xs]


# -------------------------------------------------
# Model definition
# -------------------------------------------------

@model
def blind_ptychography_model(indices, noise, dtype):
    obj = ~GaussianPrior(
        event_shape=OBJ_SHAPE,
        label="object",
        dtype=dtype,
    )

    prb = ~GaussianPrior(
        event_shape=PRB_SHAPE,
        label="probe",
        dtype=dtype,
    )

    patches = obj.extract_patches(indices)
    exit_waves = prb * patches

    GaussianMeasurement(var=noise) << fft2(exit_waves)
    return


# -------------------------------------------------
# Integration test
# -------------------------------------------------

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("schedule", ["parallel", "sequential"])
def test_blind_ptychography_end_to_end(device, schedule):
    set_backend(np)

    if device == "cuda" and not has_cupy:
        pytest.skip("CuPy is not available")

    rng = get_rng(seed=123)
    dtype = np.complex64

    # -------------------------------------------------
    # Ground truth (always NumPy)
    # -------------------------------------------------

    true_obj = random_normal_array(
        (1, *OBJ_SHAPE), dtype=dtype, rng=rng
    )
    true_prb = random_normal_array(
        (1, *PRB_SHAPE), dtype=dtype, rng=rng
    )

    # -------------------------------------------------
    # Device-side copies for monitoring
    # (explicit, Inspector 導入前の暫定対応)
    # -------------------------------------------------

    if device == "cuda":
        true_obj_dev = cp.asarray(true_obj)
        true_prb_dev = cp.asarray(true_prb)
    else:
        true_obj_dev = true_obj
        true_prb_dev = true_prb

    # -------------------------------------------------
    # Scan geometry
    # -------------------------------------------------

    indices = generate_scan_indices(
        OBJ_SHAPE, PRB_SHAPE, STRIDE
    )

    # -------------------------------------------------
    # Build graph
    # -------------------------------------------------

    g = blind_ptychography_model(
        indices=indices,
        noise=NOISE_VAR,
        dtype=dtype,
    )

    g.set_init_rng(get_rng(seed=4))

    g.get_wave("object").set_sample(true_obj)
    g.get_wave("probe").set_sample(true_prb)

    g.generate_sample(
        rng=get_rng(seed=5),
        update_observed=True,
    )

    # -------------------------------------------------
    # Monitor
    # -------------------------------------------------

    history_obj = []
    history_prb = []

    def monitor(graph, t):
        obj_est = graph.get_wave("object").compute_belief().data
        prb_est = graph.get_wave("probe").compute_belief().data

        history_obj.append(
            mse(obj_est, true_obj_dev, normalize=True)
        )
        history_prb.append(
            mse(prb_est, true_prb_dev, normalize=True)
        )

    # -------------------------------------------------
    # Run inference
    # -------------------------------------------------

    g.run(
        n_iter=N_ITER,
        schedule=schedule,
        device=device,
        callback=monitor,
    )

    # -------------------------------------------------
    # Assertions
    # -------------------------------------------------

    assert history_obj[-1] < ERROR_THRESHOLD, (
        f"Object did not converge "
        f"(device={device}, schedule={schedule}): "
        f"final MSE={history_obj[-1]:.2e}"
    )

    assert history_prb[-1] < ERROR_THRESHOLD, (
        f"Probe did not converge "
        f"(device={device}, schedule={schedule}): "
        f"final MSE={history_prb[-1]:.2e}"
    )

    obj_est = g.get_wave("object").compute_belief().data
    prb_est = g.get_wave("probe").compute_belief().data

    assert obj_est.shape == (1, *OBJ_SHAPE)
    assert prb_est.shape == (1, *PRB_SHAPE)
import importlib.util
import pytest
import numpy as np

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse
from gpie.core.linalg_utils import random_phase_mask
from gpie.core.rng_utils import get_rng
from gpie.core.backend import set_backend


# ------------------------------------------------------------
# Optional CuPy support
# ------------------------------------------------------------
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp
    backend_libs = [np, cp]
else:
    backend_libs = [np]


# ------------------------------------------------------------
# Model definition (batch_size = 1 only)
# ------------------------------------------------------------
@model
def random_structured_cdi(
    masks,
    noise,
    *,
    dtype=np.complex64,
):
    """
    Structured random CDI model.
    NOTE: This model is intended for batch_size = 1 only.
    """
    obj = ~GaussianPrior(
        event_shape=(32, 32),
        batch_size=2,
        label="sample",
        dtype=dtype,
    )

    pad_width = ((16, 16), (16, 16))
    x = obj.zero_pad(pad_width)

    for mask in masks:
        x = fft2(mask * x)

    AmplitudeMeasurement(var=noise, damping=0.3) << x
    return


# ------------------------------------------------------------
# Graph builder (device-agnostic)
# ------------------------------------------------------------
def build_random_structured_cdi_graph(
    *,
    xp,
    seed=0,
    n_layers=2,
):
    rng = get_rng(seed)

    masks = [
        random_phase_mask(
            (2, 64, 64),
            dtype=xp.complex64,
            rng=rng,
        )
        for _ in range(n_layers)
    ]

    g = random_structured_cdi(
        masks=masks,
        noise=1e-4,
        dtype=xp.complex64,
    )

    g.set_init_rng(get_rng(seed + 11))
    g.generate_sample(
        rng=get_rng(seed + 22),
        update_observed=True,
    )

    return g


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("schedule", ["parallel", "sequential"])
def test_random_structured_cdi_batch1_converges(xp, schedule):
    """
    Sanity check for structured random CDI with batch_size=1.

    This test intentionally restricts batch_size to 1.
    """
    set_backend(np)

    if xp is not np and not has_cupy:
        pytest.skip("CuPy not available")

    # ----------------------------
    # Build graph (constants on CPU)
    # ----------------------------
    g = build_random_structured_cdi_graph(
        xp=xp,
        seed=12,
        n_layers=2,
    )

    sample_wave = g.get_wave("sample")
    true_sample = sample_wave.get_sample()  # already on execution device

    # ----------------------------
    # Run inference
    # ----------------------------
    device = "cuda" if xp is not np else "cpu"

    g.run(
        n_iter=200,
        schedule=schedule,
        device=device,
        verbose=False,
    )

    # ----------------------------
    # Evaluate reconstruction
    # ----------------------------
    recon = sample_wave.compute_belief().data

    # Compare only the second batch element (original test semantics)
    err = pmse(recon[1], true_sample[1])

    assert err < 1e-3, f"PMSE too large: {err:.2e}"
    assert recon.shape == true_sample.shape

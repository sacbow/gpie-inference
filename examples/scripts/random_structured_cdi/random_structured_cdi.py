import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from gpie import (
    model,
    SupportPrior,
    fft2,
    AmplitudeMeasurement,
    pmse,
)
from gpie.core.linalg_utils import circular_aperture, random_phase_mask

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from io_utils import load_sample_image


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_random_structured_cdi")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------

@model
def random_structured_cdi(
    shape,
    support,
    n_layers,
    phase_masks,
    noise,
):
    """
    Random structured CDI model with multiple random modulation layers.
    """

    x = ~SupportPrior(
        support=support,
        event_shape=shape,
        label="object",
        dtype=np.complex64,
    )

    for i in range(n_layers):
        x = fft2(phase_masks[i] * x)

    AmplitudeMeasurement(var=noise) << x


# ------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------

def build_graph(
    shape,
    support_radius,
    n_layers,
    noise,
    rng_model,
    rng_data,
):
    H, W = shape

    support = circular_aperture(shape, radius=support_radius)

    # Ground-truth object (amplitude + phase, masked by support)
    amp = load_sample_image("camera", shape=shape)
    phase = load_sample_image("moon", shape=shape)
    x_true = amp * np.exp(1j * 2 * np.pi * phase)
    x_true *= support

    phase_masks = [
        random_phase_mask(shape, rng=rng_model, dtype=np.complex64)
        for _ in range(n_layers)
    ]

    g = random_structured_cdi(
        shape=shape,
        support=support,
        n_layers=n_layers,
        phase_masks=phase_masks,
        noise=noise,
    )

    g.set_sample("object", x_true.astype(np.complex64))
    g.generate_observations(rng=rng_data)

    return g, x_true


# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------

def run_random_structured_cdi(
    *,
    n_iter,
    size,
    n_layers,
    support_radius,
    noise,
    device,
    seed,
    save_graph,
):
    shape = (size, size)

    rng_model = np.random.default_rng(seed)
    rng_data = np.random.default_rng(seed + 1)
    rng_init = np.random.default_rng(seed + 2)

    g, x_true = build_graph(
        shape=shape,
        support_radius=support_radius,
        n_layers=n_layers,
        noise=noise,
        rng_model=rng_model,
        rng_data=rng_data,
    )

    pmse_history = []

    def monitor(graph, t):
        if t % 10 == 0 or t == n_iter - 1:
            est = graph["object"]["mean"]
            gt = graph["object"]["sample"]
            err = pmse(est, gt)
            pmse_history.append(err)
            print(f"[t={t:4d}] PMSE = {err:.5e}")

    g.set_init_rng(rng_init)

    g.run(
        n_iter=n_iter,
        device=device,
        callback=monitor,
    )

    # --------------------------------------------------------
    # Post-processing (CPU)
    # --------------------------------------------------------

    est = g["object"]["mean"][0]
    gt = x_true

    amp = np.abs(est)
    phase = np.angle(est) * (np.abs(gt) > 1e-5)

    plt.imsave(os.path.join(RESULTS_DIR, "reconstructed_amp.png"), amp, cmap="gray")
    plt.imsave(os.path.join(RESULTS_DIR, "reconstructed_phase.png"), phase, cmap="twilight")

    true_amp = np.abs(gt)
    true_phase = np.angle(gt) * (true_amp > 1e-5)

    plt.imsave(os.path.join(RESULTS_DIR, "true_amp.png"), true_amp, cmap="gray")
    plt.imsave(os.path.join(RESULTS_DIR, "true_phase.png"), true_phase, cmap="twilight")

    it = np.arange(0, len(pmse_history) * 10, 10)

    plt.figure()
    plt.plot(it, pmse_history, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("PMSE")
    plt.yscale("log")
    plt.title("Random Structured CDI Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "convergence.png"))
    plt.close()

    if save_graph:
        print("Saving factor graph visualization ...")
        g.visualize(output_path=os.path.join(RESULTS_DIR, "graph.html"))


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random Structured CDI benchmark (gPIE 0.3.1)"
    )
    parser.add_argument("--n-iter", type=int, default=400)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--support-radius", type=float, default=0.3)
    parser.add_argument("--noise", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--save-graph",
        action="store_true",
    )

    args = parser.parse_args()

    run_random_structured_cdi(
        n_iter=args.n_iter,
        size=args.size,
        n_layers=args.layers,
        support_radius=args.support_radius,
        noise=args.noise,
        device=args.device,
        seed=args.seed,
        save_graph=args.save_graph,
    )
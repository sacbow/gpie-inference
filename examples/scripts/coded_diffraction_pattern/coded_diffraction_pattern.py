import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse, replicate
from gpie.core.linalg_utils import random_phase_mask

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from io_utils import load_sample_image


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------

@model
def coded_diffraction_pattern(shape, n_measurements, phase_masks, noise):
    """
    Coded diffraction pattern model using ForkPropagator via replicate().

    phase_masks: ndarray of shape (B, H, W), always on CPU at model-build time.
    """
    H, W = shape
    B = n_measurements

    x = ~GaussianPrior(
        event_shape=shape,
        label="object",
        dtype=np.complex64,
    )

    x_batch = replicate(x, batch_size=B)
    y = fft2(phase_masks * x_batch)

    AmplitudeMeasurement(var=noise) << y


# ------------------------------------------------------------
# Graph construction (CPU only)
# ------------------------------------------------------------

def build_cdp_graph(H=256, W=256, noise=1e-4, n_measurements=4):
    rng = np.random.default_rng(seed=42)
    shape = (H, W)

    phase_masks = random_phase_mask(
        (n_measurements, *shape),
        rng=rng,
        dtype=np.complex64,
    )

    g = coded_diffraction_pattern(
        shape=shape,
        n_measurements=n_measurements,
        phase_masks=phase_masks,
        noise=noise,
    )



    amp = load_sample_image("camera", shape=shape)
    phase = load_sample_image("moon", shape=shape)
    complex_img = amp * np.exp(1j * 2 * np.pi * phase)

    g.set_sample("object",complex_img.astype(np.complex64))
    g.generate_observations(rng=np.random.default_rng(seed=999)) #rng for random noise

    return g


# ------------------------------------------------------------
# Run inference
# ------------------------------------------------------------

def run_cdp(
    n_iter=100,
    size=256,
    n_measurements=4,
    device="cpu",
    save_graph=False,
):

    # --------------------------------------------------------
    # Build graph (still CPU-side objects)
    # --------------------------------------------------------
    g = build_cdp_graph(
        H=size,
        W=size,
        n_measurements=n_measurements,
    )
    
    #residual error logging
    pmse_history = []

    # --------------------------------------------------------
    # Monitor callback
    # --------------------------------------------------------
    def monitor(graph, t):
        if t % 10 == 0 or t == n_iter - 1:
            est = graph["object"]["mean"]
            gt = graph["object"]["sample"]
            err = pmse(est, gt)
            pmse_history.append(err)
            print(f"[t={t:4d}] PMSE = {err:.5e}")

    # --------------------------------------------------------
    # Run EP inference session
    # --------------------------------------------------------
    g.set_init_rng(rng=np.random.default_rng(seed=111)) #rng for message initialization
    g.run(
        n_iter=n_iter,
        device=device,
        callback=monitor,
    )

    # --------------------------------------------------------
    # Post-processing (CPU)
    # --------------------------------------------------------
    est = g["object"]["mean"][0]
    gt = g["object"]["sample"][0]

    amp = np.abs(est)
    phase = np.angle(est)

    plt.imsave(f"{RESULTS_DIR}/reconstructed_amp.png", amp, cmap="gray")
    plt.imsave(f"{RESULTS_DIR}/reconstructed_phase.png", phase, cmap="twilight")

    true_amp = np.abs(gt)
    true_phase = np.angle(gt) * (true_amp > 1e-5)

    plt.imsave(f"{RESULTS_DIR}/true_amp.png", true_amp, cmap="gray")
    plt.imsave(f"{RESULTS_DIR}/true_phase.png", true_phase, cmap="twilight")

    plt.figure()
    plt.plot(
        np.arange(0, len(pmse_history) * 10, 10),
        pmse_history,
        marker="o",
    )
    plt.xlabel("Iteration")
    plt.ylabel("PMSE")
    plt.yscale("log")
    plt.title("Convergence Curve")
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/convergence.png")
    plt.close()

    if save_graph:
        print(f"Saving factor graph visualization to {RESULTS_DIR}/graph.html ...")
        g.visualize(output_path=os.path.join(RESULTS_DIR, "graph.html"))


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coded Diffraction Pattern demo with gPIE"
    )
    parser.add_argument("--n-iter", type=int, default=400)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--measurements", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Inference device",
    )
    parser.add_argument(
        "--save-graph",
        action="store_true",
        help="Save factor graph visualization",
    )

    args = parser.parse_args()

    run_cdp(
        n_iter=args.n_iter,
        size=args.size,
        n_measurements=args.measurements,
        device=args.device,
        save_graph=args.save_graph,
    )
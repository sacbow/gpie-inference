import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from gpie import (
    model,
    GaussianPrior,
    fft2,
    AmplitudeMeasurement,
    pmse,
    replicate,
)
from gpie.core.linalg_utils import random_phase_mask

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results_sequential_vs_parallel")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------

@model
def coded_diffraction_pattern(shape, n_measurements, phase_masks, noise):
    x = ~GaussianPrior(
        event_shape=shape,
        label="object",
        dtype=np.complex64,
    )

    x_batch = replicate(x, batch_size=n_measurements)
    y = fft2(phase_masks * x_batch)

    AmplitudeMeasurement(var=noise) << y


# ------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------

def build_graph(
    shape,
    n_measurements,
    noise,
    rng_model,
    rng_data,
):
    phase_masks = random_phase_mask(
        (n_measurements, *shape),
        rng=rng_model,
        dtype=np.complex64,
    )

    g = coded_diffraction_pattern(
        shape=shape,
        n_measurements=n_measurements,
        phase_masks=phase_masks,
        noise=noise,
    )

    # ground truth object
    amp = np.ones(shape, dtype=np.float32)
    phase = rng_model.uniform(0.0, 2.0 * np.pi, size=shape)
    x_true = amp * np.exp(1j * phase)

    g.set_sample("object", x_true.astype(np.complex64))
    g.generate_observations(rng=rng_data)

    return g


# ------------------------------------------------------------
# Single run
# ------------------------------------------------------------

def run_single(
    *,
    schedule,
    block_size,
    n_iter,
    shape,
    n_measurements,
    noise,
    seed,
    device,
):
    rng_model = np.random.default_rng(seed)
    rng_data = np.random.default_rng(seed + 1)
    rng_init = np.random.default_rng(seed + 2)

    g = build_graph(
        shape=shape,
        n_measurements=n_measurements,
        noise=noise,
        rng_model=rng_model,
        rng_data=rng_data,
    )

    history = []

    def monitor(graph, t):
        est = graph["object"]["mean"]
        gt = graph["object"]["sample"]
        history.append(pmse(est, gt))

    g.set_init_rng(rng_init)

    g.run(
        n_iter=n_iter,
        schedule=schedule,
        block_size=block_size,
        device=device,
        callback=monitor,
    )

    return np.asarray(history)


# ------------------------------------------------------------
# Multi-trial experiment
# ------------------------------------------------------------

import time

import time

def run_trials(
    *,
    n_trials,
    schedule,
    block_size,
    base_seed,
    **kwargs,
    ):
    all_histories = []

    for k in range(n_trials):
        seed = base_seed + 1000 * k

        print(
            f"[Trial {k+1:02d}/{n_trials}] "
            f"schedule={schedule}, block_size={block_size}, seed={seed}"
        )

        t0 = time.time()

        hist = run_single(
            schedule=schedule,
            block_size=block_size,
            seed=seed,        
            **kwargs,
        )

        elapsed = time.time() - t0

        print(
            f"  -> done in {elapsed:.2f} s "
            f"(final PMSE = {hist[-1]:.3e})"
        )

        all_histories.append(hist)

    return np.stack(all_histories, axis=0)


# ------------------------------------------------------------
# Statistics helper
# ------------------------------------------------------------

def summarize(histories):
    median = np.median(histories, axis=0)
    q1 = np.percentile(histories, 25, axis=0)
    q3 = np.percentile(histories, 75, axis=0)
    return median, q1, q3


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(args):
    shape = (args.size, args.size)

    common_kwargs = dict(
        n_iter=args.n_iter,
        shape=shape,
        n_measurements=args.measurements,
        noise=args.noise,
        device=args.device,
    )

    hist_parallel = run_trials(
        n_trials=args.trials,
        schedule="parallel",
        block_size=None,
        base_seed = args.seed,
        **common_kwargs,
    )

    hist_sequential = run_trials(
        n_trials=args.trials,
        schedule="sequential",
        block_size=args.block_size,
        base_seed = args.seed,
        **common_kwargs,
    )

    med_p, q1_p, q3_p = summarize(hist_parallel)
    med_s, q1_s, q3_s = summarize(hist_sequential)

    # --------------------------------------------------------
    # Save raw data
    # --------------------------------------------------------

    np.save(os.path.join(RESULTS_DIR, "pmse_parallel_trials.npy"), hist_parallel)
    np.save(os.path.join(RESULTS_DIR, "pmse_sequential_trials.npy"), hist_sequential)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    it = np.arange(args.n_iter)

    plt.figure(figsize=(6, 4))

    plt.plot(it, med_p, label="parallel", linewidth=2)
    plt.fill_between(it, q1_p, q3_p, alpha=0.2)

    plt.plot(it, med_s, label=f"sequential", linewidth=2)
    plt.fill_between(it, q1_s, q3_s, alpha=0.2)

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("PMSE")
    plt.title(f"Sequential vs Parallel EP (median ± IQR, trials={args.trials})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "convergence_median_iqr.png")
    plt.savefig(out_path)
    plt.close()

    print("Benchmark finished.")
    print(f"Results saved to: {RESULTS_DIR}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential vs parallel EP convergence (median + IQR)"
    )
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--measurements", type=int, default=3)
    parser.add_argument("--noise", type=float, default=1e-4)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    args = parser.parse_args()
    main(args)
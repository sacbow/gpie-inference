import argparse
import os
import numpy as np

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse, replicate
from gpie.core.linalg_utils import random_phase_mask
from gpie.core.rng_utils import get_rng

from benchmark_utils import run_with_timer, profile_with_cprofile


# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------

@model
def coded_diffraction_pattern(
    *,
    shape: tuple[int, int],
    n_measurements: int,
    phase_masks: np.ndarray,
    noise: float,
    damping: float,
):
    """
    Fork-based coded diffraction pattern model (batched).
    phase_masks: ndarray with shape (B, H, W) on CPU at model-build time.
    """
    x = ~GaussianPrior(event_shape=shape, label="object", dtype=np.complex64)
    x_batch = replicate(x, batch_size=n_measurements)
    y = fft2(phase_masks * x_batch)
    AmplitudeMeasurement(var=noise, damping=damping) << y


# ------------------------------------------------------------
# Graph construction (CPU-side)
# ------------------------------------------------------------

def build_cdp_graph(
    *,
    size: int,
    n_measurements: int,
    noise: float,
    damping: float,
    seed_model: int,
    seed_data: int,
):
    """
    Build a CDP graph on CPU (NumPy). Observations are generated on CPU.
    The graph is later executed by Graph.run(), which manages device/FFT backends.
    """
    rng_model = get_rng(seed=seed_model)
    rng_data = get_rng(seed=seed_data)

    shape = (size, size)

    # Batched random phase masks (CPU)
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
        damping=damping,
    )

    # Ground truth object (unit amplitude + random phase)
    phase = rng_model.uniform(0.0, 2.0 * np.pi, size=shape)
    x_true = np.exp(1j * phase).astype(np.complex64)

    g.set_sample("object", x_true)
    g.generate_observations(rng=rng_data)

    return g


# ------------------------------------------------------------
# Benchmark runner
# ------------------------------------------------------------

def run_cdp_benchmark(
    *,
    n_iter: int,
    size: int,
    n_measurements: int,
    noise: float,
    damping: float,
    device: str,
    schedule: str,
    block_size: int | None,
    fft_engine: str | None,
    fft_threads: int,
    fft_planner_effort: str,
    seed: int,
    verbose: bool,
):
    """
    Build a CDP graph (CPU), then run EP inference with the requested execution settings.
    """

    # Separate seeds for reproducibility
    seed_model = seed
    seed_data = seed + 1
    seed_init = seed + 2

    g = build_cdp_graph(
        size=size,
        n_measurements=n_measurements,
        noise=noise,
        damping=damping,
        seed_model=seed_model,
        seed_data=seed_data,
    )

    # Cache ground truth (CPU-side)
    true_x = g["object"]["sample"]

    # RNG for message initialization
    g.set_init_rng(get_rng(seed=seed_init))

    def monitor(graph, t: int):
        if not verbose:
            return
        if t % 10 == 0 or t == n_iter - 1:
            est = graph["object"]["mean"]
            err = pmse(est, true_x)
            print(f"[t={t:4d}] PMSE = {err:.5e}")

    # FFT kwargs (only used when fft_engine="fftw"; safe to pass generally)
    fft_kwargs = {"threads": fft_threads, "planner_effort": fft_planner_effort}

    g.run(
        n_iter=n_iter,
        device=device,
        schedule=schedule,
        block_size=block_size,
        fft_engine=fft_engine,
        fft_kwargs=fft_kwargs,
        callback=monitor,
        verbose=False,  # benchmark should avoid tqdm overhead by default
    )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Profiling benchmark: coded diffraction pattern (gPIE 0.3.1+ session run)"
    )

    # Inference controls
    p.add_argument("--n-iter", type=int, default=100, help="Number of EP iterations")
    p.add_argument("--size", type=int, default=512, help="Image size (H=W)")
    p.add_argument("--measurements", type=int, default=4, help="Number of CDP measurements (batch size)")
    p.add_argument("--noise", type=float, default=1e-4, help="Amplitude noise variance")
    p.add_argument("--damping", type=float, default=0.3, help="Manual damping for AmplitudeMeasurement")
    p.add_argument("--seed", type=int, default=99, help="Base RNG seed")

    # Execution session controls
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Execution device for Graph.run()")
    p.add_argument(
        "--schedule",
        choices=["parallel", "sequential"],
        default="parallel",
        help="EP schedule for Graph.run()",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="Block size for sequential schedule (ignored for parallel)",
    )

    # FFT engine (now owned by Graph.run)
    p.add_argument(
        "--fft-engine",
        choices=["numpy", "cupy", "fftw"],
        default=None,
        help="FFT backend used during Graph.run() inference session (default: device-dependent)",
    )
    p.add_argument("--fft-threads", type=int, default=1, help="FFTW threads (only meaningful for --fft-engine fftw)")
    p.add_argument(
        "--fft-planner-effort",
        type=str,
        default="FFTW_ESTIMATE",
        help="FFTW planner effort (only meaningful for --fft-engine fftw)",
    )

    # Profiling / logging
    p.add_argument("--profile", action="store_true", help="Enable cProfile profiling (CPU-side only)")
    p.add_argument("--verbose", action="store_true", help="Print PMSE every 10 iterations")

    return p.parse_args()


def main():
    args = _parse_args()

    # Normalize block_size behavior
    if args.schedule == "parallel":
        block_size = None
    else:
        block_size = args.block_size

    # Sanity checks for fft_engine
    if args.device == "cuda" and args.fft_engine == "fftw":
        raise ValueError("fft_engine='fftw' is not compatible with device='cuda'")

    run_kwargs = dict(
        n_iter=args.n_iter,
        size=args.size,
        n_measurements=args.measurements,
        noise=args.noise,
        damping=args.damping,
        schedule=args.schedule,
        block_size=block_size,
        fft_engine=args.fft_engine,
        fft_threads=args.fft_threads,
        fft_planner_effort=args.fft_planner_effort,
        seed=args.seed,
        verbose=args.verbose,
    )

    if args.profile:
        profile_with_cprofile(
            lambda: run_cdp_benchmark(device=args.device, **run_kwargs)
        )
        return

    _, elapsed = run_with_timer(
        lambda: run_cdp_benchmark(device=args.device, **run_kwargs),
        device=args.device,
        sync_gpu=True,
    )

    fft_engine = args.fft_engine
    if fft_engine is None:
        fft_engine = "cupy" if args.device == "cuda" else "numpy"

    tag = f"device={args.device}, schedule={args.schedule}, fft={fft_engine}"
    if args.schedule == "sequential":
        tag += f", block_size={args.block_size}"
    if fft_engine == "fftw":
        tag += f", threads={args.fft_threads}, effort={args.fft_planner_effort}"

    print(f"[{tag}] Total time: {elapsed:.3f} s")


if __name__ == "__main__":
    main()
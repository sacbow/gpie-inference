import argparse
import numpy as np

from gpie import (
    model,
    GaussianPrior,
    fft2,
    AmplitudeMeasurement,
    pmse,
    replicate,
)
from gpie.core.linalg_utils import random_phase_mask

from benchmark_utils import IterationTimer, IterationProfiler, resolve_fft_engine


# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------

@model
def coded_diffraction_pattern(shape, n_measurements, phase_masks, noise):
    """
    Coded diffraction pattern (CDP), batched measurements via replicate().

    phase_masks: ndarray of shape (B, H, W) on CPU at model-build time.
    """
    x = ~GaussianPrior(
        event_shape=shape,
        label="object",
        dtype=np.complex64,
    )

    x_batch = replicate(x, batch_size=n_measurements)
    y = fft2(phase_masks * x_batch)

    AmplitudeMeasurement(var=noise) << y


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def build_cdp_graph(
    *,
    size: int,
    n_measurements: int,
    noise: float,
    seed: int,
):
    rng_model = np.random.default_rng(seed)
    rng_data = np.random.default_rng(seed + 1)

    shape = (size, size)

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

    # Ground truth object
    amp = np.ones(shape, dtype=np.float32)
    phase = rng_model.uniform(0.0, 2.0 * np.pi, size=shape).astype(np.float32)
    x_true = amp * np.exp(1j * phase)

    g.set_sample("object", x_true.astype(np.complex64))
    g.generate_observations(rng=rng_data)

    return g


def run_cdp_benchmark(
    *,
    size: int,
    n_iter: int,
    n_measurements: int,
    noise: float,
    seed: int,
    device: str,
    fft_engine: str | None,
    schedule: str,
    block_size: int | None,
    verbose: bool,
    eval_pmse: bool,
    profile: bool,
    profile_iter: int,
    profile_sort: str,
    profile_limit: int,
):
    g = build_cdp_graph(
        size=size,
        n_measurements=n_measurements,
        noise=noise,
        seed=seed,
    )

    # IMPORTANT: set init rng right before run()
    g.set_init_rng(np.random.default_rng(99))

    timer = IterationTimer(sync_gpu=(device == "cuda"))
    hooks = [timer]

    if profile:
        hooks.append(
            IterationProfiler(
                target_iter=profile_iter,
                sort=profile_sort,
                limit=profile_limit,
                sync_gpu=(device == "cuda"),
            )
        )

    def combined_hook(graph, t: int, phase: str):
        for h in hooks:
            h(graph, t, phase)

    if eval_pmse:
        pmse_hist = []
        gt = g["object"]["sample"]

        def monitor(graph, t: int):
            est = graph["object"]["mean"]
            pmse_hist.append(float(pmse(est, gt)))

    else:
        pmse_hist = None
        monitor = None

    g.run(
        n_iter=n_iter,
        schedule=schedule,
        block_size=block_size,
        device=device,
        fft_engine=fft_engine,
        iteration_hook=combined_hook,
        callback=monitor,
        verbose=verbose,
    )

    times = np.asarray(timer.times)

    return {
        "mean": float(times.mean()),
        "median": float(np.median(times)),
        "min": float(times.min()),
        "max": float(times.max()),
        "all": times,
        "pmse_hist": pmse_hist,
    }


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CDP iteration-level benchmark using graph.run hooks"
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--noise", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--measurements", type=int, default=4)

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--fft-engine",
        choices=["numpy", "cupy", "fftw"],
        default=None,
        help="If omitted, inferred from device (cpu->numpy, cuda->cupy).",
    )

    parser.add_argument(
        "--schedule",
        choices=["parallel", "sequential"],
        default="parallel",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        help="Used only when schedule=sequential.",
    )

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--eval-pmse",
        action="store_true",
        help="Compute PMSE each iteration (adds overhead; not included in iteration_hook timing).",
    )

    # Profiling options (single iteration)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile for a single iteration (controlled by --profile-iter).",
    )
    parser.add_argument(
        "--profile-iter",
        type=int,
        default=0,
        help="Iteration index to profile (default: 0).",
    )
    parser.add_argument(
        "--profile-sort",
        type=str,
        default="cumtime",
        help="pstats sort key (e.g., cumtime, tottime).",
    )
    parser.add_argument(
        "--profile-limit",
        type=int,
        default=30,
        help="Number of lines to show in profiler output.",
    )

    args = parser.parse_args()

    effective_fft_engine = resolve_fft_engine(args.device, args.fft_engine)

    result = run_cdp_benchmark(
        size=args.size,
        n_iter=args.n_iter,
        n_measurements=args.measurements,
        noise=args.noise,
        seed=args.seed,
        device=args.device,
        fft_engine=args.fft_engine,
        schedule=args.schedule,
        block_size=args.block_size,
        verbose=args.verbose,
        eval_pmse=args.eval_pmse,
        profile=args.profile,
        profile_iter=args.profile_iter,
        profile_sort=args.profile_sort,
        profile_limit=args.profile_limit,
    )

    print("=== CDP iteration benchmark ===")
    print(f"device      : {args.device}")
    print(f"fft_engine  : {effective_fft_engine}")
    print(f"schedule    : {args.schedule}")
    print(f"size        : {args.size}")
    print(f"iterations  : {args.n_iter}")
    print("")

    print(f"median iter : {result['median'] * 1e3:.3f} ms")
    print(f"mean iter   : {result['mean'] * 1e3:.3f} ms")
    print(f"min / max   : {result['min'] * 1e3:.3f} / {result['max'] * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
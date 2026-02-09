import argparse
import numpy as np

from gpie import model, SupportPrior, fft2, AmplitudeMeasurement, mse
from gpie.core.linalg_utils import circular_aperture, masked_random_array

from benchmark_utils import IterationTimer, IterationProfiler, resolve_fft_engine


# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------

@model
def holography(shape, support, noise, ref_wave):
    """
    Inline holography:
        y = | FFT(ref_wave + obj) | + noise
    """
    obj = ~SupportPrior(support=support, label="obj", dtype=np.complex64)
    AmplitudeMeasurement(var=noise) << fft2(ref_wave + obj)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def build_holography_graph(
    *,
    size: int,
    noise: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    shape = (size, size)

    # Reference wave (known)
    support_ref = circular_aperture(shape, radius=0.2, center=(-0.2, -0.2))
    ref_wave = masked_random_array(support_ref, dtype=np.complex64, rng=rng)

    # Object support (unknown)
    support_obj = circular_aperture(shape, radius=0.2, center=(0.2, 0.2))

    g = holography(
        shape=shape,
        support=support_obj,
        noise=noise,
        ref_wave=ref_wave,
    )

    # Ground truth object + observations
    obj_true = masked_random_array(
        support_obj,
        dtype=np.complex64,
        rng=np.random.default_rng(seed + 1),
    )
    g.set_sample("obj", obj_true.astype(np.complex64))
    g.generate_observations(rng=np.random.default_rng(seed + 2))

    return g


def run_holography_benchmark(
    *,
    size: int,
    n_iter: int,
    device: str,
    fft_engine: str | None,
    seed: int,
    noise: float,
    schedule: str,
    block_size: int | None,
    verbose: bool,
    eval_mse: bool,
    profile: bool,
    profile_iter: int,
    profile_sort: str,
    profile_limit: int,
):
    g = build_holography_graph(
        size=size,
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

    if eval_mse:
        mse_hist = []
        gt = g["obj"]["sample"]

        def monitor(graph, t: int):
            est = graph["obj"]["mean"]
            mse_hist.append(float(mse(est, gt)))

    else:
        mse_hist = None
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
        "mse_hist": mse_hist,
    }


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Holography iteration-level benchmark using graph.run hooks"
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--noise", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)

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
        "--eval-mse",
        action="store_true",
        help="Compute MSE each iteration (adds overhead; not included in iteration_hook timing).",
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

    result = run_holography_benchmark(
        size=args.size,
        n_iter=args.n_iter,
        device=args.device,
        fft_engine=args.fft_engine,
        seed=args.seed,
        noise=args.noise,
        schedule=args.schedule,
        block_size=args.block_size,
        verbose=args.verbose,
        eval_mse=args.eval_mse,
        profile=args.profile,
        profile_iter=args.profile_iter,
        profile_sort=args.profile_sort,
        profile_limit=args.profile_limit,
    )

    print("=== Holography iteration benchmark ===")
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
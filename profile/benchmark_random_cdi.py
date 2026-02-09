import argparse
import numpy as np
from numpy.typing import NDArray

from gpie import model, SupportPrior, fft2, AmplitudeMeasurement, pmse
from gpie.core.linalg_utils import circular_aperture, random_phase_mask

from benchmark_utils import IterationTimer, IterationProfiler


# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------

@model
def random_cdi(
    support: NDArray[np.bool_],
    n_layers: int,
    phase_masks: list[NDArray],
    noise: float,
):
    """
    Structured Random Matrix CDI:
        x_{k+1} = FFT( phase_mask_k * x_k ), repeated n_layers times
        y = | x_{n_layers} | + noise
    """
    x = ~SupportPrior(support=support, label="sample", dtype=np.complex64)

    for k in range(n_layers):
        x = fft2(phase_masks[k] * x)

    AmplitudeMeasurement(var=noise, damping=0.3) << x


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def resolve_fft_engine(device: str, fft_engine: str | None) -> str:
    if fft_engine is not None:
        return fft_engine
    return "cupy" if device == "cuda" else "numpy"


def build_random_cdi_graph(
    *,
    size: int,
    n_layers: int,
    noise: float,
    support_radius: float,
    seed: int,
):
    rng_model = np.random.default_rng(seed)
    rng_data = np.random.default_rng(seed + 1)

    shape = (size, size)

    support = circular_aperture(shape, radius=support_radius)

    phase_masks = [
        random_phase_mask(shape, rng=rng_model, dtype=np.complex64)
        for _ in range(n_layers)
    ]

    g = random_cdi(
        support=support,
        n_layers=n_layers,
        phase_masks=phase_masks,
        noise=noise,
    )

    # ground truth
    amp = np.ones(shape, dtype=np.float32)
    phase = rng_model.uniform(0.0, 2.0 * np.pi, size=shape).astype(np.float32)
    x_true = amp * np.exp(1j * phase)

    g.set_sample("sample", x_true.astype(np.complex64))
    g.generate_observations(rng=rng_data)

    return g


def run_random_cdi_benchmark(
    *,
    size: int,
    n_layers: int,
    n_iter: int,
    noise: float,
    support_radius: float,
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
    g = build_random_cdi_graph(
        size=size,
        n_layers=n_layers,
        noise=noise,
        support_radius=support_radius,
        seed=seed,
    )

    # IMPORTANT: init RNG right before run()
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
        gt = g["sample"]["sample"]

        def monitor(graph, t: int):
            est = graph["sample"]["mean"]
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
        description="Structured Random Matrix CDI iteration benchmark (gPIE)"
    )

    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--noise", type=float, default=1e-4)
    parser.add_argument("--support-radius", type=float, default=0.3)
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
        help="If omitted, inferred from device.",
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
        help="Used only for sequential schedule.",
    )

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--eval-pmse",
        action="store_true",
        help="Compute PMSE each iteration (excluded from iteration timing).",
    )

    # profiling
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iter", type=int, default=0)
    parser.add_argument("--profile-sort", type=str, default="cumtime")
    parser.add_argument("--profile-limit", type=int, default=30)

    args = parser.parse_args()

    effective_fft_engine = resolve_fft_engine(args.device, args.fft_engine)

    result = run_random_cdi_benchmark(
        size=args.size,
        n_layers=args.layers,
        n_iter=args.n_iter,
        noise=args.noise,
        support_radius=args.support_radius,
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

    print("=== Random CDI iteration benchmark ===")
    print(f"device      : {args.device}")
    print(f"fft_engine  : {effective_fft_engine}")
    print(f"schedule    : {args.schedule}")
    print(f"size        : {args.size}")
    print(f"layers      : {args.layers}")
    print(f"iterations  : {args.n_iter}")
    print("")

    print(f"median iter : {result['median'] * 1e3:.3f} ms")
    print(f"mean iter   : {result['mean'] * 1e3:.3f} ms")
    print(f"min / max   : {result['min'] * 1e3:.3f} / {result['max'] * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
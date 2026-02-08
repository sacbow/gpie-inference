"""
Benchmarking utilities for gPIE.

Design policy:
- Numerical backend (NumPy / CuPy) is fully managed by Graph.run().
- FFT backend selection is also managed by Graph.run().
- This module NEVER inspects or mutates gPIE backend state.
- GPU synchronization is controlled explicitly via `device` argument.
"""

import time
import cProfile
import pstats
import io
from typing import Callable, Any, Literal, Tuple

import cupy as cp


# ------------------------------------------------------------
# Timing utilities
# ------------------------------------------------------------

def run_with_timer(
    func: Callable[..., Any],
    *args,
    device: Literal["cpu", "cuda"] = "cpu",
    sync_gpu: bool = True,
    **kwargs,
) -> Tuple[Any, float]:
    """
    Measure elapsed wall time for a function call.

    Parameters
    ----------
    func:
        Function to execute.
    *args, **kwargs:
        Arguments passed to `func`.
    device:
        Execution device used inside `func`.
        This must match the `device` argument passed to Graph.run().
    sync_gpu:
        If True and device == "cuda", synchronize the GPU before stopping the timer.

    Returns
    -------
    result:
        Return value of `func`.
    elapsed:
        Elapsed wall time in seconds.
    """

    start = time.perf_counter()
    result = func(*args, **kwargs)

    if sync_gpu and device == "cuda":
        # Ensure all GPU work launched inside Graph.run() has completed
        cp.cuda.Device().synchronize()

    elapsed = time.perf_counter() - start
    return result, elapsed


# ------------------------------------------------------------
# cProfile utilities
# ------------------------------------------------------------

def profile_with_cprofile(
    func: Callable[..., Any],
    *args,
    sort: str = "cumtime",
    limit: int = 30,
    **kwargs,
) -> None:
    """
    Run cProfile on a function and print profiling results.

    Notes
    -----
    - cProfile captures only Python-level CPU execution.
    - GPU kernel execution time is NOT included.
    """

    pr = cProfile.Profile()
    pr.enable()

    func(*args, **kwargs)

    pr.disable()

    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s).sort_stats(sort)
    stats.print_stats(limit)
    print(s.getvalue())
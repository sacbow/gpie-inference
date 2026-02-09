import time
from typing import Literal, Callable, List, Dict, Any


# ------------------------------------------------------------
# Simple timing hook (per-iteration wall-clock time)
# ------------------------------------------------------------

class IterationTimer:
    """
    Simple per-iteration wall-clock timer.

    Usage:
        timer = IterationTimer(sync_gpu=True)
        g.run(..., iteration_hook=timer)

        times = timer.times  # list of seconds per iteration
    """

    def __init__(self, sync_gpu: bool = True):
        self.sync_gpu = sync_gpu
        self._t0 = None
        self.times: List[float] = []

    def __call__(self, graph, t: int, phase: Literal["begin", "end"]):
        if phase == "begin":
            self._t0 = time.perf_counter()
        else:
            if self.sync_gpu:
                self._sync_if_needed()
            dt = time.perf_counter() - self._t0
            self.times.append(dt)

    @staticmethod
    def _sync_if_needed():
        try:
            import cupy as cp
            cp.cuda.Device().synchronize()
        except Exception:
            # CPU or CuPy not active
            pass


# ------------------------------------------------------------
# Lightweight profiler hook (aggregate counters)
# ------------------------------------------------------------
# benchmark_utils.py
import cProfile
import pstats
import io
import time


class IterationProfiler:
    """
    cProfile-based profiler for a single EP iteration.

    Designed to be passed as graph.run(iteration_hook=...).
    """

    def __init__(
        self,
        *,
        target_iter: int = 0,
        sort: str = "cumtime",
        limit: int = 30,
        sync_gpu: bool = False,
    ):
        self.target_iter = target_iter
        self.sort = sort
        self.limit = limit
        self.sync_gpu = sync_gpu

        self._profiler = cProfile.Profile()
        self._active = False
        self._done = False

    def __call__(self, graph, t: int, phase: str):
        if self._done:
            return

        if t == self.target_iter and phase == "begin":
            if self.sync_gpu:
                self._sync_gpu()
            self._profiler.enable()
            self._active = True

        elif t == self.target_iter and phase == "end" and self._active:
            if self.sync_gpu:
                self._sync_gpu()
            self._profiler.disable()
            self._active = False
            self._done = True

            self._print_stats()

    def _sync_gpu(self):
        try:
            import cupy as cp
            cp.cuda.Device().synchronize()
        except Exception:
            pass

    def _print_stats(self):
        s = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=s).sort_stats(self.sort)
        stats.print_stats(self.limit)
        print("\n=== Iteration cProfile result ===")
        print(s.getvalue())



def resolve_fft_engine(device: str, fft_engine: str | None) -> str:
    """
    Resolve the effective FFT backend name, consistent with graph.run logic.
    """
    if fft_engine is not None:
        return fft_engine
    return "cupy" if device == "cuda" else "numpy"
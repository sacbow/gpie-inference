"""
Metrics and evaluation utilities.

This module provides:
- GPU-capable core metrics for benchmarking (MSE, NMSE, PMSE, etc.)
- Optional L2 normalization for scale-invariant problems
- Phase-aligned variants for complex-valued inverse problems
- Lightweight utility functions (phase_align, support_error)

Design notes:
- Core metrics are backend-aware and GPU-friendly.
- CPU ↔ GPU transfers are minimized; only scalar results are returned to Python.
- Utility functions do NOT guarantee GPU-complete execution.
"""

from __future__ import annotations

from .backend import np, ensure_array_on_current_backend


# ============================================================
# Internal helpers (not part of public API)
# ============================================================

def _l2_normalize(x):
    """
    L2-normalize an array over all elements.
    """
    xp = np()
    norm = xp.linalg.norm(x)
    eps = xp.array(1e-12, dtype=norm.dtype)
    return x / xp.maximum(norm, eps)


def _phase_align_impl(x_est, x_true):
    """
    Internal phase alignment assuming both arrays are already
    on the same backend.
    """
    xp = np()
    inner = xp.vdot(x_true, x_est)
    phase = xp.angle(inner)
    return x_est * xp.exp(-1j * phase)


def _to_python_float(x) -> float:
    """
    Convert NumPy/CuPy scalar to Python float.
    """
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _mse_impl(
    x_est,
    x_true,
    *,
    normalize: bool = False,
    phase_align: bool = False,
) -> float:
    """
    Internal unified implementation for MSE-like metrics.
    """
    xp = np()

    # Backend unification (GPU-preferred)
    x_est = ensure_array_on_current_backend(x_est)
    x_true = ensure_array_on_current_backend(x_true)

    # Optional preprocessing
    if normalize:
        x_est = _l2_normalize(x_est)
        x_true = _l2_normalize(x_true)

    if phase_align:
        x_est = _phase_align_impl(x_est, x_true)

    err = xp.mean(xp.abs(x_est - x_true) ** 2)
    return _to_python_float(err)


def _nmse_impl(
    x_est,
    x_true,
    *,
    normalize: bool = False,
    phase_align: bool = False,
) -> float:
    """
    Internal unified implementation for NMSE-like metrics.
    """
    xp = np()

    x_est = ensure_array_on_current_backend(x_est)
    x_true = ensure_array_on_current_backend(x_true)

    if normalize:
        x_est = _l2_normalize(x_est)
        x_true = _l2_normalize(x_true)

    if phase_align:
        x_est = _phase_align_impl(x_est, x_true)

    num = xp.linalg.norm(x_est - x_true) ** 2
    den = xp.linalg.norm(x_true) ** 2
    eps = xp.array(1e-12, dtype=den.dtype)

    return _to_python_float(num / xp.maximum(den, eps))


# ============================================================
# Public core metrics (GPU-capable)
# ============================================================

def mse(x_est, x_true, *, normalize: bool = False) -> float:
    """
    Mean Squared Error (MSE).

    Parameters
    ----------
    normalize : bool
        If True, L2-normalize both inputs before computing error.
        Intended for scale-invariant problems (e.g. blind ptychography).
    """
    return _mse_impl(
        x_est,
        x_true,
        normalize=normalize,
        phase_align=False,
    )


def nmse(x_est, x_true, *, normalize: bool = False) -> float:
    """
    Normalized Mean Squared Error (NMSE).

    NMSE = ||x_est - x_true||^2 / ||x_true||^2
    """
    return _nmse_impl(
        x_est,
        x_true,
        normalize=normalize,
        phase_align=False,
    )


def pmse(x_est, x_true, *, normalize: bool = False) -> float:
    """
    Phase-aligned Mean Squared Error (PMSE).

    Removes global phase ambiguity before computing MSE.
    """
    return _mse_impl(
        x_est,
        x_true,
        normalize=normalize,
        phase_align=True,
    )


def pnmse(x_est, x_true, *, normalize: bool = False) -> float:
    """
    Phase-aligned Normalized Mean Squared Error (PNMSE).
    """
    return _nmse_impl(
        x_est,
        x_true,
        normalize=normalize,
        phase_align=True,
    )


def psnr(
    x_est,
    x_true,
    max_val: float = 1.0,
    *,
    normalize: bool = False,
) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR).

    Assumes signal range [0, max_val].
    """
    xp = np()
    mse_val = mse(x_est, x_true, normalize=normalize)
    if mse_val == 0.0:
        return float("inf")
    return 10.0 * _to_python_float(
        xp.log10((max_val ** 2) / mse_val)
    )


def ppsnr(
    x_est,
    x_true,
    max_val: float = 1.0,
    *,
    normalize: bool = False,
) -> float:
    """
    Phase-aligned PSNR.
    """
    xp = np()
    mse_val = pmse(x_est, x_true, normalize=normalize)
    if mse_val == 0.0:
        return float("inf")
    return 10.0 * _to_python_float(
        xp.log10((max_val ** 2) / mse_val)
    )


# ============================================================
# Utility functions (NOT guaranteed GPU-complete)
# ============================================================

def phase_align(x_est, x_true):
    """
    Align global phase of x_est to match x_true.

    NOTE:
        This is a low-level utility function.
        It does NOT perform backend unification and does NOT guarantee
        GPU-complete execution. For benchmarking, prefer pmse().
    """
    xp = np()
    inner_product = xp.vdot(x_true, x_est)
    phase = xp.angle(inner_product)
    return x_est * xp.exp(-1j * phase)


def support_error(x_est, x_true, threshold: float = 1e-3) -> float:
    """
    Support mismatch between estimated and true sparse signals.

    Useful for sparse recovery problems.

    NOTE:
        This metric is intended for analysis and debugging.
        It does NOT guarantee GPU-complete execution.
    """
    xp = np()
    est_support = xp.abs(x_est) > threshold
    true_support = xp.abs(x_true) > threshold
    mismatch = xp.logical_xor(est_support, true_support)
    return _to_python_float(xp.sum(mismatch) / mismatch.size)
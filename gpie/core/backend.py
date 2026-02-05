"""
Backend abstraction for numerical computation.

This module defines a swappable backend interface for NumPy-like libraries
such as NumPy, CuPy, JAX, or PyTorch. All internal math in `core/` should use
`from .backend import np` to remain backend-agnostic.
"""

import numpy as _np
from typing import Any, Optional

_backend = _np  # Default backend: NumPy


def set_backend(lib):
    """
    Set the global backend to a NumPy-compatible library (e.g., numpy, cupy, jax.numpy).

    Args:
        lib: Module object such as numpy, jax.numpy, or cupy.
    """
    from .fft import DefaultFFTBackend, CuPyFFTBackend
    global _backend, _current_fft_backend
    _backend = lib

    # Assign FFT backend depending on numerical backend
    if lib.__name__ == "cupy":
        _current_fft_backend = CuPyFFTBackend()
    else:
        _current_fft_backend = DefaultFFTBackend()


def get_backend():
    """
    Get the current backend module.

    Returns:
        The active backend module (default: numpy).
    """
    return _backend


def move_array_to_current_backend(array: Any, dtype: Optional[_np.dtype] = None) -> Any:
    """
    Ensure array is on the current backend (NumPy or CuPy), with optional dtype conversion.

    Args:
        array (Any): Input array from potentially another backend.
        dtype (np().dtype, optional): If given, cast to this dtype after transfer.

    Returns:
        backend ndarray: Array on current backend, dtype adjusted if needed.
    """
    try:
        import cupy as cp
        # If moving from CuPy to NumPy
        if isinstance(array, cp.ndarray) and np().__name__ == "numpy":
            array = array.get()
    except ImportError:
        pass

    arr = np().asarray(array)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def ensure_array_on_current_backend(array: Any):
    """
    Ensure array is on the current backend.

    Semantics:
        - If current backend is NumPy: return array as-is.
        - If current backend is CuPy:
            * NumPy array -> moved to GPU
            * CuPy array  -> kept as-is
        - GPU -> CPU transfer is NEVER performed here.
    """
    xp = np()

    # Lazy CuPy import
    try:
        import cupy as cp
        is_cupy = isinstance(array, cp.ndarray)
    except ImportError:
        is_cupy = False

    # Already on GPU → keep
    if is_cupy:
        return array

    # CPU backend → keep CPU array
    if xp.__name__ == "numpy":
        return array

    # GPU backend → move CPU array to GPU
    return xp.asarray(array)


# Aliases for convenience
np = get_backend  # use: `np().sum(...)`

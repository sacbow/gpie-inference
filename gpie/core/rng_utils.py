"""
Random number generation utilities.

Design policy:
- RNG is always NumPy-based and CPU-resident.
- RNG behavior is independent of numerical backend (NumPy / CuPy).
- Backend-specific array placement is handled elsewhere.
"""

import numpy as np
from typing import Optional


# ------------------------------------------------------------
# RNG factory
# ------------------------------------------------------------
def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Return a NumPy random number generator.

    Args:
        seed (int, optional):
            Seed for reproducible random number generation.

    Returns:
        numpy.random.Generator:
            NumPy RNG instance.
    """
    return np.random.default_rng(seed)


# ------------------------------------------------------------
# Distribution helpers (NumPy RNG only)
# ------------------------------------------------------------
def normal(
    rng: np.random.Generator,
    size,
    mean: float = 0.0,
    std: float = 1.0,
):
    """
    Draw samples from a normal (Gaussian) distribution.

    Args:
        rng:
            NumPy random number generator.
        size:
            Output shape.
        mean:
            Mean of the distribution.
        std:
            Standard deviation of the distribution.

    Returns:
        numpy.ndarray:
            CPU array of random samples.
    """
    return rng.normal(loc=mean, scale=std, size=size)


def uniform(
    rng: np.random.Generator,
    low: float = 0.0,
    high: float = 1.0,
    size=None,
):
    """
    Draw samples from a uniform distribution.

    Args:
        rng:
            NumPy random number generator.
        low:
            Lower bound.
        high:
            Upper bound.
        size:
            Output shape.

    Returns:
        numpy.ndarray:
            CPU array of random samples.
    """
    return rng.uniform(low=low, high=high, size=size)


def choice(
    rng: np.random.Generator,
    a,
    size,
    replace: bool = True,
):
    """
    Draw samples from a discrete set.

    Args:
        rng:
            NumPy random number generator.
        a:
            1-D array-like or int.
        size:
            Output shape.
        replace:
            Whether sampling is with replacement.

    Returns:
        numpy.ndarray:
            CPU array of sampled values.
    """
    return rng.choice(a, size=size, replace=replace)


def shuffle(
    rng: np.random.Generator,
    x,
):
    """
    Shuffle an array in-place along the first axis.

    Args:
        rng:
            NumPy random number generator.
        x:
            Array to be shuffled (modified in-place).

    Returns:
        numpy.ndarray:
            Shuffled array (same object as input).
    """
    rng.shuffle(x)
    return x

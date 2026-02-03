import numpy as np
import pytest

from gpie.core.rng_utils import (
    get_rng,
    normal,
    uniform,
    choice,
    shuffle,
)


# ------------------------------------------------------------
# get_rng
# ------------------------------------------------------------
def test_get_rng_returns_numpy_generator():
    """get_rng should always return a NumPy Generator."""
    rng = get_rng(seed=123)
    assert isinstance(rng, np.random.Generator)


def test_get_rng_reproducibility():
    """get_rng with the same seed should be reproducible."""
    rng1 = get_rng(seed=42)
    rng2 = get_rng(seed=42)

    x1 = rng1.normal(size=10)
    x2 = rng2.normal(size=10)

    assert np.allclose(x1, x2)


def test_get_rng_without_seed():
    """get_rng without seed should still return a valid RNG."""
    rng = get_rng()
    x = rng.normal(size=5)
    assert x.shape == (5,)


# ------------------------------------------------------------
# normal
# ------------------------------------------------------------
def test_normal_real_output():
    """normal should generate real-valued samples."""
    rng = get_rng(seed=0)
    x = normal(rng, size=10)
    assert isinstance(x, np.ndarray)
    assert x.shape == (10,)
    assert np.isrealobj(x)


def test_normal_mean_std():
    """normal should roughly respect mean and std."""
    rng = get_rng(seed=1)
    x = normal(rng, size=100_000, mean=2.0, std=3.0)
    assert abs(x.mean() - 2.0) < 0.1
    assert abs(x.std() - 3.0) < 0.1


# ------------------------------------------------------------
# uniform
# ------------------------------------------------------------
def test_uniform_range_and_shape():
    """uniform should generate values within [low, high)."""
    rng = get_rng(seed=2)
    x = uniform(rng, low=-1.0, high=1.0, size=(20,))

    assert x.shape == (20,)
    assert np.all(x >= -1.0)
    assert np.all(x < 1.0)


# ------------------------------------------------------------
# choice
# ------------------------------------------------------------
def test_choice_basic():
    """choice should sample elements from the given set."""
    rng = get_rng(seed=3)
    a = np.array([10, 20, 30, 40])
    x = choice(rng, a=a, size=5, replace=True)

    assert x.shape == (5,)
    for v in x:
        assert v in a


def test_choice_without_replacement():
    """choice with replace=False should not repeat elements."""
    rng = get_rng(seed=4)
    a = np.arange(10)
    x = choice(rng, a=a, size=5, replace=False)

    assert len(x) == 5
    assert len(set(x.tolist())) == 5


# ------------------------------------------------------------
# shuffle
# ------------------------------------------------------------
def test_shuffle_preserves_elements():
    """shuffle should permute elements without loss."""
    rng = get_rng(seed=5)
    x = np.arange(10)
    y = shuffle(rng, x.copy())

    assert y.shape == x.shape
    assert set(y.tolist()) == set(x.tolist())


def test_shuffle_inplace_semantics():
    """shuffle should operate in-place and return the same object."""
    rng = get_rng(seed=6)
    x = np.arange(5)
    y = shuffle(rng, x)

    assert y is x


# ------------------------------------------------------------
# Combined smoke test
# ------------------------------------------------------------
def test_rng_utils_smoke():
    """Smoke test for combined RNG utilities."""
    rng = get_rng(seed=7)

    x1 = normal(rng, size=3)
    x2 = uniform(rng, size=3)
    x3 = choice(rng, a=[0, 1, 2], size=2)
    x4 = shuffle(rng, np.arange(4))

    assert x1.shape == (3,)
    assert x2.shape == (3,)
    assert x3.shape == (2,)
    assert x4.shape == (4,)

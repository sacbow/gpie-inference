import pytest
import numpy as np
from gpie.graph.prior.gaussian_prior import GaussianPrior
from gpie.core.types import PrecisionMode
from gpie.core.uncertain_array import UncertainArray


def test_gaussian_prior_initialization_default_mean():
    gp = GaussianPrior(var=2.0, event_shape=(3, 3), batch_size=5, dtype=np.complex128)
    assert np.allclose(gp.var, 2.0)
    assert np.allclose(gp.precision, 0.5)
    assert gp.output.event_shape == (3, 3)
    assert gp.output.batch_size == 5
    assert gp.output.dtype == np.complex128
    # mean should default to zero array
    assert np.allclose(gp.mean, 0.0)


def test_gaussian_prior_initialization_with_scalar_mean():
    gp = GaussianPrior(mean=2.0, var=1.0, event_shape=(2, 2))
    assert gp.mean.shape == (1, 2, 2)   # 修正
    assert np.allclose(gp.mean, 2.0)
    assert np.allclose(gp.var, 1.0)
    assert np.allclose(gp.precision, 1.0)



def test_gaussian_prior_initialization_with_array_mean():
    mean_arr = np.ones((2, 2)) * 5.0
    gp = GaussianPrior(mean=mean_arr, var=1.0, event_shape=(2, 2))
    assert gp.mean.shape == (1, 2, 2)     
    assert np.allclose(gp.mean[0], mean_arr)  



def test_gaussian_prior_initialization_mean_shape_mismatch():
    with pytest.raises(ValueError, match="Mean shape mismatch"):
        GaussianPrior(mean=np.ones((3, 3)), var=1.0, event_shape=(2, 2))


def test_gaussian_prior_compute_message_scalar():
    gp = GaussianPrior(
        mean=1.0,
        var=1.0,
        event_shape=(2, 2),
        batch_size=3,
        precision_mode=PrecisionMode.SCALAR,
    )
    msg = gp._compute_message(UncertainArray.zeros(event_shape=(2, 2), batch_size=3))
    assert isinstance(msg, UncertainArray)
    assert msg.precision_mode == PrecisionMode.SCALAR
    assert msg.data.shape == (3, 2, 2)
    # mean must equal the prior mean
    assert np.allclose(msg.data, 1.0)
    # precision should broadcast correctly
    assert np.allclose(msg.precision(), 1.0)


def test_gaussian_prior_compute_message_array():
    gp = GaussianPrior(
        mean=0.5,
        var=2.0,
        event_shape=(2, 2),
        batch_size=4,
        precision_mode=PrecisionMode.ARRAY,
    )
    msg = gp._compute_message(UncertainArray.zeros(event_shape=(2, 2), batch_size=4))
    assert msg.precision_mode == PrecisionMode.ARRAY
    assert msg.data.shape == (4, 2, 2)
    assert np.allclose(msg.data, 0.5)
    assert np.allclose(msg.precision(), 1 / 2.0)


def test_gaussian_prior_get_sample_for_output_mean_and_var():
    mean = np.full((2, 2), 3.0)
    gp = GaussianPrior(mean=mean, var=4.0, event_shape=(2, 2), batch_size=10)
    s = gp.get_sample_for_output()
    assert s.shape == (10, 2, 2)
    assert s.dtype == gp.dtype
    # sample mean should be roughly near the prior mean
    assert np.allclose(s.mean(), 3.0, atol=1.0)


def test_gaussian_prior_repr_contains_mean_and_var():
    gp = GaussianPrior(mean=1.0, var=1.0, event_shape=(1,))
    r = repr(gp)
    assert "GaussianPrior" in r
    assert "var=1.0" in r
    # updated: repr now contains "mean_shape=" instead of "mean="
    assert "mean_shape=" in r




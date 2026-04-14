from .base import UncertainArray
from ..backend import np
from ..rng_utils import get_rng
from ..types import get_real_dtype
from typing import Any

def mul_ua(self: UncertainArray, other: UncertainArray) -> UncertainArray:
    """
    Combine two UncertainArrays under the additive precision model.

    Supports mixed real/complex Gaussian fusion by projecting complex inputs
    to real when dtype mismatch occurs.
    """

    xp = np()

    def is_real(dtype):
        return xp.issubdtype(dtype, xp.floating)

    def is_complex(dtype):
        return xp.issubdtype(dtype, xp.complexfloating)

    if self.dtype != other.dtype:
        # Only allow real/complex mismatch (for use in fft with real/complex conversion)
        if is_real(self.dtype) and is_complex(other.dtype):
            other = other.real
        elif is_complex(self.dtype) and is_real(other.dtype):
            self = self.real

        else:
            raise TypeError(
                f"Dtype mismatch in __mul__: {self.dtype} vs {other.dtype}"
            )
    self.assert_compatible(other, context="__mul__")

    #  Gaussian fusion
    d1, d2 = self.data, other.data
    p1 = self.precision(raw=True)
    p2 = other.precision(raw=True)

    precision_sum = p1 + p2
    result_data = (p1 * d1 + p2 * d2) / precision_sum

    return UncertainArray(result_data, dtype=self.dtype, precision=precision_sum)
    
def div_ua(self: UncertainArray, other: UncertainArray) -> UncertainArray:
    """
    Subtract a message from this UncertainArray under the additive precision model.

    This corresponds to computing a residual message in EP-style belief propagation:
        residual_precision = p1 - p2
        residual_mean = (p1 * m1 - p2 * m2) / max(p1 - p2, 1.0)
    """
    self.assert_compatible(other, context="__truediv__")

    d1, d2 = self.data, other.data
    p1, p2 = self.precision(raw = True), other.precision(raw = True)  # ← raw=False → shape == data.shape
    eps = np().asarray(1e-2, get_real_dtype(self.dtype))
    precision_diff = p1 - p2
    precision_safe = np().maximum(precision_diff, eps * p1)

    result_data = (p1 * d1 - p2 * d2) / precision_safe

    return UncertainArray(result_data, dtype=self.dtype, precision=precision_safe)
    

def fork(self, batch_size: int) -> "UncertainArray":
    """
    Replicate this UncertainArray into a new batched UncertainArray.

    This method creates a new UncertainArray in which the current single
    atomic Gaussian belief (batch_size=1) is duplicated into a batch of
    identical copies. It is typically used when one latent variable
    needs to be expanded into multiple identical instances, e.g., in
    ptychography models where the same probe illuminates multiple positions.
    """
    if self.batch_size != 1:
        raise ValueError("fork() expects batch_size=1 UncertainArray as input.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    new_data = np().broadcast_to(
            self.data,
            (batch_size,) + self.event_shape
        ).copy()

    raw_prec = self.precision(raw=True)
    # handle scalar vs array precision
    if self._scalar_precision:
        # e.g. shape (1,1,1) → (B,1,1)
        new_precision = np().broadcast_to(
                raw_prec,
                (batch_size,) + (1,) * len(self.event_shape)
            ).copy()
    else:
        # e.g. shape (1,H,W) → (B,H,W)
        new_precision = np().broadcast_to(
                raw_prec,
                (batch_size,) + self.event_shape
            ).copy()

    return UncertainArray(new_data, dtype=self.dtype, precision=new_precision)

    
def product_reduce_over_batch(self) -> "UncertainArray":
    """
    Reduce batchedd UA by fusing all atomic instances into one.

    This is equivalent to multiplying N Gaussians together:
        posterior_precision = sum_i p_i
        posterior_mean = sum_i (p_i * m_i) / sum_i p_i

    Returns:
        A new UncertainArray with batched=False.
    """

    # Get broadcasted precision of shape (N, *event_shape)
    precision = self.precision(raw = True)       # shape: (N, ...)
    weighted_data = precision * self.data  # shape: (N, ...)

    # Sum over batch axis (axis=0)
    precision_sum = np().sum(precision, axis=0)       # shape: event_shape
    weighted_data_sum = np().sum(weighted_data, axis=0)  # shape: event_shape
    reduced_data = np().divide(weighted_data_sum, precision_sum)

    return UncertainArray(reduced_data, dtype=self.dtype, precision=precision_sum, batched = False)



def damp_with(self, other: "UncertainArray", alpha: float) -> "UncertainArray":
    """
    Apply damping between this UncertainArray and another one.

    Performs convex interpolation of:
        - mean values (data)
        - standard deviation (not precision)

    This is used in EP/AMP-like updates where overcorrection is prevented.

    References:
        - Sarkar et al., "MRI Image Recovery using Damped Denoising Vector AMP", ICASSP 2021

    Args:
        other: Target UA to interpolate toward.
        alpha: Damping coefficient in [0, 1].

    Returns:
        New UA with damped mean and raw (unbroadcasted) precision.
    """
    self.assert_compatible(other, context="damp_with")

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be in [0, 1], but got {alpha}")

    # Interpolate means
    damped_data = (1 - alpha) * self.data + alpha * other.data

    # Interpolate standard deviations (use raw precision to preserve shape)
    std1 = np().sqrt(1.0 / self.precision(raw=True)).astype(get_real_dtype(self.dtype))
    std2 = np().sqrt(1.0 / other.precision(raw=True)).astype(get_real_dtype(self.dtype))
    damped_std = (1 - alpha) * std1 + alpha * std2
    damped_precision = 1.0 / (damped_std ** 2)

    return UncertainArray(damped_data, dtype=self.dtype, precision=damped_precision)

# --- monkey patch ---
UncertainArray.__mul__ = mul_ua
UncertainArray.__truediv__ = div_ua
UncertainArray.fork = fork
UncertainArray.product_reduce_over_batch = product_reduce_over_batch
UncertainArray.damp_with = damp_with
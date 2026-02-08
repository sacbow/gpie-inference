from __future__ import annotations

from typing import Optional

from .unitary_propagator import UnitaryPropagator
from ..wave import Wave
from ...core.backend import np
from ...core.uncertain_array import UncertainArray as UA
from ...core.fft import get_fft_backend
from ...core.types import UnaryPropagatorPrecisionMode


class FFT2DPropagator(UnitaryPropagator):
    """
    Centered 2D FFT-based unitary propagator for EP message passing.

    This propagator represents the unitary mapping:
        y = FFT2_centered(x)
        x = IFFT2_centered(y)

    EP-related logic (precision-mode propagation, block-wise belief/message updates)
    is implemented in the parent class `UnitaryPropagator`.

    Supported precision modes (inherited):
        - SCALAR
        - SCALAR_TO_ARRAY
        - ARRAY_TO_SCALAR

    Notes:
        - Assumes event_shape is 2D: (H, W)
        - Uses fftshifted (centered) FFTs via gPIE FFT backend.
    """

    def __init__(
        self,
        event_shape: Optional[tuple[int, int]] = None,
        precision_mode: Optional[UnaryPropagatorPrecisionMode] = None,
        dtype=np().complex64,
    ):
        """
        Args:
            event_shape:
                Optional 2D event shape. If None, inferred on first __matmul__.
            precision_mode:
                Optional UnaryPropagatorPrecisionMode.
                Typically inferred by Graph.compile() via precision propagation.
            dtype:
                Complex dtype for internal representation.
        """
        super().__init__(
            event_shape=event_shape,
            precision_mode=precision_mode,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # UnitaryPropagator hooks
    # ------------------------------------------------------------------

    def _validate_input_wave(self, wave: Wave) -> None:
        """
        Enforce that the input wave is 2D.

        Raises:
            ValueError: if wave.event_shape is not 2D.
        """
        if len(wave.event_shape) != 2:
            raise ValueError(
                f"FFT2DPropagator only supports 2D input. Got {wave.event_shape}"
            )

    def _forward_array(self, x):
        """
        Apply centered 2D FFT: y = FFT2_centered(x)

        Args:
            x:
                ndarray-like, shape (B, H, W) or (H, W) depending on usage.
                (Graph-level sampling typically uses batched arrays.)

        Returns:
            ndarray-like of same shape as x.
        """
        fft = get_fft_backend()
        return fft.fft2_centered(x)

    def _backward_array(self, y):
        """
        Apply centered 2D inverse FFT: x = IFFT2_centered(y)
        """
        fft = get_fft_backend()
        return fft.ifft2_centered(y)

    # ------------------------------------------------------------------
    # Optional UA fast-path overrides
    # ------------------------------------------------------------------

    def _forward_UA(self, msg_x: UA) -> UA:
        """
        UA-level forward transform (x -> y).

        Uses UA-native implementation `fft2_centered()`, which:
            - applies centered FFT on data
            - returns scalar-precision UA (harmonic reduction if needed)

        This is faster/cleaner than the generic UnitaryPropagator default,
        and ensures identical semantics to the previous FFT2DPropagator.
        """
        return msg_x.fft2_centered()

    def _backward_UA(self, msg_y: UA) -> UA:
        """
        UA-level backward transform (y -> x).

        Uses UA-native implementation `ifft2_centered()`, which:
            - applies centered IFFT on data
            - returns scalar-precision UA (harmonic reduction if needed)
        """
        return msg_y.ifft2_centered()

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"FFT2DProp(gen={gen}, mode={self.precision_mode})"

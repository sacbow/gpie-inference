from __future__ import annotations

from typing import Optional

from .unitary_propagator import UnitaryPropagator
from ..wave import Wave
from ...core.backend import np, move_array_to_current_backend
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import UnaryPropagatorPrecisionMode, get_complex_dtype
from ...core.fft import get_fft_backend


class PhaseMaskFFTPropagator(UnitaryPropagator):
    """
    Unitary propagator with a phase mask in Fourier domain:

        y = IFFT2_centered( phase_mask * FFT2_centered(x) )

    Adjoint (used for backward transform):
        x = IFFT2_centered( conj(phase_mask) * FFT2_centered(y) )

    Notes:
        - phase_mask can be either:
            * 2D: (H, W)   (shared across batch)
            * 3D: (B, H, W) (per-sample mask)
        - For block-wise scheduling, the 3D mask is sliced consistently.
        - phase_mask must be unit magnitude (|mask| ≈ 1).
    """

    def __init__(
        self,
        phase_mask,
        precision_mode: Optional[UnaryPropagatorPrecisionMode] = None,
        dtype=np().complex64,
    ):
        # event_shape is inferred from mask here (but __matmul__ will validate)
        phase_mask = np().asarray(phase_mask)

        if phase_mask.ndim == 2:
            event_shape = tuple(phase_mask.shape)
            self._mask_has_batch = False
        elif phase_mask.ndim == 3:
            event_shape = tuple(phase_mask.shape[-2:])
            self._mask_has_batch = True
        else:
            raise ValueError("phase_mask must be 2D (H,W) or 3D (B,H,W).")

        # unit magnitude check
        if not np().allclose(np().abs(phase_mask), 1.0, atol=1e-6):
            raise ValueError("phase_mask must be unit-magnitude (|mask| == 1).")

        super().__init__(
            event_shape=event_shape,
            precision_mode=precision_mode,
            dtype=dtype,
        )

        self.phase_mask = phase_mask
        self.phase_mask_conj = phase_mask.conj()

        # These are set per-call (full or block slice) to support block-wise scheduling.
        self._mask_block = None
        self._mask_conj_block = None

    def to_backend(self):
        """
        Move mask arrays to the current backend and resync dtype.
        """
        super().to_backend()
        self.phase_mask = move_array_to_current_backend(self.phase_mask, dtype=self.dtype)
        self.phase_mask_conj = self.phase_mask.conj()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_mask_view(self, block=None) -> None:
        """
        Prepare mask views for the current (block) computation.

        - 2D mask: keep as is (broadcast works).
        - 3D mask: slice along batch dimension when block is provided.
        """
        if not self._mask_has_batch:
            self._mask_block = self.phase_mask
            self._mask_conj_block = self.phase_mask_conj
            return

        # 3D (B,H,W)
        if block is None:
            self._mask_block = self.phase_mask
            self._mask_conj_block = self.phase_mask_conj
        else:
            if not isinstance(block, slice):
                raise TypeError("block must be a slice or None.")
            self._mask_block = self.phase_mask[block]
            self._mask_conj_block = self.phase_mask_conj[block]

    # ------------------------------------------------------------------
    # UnitaryPropagator hooks
    # ------------------------------------------------------------------

    def _validate_input_wave(self, wave: Wave) -> None:
        """
        Validate that the input wave is 2D and matches the mask event_shape.
        """
        if len(wave.event_shape) != 2:
            raise ValueError(
                f"PhaseMaskFFTPropagator expects 2D input. Got {wave.event_shape}"
            )
        if tuple(wave.event_shape) != tuple(self.event_shape):
            raise ValueError(
                f"Input wave event_shape {wave.event_shape} does not match mask shape {self.event_shape}"
            )

    def _forward_array(self, x):
        """
        Forward unitary operator:
            y = IFFT2( mask * FFT2(x) )
        """
        fft = get_fft_backend()
        return fft.ifft2_centered(self._mask_block * fft.fft2_centered(x))

    def _backward_array(self, y):
        """
        Backward (adjoint) operator:
            x = IFFT2( conj(mask) * FFT2(y) )
        """
        fft = get_fft_backend()
        return fft.ifft2_centered(self._mask_conj_block * fft.fft2_centered(y))

    def _forward_UA(self, msg_x: UA) -> UA:
        """
        UA-level forward transform (x -> y).

        Uses UA helpers:
            u = FFT2_centered(msg_x)   -> scalar precision UA
            masked(u)                 -> still scalar precision
            IFFT2_centered(masked(u)) -> scalar precision UA
        """
        u = msg_x.fft2_centered()  # scalar precision UA
        masked = UA(
            array=u.data * self._mask_block,
            dtype=self.dtype,
            precision=u.precision(raw=True),
            batched=True,
        )
        return masked.ifft2_centered()

    def _backward_UA(self, msg_y: UA) -> UA:
        """
        UA-level backward (adjoint) transform (y -> x).
        """
        u = msg_y.fft2_centered()  # scalar precision UA
        masked = UA(
            array=u.data * self._mask_conj_block,
            dtype=self.dtype,
            precision=u.precision(raw=True),
            batched=True,
        )
        return masked.ifft2_centered()

    # ------------------------------------------------------------------
    # Override EP entry points only to set mask view (block-aware)
    # ------------------------------------------------------------------

    def compute_belief(self, block=None):
        self._set_mask_view(block)
        return super().compute_belief(block=block)

    def _compute_forward(self, inputs, block=None):
        self._set_mask_view(block)
        return super()._compute_forward(inputs, block=block)

    def _compute_backward(self, output_msg, exclude, block=None):
        self._set_mask_view(block)
        return super()._compute_backward(output_msg, exclude, block=block)

    def get_sample_for_output(self, rng):
        """
        Generate a sample for the output wave given a sample from the input wave.
        """
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")

        # full batch mask view
        self._set_mask_view(block=None)
        return self._forward_array(x)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def __matmul__(self, wave: Wave) -> Wave:
        """
        Connect to an input wave.

        Special handling:
            - validate event_shape against mask
            - if mask is 3D, validate batch dimension matches wave.batch_size
            - promote dtype to complex
        """
        self._validate_input_wave(wave)

        # dtype promotion
        self.dtype = get_complex_dtype(wave.dtype)

        # If mask has batch, enforce batch alignment
        if self._mask_has_batch:
            if self.phase_mask.shape[0] != wave.batch_size:
                raise ValueError(
                    f"Batch size mismatch: phase_mask batch={self.phase_mask.shape[0]}, "
                    f"wave batch={wave.batch_size}"
                )

        # delegate common wiring to parent
        out = super().__matmul__(wave)
        return out

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"PMFFTProp(gen={gen}, mode={self.precision_mode})"

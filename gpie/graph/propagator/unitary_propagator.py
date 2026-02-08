from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from .base import Propagator
from ..wave import Wave
from ...core.backend import np
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode, get_complex_dtype
from ...core.uncertain_array.utils import reduce_precision_to_scalar


class UnitaryPropagator(Propagator, ABC):
    """
    Abstract base class for unary unitary propagators in EP message passing.

    This propagator represents a unitary mapping:
        y = U(x)
        x = U^H(y)

    Subclasses must implement:
        - _forward_array(x_nd): ndarray -> ndarray
        - _backward_array(y_nd): ndarray -> ndarray

    Optionally subclasses may override:
        - _forward_UA(msg_x): UA -> UA (scalar precision)
        - _backward_UA(msg_y): UA -> UA (scalar precision)
      to reuse UA-native implementations (e.g., msg.fft2_centered()).

    This base class provides:
        - precision_mode handling (UnaryPropagatorPrecisionMode)
        - block-aware compute_belief
        - EP message updates (_compute_forward/_compute_backward)
        - sample generation for output
        - common __matmul__ wiring logic (with overridable validation)
    """

    def __init__(
        self,
        *,
        event_shape: Optional[tuple[int, ...]] = None,
        precision_mode: Optional[UnaryPropagatorPrecisionMode] = None,
        dtype=np().complex64,
        input_name: str = "input",
    ):
        super().__init__(input_names=(input_name,), dtype=dtype, precision_mode=precision_mode)
        self.event_shape = event_shape
        self.x_belief: Optional[UA] = None
        self.y_belief: Optional[UA] = None

    def to_backend(self) -> None:
        super().to_backend()

        if self.x_belief is not None:
            self.x_belief.to_backend()

        if self.y_belief is not None:
            self.y_belief.to_backend()


    # ============================================================
    # Precision-mode restriction (common to unitary family)
    # ============================================================

    def _set_precision_mode(self, mode: str | UnaryPropagatorPrecisionMode):
        """
        Restrict precision modes to unitary-unary EP modes.

        Allowed:
            - SCALAR
            - SCALAR_TO_ARRAY
            - ARRAY_TO_SCALAR

        Enforces consistency if already set.
        """
        if isinstance(mode, str):
            mode = UnaryPropagatorPrecisionMode(mode)

        allowed = {
            UnaryPropagatorPrecisionMode.SCALAR,
            UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
            UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR,
        }
        if mode not in allowed:
            raise ValueError(f"Invalid precision_mode for UnitaryPropagator: {mode}")

        if hasattr(self, "_precision_mode") and self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict: existing='{self._precision_mode}', new='{mode}'"
            )

        self._precision_mode = mode

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        """
        Expected input precision mode given UnaryPropagatorPrecisionMode.
        """
        if self._precision_mode is None:
            return None
        if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            return PrecisionMode.ARRAY
        return PrecisionMode.SCALAR

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        """
        Expected output precision mode given UnaryPropagatorPrecisionMode.
        """
        if self._precision_mode is None:
            return None
        if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            return PrecisionMode.ARRAY
        return PrecisionMode.SCALAR

    def set_precision_mode_forward(self):
        """
        Infer propagator precision mode from input wave (forward pass).
        """
        x_wave = self.inputs["input"]
        if x_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)

    def set_precision_mode_backward(self):
        """
        Infer propagator precision mode from output wave (backward pass).
        """
        y_wave = self.output
        if y_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)

    # ============================================================
    # ndarray-level unitary operators (MUST implement)
    # ============================================================

    @abstractmethod
    def _forward_array(self, x):
        """Apply y = U(x) to ndarray-like array."""
        raise NotImplementedError

    @abstractmethod
    def _backward_array(self, y):
        """Apply x = U^H(y) to ndarray-like array."""
        raise NotImplementedError

    # ============================================================
    # UA-level unitary operators (default: scalar-precision)
    # ============================================================

    def _forward_UA(self, msg_x: UA) -> UA:
        """
        Apply U to UA.data, returning a scalar-precision UA.

        Precision rule:
            - scalar -> keep scalar precision
            - array  -> reduce to scalar by harmonic mean
        """
        data = self._forward_array(msg_x.data)

        # NOTE: adjust depending on UA implementation
        scalar = getattr(msg_x, "_scalar_precision", None)
        if scalar is None:
            scalar = msg_x.is_scalar_precision  # if available

        if scalar:
            prec = msg_x.precision(raw=True)
        else:
            prec = reduce_precision_to_scalar(msg_x.precision(raw=True))

        return UA(array=data, dtype=msg_x.dtype, precision=prec, batched=True)

    def _backward_UA(self, msg_y: UA) -> UA:
        """
        Apply U^H to UA.data, returning a scalar-precision UA.

        Precision rule matches _forward_UA.
        """
        data = self._backward_array(msg_y.data)

        scalar = getattr(msg_y, "_scalar_precision", None)
        if scalar is None:
            scalar = msg_y.is_scalar_precision

        if scalar:
            prec = msg_y.precision(raw=True)
        else:
            prec = reduce_precision_to_scalar(msg_y.precision(raw=True))

        return UA(array=data, dtype=msg_y.dtype, precision=prec, batched=True)

    # ============================================================
    # Belief computation (shared)
    # ============================================================

    def compute_belief(self, block=None):
        """
        Compute the joint belief over (x, y) under y = U x.

        Updates:
            - self.x_belief
            - self.y_belief

        Design invariants:
            - Full-batch (block=None): beliefs are replaced directly.
            - Block-wise (block!=None): beliefs must already exist.
            - No zero-fill initialization is allowed.
        """
        x_wave = self.inputs["input"]
        msg_x = self.input_messages[x_wave]
        msg_y = self.output_message

        if msg_x is None or msg_y is None:
            raise RuntimeError(
                "Both input and output messages are required to compute belief."
            )

        # ------------------------------------------------------------
        # Enforce invariant: block update requires existing belief
        # ------------------------------------------------------------
        if block is not None:
            if self.x_belief is None or self.y_belief is None:
                raise RuntimeError(
                    "Block-wise belief update requested before full-batch "
                    "belief initialization."
                )

        msg_x_blk = msg_x.extract_block(block)
        msg_y_blk = msg_y.extract_block(block)

        mode = self._precision_mode

        # ------------------------------------------------------------
        # Compute block belief
        # ------------------------------------------------------------
        if mode == UnaryPropagatorPrecisionMode.SCALAR:
            x_blk = msg_x_blk * self._backward_UA(msg_y_blk)
            y_blk = self._forward_UA(x_blk)

        elif mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            y_blk = self._forward_UA(msg_x_blk).as_array_precision() * msg_y_blk
            x_blk = self._backward_UA(y_blk)

        elif mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            x_blk = msg_x_blk * self._backward_UA(msg_y_blk).as_array_precision()
            y_blk = self._forward_UA(x_blk)

        else:
            raise ValueError(f"Unknown precision_mode: {mode}")

        # ------------------------------------------------------------
        # Full-batch case: direct replacement
        # ------------------------------------------------------------
        if block is None:
            self.x_belief = x_blk
            self.y_belief = y_blk
            return x_blk, y_blk

        # ------------------------------------------------------------
        # Block-wise merge
        # ------------------------------------------------------------
        self.x_belief.insert_block(block, x_blk)
        self.y_belief.insert_block(block, y_blk)

        return x_blk, y_blk


    # ============================================================
    # EP message updates (shared)
    # ============================================================

    def _compute_forward(self, inputs, block=None):
        """
        EP forward update (x -> y).

        Initial:
            m_y = U(m_x)  (and cast to array precision if output requires it)
        Steady-state:
            m_y = y_belief / m_y_old

        Returns:
            msg_block (UA): message restricted to the given block
        """


        msg_x = inputs["input"]
        out_msg = self.output_message
        yb = self.y_belief

        msg_x_blk = msg_x.extract_block(block)

        # Initial forward
        if out_msg is None and yb is None:
            msg = self._forward_UA(msg_x_blk)
            if self.output.precision_mode_enum == PrecisionMode.ARRAY:
                msg = msg.as_array_precision()
            return msg

        # Steady-state EP update
        out_msg_blk = out_msg.extract_block(block)
        yb_blk = yb.extract_block(block)

        return yb_blk / out_msg_blk

    def _compute_backward(self, output_msg, exclude, block=None):
        """
        EP backward update (y -> x).

        Implements:
            m_x = x_belief / m_x_old

        Returns:
            msg_block (UA): message restricted to the given block
        """
        if exclude != "input":
            raise RuntimeError("UnitaryPropagator has only one input: 'input'.")

        x_blk, _ = self.compute_belief(block=block)

        x_wave = self.inputs["input"]
        msg_x_old = self.input_messages[x_wave]
        msg_x_old_blk = msg_x_old.extract_block(block)

        return x_blk / msg_x_old_blk

    # ============================================================
    # Sample generation (shared)
    # ============================================================

    def get_sample_for_output(self, rng=None):
        """
        Generate a sample for the output wave from the input sample via y = U(x).

        rng is accepted for interface compatibility but is not required.
        """
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        return self._forward_array(x)

    # ============================================================
    # Wiring / graph DSL (shared with overridable validation)
    # ============================================================

    def _validate_input_wave(self, wave: Wave) -> None:
        """
        Hook for subclasses to enforce input constraints (e.g., ndim).
        Default: no-op.
        """
        return

    def __matmul__(self, wave: Wave) -> Wave:
        """
        Connect propagator to an input Wave and create an output Wave.

        Subclasses may override _validate_input_wave() to enforce constraints.
        """
        self._validate_input_wave(wave)

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        self.dtype = get_complex_dtype(wave.dtype)
        self.event_shape = wave.event_shape
        self.batch_size = wave.batch_size

        out_wave = Wave(
            event_shape=self.event_shape,
            batch_size=wave.batch_size,
            dtype=self.dtype,
        )
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

from typing import Optional, Any
from ...core.backend import np, move_array_to_current_backend
from ...core.rng_utils import get_rng

from .base import Prior
from ...core.uncertain_array import UncertainArray as UA
from ...core.linalg_utils import random_normal_array
from ...core.types import PrecisionMode, get_real_dtype


class SupportPrior(Prior):
    """
    A structured prior that enforces known support constraints on the latent variable.

    This prior models a variable as:
        - Gaussian CN(0, 1) or N(0, 1) on the support region
        - Deterministically zero (delta function) elsewhere

    Internally, this is implemented via an UncertainArray with:
        - mean = 0 everywhere
        - precision = 1 on support=True
        - precision = large_value on support=False

    Notes:
        - Precision mode is always ARRAY (scalar mode is not supported).
        - Behavior is block-agnostic: the same message is sent every iteration.
    """

    def __init__(
        self,
        support: Any,          # ndarray(bool)
        event_shape: tuple[int, ...] = None,
        *,
        batch_size: int = 1,
        dtype: np().dtype = np().complex64,
        label: Optional[str] = None,
    ) -> None:

        # ----------------------------
        # Input validation
        # ----------------------------
        if support.dtype != bool:
            raise ValueError("Support mask must be a boolean array.")

        # Infer event_shape
        if event_shape is None:
            event_shape = support.shape

        expected_shape = (batch_size,) + event_shape

        # Broadcast or validate
        if support.shape == event_shape:
            support = np().broadcast_to(support, expected_shape)
        elif support.shape != expected_shape:
            raise ValueError(
                f"Support shape {support.shape} is invalid. "
                f"Must be {event_shape} or {expected_shape}."
            )

        self.support = support
        self.large_value = get_real_dtype(dtype)(1e10)

        # ----------------------------
        # Force precision mode = ARRAY
        # ----------------------------
        super().__init__(
            event_shape=event_shape,
            batch_size=batch_size,
            dtype=dtype,
            precision_mode=PrecisionMode.ARRAY,
            label=label,
        )

        # Constant message cached
        self.const_msg: UA = self._create_fixed_array(dtype)

    # ----------------------------------------------------------
    # Create constant array used every iteration
    # ----------------------------------------------------------
    def _create_fixed_array(self, dtype: np().dtype) -> UA:
        real_dtype = get_real_dtype(dtype)

        mean = np().zeros_like(self.support, dtype=dtype)
        precision = np().where(self.support, real_dtype(1), self.large_value)

        return UA(mean, dtype=dtype, precision=precision)

    # ----------------------------------------------------------
    # Compute message (not used during block-wise scheduling)
    # ----------------------------------------------------------
    def _compute_message(self, incoming: UA) -> UA:
        """
        For API compatibility. SupportPrior ignores incoming messages and
        always returns the constant prior message in ARRAY precision mode.
        """
        return self.const_msg

    # ----------------------------------------------------------
    # Block-agnostic forward
    # ----------------------------------------------------------
    def forward(self, block: slice | None = None) -> None:
        """
        Forward pass:
            - First iteration uses Prior base initialization.
            - Later iterations always send the constant message.
        Block parameter is ignored (support is deterministic).
        """

        # First iteration: base Prior initialization
        if self.output_message is None:
            if self._init_rng is None:
                raise RuntimeError(
                    "Initial RNG not configured for Prior. "
                    "Call graph.set_init_rng(...) before run()."
                )
            msg = self._get_initial_message(self._init_rng)
            self._store_forward_message(msg)
            self.output.receive_message(self, msg)
            return

        # Later iterations: constant message
        msg = self.const_msg
        self._store_forward_message(msg)
        self.output.receive_message(self, msg)

    # ----------------------------------------------------------
    # Sampling (for initialization)
    # ----------------------------------------------------------
    def get_sample_for_output(self, rng: Optional[Any] = None) -> np().ndarray:
        """
        Return a sample with N(0,1)/CN(0,1) on support=True and 0 elsewhere.
        """
        if rng is None:
            rng = get_rng()

        sample = np().zeros(self.support.shape, dtype=self.dtype)
        values = random_normal_array(self.support.shape, dtype=self.dtype, rng=rng)
        sample[self.support] = values[self.support]
        return sample

    # ----------------------------------------------------------
    # Backend conversion
    # ----------------------------------------------------------
    def to_backend(self) -> None:
        """
        Convert internal arrays to the active backend.
        """
        self.support = move_array_to_current_backend(self.support, dtype=bool)
        self.const_msg = self._create_fixed_array(self.dtype)
        self.dtype = self.const_msg.dtype
        if self._manual_init_msg is not None:
            self._manual_init_msg.to_backend()

    # ----------------------------------------------------------
    # Representation
    # ----------------------------------------------------------
    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        return f"SupportPrior(gen={gen}, mode=ARRAY)"

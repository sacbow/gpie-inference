from typing import Optional, Tuple, Any, Union

from ...core.backend import np, move_array_to_current_backend
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, get_real_dtype
from ...core.linalg_utils import random_normal_array
from ...core.rng_utils import get_rng

from .base import Prior


class GaussianPrior(Prior):
    """
    Gaussian prior:
        x ~ N(mean, var)  (real case)
        x ~ CN(mean, var) (complex case)

    This prior produces a constant EP message after initialization.
    The message does not depend on incoming beliefs, so block-aware
    operations are unnecessary and skipped entirely.
    """

    def __init__(
        self,
        mean: Optional[Union[float, np().ndarray]] = 0.0,
        var: float = 1.0,
        event_shape: Tuple[int, ...] = (1,),
        *,
        batch_size: int = 1,
        dtype: np().dtype = np().complex64,
        precision_mode: Optional[PrecisionMode] = None,
        label: Optional[str] = None,
    ) -> None:

        if var <= 0:
            raise ValueError("Variance must be positive.")

        real_dtype = get_real_dtype(dtype)
        self.var: float = real_dtype(var)
        self.precision: float = np().asarray(1.0 / var)

        # -----------------------------------------------------------
        # Mean handling: create full shape (batch_size, *event_shape)
        # -----------------------------------------------------------
        if mean is None:
            base = np().zeros(event_shape, dtype=dtype)
        elif np().isscalar(mean):
            base = np().full(event_shape, mean, dtype=dtype)
        else:
            arr = np().asarray(mean, dtype=dtype)
            if arr.shape != event_shape:
                raise ValueError(
                    f"Mean shape mismatch: expected {event_shape}, got {arr.shape}"
                )
            base = arr

        # Expand to full batch shape
        self.mean = np().broadcast_to(base, (batch_size,) + event_shape).copy()

        # Cached constant message (constructed lazily)
        self.const_msg: Optional[UA] = None

        # Now construct the wave and link via Prior.__init__
        super().__init__(
            event_shape=event_shape,
            batch_size=batch_size,
            dtype=dtype,
            precision_mode=precision_mode,
            label=label,
        )

    # -----------------------------------------------------------
    # Backend migration
    # -----------------------------------------------------------
    def to_backend(self) -> None:
        """Move internal arrays to the current backend."""
        self.mean = move_array_to_current_backend(self.mean, dtype=self.dtype)
        self.precision = move_array_to_current_backend(self.precision)

        if self.const_msg is not None:
            self.const_msg.to_backend()

        if self._manual_init_msg is not None:
            self._manual_init_msg.to_backend()

    # -----------------------------------------------------------
    # Constant message construction
    # -----------------------------------------------------------
    def _ensure_const_msg(self) -> None:
        """
        Construct a constant UncertainArray message for all later iterations.
        Error if precision mode is not available (should be set by graph).
        """
        if self.const_msg is not None:
            return

        mode = self.output.precision_mode_enum
        if mode is None:
            raise RuntimeError(
                "GaussianPrior: precision_mode must be resolved before constructing const_msg."
            )

        scalar_precision = (mode == PrecisionMode.SCALAR)

        # UA handles precision broadcasting internally
        if scalar_precision:
            prec = self.precision  # scalar
        else:
            # array precision → shape (batch_size, *event_shape)
            prec = np().full(
                (self.batch_size,) + self.event_shape,
                self.precision,
                dtype=get_real_dtype(self.dtype),
            )

        msg = UA(self.mean, dtype=self.dtype, precision=prec, batched=True)
        self.const_msg = msg

    # -----------------------------------------------------------
    # Forward EP message passing
    # -----------------------------------------------------------
    def forward(self, block: slice | None = None) -> None:
        """
        Forward pass:
            - First iteration: use Prior base-class initialization.
            - Later iterations: ignore block and incoming beliefs; send const_msg.
        """
        if self.output_message is None:
            if self._init_rng is None and self._manual_init_msg is None:
                raise RuntimeError(
                    "Initial RNG not configured for Prior. "
                    "Call graph.set_init_rng(...) before run()."
                )
            # Base Prior logic (samples / uninformative / manual)
            msg = self._get_initial_message(self._init_rng)
            self._store_forward_message(msg)
            self.output.receive_message(self, msg)
            return

        # Later iterations: always constant
        self._ensure_const_msg()
        msg = self.const_msg

        # Send & store
        self._store_forward_message(msg)
        self.output.receive_message(self, msg)

    # -----------------------------------------------------------
    # _compute_message required by abstract base class
    # -----------------------------------------------------------
    def _compute_message(self, incoming: UA) -> UA:
        """Return the constant EP message."""
        self._ensure_const_msg()
        return self.const_msg

    # -----------------------------------------------------------
    # Sampling for "sample" initialization mode
    # -----------------------------------------------------------
    def get_sample_for_output(self, rng: Optional[Any] = None) -> np().ndarray:
        """Return mean + sqrt(var) * ε."""
        if rng is None:
            rng = get_rng()

        shape = (self.batch_size,) + self.event_shape
        noise = random_normal_array(shape, dtype=self.dtype, rng=rng)
        return self.mean + np().sqrt(self.var) * noise

    # -----------------------------------------------------------
    # Debug representation
    # -----------------------------------------------------------
    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return (
            f"GaussianPrior(gen={gen}, mode={mode}, "
            f"mean_shape={self.mean.shape}, var={self.var})"
        )

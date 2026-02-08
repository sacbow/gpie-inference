from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple
from ...core.backend import np
from ...core.rng_utils import get_rng

from ..factor import Factor
from ..wave import Wave
from ...core.uncertain_array import UncertainArray
from ...core.types import PrecisionMode


class Prior(Factor, ABC):
    """
    Abstract base class for prior factors in a Computational Factor Graph (CFG).

    A `Prior` defines the generative origin of a `Wave` node by specifying
    its initial distribution (e.g., standard Gaussian, sparse, structured).

    Responsibilities:
        - Owns and connects a single output Wave (no inputs)
        - Sends initial messages during the forward pass
        - Optionally refines messages if feedback is available (e.g., structured priors)
        - Manages precision mode (scalar/array) based on Wave or user input
        - Supports multiple initialization strategies

    Initialization Strategies:
        - "uninformative": default, random Gaussian (UncertainArray.random)
        - "sample": use subclass-defined get_sample_for_output()
        - "manual": use set_manual_init() defined by the user

    DSL Integration:
        Supports syntactic sugar:
            >> x = ~MyPrior(...)   # equivalent to: x = prior.output
    """

    def __invert__(self) -> Wave:
        """Enable `x = ~MyPrior(...)` syntax for DSL-like expression."""
        return self.output

    def __init__(
        self,
        event_shape: Tuple[int, ...],
        *,
        batch_size: int = 1,
        dtype: np().dtype = np().complex64,
        precision_mode: Optional[PrecisionMode] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.event_shape = event_shape
        self.batch_size = batch_size
        self.dtype = dtype
        self._init_rng: Optional[Any] = None
        self._manual_init_msg: Optional[UncertainArray] = None
        self._init_strategy: str = "uninformative"  # default

        if precision_mode is not None:
            self._set_precision_mode(precision_mode)

        wave = Wave(
            event_shape=event_shape,
            batch_size=batch_size,
            dtype=dtype,
            precision_mode=precision_mode,
            label=label,
        )
        self.connect_output(wave)

    # ---------- Precision mode utilities ----------

    def set_precision_mode_backward(self) -> None:
        """If the output wave's mode is externally fixed, adopt it into this Prior."""
        if self.output.precision_mode_enum is not None:
            self._set_precision_mode(self.output.precision_mode_enum)

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        """Return this Prior's precision mode (as Enum)."""
        return self._precision_mode

    # ---------- Initialization controls (public API) ----------

    def set_init_rng(self, rng: Optional[Any]) -> None:
        """Set RNG used for initial sampling of this prior."""
        self._init_rng = rng

    def set_manual_init(
        self,
        data: np().ndarray,
        *,
        precision: float | np().ndarray = 1.0,
        batched: bool = True,
    ) -> None:
        """
        Set manual initialization by ndarray (UA is constructed internally).

        Args:
            data: ndarray with shape (batch_size, *event_shape) or (*event_shape,)
                  This is interpreted as the mean parameter of the initial message.
            precision: Optional scalar or array precision for the initial message (default: 1.0).
            batched: Whether to treat the first dimension as the batch dimension (default: True).

        Raises:
            ValueError: If the provided data shape is incompatible with (batch_size, *event_shape).
        """
        expected = (self.batch_size,) + self.event_shape

        # batch_size > 1 and single sample provided -> mismatch
        if self.batch_size > 1 and data.shape == self.event_shape:
            raise ValueError(
                f"Manual init shape mismatch: expected {expected}, got {data.shape}"
            )

        # Coerce data to batched shape only if batch_size == 1
        if self.batch_size == 1 and data.shape == self.event_shape:
            data = data.reshape(expected)
        elif data.shape != expected:
            raise ValueError(f"Manual init shape mismatch: expected {expected}, got {data.shape}")

        ua = UncertainArray(
            array=data,
            dtype=self.dtype,
            precision=precision,
            batched=batched,
        )
        self._manual_init_msg = ua

    def set_init_strategy(self, mode: str) -> None:
        """
        Select initialization strategy for the first forward iteration.

        Args:
            mode: One of {"uninformative", "sample", "manual"}.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        allowed = {"uninformative", "sample", "manual"}
        if mode not in allowed:
            raise ValueError(f"Invalid init strategy '{mode}'. Must be one of {allowed}.")
        self._init_strategy = mode

    # ---------- Core message passing ----------

    def forward(self, block: slice | None = None) -> None:
        """
        Send the forward message to the output wave.

        Behavior:
            - First iteration:
                If `self.output_message is None`, create a full-batch initial message
                using the selected initialization strategy. Block information is ignored.
                Store the full message via `_store_forward_message()`.

            - Later iterations:
                * If `block is None`, fall back to the legacy full-batch EP update.
                That is, compute `_compute_message(self.output_message)` for the entire
                batch, store it, and send it to the output.

                * If `block` is a slice, perform block-wise update:
                    (1) Extract the corresponding block from `self.output_message`
                    (2) Apply `_compute_message()` to the block
                    (3) Insert the updated block into `self.last_forward_message`
                    (4) Store and send the updated full message

        Notes:
            - `self.last_forward_message` always holds the most recent full outgoing
            forward message and acts as the accumulation buffer for block updates.
            - Prior has no backward message, so backward() remains a no-op.
        """

        # --- First iteration: no incoming belief from the output yet ---
        if self.output_message is None:
            if self._init_rng is None and self._manual_init_msg is None:
                raise RuntimeError(
                    "Initial RNG not configured for Prior. "
                    "Call graph.set_init_rng(...) before run()."
                )
            # Produce full-batch initial message
            msg = self._get_initial_message(self._init_rng)
            # Cache the outgoing message
            self._store_forward_message(msg)
            # Send to output wave
            self.output.receive_message(self, msg)
            return

        # --- Later iterations, no block specified (full update) ---
        if block is None or self.last_forward_message is None:
            # Compute full-batch updated message
            msg = self._compute_message(self.output_message)
            # Cache full outgoing message
            self._store_forward_message(msg)
            # Send to output wave
            self.output.receive_message(self, msg)
            return

        # --- Block-wise update ---
        # 1) Extract incoming belief restricted to this block
        incoming_block = self.output_message.extract_block(block)
        # 2) Apply update rule to the block
        updated_block = self._compute_message(incoming_block)
        # 3) Insert updated block into the full cached message
        full_msg = self.last_forward_message
        full_msg.insert_block(block, updated_block)
        # 4) Cache updated full message
        self._store_forward_message(full_msg)
        # 5) Send updated full message to the output wave
        self.output.receive_message(self, full_msg)


    def backward(self, block = None) -> None:
        """No backward message from Prior."""
        pass

    # ---------- Unified initialization (internal) ----------

    def _get_initial_message(self, rng: Any) -> UncertainArray:
        """
        Unified initializer for the first iteration.

        Strategy depends on `self._init_strategy`:
            - "manual": use user-provided message
            - "sample": use get_sample_for_output()
            - "uninformative": fallback Gaussian random initialization
        """
        # --- Resolve precision mode first ---
        mode = self._precision_mode or self.output.precision_mode_enum
        if mode is None:
            self.set_precision_mode_backward()
            mode = self._precision_mode or self.output.precision_mode_enum
        if mode is None:
            raise RuntimeError("Precision mode must be set or inferrable before initialization.")
        scalar_precision = mode == PrecisionMode.SCALAR

        # --- Strategy dispatch ---
        if self._init_strategy == "manual":
            if self._manual_init_msg is None:
                raise RuntimeError("Manual initialization selected but no manual message is set.")
            msg = self._manual_init_msg
            msg = msg.as_scalar_precision() if scalar_precision else msg.as_array_precision()
            return msg

        elif self._init_strategy == "sample":
            try:
                sample = self.get_sample_for_output(rng)
            except NotImplementedError:
                raise RuntimeError(
                    f"Sampling initialization requested, but {type(self).__name__} "
                    "does not implement get_sample_for_output()."
                )

            expected = (self.batch_size,) + self.event_shape
            if sample.shape == self.event_shape:
                sample = sample.reshape(expected)
            elif sample.shape != expected:
                raise ValueError(
                    f"Sample shape mismatch from get_sample_for_output: expected {expected}, got {sample.shape}"
                )

            msg = UncertainArray(
                array=sample,
                dtype=self.dtype,
                precision=1.0,
                batched=True,
            )
            return msg.as_scalar_precision() if scalar_precision else msg.as_array_precision()

        elif self._init_strategy == "uninformative":
            return UncertainArray.random(
                event_shape=self.event_shape,
                batch_size=self.batch_size,
                dtype=self.dtype,
                scalar_precision=scalar_precision,
                rng=rng,
            )

        else:
            raise ValueError(f"Unknown init strategy '{self._init_strategy}'")

    # ---------- Hooks for subclasses ----------

    def get_sample_for_output(self, rng: Optional[Any] = None) -> np().ndarray:
        """
        Optional hook for subclasses to provide a distribution-aware sample
        used for the initial message.

        Implementations should return an ndarray of shape:
            (batch_size, *event_shape) or (*event_shape,)

        By default, this method is not implemented and the Prior will fallback
        to random initialization.

        Args:
            rng: Backend RNG to be used for sampling. If None, a default RNG is created.

        Returns:
            ndarray sample or raises NotImplementedError to signal fallback.
        """
        raise NotImplementedError("get_sample_for_output is not implemented for this Prior.")

    @abstractmethod
    def _compute_message(self, incoming: UncertainArray) -> UncertainArray:
        """Compute the message based on incoming observation (used in structured priors)."""
        pass

    def to_backend(self) -> None:
        """Move internal UA to the current backend."""
        super().to_backend()
        if self._manual_init_msg is not None:
            self._manual_init_msg.to_backend()
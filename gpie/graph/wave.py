from __future__ import annotations
import warnings
from typing import Optional, TYPE_CHECKING, Any, List
from ..core.rng_utils import get_rng
from ..core.backend import np
from ..core.types import ArrayLike, PrecisionMode, Precision
from ..core.linalg_utils import reduce_precision_to_scalar, random_normal_array
from numpy.typing import NDArray
from ..core.uncertain_array import UncertainArray

if TYPE_CHECKING:
    from .propagator.add_propagator import AddPropagator
    from .propagator.multiply_propagator import MultiplyPropagator
    from .structure.graph import Factor


class Wave:
    """
    Represents a latent variable node in a Computational Factor Graph (CFG),
    used in Expectation Propagation-based inference.

    Each Wave corresponds to a (batched) vector-shaped random variable whose belief
    is Gaussian-distributed and updated via message passing with connected factors.

    Key Features:
    - Supports vectorization via `batch_size`, enabling efficient modeling of
      multiple independent subgraphs (batched Wave nodes)
    - Maintains belief state as an `UncertainArray`
    - Participates in forward and backward message passing

    Message Passing Semantics:
        - `forward()`: sends messages to child factors based on current belief
        - `backward()`: combines messages from children and sends to parent
        - Belief update follows:
              belief = parent_message * combine(child_messages)

    Precision Mode:
        - 'scalar': assumes isotropic uncertainty per instance
        - 'array' : allows per-element uncertainty
        - The mode is inferred from connected factors via
            `set_precision_mode_forward()` / `backward()`, or can be manually set.

    Attributes:
        event_shape (tuple[int, ...]):
            Shape of the variable excluding batch dimension.
        batch_size (int):
            Number of vectorized instances (default: 1).
        dtype (np().dtype):
            Data type of the variable (e.g., np().complex64).
        label (str | None):
            Optional name for visualization or debugging.
        belief (UncertainArray | None):
            Current fused belief state.
        parent_message (UncertainArray | None):
            Latest message received from parent factor.
        child_messages (dict[Factor, UncertainArray]):
            Latest messages from child factors.
        parent (Factor | None):
            Connected parent factor.
        children (list[Factor]):
            Connected child factors.
        _precision_mode (PrecisionMode | None):
            Required or inferred precision mode.
        _sample (NDArray | None):
            Sample used in generative models (optional).

    Example:
        >>> x = Wave((64, 64), batch_size=8)
        >>> y = Wave((64, 64))
        >>> z = x + y  # internally creates AddPropagator
    """

    __array_priority__ = 1000

    def __init__(
        self,
        event_shape: tuple[int, ...],
        *,
        batch_size: int = 1,
        dtype: np().dtype = np().complex64,
        precision_mode: Optional[str | PrecisionMode] = None,
        label: Optional[str] = None,
    ) -> None:
        """
        Initialize a Wave representing a latent variable node in a computational factor graph.

        Each Wave models a Gaussian-distributed vector-valued random variable, which may be
        batched (i.e., represent multiple independent instances of the same structure).
        The belief associated with this Wave is stored as an `UncertainArray`, and messages
        from connected factors are used to update it via Expectation Propagation.

        This constructor sets up the static properties of the variable such as shape,
        data type, vectorization level, and optionally its precision mode.

        Args:
            event_shape (tuple[int, ...]):
                Shape of each atomic variable (excluding batch dimension),
                e.g., (64,), (32, 32), etc.
            
            batch_size (int, optional):
                Number of independent vectorized instances of this variable.
                Defaults to 1 (non-vectorized). If >1, the Wave participates in
                batched inference where messages and beliefs are broadcast over batch.

            dtype (np().dtype, optional):
                Data type for the variable and messages (e.g., np().float32 or np().complex64).
                Defaults to complex64 for GPU efficiency.

            precision_mode (str | PrecisionMode, optional):
                Either 'scalar' or 'array'. This sets the expected format of the precision
                (isotropic vs anisotropic uncertainty). If left as None, the mode will be
                inferred during graph compilation based on the requirements of connected
                factors (via `set_precision_mode_forward()` / `backward()`).

            label (str, optional):
                Human-readable name for the variable. Used in debugging and graph visualization.

        Raises:
            ValueError: If invalid precision mode is provided.
        """

        self.event_shape = event_shape
        self.batch_size = batch_size
        self.dtype = dtype
        self._precision_mode: Optional[PrecisionMode] = (
            PrecisionMode(precision_mode) if isinstance(precision_mode, str) else precision_mode
        )
        self.label = label
        self._init_rng: Optional[Any] = None

        self.parent: Optional["Factor"] = None
        self.parent_message: Optional[UncertainArray] = None
        self.children: list["Factor"] = []

        self.child_messages: dict["Factor", UncertainArray] = {}

        self.belief: Optional[UncertainArray] = None
        self._generation: int = 0
        self._sample: Optional[NDArray] = None

    
    def to_backend(self) -> None:
        """
        Convert all internal UncertainArrays (parent, children, belief) to current backend.

        This should be called when switching between NumPy and CuPy backends.
        """
        # Convert child messages
        for msg in self.child_messages.values():
            if msg is not None:
                msg.to_backend()

        # Convert belief if it exists
        if self.belief is not None:
            self.belief.to_backend()
            self.dtype = self.belief.dtype  # Ensure dtype consistency

        # Convert parent message if it exists
        if self.parent_message is not None:
            self.parent_message.to_backend()


    def set_label(self, label: str) -> None:
        """Assign label to this wave (for debugging or visualization)."""
        self.label = label

    def _set_generation(self, generation: int) -> None:
        """Internal: Assign scheduling generation index."""
        self._generation = generation

    @property
    def generation(self) -> int:
        """Topological generation index for inference scheduling."""
        return self._generation
    
    @property
    def precision_mode_enum(self) -> Optional[PrecisionMode]:
        """
        Return the internal precision mode as an Enum (recommended for new code).

        Returns:
            PrecisionMode or None
        """
        return self._precision_mode

    @property
    def precision_mode(self) -> Optional[str]:
        """
        Return the precision mode as a string ("scalar" or "array").

        This is kept for backward compatibility. Use `precision_mode_enum` for new code.

        Returns:
            "scalar", "array", or None
        """
        return self._precision_mode.value if self._precision_mode else None

    def _set_precision_mode(self, mode: str | PrecisionMode) -> None:
        """
        Set the precision mode for this wave, ensuring consistency if already set.

        Raises:
            ValueError: If conflicting precision mode already assigned.
        """
        if isinstance(mode, str):
            mode = PrecisionMode(mode)
        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict for Wave: existing='{self._precision_mode}', requested='{mode}'"
            )
        self._precision_mode = mode

    def set_precision_mode_forward(self) -> None:
        """
        Infer precision mode from parent factor's output requirements.
        Called during graph compilation.
        """
        if self.parent is not None:
            parent_mode = self.parent.get_output_precision_mode()
            if parent_mode is not None:
                self._set_precision_mode(parent_mode)

    def set_precision_mode_backward(self) -> None:
        """
        Infer precision mode from child factors' input requirements.
        Called during graph compilation.
        """
        for factor in self.children:
            child_mode = factor.get_input_precision_mode(self)
            if child_mode is not None:
                self._set_precision_mode(child_mode)

    def set_parent(self, factor: Factor) -> None:
        """Assign a parent factor. Each wave can have only one parent."""
        if self.parent is not None:
            raise ValueError("Parent factor is already set for this Wave.")
        self.parent = factor
        self.parent_message = None

    def add_child(self, factor: Factor) -> None:
        """Register a child factor to this wave."""
        if factor in self.child_messages:
            raise ValueError(f"Factor {factor} already registered as child.")
        self.children.append(factor)
        self.child_messages[factor] = None

    def receive_message(self, factor: Factor, message: UncertainArray) -> None:
        """
        Receive a message from either the parent or a child.

        If the message's dtype does not match the Wave's dtype:
            - If Wave expects real and message is complex → apply UA.real
            - If Wave expects complex and message is real → apply UA.astype(complex)

        Raises:
            TypeError: If dtype mismatch cannot be reconciled.
            ValueError: If factor is not connected to this wave.
        """
        # --- Dtype reconciliation ---
        if message.dtype != self.dtype:
            if np().issubdtype(self.dtype, np().floating) and np().issubdtype(message.dtype, np().complexfloating):
                message = message.real  # Complex → Real
            elif np().issubdtype(self.dtype, np().complexfloating) and np().issubdtype(message.dtype, np().floating):
                message = message.astype(self.dtype)  # Real → Complex
            else:
                raise TypeError(
                    f"UncertainArray dtype {message.dtype} does not match Wave dtype {self.dtype}, "
                    f"and cannot be safely converted."
                )

        # --- Assign message ---
        if factor == self.parent:
            self.parent_message = message
        elif factor in self.children:
            self.child_messages[factor] = message
        else:
            raise ValueError(
                f"Received message from unregistered factor: {factor}. "
                f"Expected parent: {self.parent}, or one of children: {list(self.children)}"
            )

    def combine_child_messages(self) -> UncertainArray:
        """
        Combine all incoming messages from child factors into a single UncertainArray belief.
        Assumes all messages are pre-initialized (i.e., no None entries).
        """
        if not self.child_messages:
            raise RuntimeError("No child messages to combine.")

        iterator = iter(self.child_messages.values())
        first = next(iterator)
        dtype = first.dtype
        p = first.precision(raw=True).copy()
        weighted = p * first.data

        for ua in iterator:
            p_i = ua.precision(raw=True)
            weighted += p_i * ua.data
            p += p_i

        mean = weighted / (p + 1e-12)
        return UncertainArray(mean, dtype=dtype, precision=p)

    

    def set_belief(self, belief: UncertainArray) -> None:
        """Manually assign the belief (used in propagators with internal computation)."""
        if belief.batch_size != self.batch_size:
            raise ValueError(f"Belief batch_size mismatch: expected {self.batch_size}, got {belief.batch_size}")
        if belief.event_shape != self.event_shape:
            raise ValueError(f"Belief shape mismatch: expected {self.event_shape}, got {belief.event_shape}")
        if belief.dtype != self.dtype:
            raise ValueError(f"Belief dtype mismatch: expected {self.dtype}, got {belief.dtype}")
        self.belief = belief


    def compute_belief(self) -> UncertainArray:
        """
        Compute current belief by combining parent and child messages.

        Returns:
            Fused `UncertainArray` belief.
        """
        child_belief = self.combine_child_messages()

        if self.parent_message is not None:
            combined = self.parent_message * child_belief
        else:
            raise RuntimeError("Cannot compute belief without parent message.")

        self.set_belief(combined)
        return combined


    def forward(self, block = None) -> None:
        """
        Send messages to all child factors using EP-style division.
        forward(): sends (belief / child_message) to each child
        Requires that parent message has already been received.
        """
        if self.parent_message is None:
            raise RuntimeError("Cannot forward without parent message.")

        if len(self.children) == 1:
            self.children[0].receive_message(self, self.parent_message, block)
        else:
            belief = self.compute_belief()
            for factor in self.children:
                msg = belief / self.child_messages[factor]
                factor.receive_message(self, msg, block)

    def backward(self, block = None) -> None:
        """
        Send message to parent by combining all child messages.
        backward(): sends combined(child_messages) to parent
        If there's only one child, reuse its message directly.
        """
        if self.parent is None:
            return

        if len(self.children) == 1:
            msg = self.child_messages[self.children[0]]
        else:
            msg = self.combine_child_messages()

        self.parent.receive_message(self, msg, block)


    def set_init_rng(self, rng) -> None:
        """Set backend-agnostic random generator."""
        self._init_rng = rng

    
    def _generate_sample(self, rng) -> None:
        """Pull sample from parent factor if not already set."""
        if self._sample is not None:
            return
        if self.parent and hasattr(self.parent, "get_sample_for_output"):
            sample = self.parent.get_sample_for_output(rng = rng)
            self.set_sample(sample)
    

    def get_sample(self) -> Optional[NDArray]:
        """Return the current sample (if set). To be deplicated."""
        return self._sample

    def set_sample(self, sample: NDArray) -> None:
        """Set sample value explicitly, allowing broadcast to expected shape."""
        expected_shape = (self.batch_size,) + self.event_shape
        try:
            # Attempt to broadcast to expected shape
            broadcasted = np().broadcast_to(sample, expected_shape)
        except ValueError as e:
            raise ValueError(
                f"Sample shape mismatch: expected broadcastable to {expected_shape}, "
                f"but got {sample.shape}"
            ) from e

        self._sample = broadcasted.copy()

    def clear_sample(self) -> None:
        """Clear the stored sample."""
        self._sample = None
    
    def extract_patches(self, indices: list[tuple[slice, ...]]) -> "Wave":
        """
        Extract multiple patches from this Wave using SlicePropagator.

        Args:
            indices (list of tuple[slice,...]): List of slice tuples.
                Each tuple must match the rank of event_shape.

        Returns:
            Wave: Output wave whose batch_size = len(indices),
                  and event_shape = shape of each patch.
        """
        from .propagator.slice_propagator import SlicePropagator

        return SlicePropagator(indices) @ self
    
    def __getitem__(self, index) -> "Wave":
        """
        Extract a single patch via slicing syntax.

        Supports both slice objects and integer indices.

        Example:
            >>> x = Wave(event_shape=(32,32), batch_size=1)
            >>> y1 = x[0:16, 0:16]   # slice
            >>> y2 = x[0, 0:16]      # int + slice
        """
        if not isinstance(index, tuple):
            index = (index,)

        norm_index = []
        for dim, idx in enumerate(index):
            if isinstance(idx, bool):
                raise TypeError(
                    f"Invalid index type bool at dim {dim}. "
                    f"Must be int or slice."
                )
            if isinstance(idx, int):
                # Convert int to slice(i, i+1)
                if idx < 0:
                    idx += self.event_shape[dim]  # support negative indexing
                if idx < 0 or idx >= self.event_shape[dim]:
                    raise IndexError(
                        f"Index {idx} out of bounds for dimension {dim} "
                        f"with size {self.event_shape[dim]}"
                    )
                norm_index.append(slice(idx, idx + 1))
            elif isinstance(idx, slice):
                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else self.event_shape[dim]
                step = idx.step
                norm_index.append(slice(start, stop, step))
            else:
                raise TypeError(
                    f"Invalid index type {type(idx)} at dim {dim}. "
                    f"Must be int or slice."
                )

        return self.extract_patches([tuple(norm_index)])


    def zero_pad(self, pad_width: tuple[tuple[int, int], ...]) -> "Wave":
        """
        Zero-pad this Wave along its event dimensions.

        Args:
            pad_width (tuple[tuple[int,int], ...]):
                Same format as numpy.pad, but excluding the batch dimension.
                Example: ((2,2), (3,3)) will pad rows by 2 top/bottom
                and cols by 3 left/right.

        Returns:
            Wave: A new Wave with event_shape expanded according to pad_width.

        Example:
            >>> x = Wave(event_shape=(32, 32), batch_size=1)
            >>> y = x.zero_pad(((2,2), (2,2)))
            >>> y.event_shape
            (36, 36)
        """
        from gpie.graph.propagator.zero_pad_propagator import ZeroPadPropagator

        return ZeroPadPropagator(pad_width) @ self



    def __add__(self, other):
        """
        x + other → AddConstPropagator if other is scalar or ndarray,
                    otherwise AddPropagator.
        Includes implicit replicate() if batch sizes differ.
        """
        from .propagator.add_propagator import AddPropagator
        from .propagator.add_const_propagator import AddConstPropagator
        from .shortcuts import replicate

        if isinstance(other, Wave):
            # --- handle batch mismatch ---
            if self.batch_size != other.batch_size:
                if self.batch_size == 1:
                    self = replicate(self, batch_size=other.batch_size)
                elif other.batch_size == 1:
                    other = replicate(other, batch_size=self.batch_size)
                else:
                    raise ValueError(
                        f"Cannot add waves with mismatched batch sizes: "
                        f"{self.batch_size} vs {other.batch_size}"
                    )
            return AddPropagator() @ (self, other)

        if np().isscalar(other) or isinstance(other, np().ndarray):
            return AddConstPropagator(const=other) @ self

        return NotImplemented


    def __radd__(self, other):
        """other + x → same as x + other."""
        return self.__add__(other)


    def __mul__(self, other) -> Wave:
        """
        Overloaded elementwise multiplication.

        Supports:
            - Wave * Wave → MultiplyPropagator
            - Wave * ndarray/scalar → MultiplyConstPropagator
        Includes implicit replicate() if batch sizes differ.
        """
        from .propagator.multiply_const_propagator import MultiplyConstPropagator
        from .propagator.multiply_propagator import MultiplyPropagator
        from .shortcuts import replicate

        if isinstance(other, Wave):
            # --- handle batch mismatch ---
            if self.batch_size != other.batch_size:
                if self.batch_size == 1:
                    self = replicate(self, batch_size=other.batch_size)
                elif other.batch_size == 1:
                    other = replicate(other, batch_size=self.batch_size)
                else:
                    raise ValueError(
                        f"Cannot multiply waves with mismatched batch sizes: "
                        f"{self.batch_size} vs {other.batch_size}"
                    )
            return MultiplyPropagator() @ (self, other)

        elif isinstance(other, (int, float, complex, np().ndarray)):
            return MultiplyConstPropagator(other) @ self

        return NotImplemented


    def __rmul__(self, other) -> Wave:
        """Right-side multiplication (scalar/ndarray * Wave)."""
        return self.__mul__(other)


    def __repr__(self) -> str: 
        label_str = f", label='{self.label}'" if self.label else ""
        dtype_str = f", dtype={np().dtype(self.dtype).name}" if self.dtype else ""
        precision_str = f", precision={self.precision_mode}" if self.precision_mode else ""

        if self.batch_size == 1:
            return f"Wave(event_shape={self.event_shape}{precision_str}{label_str}{dtype_str})"
        else:
            return (
                f"Wave(batch_size={self.batch_size}, "
                f"event_shape={self.event_shape}"
                f"{precision_str}{label_str}{dtype_str})"
            )
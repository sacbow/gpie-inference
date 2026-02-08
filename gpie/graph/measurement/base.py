from abc import ABC, abstractmethod
import warnings
from typing import Optional, Union, Any
from ...core.backend import np, move_array_to_current_backend
from ...core.types import PrecisionMode, get_real_dtype
from ...core.uncertain_array import UncertainArray
from ..factor import Factor
from ..wave import Wave
from ..structure.graph import Graph


class Measurement(Factor, ABC):
    """
    Abstract base class for measurement factors in a computational factor graph.

    A Measurement defines the probabilistic relationship between a latent Wave variable
    and observed data. It handles observation injection, backward message computation,
    and synthetic data generation.

    Subclasses must implement:
        - _compute_message: backward message from observed data
        - _generate_sample: forward sampling of observed data
        - _infer_observed_dtype_from_input: determine observed dtype from input dtype
    """


    expected_input_dtype: Optional[Any] = None     # e.g. np.floating or np.complexfloating
    expected_observed_dtype: Optional[Any] = None  # e.g. np.floating or np.complexfloating

    def __init__(self, with_mask: bool = False, label: str = None) -> None:
        super().__init__()

        self._sample: Optional[np().ndarray] = None
        self.observed: Optional[UncertainArray] = None
        self._mask: Optional[np().ndarray] = None
        self.label: Optional[str] = label
        self._with_mask = with_mask
        self.input_dtype: Optional[np().dtype] = None
        self.observed_dtype: Optional[np().dtype] = None
        self.input: Optional[Wave] = None

        if self._with_mask:
            self._precision_mode: Optional[PrecisionMode] = PrecisionMode.ARRAY
        else:
            self._precision_mode: Optional[PrecisionMode] = None

    def __lshift__(self, wave: Wave) -> "Measurement":
        """
        Connect the measurement node to a latent Wave variable.

        Infers dtype relationships, registers the node in the active graph,
        and performs type validation against expected dtypes.
        """

        self.input_dtype = wave.dtype
        self.batch_size = wave.batch_size
        if self.expected_input_dtype is not None:
            if not np().issubdtype(self.input_dtype, self.expected_input_dtype):
                raise TypeError(
                    f"{type(self).__name__} expects input dtype compatible with "
                    f"{self.expected_input_dtype}, but got {self.input_dtype}"
                )

        self.observed_dtype = self._infer_observed_dtype_from_input(self.input_dtype)

        if self.expected_observed_dtype is not None:
            if not np().issubdtype(self.observed_dtype, self.expected_observed_dtype):
                raise TypeError(
                    f"{type(self).__name__} expects observed dtype compatible with "
                    f"{self.expected_observed_dtype}, but inferred {self.observed_dtype}"
                )

        self.input = wave
        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        graph = Graph.get_active_graph()
        if graph is not None:
            label = getattr(self, "label", None)
            if label is not None:
                setattr(graph, label, self)
            else:
                i = 0
                while True:
                    candidate = f"measurement_{i}"
                    if not hasattr(graph, candidate):
                        setattr(graph, candidate, self)
                        self.label = candidate
                        break
                    i += 1

        return self
    

    def set_observed(
        self,
        data: np().ndarray,
        precision: Union[float, np().ndarray, None] = None,
        mask: Optional[np().ndarray] = None,
        batched: bool = True,
    ) -> None:
        """
        Attach observed data and its precision to this measurement node.

        Args:
            data: Observation array, shape must match input Wave.
            precision: Optional scalar or array precision. Defaults to 1/var.
            mask: Optional boolean mask to mark observed regions.
            batched: Whether the input is batched (default: True).

        Raises:
            TypeError: If dtype or mask format is invalid.
            ValueError: If shape mismatches occur.
        """

        if self.input is None:
            raise RuntimeError("Cannot set observed before connecting input wave.")

        dtype = data.dtype
        var = getattr(self, "_var", 1.0)
        prec = precision if precision is not None else 1.0 / var

        if not batched:
            data = data.reshape((1,) + data.shape)

        expected_shape = (self.input.batch_size,) + self.input.event_shape
        if data.shape != expected_shape:
            raise ValueError(f"Observed data shape mismatch: expected {expected_shape}, got {data.shape}")

        if mask is not None:
            if mask.shape == self.input.event_shape:
                # Automatically add batch dimension
                mask = mask.reshape((1,) + mask.shape)

            if mask.shape != expected_shape:
                raise ValueError(f"Mask shape mismatch: expected {expected_shape}, got {mask.shape}")

            if mask.dtype != np().bool_:
                raise TypeError(f"Mask must have dtype=bool, but got {mask.dtype}")

            self._mask = mask


        if not np().issubdtype(dtype, self.observed_dtype):
            # Allow float/complex mismatch within same family (e.g., float32 vs float64)
            if (
                np().issubdtype(dtype, np().floating) and np().issubdtype(self.observed_dtype, np().floating)
            ) or (
                np().issubdtype(dtype, np().complexfloating) and np().issubdtype(self.observed_dtype, np().complexfloating)
            ):
                warnings.warn(
                    f"Observed dtype {dtype} does not exactly match expected dtype {self.observed_dtype}. "
                    f"Automatic casting will be applied.",
                    category=UserWarning
                )
                dtype = self.observed_dtype  # Cast below
            else:
                raise TypeError(
                    f"{type(self).__name__} expects observed dtype compatible with "
                    f"{self.observed_dtype}, but got {dtype}"
                )

        # Build precision array
        if isinstance(prec, float):
            if self._mask is not None:
                precision = np().where(self._mask, prec, 0.0)
            elif self.precision_mode_enum == PrecisionMode.SCALAR:
                precision = prec
            else:
                precision = np().full_like(data, fill_value=prec, dtype=get_real_dtype(dtype))
        else:
            precision = prec

        ua = UncertainArray(data, dtype=dtype, precision=precision, batched=True)
        if ua.dtype != self.observed_dtype:
            ua = ua.astype(self.observed_dtype)

        self.observed = ua
        self.observed_dtype = ua.dtype

    def update_observed_from_sample(self, mask: Optional[np().ndarray] = None) -> None:
        """
        Promote the stored synthetic sample to observed data.

        Optionally overrides the existing mask.

        Raises:
            RuntimeError: If no sample is available.
        """

        if self._sample is None:
            raise RuntimeError("No sample available to update observed.")

        if self._with_mask and mask is not None:
            self._mask = mask

        self.set_observed(self._sample, mask=self._mask)


    def to_backend(self) -> None:
        """
        Move internal arrays and dtype references to current backend.

        This ensures consistency when switching between NumPy/CuPy.
        """
        super().to_backend()
        if self.observed is not None:
            self.observed.to_backend()

        if self._mask is not None:
            self._mask = move_array_to_current_backend(self._mask, dtype=bool)

        if self.input_dtype is not None:
            self.input_dtype = np().dtype(self.input_dtype)

        if self.observed_dtype is not None:
            self.observed_dtype = np().dtype(self.observed_dtype)


    def set_precision_mode_forward(self) -> None:
        if self.input.precision_mode_enum is not None:
            self._set_precision_mode(self.input.precision_mode_enum)
    
    def get_input_precision_mode(self, wave : Wave) -> Optional[str]:
        return self.precision_mode

    def forward(self, block = None) -> None:
        pass

    def backward(self, block=None) -> None:
        self._check_observed()

        incoming = self.input_messages[self.input]

        # block-aware compute
        msg_blk = self._compute_message(incoming, block=block)

        # block=None: standard EP behavior
        if block is None:
            self.last_backward_messages[self.input] = msg_blk
            self.input.receive_message(self, msg_blk)
            return
        else:
            if self.input not in self.last_backward_messages:
                raise RuntimeError(
                    "Block-wise forward called before full-batch initialization. "
                    "Run a full forward() pass before sequential updates."
                    )
            # update cache
            full_msg = self.last_backward_messages[self.input]
            full_msg.insert_block(block, msg_blk)
            self.input.receive_message(self, full_msg)


    def _check_observed(self) -> None:
        if self.observed is None:
            raise RuntimeError("Observed data is not set.")

    def get_sample(self) -> Optional[np().ndarray]:
        return self._sample

    def set_sample(self, sample: np().ndarray) -> None:
        expected_shape = (self.input.batch_size,) + self.input.event_shape
        if sample.shape != expected_shape:
            raise ValueError(f"Sample shape mismatch: expected {expected_shape}, got {sample.shape}")
        self._sample = sample

    def clear_sample(self) -> None:
        self._sample = None

    @property
    def mask(self) -> Optional[np().ndarray]:
        return self._mask

    @property
    def precision_mode_enum(self) -> Optional[PrecisionMode]:
        return self._precision_mode

    @property
    def precision_mode(self) -> Optional[str]:
        return self._precision_mode.value if self._precision_mode else None
    
    @abstractmethod
    def compute_belief(self) -> UncertainArray:
        """
        Return the current belief (posterior approximation) over the latent variable z.

        Each subclass defines how to compute this belief — e.g.,
        by multiplying messages (GaussianMeasurement) or by performing
        a nonlinear Laplace approximation (AmplitudeMeasurement).
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_message(self, incoming: UncertainArray, block=None) -> UncertainArray:
        """
        Compute the backward message from the observed data to the latent variable.
        """
        pass

    @abstractmethod
    def _generate_sample(self, rng: Any) -> None:
        pass

    @abstractmethod
    def _infer_observed_dtype_from_input(self, input_dtype: np().dtype) -> np().dtype:
        """
        Determine the dtype of the observed data given the input dtype.
        """
        pass

    def compute_fitness(self):
        """
        Compute and return a scalar data-fit measure for this measurement node.

        Returns
        -------
        float
            The fitness value (smaller is better).

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.compute_fitness() is not implemented."
        )
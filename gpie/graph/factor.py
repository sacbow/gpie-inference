from abc import ABC
from typing import Optional, Union
from .wave import Wave
from ..core.uncertain_array import UncertainArray
from ..core.types import PrecisionMode
from ..core.backend import np


class Factor(ABC):
    """
    Abstract base class for factor nodes in the Computational Factor Graph (CFG).

    A Factor represents a probabilistic dependency or transformation among one or more
    latent variables (`Wave` nodes). It serves as the core unit of computation and message
    propagation in Expectation Propagation (EP) and related inference algorithms.

    Responsibilities:
        - Receives and sends messages to connected Waves
        - Defines forward and backward message-passing logic
        - Specifies precision mode requirements for connected Waves
        - Participates in topological scheduling and sampling

    Typical subclasses:
        - Prior:    Defines a distribution from which a Wave is generated (no inputs)
        - Propagator: Maps input Wave(s) to an output Wave (e.g., Add, Multiply)
        - Measurement: Applies an observation likelihood to a Wave (no output)

    Message Passing Semantics:
        - receive_message(): stores incoming UncertainArray from Wave
        - forward(): sends message from inputs to output (to be implemented in subclass)
        - backward(): sends message from output to inputs (to be implemented in subclass)

    Precision Modes:
        Factors may require a specific precision model ("scalar" or "array").
        These are propagated via `set_precision_mode_forward()` and `set_precision_mode_backward()`.

    Attributes:
        inputs (dict[str, Wave]):
            Mapping from string keys (e.g., "a", "x") to connected input Waves.
        output (Wave | None):
            Output wave node (optional, depending on subclass).
        input_messages (dict[Wave, UncertainArray | None]):
            Cached incoming messages from input Waves.
        output_message (UncertainArray | None):
            Cached message from output Wave.
        _precision_mode (PrecisionMode | None):
            Required precision mode for this factor. Can be inferred or user-specified.
        _generation (int | None):
            Topological depth index for scheduling in the graph.
    """

    def __init__(self):
        # Connected wave nodes
        self.inputs: dict[str, Wave] = {}
        self.output: Optional[Wave] = None

        # Messages coming inward
        self.input_messages: dict[Wave, Optional[UncertainArray]] = {}
        self.output_message: Optional[UncertainArray] = None

        # Messages goint outward
        self.last_forward_message: Optional[UncertainArray] = None
        self.last_backward_messages: dict[Wave, UncertainArray] = {}

        # Scheduling & precision
        self._generation: Optional[int] = None
        self._precision_mode: Optional[PrecisionMode] = None

        # Batch size metadata
        self.batch_size: int = 1  # default, updated when connecting Waves

    def _set_generation(self, gen: int):
        """Set scheduling index (used during graph compilation)."""
        self._generation = gen

    @property
    def generation(self) -> Optional[int]:
        """Return topological scheduling index."""
        return self._generation

    @property
    def precision_mode(self) -> Optional[PrecisionMode]:
        """Return the current precision mode of the factor (scalar or array)."""
        return self._precision_mode

    def _set_precision_mode(self, mode: Union[str, PrecisionMode]):
        """
        Set the precision mode for the factor, with consistency checks.

        Args:
            mode (str | PrecisionMode): Either a string ("scalar", "array") or Enum value.

        Raises:
            ValueError: If the string is invalid or mode conflicts with existing value.
            TypeError: If input is not a valid type.
        """
        if isinstance(mode, str):
            try:
                mode = PrecisionMode(mode)
            except ValueError:
                raise ValueError(f"Invalid precision mode string: {mode}")

        if not isinstance(mode, PrecisionMode):
            raise TypeError(f"Invalid precision mode type: {type(mode)}")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict for {type(self).__name__}: "
                f"existing='{self._precision_mode}', requested='{mode}'"
            )

        self._precision_mode = mode

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        """
        Override in subclasses to specify required precision mode for the output Wave.

        Returns:
            PrecisionMode or None
        """
        return None

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        """
        Override in subclasses to specify required precision mode for a given input Wave.

        Args:
            wave (Wave): Input wave to query.

        Returns:
            PrecisionMode or None
        """
        return None

    def set_precision_mode_forward(self):
        """
        Optionally propagate precision mode forward: from inputs to output.

        Used during graph compilation to coordinate Wave-Factor consistency.
        Override in subclasses.
        """
        pass

    def set_precision_mode_backward(self):
        """
        Optionally propagate precision mode backward: from output to inputs.

        Used during graph compilation to coordinate Wave-Factor consistency.
        Override in subclasses.
        """
        pass

    def add_input(self, name: str, wave: Wave):
        """
        Connect an input Wave to this factor under a given name.

        Args:
            name (str): Name/key to refer to this input (e.g., "x", "lhs").
            wave (Wave): Wave instance to connect.
        """
        if name in self.inputs:
            raise KeyError(f"Input name '{name}' is already registered.")
        self.inputs[name] = wave
        self.input_messages[wave] = None
        wave.add_child(self)

    def connect_output(self, wave: Wave):
        """
        Connect a Wave as the output of this factor.

        This sets the parent/child links and also updates generation indices.

        Args:
            wave (Wave): Output Wave node to connect.
        """
        if self.output is not None:
            raise ValueError(f"Output wave is already connected: {self.output}")
        self.output = wave
        max_gen = max(
            (w._generation for w in self.inputs.values() if w._generation is not None),
            default=0
        )
        self._set_generation(max_gen + 1)
        wave._set_generation(self._generation + 1)
        wave.set_parent(self)
        self.batch_size = wave.batch_size

    def receive_message(self, wave: Wave, message: UncertainArray, block = None):
        """
        Receive a message from a connected wave.

        Args:
            wave (Wave): The sender Wave.
            message (UncertainArray): The message to store.

        Raises:
            ValueError: If wave is not connected to this factor.
        """
        if wave in self.inputs.values():
            self.input_messages[wave] = message
        elif wave == self.output:
            self.output_message = message
        else:
            raise ValueError("Received message from unconnected Wave.")

    def _store_forward_message(self, msg: UncertainArray):
        self.last_forward_message = msg

    def _store_backward_message(self, wave: Wave, msg: UncertainArray):
        self.last_backward_messages[wave] = msg


    def get_sample_for_output(self, rng) -> np().ndarray:
        """
        Optionally return a sample corresponding to this factor's generative distribution.

        This method is only expected to be implemented by factors that serve as
        generative sources (e.g., Prior, Propagator).

        Measurement factors (which have no output Wave) should NOT implement this method.

        If not overridden, calling this will raise an informative error.

        Returns:
            np().ndarray: A sampled array matching output Wave's shape and dtype.

        Raises:
            NotImplementedError: If the factor does not support generative sampling.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement `get_sample_for_output()`.")
    
    def to_backend(self) -> None:
        """
        Move all cached UncertainArray objects owned by this factor
        to the current backend.
        """
        # Input messages
        for msg in self.input_messages.values():
            if msg is not None:
                msg.to_backend()

        # Output message
        if self.output_message is not None:
            self.output_message.to_backend()

        # Cached forward/backward messages
        if self.last_forward_message is not None:
            self.last_forward_message.to_backend()

        for msg in self.last_backward_messages.values():
            if msg is not None:
                msg.to_backend()
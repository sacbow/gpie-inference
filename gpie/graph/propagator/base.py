from abc import ABC, abstractmethod
from typing import Optional, Union, Dict

from ..factor import Factor
from ..wave import Wave
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import UnaryPropagatorPrecisionMode, PrecisionMode
from ...core.backend import np


class Propagator(Factor, ABC):
    """
    Base class for deterministic mappings (Propagators) in EP message passing.

    This block-aware version adds:
        forward(self, block=None)
        backward(self, block=None)

    where block=None → full-batch update (legacy behavior),
          block=slice → update only the slice and merge via UA.insert_block().
    """

    def __init__(
        self,
        input_names: tuple[str, ...] = ("input",),
        dtype: np().dtype = np().complex64,
        precision_mode: Optional[Union[str, UnaryPropagatorPrecisionMode]] = None,
    ):
        super().__init__()
        self.dtype = dtype
        self.input_names = input_names

        if precision_mode is not None:
            self._set_precision_mode(precision_mode)

    # ------------------------------------------------------------------
    # Backend
    # ------------------------------------------------------------------
    def to_backend(self) -> None:
        super().to_backend()
        current_backend = np()
        if self.dtype is not None:
            self.dtype = current_backend.dtype(self.dtype)

    # ------------------------------------------------------------------
    # Precision mode utilities
    # ------------------------------------------------------------------
    @property
    def precision_mode_enum(self) -> Optional[UnaryPropagatorPrecisionMode]:
        return self._precision_mode

    @property
    def precision_mode(self) -> Optional[str]:
        return str(self._precision_mode) if self._precision_mode else None

    def _set_precision_mode(self, mode: Union[str, UnaryPropagatorPrecisionMode]) -> None:
        if isinstance(mode, str):
            try:
                mode = UnaryPropagatorPrecisionMode(mode)
            except ValueError:
                raise ValueError(f"Invalid precision mode for Propagator: {mode}")

        if not isinstance(mode, UnaryPropagatorPrecisionMode):
            raise TypeError("Precision mode must be a string or UnaryPropagatorPrecisionMode")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict: "
                f"existing='{self._precision_mode}', requested='{mode}'"
            )

        self._precision_mode = mode

    # ------------------------------------------------------------------
    # Abstract precision mode propagation
    # ------------------------------------------------------------------
    @abstractmethod
    def set_precision_mode_forward(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_precision_mode_backward(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Block-aware FORWARD
    # ------------------------------------------------------------------
    def forward(self, block=None):
        """
        Compute and send a (possibly block-wise) message to the output wave.
        """

        # Ensure all inputs exist
        if not all(self.inputs.get(name) for name in self.input_names):
            raise RuntimeError("Inputs not fully connected.")

        # Gather input messages
        messages = {
            name: self.input_messages[self.inputs[name]]
            for name in self.input_names
        }
        if any(msg is None for msg in messages.values()):
            raise RuntimeError("Missing input message(s) for forward.")

        # --- Compute (block) output ---
        msg_block = self._compute_forward(messages, block=block)

        # --- Full batch case ---
        if block is None:
            msg_full = msg_block
        else:
            # Merge into full outgoing message
            if self.last_forward_message is None:
                raise RuntimeError(
                    "Block-wise forward called before full-batch initialization. "
                    "Run a full forward() pass before sequential updates."
                )
            else:
                msg_full = self.last_forward_message

            msg_full.insert_block(block, msg_block)

        # send to wave
        self.output.receive_message(self, msg_full)

        # store outgoing message
        self._store_forward_message(msg_full)


    # ------------------------------------------------------------------
    # Block-aware BACKWARD
    # ------------------------------------------------------------------
    def backward(self, block=None):
        """
        Compute and send (possibly block-wise) messages to each input wave.
        """

        out_msg = self.output_message
        if out_msg is None:
            raise RuntimeError("Missing output message for backward.")

        for name, wave in self.inputs.items():
            if wave is None:
                raise RuntimeError(f"Input wave '{name}' not connected.")

            # compute block message
            msg_block = self._compute_backward(out_msg, exclude=name, block=block)

            # --- Full-batch case ---
            if block is None:
                msg_full = msg_block
            else:
                # merge into full backward message buffer
                last = self.last_backward_messages.get(wave)
                if last is None:
                    raise RuntimeError(
                    "Block-wise backward called before full-batch initialization. "
                    "Run a full backward() pass before sequential updates."
                    )
                else:
                    msg_full = last

            msg_full.insert_block(block, msg_block)

            # send message to input wave
            wave.receive_message(self, msg_full)

            # store outgoing backward message
            self._store_backward_message(wave, msg_full)


    # ------------------------------------------------------------------
    # Subclass must implement block-aware kernels
    # ------------------------------------------------------------------

    def _compute_forward(self, inputs: Dict[str, UA], block=None) -> UA:
        """
        Compute outgoing message for the given block.
        Must return UA of shape (block_size, *event_shape) when block is not None.
        """
        raise NotImplementedError

    def _compute_backward(self, output_msg: UA, exclude: str, block=None) -> UA:
        """
        Compute backward message for one input wave over given block.
        Must return UA of shape (block_size, *event_shape) when block is not None.
        """
        raise NotImplementedError

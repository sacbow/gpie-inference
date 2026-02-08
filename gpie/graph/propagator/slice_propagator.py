from ...core.backend import np
from typing import Optional
from ..wave import Wave
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode
from ...core.accumulative_uncertain_array import AccumulativeUncertainArray as AUA
from ...core.uncertain_array import UncertainArray as UA
from .accumulative_propagator import AccumulativePropagator

class SlicePropagator(AccumulativePropagator):
    """
    Extracts fixed-size patches from a single input wave.

    Each patch is defined by a tuple of `slice` objects. Multiple patches
    can be specified by passing a list of slice-tuples, in which case the
    output wave has one batch entry per patch.

    Constraints:
        - Input wave must have `batch_size == 1`
        - All provided slices must yield the same patch shape

    Example:
        >>> x = Wave(event_shape=(32, 32), batch_size=1)
        >>> prop = SlicePropagator([(slice(0, 16), slice(0, 16)),
        ...                         (slice(16, 32), slice(16, 32))])
        >>> y = prop @ x
        >>> y.batch_size   # 2 patches
        2
        >>> y.event_shape  # (16, 16)
        (16, 16)
    """

    def __init__(self, indices):
        super().__init__(input_names=("input",), precision_mode=UnaryPropagatorPrecisionMode.ARRAY)

        # normalize indices to a list of tuples of slices
        if isinstance(indices, tuple) and all(isinstance(s, slice) for s in indices):
            self.indices = [indices]
        elif isinstance(indices, list):
            if not all(isinstance(idx, tuple) and all(isinstance(s, slice) for s in idx) for idx in indices):
                raise TypeError("indices must be a tuple of slices or a list of tuples of slices.")
            self.indices = indices
        else:
            raise TypeError("indices must be a tuple of slices or a list of tuples of slices.")

        # check shapes
        shapes = [tuple(s.stop - s.start for s in idx) for idx in self.indices]
        if not all(sh == shapes[0] for sh in shapes):
            raise ValueError("All slice indices must produce patches of the same shape.")
        self.patch_shape = shapes[0]

        # AUA will be initialized later
        self.output_product: Optional[AUA] = None
    
    def to_backend(self) -> None:
        """
        Ensures that this propagator and its associated AccumulativeUncertainArray (AUA)
        remain consistent when switching between NumPy and CuPy backends.
        """
        super().to_backend() 
        if self.output_product is not None:
            self.output_product.to_backend()


    def __matmul__(self, wave: Wave) -> Wave:
        """
        Connect this propagator to an input wave and construct the output wave.

        This performs the graph-building step: the propagator registers its input,
        validates that all slice indices are compatible with the input's event_shape,
        and then creates the output wave.

        Args:
            wave (Wave): Input wave. Must have batch_size=1.

        Returns:
            Wave: Output wave with event_shape = self.patch_shape and
                batch_size = len(self.indices).

        Raises:
            ValueError: if batch_size != 1, slice rank mismatch, or indices
                        are out of range for the input's event_shape.
        """

        # input wave must have batch_size == 1
        if wave.batch_size != 1:
            raise ValueError("SlicePropagator only accepts input waves with batch_size=1.")

        # check that indices fit within wave.event_shape
        for idx in self.indices:
            if len(idx) != len(wave.event_shape):
                raise ValueError(
                    f"Slice rank mismatch: got {len(idx)} slices, "
                    f"but wave.event_shape={wave.event_shape}"
                )
            for s, dim in zip(idx, wave.event_shape):
                if s.start < 0 or s.stop > dim:
                    raise ValueError(
                        f"Slice {s} out of range for dimension {dim} "
                        f"(wave.event_shape={wave.event_shape})"
                    )

        # register input
        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)
        self.batch_size = len(self.indices)
        # output wave: batch_size = number of slices, event_shape = patch_shape
        self.dtype = wave.dtype
        out_wave = Wave(
            event_shape=self.patch_shape,
            batch_size=len(self.indices),
            dtype=self.dtype
        )
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        self.output_product = AUA(
            event_shape=wave.event_shape,
            indices=self.indices,
            dtype=self.dtype
        )
        return self.output

    def get_sample_for_output(self, rng=None):
        """
        Return deterministic patches from the input sample.

        Uses the provided slice indices to extract patches from the input
        wave's sample. The resulting array has shape:
            (num_patches, *patch_shape)

        Raises:
            RuntimeError: if the input wave has no sample set.
        """
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")

        patches = [x[(0,) + idx] for idx in self.indices]
        return np().stack(patches, axis=0)
    
    
    def set_precision_mode_forward(self):
        return

    def set_precision_mode_backward(self):
        return

    def get_input_precision_mode(self, wave: Wave) -> PrecisionMode:
        return PrecisionMode.ARRAY

    def get_output_precision_mode(self) -> PrecisionMode:
        return PrecisionMode.ARRAY
    

    
    def _compute_forward(self, inputs: dict[str, UA], block=None) -> UA:
        """
        Compute forward message: input -> patches (block-aware).

        Forward semantics:
            - Warm-start (output_message is None):
                * Only full-batch forward is allowed.
                * Deterministically slice input UA into patches.
            - After warm-start:
                * AUA (self.output_product) is treated as read-only.
                * Forward message is computed purely at UA level.

        Args:
            inputs: {"input": UA} with batch_size == 1.
            block: None (full-batch / parallel) or slice over patch index dimension.

        Returns:
            UA: forward message to output wave.
        """
        x_msg = inputs["input"]
        if x_msg.batch_size != 1:
            raise ValueError("SlicePropagator expects batch_size=1 input message.")

        # ------------------------------------------------------------
        # Warm-start: deterministic slicing only
        # ------------------------------------------------------------
        if self.output_message is None:
            if block is not None:
                raise RuntimeError(
                    "Block-wise forward called before warm-start. "
                    "Run a full-batch forward() once before sequential updates."
                )
            return x_msg.extract_patches(self.indices)

        # ------------------------------------------------------------
        # Full-batch / parallel forward
        # ------------------------------------------------------------
        if block is None:
            # 1. Convert backprojected output (AUA) to full-size UA
            backproj_ua = self.output_product.as_uncertain_array()
            # 2. Fuse with input belief (UA-level multiplication)
            belief_ua = backproj_ua * x_msg
            # 3. Slice belief into patches (UA method)
            belief_patches = belief_ua.extract_patches(self.indices)
            # 4. EP residual: divide by current output message
            msg_to_send = belief_patches / self.output_message
            return msg_to_send

        # ------------------------------------------------------------
        # Block-wise / sequential forward
        # ------------------------------------------------------------
        if not isinstance(block, slice):
            raise TypeError(f"block must be a slice or None, got {type(block)}")

        blk = self._normalize_block(block)
        indices_blk = self.indices[blk.start:blk.stop]

        if len(indices_blk) == 0:
            raise ValueError(f"Empty block slice {block} for {len(self.indices)} patches.")

        # 1. Backprojected output restricted to this block (AUA -> UA, block-wise)
        backproj_blk = self.output_product.extract_patches(block=block)
        # 2. Input belief restricted to this block
        x_blk = x_msg.extract_patches(indices_blk)
        # 3. Fuse in patch domain
        belief_blk = backproj_blk * x_blk
        # 4. EP residual: divide by corresponding output message block
        out_blk = self.output_message.extract_block(block)
        msg_blk = belief_blk / out_blk

        return msg_blk


    def _rebuild_accumulator_from_output(self, out_msg: UA) -> None:
        self.output_product.clear()
        self.output_product.scatter_mul(out_msg)

    def _apply_incremental_update(self, new_blk: UA, old_blk: UA, blk: slice) -> None:
        self.output_product.scatter_add_ua(new_blk, block=blk)
        self.output_product.scatter_sub_ua(old_blk, block=blk)


    def backward(self, block=None) -> None:
        out_msg = self.output_message
        if out_msg is None:
            raise RuntimeError("Missing output message for backward.")

        input_wave = next(iter(self.inputs.values()))
        if input_wave is None:
            raise RuntimeError("Input wave not connected.")

        if self.output_product is None:
            raise RuntimeError("Forward pass must be run before backward.")

        if block is None:
            if out_msg.batch_size != self.batch_size:
                raise ValueError("Output message batch size mismatch.")
            self._rebuild_accumulator_from_output(out_msg)
            msg_in = self.output_product.as_uncertain_array()
            input_wave.receive_message(self, msg_in)
            self._store_backward_message(input_wave, msg_in)
            self._clear_spec_cache()
            return

        blk = self._normalize_block(block)
        self._backward_with_speculation(out_msg, blk)

        msg_in = self.output_product.as_uncertain_array()
        input_wave.receive_message(self, msg_in)
        self._store_backward_message(input_wave, msg_in)
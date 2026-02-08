from __future__ import annotations
from typing import Optional
from .binary_propagator import BinaryPropagator
from ..wave import Wave
from ...core.backend import np
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, BinaryPropagatorPrecisionMode as BPM, get_lower_precision_dtype, get_real_dtype


class MultiplyPropagator(BinaryPropagator):
    """
    A propagator implementing Z = A * B under complex Gaussian belief propagation.

    This module supports multiplicative interactions between two latent variables
    within the Expectation Propagation (EP) framework.

    Unlike additive propagators, the multiplication is nonlinear and leads to
    non-Gaussian true posteriors. This module approximates them as Gaussians
    by moment matching (mean & variance).

    Precision modes:
        - Supported: ARRAY, SCALAR_AND_ARRAY_TO_ARRAY, ARRAY_AND_SCALAR_TO_ARRAY
        - Not supported: SCALAR (this is a bad approximation)

    Key operations:
        - Forward: Combines beliefs of A and B to estimate Z ≈ A * B
        - Backward: Sends messages to A or B by conditioning on Z and the other

    Note:
        Belief estimates are required on both inputs before forward propagation.

    Typical use cases:
        - Gain-modulated signal modeling
        - Elementwise multiplicative interaction (e.g., masks, amplitude scaling)
    """


    def __init__(self, precision_mode: Optional[BPM] = None, num_inner_loop: int = 1):
        super().__init__(precision_mode=precision_mode)
        # Beliefs for input and output variables
        self.input_beliefs = {"a": None, "b": None}
        self.output_belief: Optional[UA] = None
        # Number of inner-loop updates
        self.num_inner_loop = num_inner_loop
    
    def to_backend(self):
        super().to_backend()
        for b in self.input_beliefs.values():
            if b is not None:
                b.to_backend()
        if self.output_belief is not None:
            self.output_belief.to_backend()


    def _set_precision_mode(self, mode: BPM) -> None:
        allowed = {
            BPM.ARRAY,
            BPM.SCALAR_AND_ARRAY_TO_ARRAY,
            BPM.ARRAY_AND_SCALAR_TO_ARRAY,
            BPM.ARRAY_AND_ARRAY_TO_SCALAR,
        }
        if mode not in allowed:
            raise ValueError(f"Invalid precision_mode for MultiplyPropagator: '{mode}'")
        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict in MultiplyPropagator: "
                f"existing='{self._precision_mode}', requested='{mode}'"
            )
        self._precision_mode = mode

    def set_precision_mode_forward(self) -> None:
        a_mode = self.inputs["a"].precision_mode_enum
        b_mode = self.inputs["b"].precision_mode_enum

        if a_mode == PrecisionMode.SCALAR and b_mode == PrecisionMode.SCALAR:
            raise ValueError("MultiplyPropagator does not support scalar × scalar mode.")

        if a_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(BPM.SCALAR_AND_ARRAY_TO_ARRAY)
        elif b_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(BPM.ARRAY_AND_SCALAR_TO_ARRAY)
    
    def set_precision_mode_backward(self) -> None:
        z_mode = self.output.precision_mode_enum
        a_mode = self.inputs["a"].precision_mode_enum
        b_mode = self.inputs["b"].precision_mode_enum

        if z_mode is None or z_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(BPM.ARRAY_AND_ARRAY_TO_SCALAR)
            return

        if z_mode == PrecisionMode.ARRAY:
            if a_mode == PrecisionMode.SCALAR:
                self._set_precision_mode(BPM.SCALAR_AND_ARRAY_TO_ARRAY)
            elif b_mode == PrecisionMode.SCALAR:
                self._set_precision_mode(BPM.ARRAY_AND_SCALAR_TO_ARRAY)
            else:
                self._set_precision_mode(BPM.ARRAY)


    def compute_variational_inference(self, block: slice | None = None) -> None:
        """
        Perform (block-aware) inner-loop VMP updates for MultiplyPropagator.

        Contract:
            - Updates ONLY the specified block for:
                * self.input_beliefs["a"], self.input_beliefs["b"]
                * self.output_belief
                * self._last_backward_msgs ("a" and "b": raw VMP messages)
            - For block-wise execution (block is not None), full-batch initialization MUST
            have already populated:
                * self.output_message
                * self.output_belief
                * self.input_beliefs["a"], self.input_beliefs["b"]
                * self._last_backward_msgs["a"], self._last_backward_msgs["b"]
            - This method must not silently zero-initialize any of the above buffers in
            block-wise mode; it raises RuntimeError instead.
        """

        # ------------------------------------------------------------
        # Preconditions (common)
        # ------------------------------------------------------------
        if self.output_message is None:
            raise RuntimeError("compute_variational_inference(): output_message is None.")

        if self.input_beliefs.get("a") is None or self.input_beliefs.get("b") is None:
            raise RuntimeError("compute_variational_inference(): input_beliefs not initialized.")

        if self.output_belief is None:
            raise RuntimeError(
                "compute_variational_inference(): output_belief is None. "
                "Full-batch forward initialization must run before VMP."
            )

        # In block-wise mode, raw backward-message buffers must already exist
        if block is not None:
            if (
                getattr(self, "_last_backward_msgs", None) is None
                or self._last_backward_msgs.get("a") is None
                or self._last_backward_msgs.get("b") is None
            ):
                raise RuntimeError(
                    "compute_variational_inference(): block-wise update called before "
                    "full-batch initialization of _last_backward_msgs."
                )

        # ------------------------------------------------------------
        # Extract block-local beliefs and messages
        # ------------------------------------------------------------
        qx_blk = self.input_beliefs["a"].extract_block(block)
        qy_blk = self.input_beliefs["b"].extract_block(block)
        z_msg_blk = self.output_message.extract_block(block)

        z_m = z_msg_blk.data
        gamma_z = z_msg_blk.precision(raw=False)

        # Incoming messages for EP/VMP fusion (array precision, block-local)
        msg_a_blk = self.input_messages[self.inputs["a"]].extract_block(block).as_array_precision()
        msg_b_blk = self.input_messages[self.inputs["b"]].extract_block(block).as_array_precision()

        last_msg_to_x: UA | None = None
        last_msg_to_y: UA | None = None

        # ------------------------------------------------------------
        # Inner-loop VMP updates (block-local)
        # ------------------------------------------------------------
        for _ in range(self.num_inner_loop):
            # --- Q_x update ---
            y_m = qy_blk.data
            sy2 = 1.0 / qy_blk.precision(raw=False)
            abs_y2_plus_var = np().abs(y_m) ** 2 + sy2

            mean_x = np().conj(y_m) * z_m / abs_y2_plus_var
            prec_x = gamma_z * abs_y2_plus_var
            msg_to_x = UA(mean_x, dtype=self.dtype, precision=prec_x)  # raw VMP message

            qx_blk = msg_to_x * msg_a_blk
            last_msg_to_x = msg_to_x

            # --- Q_y update ---
            x_m = qx_blk.data
            sx2 = 1.0 / qx_blk.precision(raw=False)
            abs_x2_plus_var = np().abs(x_m) ** 2 + sx2

            mean_y = np().conj(x_m) * z_m / abs_x2_plus_var
            prec_y = gamma_z * abs_x2_plus_var
            msg_to_y = UA(mean_y, dtype=self.dtype, precision=prec_y)  # raw VMP message

            qy_blk = msg_to_y * msg_b_blk
            last_msg_to_y = msg_to_y

        # ------------------------------------------------------------
        # Register/update raw backward messages (no UA division)
        # ------------------------------------------------------------
        if block is None:
            # Full-batch assignment
            self._last_backward_msgs = {"a": last_msg_to_x, "b": last_msg_to_y}
        else:
            # Block-wise overwrite into existing full-batch buffers
            self._last_backward_msgs["a"].insert_block(block, last_msg_to_x)
            self._last_backward_msgs["b"].insert_block(block, last_msg_to_y)

        # ------------------------------------------------------------
        # Compute output belief on this block and insert into full belief buffer
        # ------------------------------------------------------------
        mu_z = qx_blk.data * qy_blk.data
        var_z = (
            (np().abs(qx_blk.data) ** 2 + 1.0 / qx_blk.precision(raw=False))
            * (np().abs(qy_blk.data) ** 2 + 1.0 / qy_blk.precision(raw=False))
            - np().abs(mu_z) ** 2
        )

        eps = np().array(1e-8, dtype=get_real_dtype(self.dtype))
        prec_z = 1.0 / np().maximum(var_z, eps)
        out_blk = UA(mu_z, dtype=self.dtype, precision=prec_z)

        # output_belief is a full-batch UA buffer (must already exist)
        if block is None:
            self.output_belief = out_blk
            self.input_beliefs["a"] = qx_blk
            self.input_beliefs["b"] = qy_blk
        else:
            self.output_belief.insert_block(block, out_blk)
            self.input_beliefs["a"].insert_block(block, qx_blk)
            self.input_beliefs["b"].insert_block(block, qy_blk)


    def _compute_forward(
        self,
        inputs: dict[str, UA],
        block: slice | None = None
    ) -> UA:
        """
        Block-aware forward kernel for MultiplyPropagator.

        Semantics:
            - Initialization (full-batch only):
                If output_message is None and output_belief is None, build an initial
                Gaussian approximation for Z = A * B by moment matching, store it
                into self.output_belief, and return the full-batch message.

            - EP update (full-batch or block-wise):
                If both output_message and output_belief exist, return the EP-style
                outgoing message for the requested block:
                    msg = output_belief / output_message
                with scalar/array precision alignment.

        Notes:
            - This method must NOT call receive_message(); Propagator.forward() handles that.
            - For block-wise execution, full-batch initialization must have already happened.
        """
        a_msg = inputs.get("a")
        b_msg = inputs.get("b")
        if a_msg is None or b_msg is None:
            raise RuntimeError("MultiplyPropagator: missing input messages in _compute_forward().")

        out_msg = self.output_message
        out_belief = self.output_belief

        # ------------------------------------------------------------
        # Ensure input beliefs are initialized (full-batch buffers)
        # ------------------------------------------------------------
        if self.input_beliefs.get("a") is None:
            self.input_beliefs["a"] = a_msg
        if self.input_beliefs.get("b") is None:
            self.input_beliefs["b"] = b_msg

        # Cast to factor dtype (keep computation consistent)
        qa_full = self.input_beliefs["a"].astype(self.dtype)
        qb_full = self.input_beliefs["b"].astype(self.dtype)

        z_wave = self.output
        if z_wave is None:
            raise RuntimeError("MultiplyPropagator._compute_forward(): output wave is not connected.")

        # ------------------------------------------------------------
        # Case A: initialization (full-batch only)
        # ------------------------------------------------------------
        if out_msg is None and out_belief is None:
            if block is not None:
                raise RuntimeError(
                    "MultiplyPropagator._compute_forward(): initialization must be full-batch (block=None)."
                )

            # Moment-matching approximation for Z = A * B
            mu_z = qa_full.data * qb_full.data
            var_z = (
                (np().abs(qa_full.data) ** 2 + 1.0 / qa_full.precision(raw=False))
                * (np().abs(qb_full.data) ** 2 + 1.0 / qb_full.precision(raw=False))
                - np().abs(mu_z) ** 2
            )

            eps = np().array(1e-8, dtype=get_real_dtype(self.dtype))
            prec_z = 1.0 / np().maximum(var_z, eps)

            msg_full = UA(mu_z, dtype=self.dtype, precision=prec_z)

            # Align precision mode with output wave
            if z_wave.precision_mode_enum == PrecisionMode.SCALAR:
                msg_full = msg_full.as_scalar_precision()
            else:
                msg_full = msg_full.as_array_precision()

            # Cache as the current output belief (full batch)
            self.output_belief = msg_full

            # Return the full-batch outgoing message
            return msg_full

        # ------------------------------------------------------------
        # Case B: EP update (full-batch or block-wise)
        # ------------------------------------------------------------
        if out_msg is not None and out_belief is not None:
            out_msg_blk = out_msg.extract_block(block)
            belief_blk = out_belief.extract_block(block)

            if z_wave.precision_mode_enum == PrecisionMode.SCALAR:
                return belief_blk.as_scalar_precision() / out_msg_blk
            return belief_blk / out_msg_blk

        # ------------------------------------------------------------
        # Case C: inconsistent internal state
        # ------------------------------------------------------------
        raise RuntimeError(
            "MultiplyPropagator._compute_forward(): inconsistent state — "
            "expected (output_message, output_belief) to be both None (init) or both present (EP update)."
        )



    def _compute_backward(
        self,
        output_msg: UA,
        exclude: str,
        block: slice | None = None
    ) -> UA:
        """
        Block-aware backward kernel for MultiplyPropagator.

        Semantics:
            - Runs block-aware VMP updates via compute_variational_inference(block).
            - For ARRAY-precision input waves:
                returns raw VMP backward message (no UA division).
            - For SCALAR-precision input waves:
                returns scalarized belief divided by incoming message.
        """

        wave = self.inputs.get(exclude)
        msg_in_full = self.input_messages.get(wave)

        if msg_in_full is None:
            raise RuntimeError(f"_compute_backward(): missing input message for '{exclude}'.")

        if self.output_message is None:
            raise RuntimeError("_compute_backward(): output_message is None.")

        if self.input_beliefs.get(exclude) is None:
            raise RuntimeError("_compute_backward(): input_belief not initialized.")

        # ------------------------------------------------------------
        # Run block-aware VMP updates (updates beliefs and raw messages)
        # ------------------------------------------------------------
        self.compute_variational_inference(block=block)

        # ------------------------------------------------------------
        # Extract block-local quantities
        # ------------------------------------------------------------
        msg_in_blk = msg_in_full.extract_block(block)
        belief_blk = self.input_beliefs[exclude].extract_block(block)

        # ------------------------------------------------------------
        # Construct backward EP message
        # ------------------------------------------------------------
        if wave.precision_mode_enum == PrecisionMode.SCALAR:
            # Scalar precision: belief / incoming message
            return belief_blk.as_scalar_precision() / msg_in_blk

        # Array precision: use raw VMP message directly
        raw_msg = self._last_backward_msgs.get(exclude)
        if raw_msg is None:
            raise RuntimeError(
                "_compute_backward(): raw backward message missing for ARRAY-precision wave."
            )

        return raw_msg.extract_block(block)



    def get_sample_for_output(self, rng):
        a = self.inputs["a"].get_sample()
        b = self.inputs["b"].get_sample()
        if a is None or b is None:
            raise RuntimeError("Input sample(s) not set for MultiplyPropagator.")
        return a * b

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"Mul(gen={gen}, mode={self.precision_mode})"
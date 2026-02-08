from typing import Union, Optional, Dict
from ..wave import Wave
from .base import Propagator
from ...core.uncertain_array import UncertainArray as UA
from ...core.backend import np, move_array_to_current_backend
from ...core.types import (
    PrecisionMode,
    UnaryPropagatorPrecisionMode,
    get_lower_precision_dtype,
    get_real_dtype,
)


class MultiplyConstPropagator(Propagator):
    """
    MultiplyConstPropagator
    -----------------------
    Deterministic unary propagator that multiplies the incoming wave
    by a fixed constant or field (e.g., a probe illumination function in ptychography).

    This propagator performs elementwise multiplication:
        Forward :  μ_out = μ_in × const
        Backward:  μ_in  = μ_out × conj(const) / (|const|² + ε)

    Precision (inverse variance) propagation:
        Forward :  prec_out = prec_in / (|const|² + ε)
        Backward:  prec_in  = prec_out × |const|²

    where ε > 0 is a small stabilizer added to the denominator to avoid
    precision explosion when |const| → 0.

    Parameters
    ----------
    const : float | complex | ndarray
        Constant multiplier (illumination field). Can be scalar or any array
        broadcastable to the input wave shape.
    eps : float, optional (default: 1e-8)
        Positive stabilizer added to |const|² in divisions for numerical stability.

    Attributes
    ----------
    const : ndarray
        The complex multiplicative field.
    const_conj : ndarray
        Complex conjugate of `const`.
    const_abs_sq : ndarray
        Squared amplitude |const|² used for precision scaling.
    _eps : ndarray
        Stabilization constant stored as a scalar array on the current backend.
    const_dtype : dtype
        Data type of the constant field.
    """

    def __init__(self, 
                 const: Union[float, complex, np().ndarray],
                 *, 
                 eps: float = 1e-12, dtype = np().complex64
                 ):
        """
        Initialize a propagator that multiplies the incoming message by a fixed complex field.

        Parameters
        ----------
        const : float | complex | ndarray
            The illumination field (probe) or any constant multiplicative field.
            It can be scalar or an array broadcastable to the wave shape later in `__matmul__`.
        eps : float, optional (default: 1e-8)
            Non-negative stabilizer added to |const|^2 in divisions to avoid precision blow-ups:
                forward  : precision_out = precision_in / (|const|^2 + eps)
                backward : mean_in = mean_out * conj(const) / (|const|^2 + eps)
            Note: this does NOT clamp const itself; it regularizes only the denominators.
        """
        super().__init__(input_names=("input",))
        # Store the raw constant as provided (no clamping). Keep dtype for later synchronization.
        self.const = np().asarray(const, dtype = dtype)
        self.const_dtype = self.const.dtype
        self.support_threshold = np().max(np().abs(self.const)) * 1e-3

        # Validate and store stabilizer epsilon in a dtype-consistent 0-D array.
        if eps < 0:
            raise ValueError("eps must be non-negative.")
        self._eps = np().array(eps, dtype=get_real_dtype(self.const_dtype))

        # Precompute caches used by forward/backward:
        #   - const_conj  : complex conjugate of const
        #   - const_abs_sq: |const|^2 (real)
        self._rebuild_cached_fields()


    def _rebuild_cached_fields(self) -> None:
        abs_const = np().abs(self.const)

        if self.support_threshold is not None:
            # Build support mask ON CURRENT BACKEND
            self._support_mask = abs_const >= self.support_threshold

            zero_c = np().zeros((), dtype=self.const.dtype)
            zero_r = np().zeros((), dtype=get_real_dtype(self.const.dtype))

            const = np().where(self._support_mask, self.const, zero_c)

            self.const = const
            self.const_conj = np().conj(const)
            self.const_abs_sq = np().where(
                self._support_mask,
                abs_const ** 2,
                zero_r,
            )
        else:
            self._support_mask = None
            self.const_conj = np().conj(self.const)
            self.const_abs_sq = abs_const ** 2


    def to_backend(self):
        super().to_backend() 
        self.const = move_array_to_current_backend(self.const, dtype=self.const_dtype)
        self.const_dtype = self.const.dtype
        self.support_threshold = move_array_to_current_backend(
            self.support_threshold,
            dtype=get_real_dtype(self.const_dtype),
        )

        self._rebuild_cached_fields()

        self._eps = move_array_to_current_backend(
            self._eps,
            dtype=get_real_dtype(self.const_dtype),
        )

    

    def _set_precision_mode(self, mode: Union[str, UnaryPropagatorPrecisionMode]) -> None:
        if isinstance(mode, str):
            mode = UnaryPropagatorPrecisionMode(mode)

        if mode not in {
            UnaryPropagatorPrecisionMode.ARRAY,
            UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR,
            UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
        }:
            raise ValueError(f"Unsupported precision mode for MultiplyConstPropagator: {mode}")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict: existing='{self._precision_mode}', requested='{mode}'"
            )
        self._precision_mode = mode

    def set_precision_mode_forward(self) -> None:
        mode = self.inputs["input"].precision_mode_enum
        if mode == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)

    def set_precision_mode_backward(self):
        input_mode = self.inputs["input"].precision_mode_enum
        output_mode = self.output.precision_mode_enum
        if output_mode is None or output_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)
        elif input_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)
        else:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)


    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        if self.precision_mode_enum is None:
            return None
        elif self.precision_mode_enum == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            return PrecisionMode.SCALAR
        else:
            return PrecisionMode.ARRAY


    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self.precision_mode_enum is None:
            return None
        elif self.precision_mode_enum == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            return PrecisionMode.SCALAR
        else:
            return PrecisionMode.ARRAY

    def _compute_forward(self, inputs: Dict[str, UA], block=None) -> UA:
        """
        Block-aware forward kernel for MultiplyConstPropagator.

        Returns:
            UA restricted to the given block (batch slice).
            - If block is None: full-batch UA
            - If block is a slice: UA with batch_size = block.stop - block.start

        Mode-specific behavior:
            - ARRAY_TO_SCALAR:
                - initial (output_message is None): return ua.as_scalar_precision()
                - steady-state: q = (ua * out_msg.as_array_precision()).as_scalar_precision()
                                return q / out_msg
            - otherwise (ARRAY, SCALAR_TO_ARRAY): return ua (already array precision)
        """
        x_msg = inputs["input"].extract_block(block)

        # slice constant fields (const is already shaped (B, *event_shape) after __matmul__)
        const = self.const if block is None else self.const[block]
        abs_sq = self.const_abs_sq if block is None else self.const_abs_sq[block]
        eps = self._eps  # 0-d real scalar array on backend

        # dtype negotiation
        dtype = np().result_type(x_msg.dtype, const.dtype)
        x_msg = x_msg.astype(dtype)
        const = const.astype(dtype)

        real_dtype = get_real_dtype(dtype)
        abs_sq = abs_sq.astype(real_dtype)
        eps = eps.astype(real_dtype)

        # deterministic propagation (always ARRAY precision)
        mu = x_msg.data * const
        prec = x_msg.precision(raw=True) / (abs_sq + eps)
        ua = UA(mu, dtype=dtype, precision=prec)  # array precision by construction

        # special EP handling only for ARRAY_TO_SCALAR
        if self.precision_mode_enum == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            out_msg = self.output_message
            out_blk = None if out_msg is None else out_msg.extract_block(block)

            if out_blk is None:
                # initial iteration: no EP correction possible
                return ua.as_scalar_precision()

            q = (ua * out_blk.as_array_precision()).as_scalar_precision()
            return q / out_blk

        # ARRAY / SCALAR_TO_ARRAY: return as-is (ua already array precision)
        return ua


    def _compute_backward(self, output_msg: UA, exclude: str, block=None) -> UA:
        """
        Block-aware backward kernel for MultiplyConstPropagator.

        Backward mapping (adjoint-like):
            mu_in  = mu_out * conj(const) / (|const|^2 + eps)
            prec_in = prec_out * |const|^2

        Mode-specific behavior:
            - SCALAR_TO_ARRAY (input wave is scalar):
                - initial (input_message is None): return ua.as_scalar_precision()
                - steady-state: q = (ua * in_msg.as_array_precision()).as_scalar_precision()
                                return q / in_msg
            - otherwise (ARRAY, ARRAY_TO_SCALAR): return ua (array precision)
        """
        if exclude != "input":
            raise RuntimeError("MultiplyConstPropagator has only one input: 'input'.")

        out_blk = output_msg.extract_block(block)

        # slice constant fields (const is already shaped (B, *event_shape) after __matmul__)
        const_conj = self.const_conj if block is None else self.const_conj[block]
        abs_sq = self.const_abs_sq if block is None else self.const_abs_sq[block]
        eps = self._eps

        # dtype negotiation
        dtype = np().result_type(out_blk.dtype, const_conj.dtype)
        out_blk = out_blk.astype(dtype)
        const_conj = const_conj.astype(dtype)

        real_dtype = get_real_dtype(dtype)
        abs_sq = abs_sq.astype(real_dtype)
        eps = eps.astype(real_dtype)

        # deterministic backward propagation (always ARRAY precision)
        mu = out_blk.data * const_conj / (abs_sq + eps)
        prec = out_blk.precision(raw=True) * abs_sq
        ua = UA(mu, dtype=dtype, precision=prec)  # array precision by construction

        #for numerical stability

        ua = UA(mu, dtype=dtype, precision=prec)  # array precision by construction

        # special EP handling only for SCALAR_TO_ARRAY (input scalar)
        if self.precision_mode_enum == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            x_wave = self.inputs["input"]
            in_msg = self.input_messages.get(x_wave)
            in_blk = None if in_msg is None else in_msg.extract_block(block)

            if in_blk is None:
                # very first backward before any input cache exists (rare but consistent)
                return ua.as_scalar_precision()

            q = (ua * in_blk.as_array_precision()).as_scalar_precision()
            return q / in_blk

        # ARRAY / ARRAY_TO_SCALAR: return as-is
        return ua

    def get_sample_for_output(self, rng=None):
        x = self.inputs["input"].get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        const = self.const.astype(x.dtype) if self.const_dtype != x.dtype else self.const
        return x * const

    def __matmul__(self, wave: Wave) -> Wave:
        """
        Connect this propagator to a Wave via `@`.

        Responsibilities:
            - register graph connectivity
            - synchronize dtype with wave/const
            - broadcast const to (B, *event_shape) so that block slicing works
            - set self.batch_size for scheduling (Factor default is 1)
            - create output Wave
        """
        # --- connect ---
        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        # --- metadata ---
        self.event_shape = wave.event_shape
        self.batch_size = wave.batch_size  # IMPORTANT: enable scheduling logic in Graph/Propagator

        # --- dtype negotiation ---
        self.dtype = get_lower_precision_dtype(wave.dtype, self.const_dtype)

        self.const = np().asarray(self.const, dtype=self.dtype)
        self.const_dtype = self.const.dtype

        # --- broadcast const to full batched shape ---
        expected_shape = (wave.batch_size, *wave.event_shape)
        try:
            self.const = np().broadcast_to(self.const, expected_shape)
        except ValueError as e:
            raise ValueError(
                f"MultiplyConstPropagator: const shape {self.const.shape} "
                f"is not broadcastable to expected shape {expected_shape}."
            ) from e

        # rebuild cached fields on the (possibly new) backend/dtype
        self._rebuild_cached_fields()

        # ensure eps is compatible with backend and real dtype
        self._eps = move_array_to_current_backend(self._eps, dtype=get_real_dtype(self.const_dtype))

        # --- create output wave ---
        out_wave = Wave(
            event_shape=wave.event_shape,
            batch_size=wave.batch_size,
            dtype=self.dtype,
        )
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave

        return self.output


    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "unset"
        return (
            f"MultiplyConstProp("
            f"gen={gen}, "
            f"mode={mode}, "
            f"batch={self.batch_size}, "
            f")"
        )
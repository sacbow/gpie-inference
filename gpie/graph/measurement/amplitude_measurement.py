from typing import Optional, Union, Any

from .base import Measurement
from ...core.backend import np
from ...core.adaptive_damping import AdaptiveDamping, DampingScheduleConfig
from ...core.linalg_utils import random_normal_array
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, get_real_dtype


class AmplitudeMeasurement(Measurement):
    """
    Nonlinear amplitude measurement model: y = |z| + ε, with ε ~ N(0, var)

    Observes the magnitude of a complex-valued latent variable plus additive Gaussian noise.
    """

    expected_input_dtype = np().complexfloating
    expected_observed_dtype = np().floating

    def __init__(
        self,
        var: float = 1e-4,
        damping: Union[float, str] = "auto",
        precision_mode: Optional[Union[str, PrecisionMode]] = None,
        with_mask: bool = False,
        label: str = None,
        adaptive_cfg: Optional[DampingScheduleConfig] = None,
    ) -> None:
        self._var = var
        self.damping = damping
        self.belief = None

        # --- Adaptive damping setup ---
        if damping == "auto":
            self._adaptive = True
            self._scheduler = AdaptiveDamping(adaptive_cfg or DampingScheduleConfig())
            self.damping = 1.0 - self._scheduler.beta
        else:
            self._adaptive = False
            self.damping = float(damping)

        if isinstance(precision_mode, str):
            precision_mode = PrecisionMode(precision_mode)

        super().__init__(with_mask=with_mask, label = label)
        if precision_mode is not None:
            self._set_precision_mode(precision_mode)

    def _infer_observed_dtype_from_input(self, input_dtype: np().dtype) -> np().dtype:
        return get_real_dtype(input_dtype)

    def _generate_sample(self, rng: Any) -> None:
        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")

        abs_x = np().abs(x)
        noise = random_normal_array(
                    abs_x.shape,
                    dtype=abs_x.dtype,
                    rng=rng,
                ) * self._var**0.5
        
        self._sample = (abs_x + noise).astype(self.observed_dtype)


    def compute_belief(self, incoming: UA, observed: Optional[UA] = None, block = None) -> UA:
        """
        Compute approximate posterior using Laplace approximation.
        If `observed` is provided, the computation is done using
        the given UncertainArray (block-aware).
        """

        if observed is None:
            observed = self.observed

        z0 = incoming.data
        tau = incoming.precision(raw=True)
        v0 = np().reciprocal(tau)

        y = observed.data
        eps = np().array(1e-12, dtype=v0.dtype)
        v = np().reciprocal(observed.precision(raw=True) + eps)

        abs_z0 = np().abs(z0)
        abs_z0_safe = np().maximum(abs_z0, eps)
        unit_phase = z0 / abs_z0_safe

        z_hat = (v0 * y + 2 * v * abs_z0_safe) / (v0 + 2 * v) * unit_phase
        v_hat = (v0 * (v0 * y + 4 * v * abs_z0_safe)) / (2 * abs_z0_safe * (v0 + 2 * v))
        v_hat = np().maximum(v_hat, eps)

        posterior = UA(z_hat, dtype=self.input_dtype, precision=np().reciprocal(v_hat))

        if self.precision_mode_enum == PrecisionMode.SCALAR:
            posterior = posterior.as_scalar_precision()
        
        if block is None:
            self.belief = posterior
        else:
            if self.belief is None:
                self.belief = UA.zeros(
                event_shape=self.input.event_shape,
                batch_size=self.batch_size,
                dtype=self.input.dtype,
                precision=1.0,
                scalar_precision=(self.precision_mode_enum == PrecisionMode.SCALAR),
            )
            self.belief.insert_block(block, posterior)

        return posterior


    def _compute_message(self, incoming: UA, block=None) -> UA:
        self._check_observed()

        incoming_blk = incoming.extract_block(block)
        observed_blk = self.observed.extract_block(block)

        # --- compute block belief ---
        belief_blk = self.compute_belief(incoming_blk, observed=observed_blk, block = block)

        # --- raw message ---
        full_msg_blk = belief_blk / incoming_blk

        # --- apply mask (block-wise) ---
        if self._mask is not None:
            mask_blk = self._mask if block is None else self._mask[block]
            data = np().zeros_like(full_msg_blk.data)
            prec = np().zeros_like(full_msg_blk.precision(raw=True))
            data[mask_blk] = full_msg_blk.data[mask_blk]
            prec[mask_blk] = full_msg_blk.precision(raw=True)[mask_blk]
            msg_blk = UA(data, dtype=self.input_dtype, precision=prec)
        else:
            msg_blk = full_msg_blk

        # --- damping using previous full backward message ---
        if self.damping > 0 and self.input in self.last_backward_messages:
            prev_full = self.last_backward_messages[self.input]
            prev_blk = prev_full.extract_block(block)
            msg_blk = msg_blk.damp_with(prev_blk, alpha=self.damping)

        return msg_blk

    
    def backward(self, block=None) -> None:
        """
        Block-aware backward pass.
        Damping is fully handled inside _compute_message(),
        which receives the previous full backward message
        through self.last_backward_messages[self.input].
        """

        self._check_observed()
        incoming = self.input_messages[self.input]

        # --- Block-aware message computation (includes damping internally) ---
        msg_blk = self._compute_message(incoming, block=block)

        # -----------------------------------------------------------
        # Non-sequential mode (block=None)
        # -----------------------------------------------------------
        if block is None:
            # Cache full backward message
            self.last_backward_messages[self.input] = msg_blk
            # Send to wave
            self.input.receive_message(self, msg_blk)

            # Update adaptive damping parameter once per iteration
            if self._adaptive:
                J = self.compute_fitness()
                new_damp, repeat = self._scheduler.step(J)
                self.damping = new_damp
            return

        # -----------------------------------------------------------
        # Sequential (block-wise) mode
        # -----------------------------------------------------------
        # Initialize the full backward message cache if needed
        
        if self.input not in self.last_backward_messages:
            raise RuntimeError(
                "Block-wise forward called before full-batch initialization. "
                "Run a full forward() pass before sequential updates."
                )

        full_msg = self.last_backward_messages[self.input]
        # Insert block update into full cached message
        full_msg.insert_block(block, msg_blk)

        # Send the updated full message
        self.input.receive_message(self, full_msg)
        # -----------------------------------------------------------
        # Adaptive damping update only at the final block
        # -----------------------------------------------------------
        if self._adaptive and block.stop == self.batch_size:
            J = self.compute_fitness()
            new_damp, repeat = self._scheduler.step(J)
            self.damping = new_damp


    
    def compute_fitness(self) -> float:
        """
        Compute precision-weighted mean squared error between
        the magnitude of the belief mean and the observed amplitude.

        fitness = mean_i [ γ_i * (|μ_belief_i| - y_i)^2 ]
        """
        xp = np()
        if self.belief is None:
            if self.input is None or self.input not in self.input_messages:
                raise RuntimeError("Cannot compute belief: missing input message.")
            self.compute_belief(self.input_messages[self.input])

        mu_belief = xp.abs(self.belief.data)
        y = self.observed.data
        gamma = self.observed.precision(raw=True)

        diff2 = (mu_belief - y) ** 2
        weighted = gamma * diff2
        fitness = xp.mean(weighted)
        return float(fitness)

    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        return f"AmplitudeMeas(gen={gen}, mode={self.precision_mode})"

from typing import Optional, Any, Union

from ...core.backend import np
from ...core.rng_utils import get_rng
from ...core.adaptive_damping import AdaptiveDamping, DampingScheduleConfig
from ...core.uncertain_array import UncertainArray as UA
from ...core.linalg_utils import reduce_precision_to_scalar, random_normal_array
from ...core.types import PrecisionMode, get_real_dtype

from .base import Prior


class SparsePrior(Prior):
    """
    A spike-and-slab prior promoting sparsity in the latent variable.

    Model (per element):
        x_i ~ (1 - rho) * delta(0) + rho * CN(0, 1)
    or N(0, 1) in the real-valued case.

    Inference behavior:
        - Approximates the spike-and-slab posterior elementwise.
        - Forms an EP message as (posterior / incoming).
        - Supports block-wise EP updates along the batch dimension.
        - Supports optional (fixed or adaptive) damping.

    Damping:
        - Performed block-wise inside `_compute_message(...)`.
        - Uses `last_forward_message` as the "old" message.
        - Adaptive damping uses a scalar proxy J = -logZ, where logZ
          is accumulated over all blocks within an iteration.

    Args:
        rho: Probability of non-zero entry (sparsity level).
        event_shape: Shape of the latent variable (excluding batch).
        batch_size: Batch size.
        dtype: Backend dtype (real or complex).
        damping: Either a fixed damping coefficient in [0, 1] or "auto"
                 to enable adaptive damping.
        precision_mode: Scalar or array precision for the connected Wave.
        label: Optional label for the Wave.
        adaptive_cfg: Optional configuration for the adaptive damping scheduler.
    """

    def __init__(
        self,
        rho: float = 0.5,
        event_shape: tuple[int, ...] = (1,),
        *,
        batch_size: int = 1,
        dtype: np().dtype = np().complex64,
        damping: Union[float, str] = "auto",
        precision_mode: Optional[PrecisionMode] = None,
        label: Optional[str] = None,
        adaptive_cfg: Optional[DampingScheduleConfig] = None,
    ) -> None:
        real_dtype = get_real_dtype(dtype)
        self.rho = real_dtype(rho)

        # Accumulated log-evidence proxy over current iteration
        # (sum over blocks). Used for adaptive damping.
        self.logZ: float = 0.0

        super().__init__(
            event_shape=event_shape,
            batch_size=batch_size,
            dtype=dtype,
            precision_mode=precision_mode,
            label=label,
        )

        if damping == "auto":
            self._adaptive = True
            self._scheduler = AdaptiveDamping(adaptive_cfg or DampingScheduleConfig())
            # Start from a value consistent with the scheduler's beta
            self.damping = real_dtype(1.0 - self._scheduler.beta)
        else:
            self._adaptive = False
            self.damping = real_dtype(damping)

    # ------------------------------------------------------------------
    # Core EP message computation (block-wise)
    # ------------------------------------------------------------------
    def _compute_message(self, incoming: UA, *, block: slice | None = None) -> UA:
        """
        Compute the EP message for a (possibly block-restricted) incoming belief.

        Steps:
            1) Compute approximate posterior q(x | incoming) for this block.
            2) Form the undamped EP message: new = posterior / incoming.
            3) If damping is enabled and a previous forward message exists:
                   - For block=None: damp with the full previous message.
                   - For block=slice: damp with the corresponding slice
                     of the previous message.
        The method also relies on `approximate_posterior()` to accumulate
        contributions to `self.logZ` for adaptive damping.
        """
        # 1) Posterior approximation for this block (accumulates logZ)
        posterior = self.approximate_posterior(incoming)

        # 2) Undamped EP message
        new_msg = posterior / incoming

        prev_full = self.last_forward_message
        if prev_full is None or self.damping <= 0:
            return new_msg

        # 3) Block-wise damping
        if block is None:
            # Full-batch damping
            return new_msg.damp_with(prev_full, alpha=self.damping)
        else:
            # Damping against the previous message restricted to this block
            prev_block = prev_full.extract_block(block)
            return new_msg.damp_with(prev_block, alpha=self.damping)

    # ------------------------------------------------------------------
    # Forward pass (block-aware)
    # ------------------------------------------------------------------
    def forward(self, block: slice | None = None) -> None:
        """
        Block-aware forward message passing.

        Cases:
            - First iteration (no output_message yet):
                Uses Prior's initialization mechanism and ignores `block`.

            - Later iterations, full-batch (block is None):
                * Reset `logZ`.
                * Call `_compute_message` on the full incoming belief.
                * Send the resulting full-batch message.
                * If adaptive, update damping based on accumulated `logZ`.

            - Later iterations, block-wise (block is a slice):
                * For each block:
                    - Extract incoming block.
                    - Compute block-wise EP message with damping inside
                      `_compute_message(incoming_block, block=block)`.
                    - Insert the updated block into `last_forward_message`.
                    - Send the full-batch message after each block.
                * Only when processing the last block (block.stop == batch_size):
                    - Perform adaptive damping update using accumulated `logZ`.
                    - Reset `logZ` for the next iteration.
        """
        # -------------------------
        # First iteration
        # -------------------------
        if self.output_message is None:
            if self._init_rng is None:
                raise RuntimeError(
                    "Initial RNG not configured for Prior. "
                    "Call graph.set_init_rng(...) before run()."
                )

            msg = self._get_initial_message(self._init_rng)
            self._store_forward_message(msg)
            self.output.receive_message(self, msg)
            return

        # -------------------------
        # Full-batch EP update
        # -------------------------
        if block is None:
            # Reset logZ accumulator for this iteration
            self.logZ = 0.0

            msg = self._compute_message(self.output_message, block=None)

            # Cache and send
            self._store_forward_message(msg)
            self.output.receive_message(self, msg)

            # Adaptive damping update (once per iteration)
            if self._adaptive:
                try:
                    J = -self.logZ
                    new_damp, repeat = self._scheduler.step(J)
                    self.damping = new_damp
                except Exception:
                    pass
            return

        # -------------------------
        # Block-wise EP update
        # -------------------------
        is_last_block = (block.stop == self.batch_size)

        if self.last_forward_message is None:
            raise RuntimeError(
                "SparsePrior.forward(): last_forward_message is None during block-wise update."
            )

        # 1) Extract incoming block
        incoming_block = self.output_message.extract_block(block)

        # 2) Compute block-wise EP message with internal damping
        new_block = self._compute_message(incoming_block, block=block)

        # 3) Insert into full-batch message buffer
        self.last_forward_message.insert_block(block, new_block)

        # 4) Send the full-batch message every time
        msg = self.last_forward_message
        self._store_forward_message(msg)
        self.output.receive_message(self, msg)

        # 5) Adaptive damping only after the final block
        if is_last_block and self._adaptive:
            try:
                J = -self.logZ
                new_damp, repeat = self._scheduler.step(J)
                self.damping = new_damp
            except Exception:
                pass

            # Reset for the next iteration
            self.logZ = 0

    # ------------------------------------------------------------------
    # Posterior approximation (per block, accumulates logZ)
    # ------------------------------------------------------------------
    def approximate_posterior(self, incoming: UA) -> UA:
        """
        Approximate the spike-and-slab posterior for the given incoming message.

        This method operates on the full contents of `incoming`, which may
        correspond to either the whole batch or a single contiguous block
        (when used in block-wise scheduling).

        It also updates `self.logZ` by adding a scalar log-likelihood proxy
        derived from the current block.
        """
        m = incoming.data
        v = 1 / incoming.precision(raw=True)

        prec_post = 1 + 1 / v
        v_post = 1 / prec_post
        m_post = v_post * (m / v)

        is_real = incoming.is_real()
        eps = np().array(1e-12, dtype=v.dtype)

        if is_real:
            slab = self.rho * (1 / np().sqrt(1 + v)) * np().exp(-m**2 / (2 * (1 + v)))
            spike = (1 - self.rho) * (1 / np().sqrt(v)) * np().exp(-m**2 / (2 * v))
        else:
            slab = self.rho * np().exp(-np().abs(m) ** 2 / (1 + v)) / (1 + v)
            spike = (1 - self.rho) * np().exp(-np().abs(m) ** 2 / v) / v

        Z = slab + spike + eps  # prevent division by zero

        # Accumulate scalar log-likelihood proxy over this block
        self.logZ += float(np().sum(np().log(Z + eps)))

        mu = (slab / Z) * m_post
        e_x2 = (slab / Z) * (np().abs(m_post) ** 2 + v_post)
        var = np().maximum(e_x2 - np().abs(mu) ** 2, eps)
        precision = 1 / var

        if self.precision_mode == PrecisionMode.SCALAR:
            precision = reduce_precision_to_scalar(precision)

        return UA(mu, dtype=self.dtype, precision=precision)

    # ------------------------------------------------------------------
    # Sampling for initialization
    # ------------------------------------------------------------------
    def get_sample_for_output(self, rng: Optional[Any] = None) -> np().ndarray:
        """
        Draw a sample from the spike-and-slab prior.

        Each element is zero with probability (1 - rho) and drawn from
        a standard (real or complex) Gaussian with probability rho.
        """
        if rng is None:
            rng = get_rng()

        shape = (self.batch_size,) + self.event_shape

        mask = rng.uniform(size=shape) < self.rho
        sample = np().zeros(shape, dtype=self.dtype)
        values = random_normal_array(shape, dtype=self.dtype, rng=rng)
        sample[mask] = values[mask]
        return sample

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return f"SPrior(gen={gen}, mode={mode}, rho={self.rho})"

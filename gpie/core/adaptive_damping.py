"""
Adaptive damping scheduler for expectation propagation.

This module implements an AD-GAMP-like adaptive damping rule that adjusts the
damping parameter (or equivalently, step size) based on the recent trajectory of
a any user-defined scalar cost.

The adaptive rule follows the method proposed in:

    J. Vila, P. Schniter, S. Rangan, F. Krzakala and L. Zdeborová,
    "Adaptive damping and mean removal for the generalized approximate message passing algorithm,"
    2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
    South Brisbane, QLD, Australia, 2015, pp. 2021–2025,
    doi: 10.1109/ICASSP.2015.7178325.

In that paper, the adaptive rule monitors a scalar objective J_t and increases or decreases 
the damping factor β ∈ (0,1] depending on whether J_t improves compared to recent iterations. 
This implementation is general and can be plugged into arbitrary factor graph nodes in gPIE.

The mapping between AMP's β and gPIE's `damping` parameter is:
    damping = 1 - β
where gPIE's `damping` = 0 corresponds to "no damping", and `damping` = 1 corresponds
to "fully damped" (frozen) updates.

Typical default parameters:
    G_pass = 1.1, G_fail = 0.5, β_min = 0.01, β_max = 1.0, T_β = 3
"""
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DampingScheduleConfig:
    """
    Configuration parameters for the adaptive damping scheduler.
    """
    G_pass: float = 1.4
    G_fail: float = 0.8
    beta_min: float = 0.05
    beta_max: float = 1.0
    T_beta: int = 3


class AdaptiveDamping:
    """
    Adaptive damping controller (float32 internal precision).
    """

    def __init__(self, cfg: DampingScheduleConfig):
        self.cfg = cfg
        self.beta = np.float32(cfg.beta_max)  # enforce single precision
        self.hist: deque[np.float32] = deque(maxlen=max(1, cfg.T_beta))

    def step(self, J: float) -> Tuple[np.float32, bool]:
        """
        Update internal β and compute corresponding damping (single precision).
        """
        J = np.float32(J)  # cast incoming scalar to float32

        worst_recent = np.max(self.hist) if self.hist else np.float32(np.inf)
        passed = (J <= worst_recent) or (self.beta <= np.float32(self.cfg.beta_min))

        if passed:
            self.beta = np.minimum(
                np.float32(self.cfg.beta_max),
                np.float32(self.cfg.G_pass) * self.beta,
            )
            self.hist.append(J)
            repeat = False
        else:
            self.beta = np.maximum(
                np.float32(self.cfg.beta_min),
                np.float32(self.cfg.G_fail) * self.beta,
            )
            repeat = True

        # Compute damping = 1 - β (in float32)
        damping = np.float32(1.0) - self.beta
        return damping, repeat

    def reset(self):
        self.beta = np.float32(self.cfg.beta_max)
        self.hist.clear()

    def __repr__(self) -> str:
        return (
            f"AdaptiveDamping(beta={float(self.beta):.4f}, "
            f"hist_len={len(self.hist)}, cfg={self.cfg})"
        )

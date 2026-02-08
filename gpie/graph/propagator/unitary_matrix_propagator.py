from typing import Optional
from .unitary_propagator import UnitaryPropagator
from ..wave import Wave
from ...core.backend import np, move_array_to_current_backend
from ...core.types import get_complex_dtype


class UnitaryMatrixPropagator(UnitaryPropagator):
    """
    Unitary propagator defined by a fixed unitary matrix U.

    Models:
        y = U @ x
        x = Uᴴ @ y

    Supports batch-wise operation:
        - U shape (N, N)  → broadcast to (B, N, N)
        - U shape (B, N, N) → batch-specific unitary
    """

    def __init__(
        self,
        U,
        *,
        precision_mode=None,
        dtype=np().complex64,
    ):
        super().__init__(dtype=dtype, precision_mode=precision_mode)

        if U is None:
            raise ValueError("Unitary matrix U must be provided.")

        U = np().asarray(U)
        if U.ndim == 2:
            self._needs_batch = True
            U = U[None, ...]              # (1, N, N)
        elif U.ndim == 3:
            self._needs_batch = False
        else:
            raise ValueError("U must be 2D or 3D array.")

        if U.shape[-1] != U.shape[-2]:
            raise ValueError("U must be square.")

        self.U = U.astype(dtype, copy=False)
        self.Uh = self.U.conj().transpose(0, 2, 1)

        self.event_shape = (U.shape[-1],)

    # ------------------------------------------------------------------
    # Backend handling
    # ------------------------------------------------------------------
    def to_backend(self):
        super().to_backend()
        self.U = move_array_to_current_backend(self.U, dtype=self.dtype)
        self.Uh = move_array_to_current_backend(self.Uh, dtype=self.dtype)
        self.dtype = np().dtype(self.dtype)

    # ------------------------------------------------------------------
    # Required by UnitaryPropagator
    # ------------------------------------------------------------------
    def _forward_array(self, x):
        """
        Apply y = U @ x.

        Args:
            x: ndarray of shape (B, N)
        Returns:
            ndarray of shape (B, N)
        """
        return (self.U @ x[..., None])[..., 0]

    def _backward_array(self, y):
        """
        Apply x = Uᴴ @ y.

        Args:
            y: ndarray of shape (B, N)
        Returns:
            ndarray of shape (B, N)
        """
        return (self.Uh @ y[..., None])[..., 0]

    # ------------------------------------------------------------------
    # Graph DSL integration
    # ------------------------------------------------------------------
    def __matmul__(self, wave: Wave) -> Wave:
        """
        Connect this propagator to a 1D Wave.
        """
        if len(wave.event_shape) != 1:
            raise ValueError(
                f"UnitaryMatrixPropagator expects 1D wave, got {wave.event_shape}"
            )

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)
        self.dtype = get_complex_dtype(wave.dtype)

        B = wave.batch_size
        N = wave.event_shape[0]

        # Expand U to batch if needed
        if self._needs_batch:
            self.U = np().broadcast_to(self.U, (B, N, N))
            self.Uh = self.U.conj().transpose(0, 2, 1)
            self._needs_batch = False

        if self.U.shape != (B, N, N):
            raise ValueError(
                f"U shape {self.U.shape} incompatible with wave (batch={B}, N={N})"
            )

        out = Wave(
            event_shape=wave.event_shape,
            batch_size=B,
            dtype=self.dtype,
        )
        out._set_generation(self._generation + 1)
        out.set_parent(self)
        self.output = out
        return out

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"UnitaryMatrixProp(gen={gen}, mode={self.precision_mode})"

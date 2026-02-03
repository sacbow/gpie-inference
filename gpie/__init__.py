__version__ = "0.3.0"

# core modules
from .core import (
    UncertainArray,
    mse,
    nmse,
    pmse,
    psnr,
    support_error,
    PrecisionMode,
    UnaryPropagatorPrecisionMode,
    BinaryPropagatorPrecisionMode,
)

from .core.linalg_utils import (
    random_normal_array,
    random_unitary_matrix,
    random_binary_mask,
    random_phase_mask,
    masked_random_array
)

# Graph structure and base components
from .graph.structure import Graph, model, observe
from .graph.wave import Wave
from .graph.factor import Factor

# Priors
from .graph.prior import (
    GaussianPrior,
    SparsePrior,
    SupportPrior
)

# Propagators
from .graph.propagator import (
    UnitaryMatrixPropagator,
    FFT2DPropagator,
    IFFT2DPropagator,
    PhaseMaskFFTPropagator,
    AddPropagator,
    MultiplyPropagator,
    AddConstPropagator,
    MultiplyConstPropagator
)

# Measurements
from .graph.measurement import (
    GaussianMeasurement,
    AmplitudeMeasurement,
)

# Shortcuts
from .graph.shortcuts import (
    fft2,
    ifft2,
    replicate
)

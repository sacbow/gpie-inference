# gPIE: Graph-based Probabilistic Inference Engine

[![Tests](https://github.com/sacbow/gpie-inference/actions/workflows/tests.yml/badge.svg)](https://github.com/sacbow/gpie-inference/actions/workflows/tests.yml)

[![codecov](https://codecov.io/gh/sacbow/gpie-inference/graph/badge.svg?token=OVKYM0YQZ4)](https://codecov.io/gh/sacbow/gpie-inference)

**gPIE** is a modular, extensible Python framework for structured probabilistic inference via **Expectation Propagation (EP)** on factor graphs, with applications to inverse problems in computational photonics.

## Quick Start: Defining an Inverse Problem & Running Approximate Bayes
In gPIE, a Bayesian inverse problem is defined by writing its forward model using a small domain-specific language (DSL) .

### Model Definition
Models are defined under the `@model` decorator:

```python
@model
def coded_diffraction_pattern(shape, n_measurements, phase_masks, noise):
    x = ~GaussianPrior(event_shape=shape, label="object", dtype=np.complex64)
    x_batch = replicate(x, batch_size=n_measurements)
    y = fft2(phase_masks * x_batch)
    AmplitudeMeasurement(var=noise) << y
```

- The model definition describes only the generative structure:
  - latent variables (`Wave`)
  - deterministic operators (e.g., FFT, addition, multiplication, replication of variable)
  - measurement likelihoods

- No data, or device are specified at this stage.

Calling the model function instantiates and compiles a factor graph:

```python
g = coded_diffraction_pattern(...)
```

### Injecting data
For generating synthetic data, one can set the ground truth by accessing the labeled latent variables.

```python
g.set_sample("object", complex_img.astype(np.complex64))
g.generate_observations(rng=np.random.default_rng(seed=999))
```

- `set_sample(label, value)` assigns ground truth to a latent variable.
- `generate_observations()` synthesizes noisy measurements from the forward model.
- Randomness for data generation is fully controlled via an explicit RNG.

### Running EP Inference
Inference is started by calling `g.run()`:

```python
g.set_init_rng(rng=np.random.default_rng(seed=111))

g.run(
    n_iter=400,
    device="cpu",   # or "cuda"
    callback=monitor,
)
```
- `set_init_rng()` controls random initialization of EP messages
- `device` specifies the execution backend (`cpu` / `cuda`)
- `callback(graph, iteration)` enables monitoring during inference

All arrays are transferred to the target device at the start of `run()`, and results are transferred back to CPU when inference finishes.

### Inspecting Posterior Estimates
After inference, posterior estimates are accessible via labeled variables:

```python
object_mean = g["object"]["mean"] #posterior mean
object_variance  = g["object"]["variance"] #posterior variance
```

## Advanced: Inference Schedules
gPIE supports both synchronous and asynchronous scheduling of message passing algorithm, when the model involves `replicate` or `extract_patches` operations (for e.g., in Coded Diffraction Pattern and Ptychography.)

### Parallel Schedule (Default)
```python
g.run(
    n_iter=400,
    device="cuda",
    schedule="parallel"
)
```
- All batch elements are updated simultaneously in each EP iteration
- Best choice for exploiting GPU acceleration

### Sequential Schedule (Block-wise EP)
```python
g.run(
    n_iter=400,
    device="cuda",
    schedule="sequential"
)
```
- When the observed data consists of several measurements, each data is visited sequentially to update posterior distribution.
- Often useful for stabilizing the convergence in challenging inverse problems such as phase retrieval.



## Installation

### Minimal installation
gPIE is currently distributed as a research prototype and is recommended to be installed from source.

```bash
git clone https://github.com/sacbow/gpie-inference
cd gpie-inference
pip install -e .
```

### Core dependencies
Core inference functionality depends only on [NumPy](https://numpy.org/).

- Python >= 3.9
- NumPy >= 2.2.6

Since gPIE returns posterior estimates as `numpy.ndarray` objects, users may freely choose any downstream visualization or analysis libraries compatible with NumPy.

### GPU acceleration
GPU execution is supported via [CuPy](https://cupy.dev/). Install a CuPy build compatible with your local CUDA toolkit, then run

```python
g.run(device="cuda") # default: device = "cpu"
```

### FFT engines
The default FFT engine of gPIE is [pocketfft](https://github.com/mreineck/pocketfft) (when `device = "cpu"`) and [cuFFT](https://developer.nvidia.com/cufft) (when `device = "cuda"`).

On CPU backend, users can install [pyfftw](https://pyfftw.readthedocs.io/en/latest/) and run

```python
g.run(fft_engine = "fftw")
```

[FFTW](https://www.fftw.org/) is an auto-tuning library that finds the best plan for computing FFT from candidate plans at runtime. 

### Graph visualization
Factor graph visualization is provided through:

```python
graph.visualize(layout=..., backend=...)
```

Layout engines:
- [networkx](https://networkx.org/en/) ("spring", "shell", "circular", "kamada_kawai")
- [pygraphviz](https://pygraphviz.github.io/) (Graphviz-based layout)

Rendering backend:
- [matplotlib](https://matplotlib.org/)
- [bokeh](https://bokeh.org/)

These libraries are optional and must be installed separately when using visualization API.


## Project Structure
```
gpie-inference/
├── CHANGELOG.md
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── requirements-optional.txt
│
├── gpie/                         # Core Python package (importable as `gpie`)
│   ├── __init__.py
│   ├── core/                     # Core numerical and EP utilities
│   │   ├── __init__.py
│   │   ├── backend.py            # NumPy / CuPy backend switching
│   │   ├── blocks.py             # Block-wise scheduling utilities
│   │   ├── fft.py                # FFT backend abstraction
│   │   ├── linalg_utils.py       # Linear algebra helpers
│   │   ├── metrics.py            # Error metrics (MSE, PMSE, PSNR, ...)
│   │   ├── rng_utils.py          # Random number generation utilities
│   │   ├── types.py              # Common type definitions
│   │   ├── adaptive_damping.py   # Adaptive damping controller
│   │   ├── accumulative_uncertain_array.py
│   │   └── uncertain_array/      # UncertainArray abstraction
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── ops.py
│   │       └── utils.py
│   │
│   ├── graph/                    # Factor graph and EP engine
│   │   ├── __init__.py
│   │   ├── wave.py               # Latent variable representation
│   │   ├── factor.py             # Factor base class
│   │   ├── shortcuts.py          # High-level DSL shortcuts (fft2, replicate, ...)
│   │   │
│   │   ├── prior/                # Prior factors
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── gaussian_prior.py
│   │   │   ├── sparse_prior.py
│   │   │   └── support_prior.py
│   │   │
│   │   ├── propagator/           # Deterministic forward operators
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── fft_2d_propagator.py
│   │   │   ├── ifft_2d_propagator.py
│   │   │   ├── phase_mask_fft_propagator.py
│   │   │   ├── fork_propagator.py
│   │   │   ├── slice_propagator.py
│   │   │   ├── zero_pad_propagator.py
│   │   │   ├── unitary_propagator.py
│   │   │   ├── unitary_matrix_propagator.py
│   │   │   ├── add_propagator.py
│   │   │   ├── add_const_propagator.py
│   │   │   ├── multiply_propagator.py
│   │   │   ├── multiply_const_propagator.py
│   │   │   ├── binary_propagator.py
│   │   │   └── accumulative_propagator.py
│   │   │
│   │   ├── measurement/          # Measurement likelihoods
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── gaussian_measurement.py
│   │   │   └── amplitude_measurement.py
│   │   │
│   │   └── structure/            # Graph structure, DSL, visualization
│   │       ├── __init__.py
│   │       ├── graph.py          # orchestrates inference
│   │       ├── wave_view.py      # help inspection of the graph state
│   │       ├── model.py          # decorator for constructing graph from Domain Specific Language(DSL)
│   │       ├── visualization.py
│   │       ├── _bokeh_vis.py
│   │       └── _matplotlib_vis.py
│
├── examples/                     # Numerical experiments and demos
│   ├── scripts/
│   ├── notebooks/
│   └── sample_data/
│
├── profile/                      # Profiling and performance benchmarks
│
├── test/                         # Unit and integration tests
│
└── gpie.egg-info/                # Package metadata (generated)

```


## Related libraries

### Message Passing Libraries
gPIE shares common ground with several existing frameworks for message passing inference:

#### [ForneyLab (Julia)](https://biaslab.github.io/project/forneylab/)
ForneyLab is a declarative probabilistic programming framework built around factor graphs, with a strong emphasis on flexibility in inference algorithm design.

- **Strength**: Supports multiple inference paradigms—including sum–product, expectation propagation, and variational Bayes—and allows users to explicitly choose and compare inference strategies using free-energy-based criteria.
- **Difference**: gPIE adopts a more constrained but scalable design: expectation propagation is used as the default inference mechanism, with variational updates introduced only where necessary for tractability. Rather than exposing algorithm selection as a primary user choice, gPIE focuses on stability and scalability through adaptive damping and scheduling.


#### [Tree-AMP (Python)](https://sphinxteam.github.io/tramp.docs/0.1/html/index.html)
Tree-AMP is a Python framework for approximate message passing algorithms, primarily intended as a platform for numerical experimentation and theoretical analysis of AMP.

- **Strength**: Well-suited for constructing and analyzing AMP-style algorithms, and provides dedicated support for theoretical tools such as state evolution and free entropy.
- **Difference**: In gPIE the choice of the exponential-family approximation for each variable is automatically determined by the compiler based on the surrounding factor graph, whereas in Tree-AMP this choice must be explicitly specified by the user.
Second, gPIE supports multiplication between variables through factor nodes implemented using variational message passing (VMP), enabling models that involve products of latent variables (for e.g., in blind ptychography). 

#### [Dimple (Java/Matlab)](https://github.com/analog-garage/dimple)
Dimple is an early and influential factor-graph-based inference system supporting both discrete and continuous variables, with a strong focus on large-scale discrete inference.

- **Strength**: Achieves scalability through dedicated accelerator (GP5).
- **Difference**: gPIE targets continuous-valued inference problems and GPU-accelerated scientific Python workflows, rather than discrete inference or hardware-specific execution.

#### [Infer.NET (C#)](https://dotnet.github.io/infer/)
Infer.NET is a mature probabilistic programming framework developed at Microsoft, supporting expectation propagation, variational message passing, and Gibbs sampling.

- **Strength**: Industrial-grade implementation with broad model expressiveness and extensive use in both academic and industrial settings.
- **Difference**: Infer.NET relies on a compilation-based execution model and requires a global choice of inference algorithm. In contrast, gPIE allows compositional combinations of message passing rules (EP and VMP) within a single model and emphasizes runtime control of scheduling.


### Inverse Probelem Solvers for imaging
Several modern libraries address inverse problems in computational imaging, founded in different algorithmic paradigms.

#### [ProxImaL (Python)](https://www.proximal-lang.org/en/latest/)
ProxImaL provides a domain-specific language for expressing inverse problems in a style similar to CVXPY.
It supports proximal splitting methods (e.g., ADMM, primal–dual algorithms) and includes GPU acceleration as well as proximal operators for phase retrieval and total variation regularization.

- **Strength**: Flexible DSL that allows users to construct objective functions that include imaging-specific proximal terms.
- **Difference**: ProxImaL formulates inverse problems as optimization problem, whereas gPIE adopts a probabilistic factor-graph formulation and constructs inference algorithms via expectation propagation.

#### [CUQIpy (Python)](https://cuqi-dtu.github.io/CUQIpy/)
CUQIpy follows a Bayesian approach and emphasizes sampling-based inference, particularly Markov chain Monte Carlo (MCMC), with automated selection of sampling strategies.

- **Strength**: Uncertainty quantification via posterior sampling, and mechanism for choosing efficient MCMC method from specified model.
- **Difference**: While CUQIPy can capture the correlation of the uncertainty in each pixel value, gPIE neglects the pixel-wise correlations to achieve scalability comparable with optimization methods.

#### [DeepInverse (Python / PyTorch)](https://deepinv.github.io/deepinv/)
DeepInverse unifies optimization-based, learning-based, and sampling-based approaches within a PyTorch ecosystem, with domain-specific physics modules for modalities such as MRI, tomography, and phase retrieval.

- **Strength**: Deep learning integration and rich ecosystem of physics operators.
- **Difference**: DeepInverse emphasizes an ecosystem of physics classes corresponding to particular imaging modalities. In contrast, gPIE adopts a compositional domain-specific language that allows users to construct forward models from primitive operations (e.g., FFT, slicing, multiplication, replication), independent of any fixed modality.


##  License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.


## Contact
For questions, please open an issue or contact:
- Hajime Ueda (ueda@mns.k.u-tokyo.ac.jp)


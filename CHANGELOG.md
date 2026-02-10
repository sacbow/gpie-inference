# Changelog

## [v0.3.1] — 2026-02-10

### Added
- Graph inspection API
  - Latent variables (Wave) can now be inspected directly via `graph[label]`
    - e.g. posterior mean: g["object"]["mean"]
    - posterior variance: g["object"]["variance"]
  - Improves usability for monitoring, debugging, and post-processing inference results

- Synthetic data generation API
  - Added `Graph.generate_observations()`
  - Randomness for data generation is fully controlled via a user-supplied RNG

### Changed
- Unified device policy for inference sessions
  - Accelerator devices (e.g. CUDA) are now used only inside an inference session.
  - Outside of run(), all graph objects and arrays remain CPU-resident

## [v0.3.0] — 2026-01-14

### Added
- **Explicit and flexible message scheduling for Expectation Propagation (EP)**
  - Parallel (Jacobi-style) scheduling
  - Sequential (Gauss–Seidel-style) scheduling
  - Block-wise asynchronous scheduling
  - Scheduling can be selected *at runtime* without modifying model definitions

- **Block-wise scheduling utilities**
  - New scheduling logic in `gpie/core/blocks.py`
  - Enables asynchronous and partially-updated EP execution

- **Integration tests for scheduling behavior**
  - Added end-to-end reconstruction tests covering parallel and sequential schedules
  - Improved coverage of graph execution, propagators, and measurement updates

### Changed
- **Project structure cleanup**
  - Imaging-specific utilities (e.g., ptychography datasets and simulators) removed from the core package
  - Core `gpie` package now focuses exclusively on generic EP inference infrastructure

- **Examples and benchmarks updated**
  - Benchmark scripts extended to compare scheduling strategies

### Improved
- Test coverage increased to ~91%, including integration-level EP convergence tests

### Notes
- This release represents a **major architectural update** rather than a change in inference algorithms.
- While the underlying EP / AMP-style updates remain similar to previous versions, the new scheduling framework enables systematic experimentation with synchronous and asynchronous execution strategies.
- Imaging-specific extensions (e.g., ptychography datasets and simulators) are now developed in separate repositories to keep the core engine lightweight and modular.



## [v0.2.2] — 2025-11-06

### Added 
- Adaptive Damping Mechanism (`gpie/core/adaptive_damping.py`)
Introduced an AD-GAMP–like adaptive damping controller for Expectation Propagation (EP).
  - `AmplitudeMeasurement`: fitness-based auto-tuning (`damping="auto"`)
  - `SparsePrior`: log-evidence (logZ)–based auto-tuning (`damping="auto"`)
The damping parameter now self-adjusts during inference, removing the need for manual tuning.

- Multiple Initialization Strategies
Priors now support multiple message initialization schemes, from non-informative random initialization to use-specified initial value.

### Changed
- All benchmark scripts and notebooks in `examples/` now use adaptive damping as the default configuration instead of manually specified damping values.

### Notes
- The default adaptive damping configuration has been hand-tuned by the developer to ensure stable convergence across all benchmark tasks,
including ptychography, holography, coded diffraction pattern, layered optics, and compressive sensing.


## [v0.2.1] — 2025-10-26

### Added
- New demo: `examples/scripts/blind_ptychography_with_phase.py`  
  Demonstrates bilinear inference (object × probe) under phase observation.

### Improved
- `MultiplyPropagator`: introduced inner-loop variational updates for more stable VMP convergence.

### Notes
This release improves numerical stability for bilinear models and adds a reproducible example for hybrid EP/VMP inference.



## [0.2.0] - 2025-10-10
### Added
- Ptychography support via new factor-graph modules:

  - PtychographyDataset: unified container for object, probe, and diffraction data.

  - SlicePropagator + AccumulativeUncertainArray: A syntax to describe the physical model of ptychography.

- Example script:
  - examples/notebook/ptychography_demo.ipynb — An introduction to ptychographic phase retrieval via gPIE

  - examples/scripts/ptychography/ptychography.py — complete forward and reconstruction workflow.

### Note
- Ptychographic reconstruction via gPIE is seen as an implementation of the Ptycho-EP algorithm proposed in our paper **Ueda, H., Katakami, S., & Okada, M. (2025). A Message-Passing Perspective on Ptychographic Phase Retrieval** on [Arxiv](https://arxiv.org/abs/2504.05668).


## [0.1.2] - 2025-10-01
### Added
- New propagators:
  - **ForkPropagator**: replicate input waves across batch dimension.
  - **SlicePropagator**: extract fixed-size patches (for ptychography, etc.).
  - **ZeroPadPropagator**: apply zero-padding.
- `Wave` class convenience methods:
  - `Wave.extract_patches()` and `Wave.__getitem__` for intuitive slicing.
  - `Wave.zero_pad()` for easy zero-padding in models.


### Changed
- Improved coverage of `Wave` error handling, message passing, and sampling logic.
- Refactored tests for slice and zero-padding propagators to ensure >90% coverage.

### Notes
- Reported test coverage may differ between environments:
  - On local machines with **CuPy, pygraphviz, and bokeh** installed, both all backends are tested, yielding >91% coverage.
  - On CI environments without CuPy, coverage may appear slightly lower.



## [0.1.1] - 2025-09-25
### Added
- CuPy backend now uses `CuPyFFTBackend` with plan caching (faster FFT on GPU).
- Benchmark scripts now support `--fftw`, `--threads`, and `--planner-effort` options.
- Profiling insights added for Holography, Random CDI, and CDP (1024×1024 scale).

### Changed
- `set_backend()` now automatically chooses the right FFT backend (NumPy → DefaultFFT, CuPy → CuPyFFT).
- Default FFTW planner effort changed from `FFTW_MEASURE` to `FFTW_ESTIMATE` for faster startup.

### Fixed
- Minor docstring improvements.
- Profiling README clarified.


---

## [0.1.0] - 2025-09-21
### Added
- `@model` syntax: drastically simplified model description.

### Changed
- `UncertainArray` now retains a batch of arrays.
- `UncertainArrayTensor` class removed (merged into `UncertainArray`).
- Refactored dtype management in Measurement classes.

---

## [0.0.0] - 2025-09-01
### Added
- Initial public release.

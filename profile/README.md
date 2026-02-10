# gPIE Profiling & Benchmarking

This directory contains **iteration-level benchmarks and profiling utilities** for evaluating the computational performance of **gPIE**.

All benchmarks are designed to measure the **core EP iteration cost**, separating it from one-time overheads such as graph construction, data generation, and device transfer.[

---

## File Structure
```
profile/
├─ benchmark_utils.py # Timing and cProfile utilities
├─ benchmark_holography.py # Holography benchmark
├─ benchmark_random_cdi.py # Random CDI benchmark
├─ benchmark_coded_diffraction_pattern.py # CDP benchmark
└─ README.md # This file                               
```

---

## Benchmarking Philosophy

All benchmarks in this directory follow the same principles:

- **Iteration-level timing**
  - Only the body of one EP iteration is measured
  - Warm-up, device transfer, FFT plan creation, and RNG setup are excluded
- **Unified execution path**
  - All benchmarks use `Graph.run(..., iteration_hook=...)`
- **Reproducibility**
  - `g.set_init_rng(np.random.default_rng(...))` is set immediately before `run()`
- **Device-agnostic design**
  - CPU / GPU behavior is controlled via `--device`
  - FFT backend is inferred automatically unless explicitly specified

This makes the results comparable across models, devices, and schedules.

---

## Target Models

We benchmark the following representative computational imaging models:

1. **Holography**
   - Serves as a minimal baseline (1 forward + 1 backward FFT per iteration)

2. **Random Structured Matrix CDI**
   - Multi-layer CDI with sequential FFT and random mask layers

3. **Coded Diffraction Pattern (CDP)**
   - Highlights differences between `parallel` and `sequential` schedules

---

## How to Run Benchmarks

### Common Options

All benchmark scripts support the following core options:

- `--device {cpu,cuda}`
- `--size <int>` (image size, e.g. 128 / 256 / 512 / 1024)
- `--n-iter <int>` (number of EP iterations)
- `--schedule {parallel,sequential}`

FFT backend selection:

- If `--fft-engine` is **omitted**:
  - `cpu` → NumPy FFT
  - `cuda` → CuPy FFT
- Explicit selection is possible via `--fft-engine {numpy,cupy,fftw}`

---

### CPU Benchmark

```bash
python profile/benchmark_holography.py --device cpu
python profile/benchmark_random_cdi.py --device cpu --fft-engine fftw --layers 2
python profile/benchmark_coded_diffraction_pattern.py --device cpu --schedule sequential
```

### GPU (CUDA / CuPy) Benchmark

```bash
python profile/benchmark_holography.py --device cuda
python profile/benchmark_random_cdi.py --device cuda
python profile/benchmark_coded_diffraction_pattern.py --device cuda 
```

### Iteration-Level Profiling (cProfile)

```bash
python profile/benchmark_coded_diffraction_pattern.py  --device cuda --profile 
```

## Benchmark Environment of the developer

- **OS:** Windows 11 Home, ver. 24H2, OS build 26100 4652
- **CPU:** Intel Core i7-14650K (16 cores, 24 threads)  
- **RAM:** 32 GB 
- **GPU:** NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)  
- **NVIDIA Driver:** 576.02
- **CUDA Toolkit:** 12.9
- **Python:** 3.10.5 (venv)  
- **Libraries:**
  - NumPy: 2.2.6
  - CuPy: 13.5.1

- **Note**: Results are device-dependent and may vary on different hardware or driver configurations.

##  Benchmark Results (512 x 512 pixels, per-iteration time)

| Model                  | NumPy (default FFT) | NumPy + FFTW       | CuPy (GPU)       |
|------------------------ |------------------- |------------------------ |--------------------|
| **Holography**          | 26-30 ms             | 24-28 ms                  | 2-8 ms             |
| **Random CDI (2 layers)** | 60-68 ms           | 54-64 ms                  | 4-10 ms            |
| **CDP (4 masks, parallel)** | 160-190 ms            | 140-170 ms                   | 4-7 ms              |
| **CDP (4 masks, sequential)** | 210-280 ms            | 200-220 ms                   | 15-33 ms              |

## Profiling insights
- Across all benchmarks (Holography, Random CDI, CDP), FFT accounts for ~20–25% of per-iteration time on CPU.
- Together, FFT + Uncertain Array algebra (`UA.__mul__` and `UA.__truediv__`) consistently explain ~2/3 of total iteration time.
- These operations are data-parallel and benefit directly from GPU acceleration.
- In CDP on CPU, the difference between parallel and sequential schedules comes from incremental updates in `fork propagator`in sequential mode.
- In CDP on GPU (CUDA), sequential scheduling is ~batch_size times slower than parallel, indicating that the runtime is dominated by kernel launch overhead.



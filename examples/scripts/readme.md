# Examples

This directory contains example experiments using the **gPIE** framework.

Each subdirectory corresponds to a different inverse problem or imaging model, implemented with Expectation Propagation (EP) on a factor graph.

---

## 1. Holography

Inline holography is modeled using:

- **Object**: Complex-valued image (ground truth or synthetic)
- **Reference wave**: Known reference pattern
- **Model**: Interference + FFT + amplitude detection
- **Inference**: EP-based recovery from amplitude-only measurements

### 🔧 Script

```bash
python examples/scripts/holography/holography.py --obj-img camera --ref-img moon --obj-radius 0.15 --ref-radius 0.2 --n-iter 100 --save-graph
```

| Option         | Description                                         |
|----------------|-----------------------------------------------------|
| `--obj-img`    | Object image name from `skimage.data`              |
| `--ref-img`    | Reference image (optional)                         |
| `--obj-radius` | Support mask radius for object (if not using image)|
| `--ref-radius` | Support mask radius for reference wave             |
| `--n-iter`     | Number of EP iterations                            |
| `--save-graph` | Save factor graph as HTML                          |

### 💾 Outputs (`examples/scripts/holography/results/`)

- `true_amp.png`, `true_phase.png`: Ground truth
- `reconstructed_amp.png`, `reconstructed_phase.png`: Estimate
- `ref_amp.png`, `ref_phase.png`: Reference wave
- `convergence.png`: MSE plot
- `graph.html`: Factor graph (optional)

---

## 2. Coded Diffraction Pattern (CDP)

CDP performs phase retrieval from multiple masked FFT amplitude observations.

- Random phase masks modulate input
- FFT applied before measurement
- Ground truth: camera (amplitude), moon (phase)

### 🔧 Script

```bash
python examples/scripts/coded_diffraction_pattern/coded_diffraction_pattern.py --n-iter 300 --size 256 --measurements 4 --save-graph
```

| Option           | Description                                      |
|------------------|--------------------------------------------------|
| `--n-iter`       | Number of EP iterations                          |
| `--size`         | Image shape (H=W)                                |
| `--measurements` | Number of phase masks                            |
| `--save-graph`   | Save factor graph visualization                  |

### 💾 Outputs (`examples/scripts/coded_diffraction_pattern/results/`)

- `true_amp.png`, `true_phase.png`
- `reconstructed_amp.png`, `reconstructed_phase.png`
- `convergence.png`
- `graph.html`

### 🔬 Scheduling Benchmark (Sequential vs Parallel)

In addition to single-run reconstructions, this example includes a benchmark comparing **sequential** and **parallel** scheduling strategies in EP.

The benchmark repeatedly solves the same CDP instance with different random seeds and visualizes the distribution of reconstruction error over iterations.

#### 🔧 Script

```bash
python examples/scripts/coded_diffraction_pattern/sequential_vs_parallel.py --trials 10 --measurements 3 --size 64 --n-iter 400
```


### 📖 Reference

Candès, E. J., Li, X., & Soltanolkotabi, M. (2015).  
**Phase retrieval from coded diffraction patterns**, *ACHA* 39(2), 277–299.  
[DOI](https://doi.org/10.1016/j.acha.2014.09.004)

---

## 3. Random Structured CDI

Structured CDI applies multiple phase layers before amplitude detection.

- Complex sample with known support
- Forward model: Layered phase mask → FFT → amplitude

### 🔧 Script

```bash
python examples/scripts/random_structured_cdi/random_structured_cdi.py --n-iter 300 --size 256 --layers 2 --support-radius 0.3 --save-graph
```

| Option         | Description                                      |
|----------------|--------------------------------------------------|
| `--n-iter`     | Number of EP iterations                          |
| `--size`       | Image shape                                      |
| `--layers`     | Number of phase masks                            |
| `--radius`     | Support mask radius                              |
| `--save-graph` | Save graph as HTML                               |

### 💾 Outputs (`examples/scripts/random_structured_cdi/results/`)

- `true_amp.png`, `true_phase.png`
- `reconstructed_amp.png`, `reconstructed_phase.png`
- `convergence.png`
- `graph.html`

### 📖 Reference

Hu, Z., Tachella, J., Unser, M., Dong, J. (2025).  
**Structured Random Model for Fast and Robust Phase Retrieval**, *ICASSP 2025*.

---

## 4. Compressed Sensing

Compressed sensing using sparsity in Fourier domain.

- SparsePrior → FFT → subsampled measurement (Gaussian)
- Sample is pre-sparsified by retaining top-ρ values

### 🔧 Script

```bash
python examples/scripts/compressed_sensing/compressed_sensing.py --n-iter 100 --rho 0.1 --subsample-rate 0.3 --size 256 --image camera --save-graph
```

| Option              | Description                                   |
|---------------------|-----------------------------------------------|
| `--n-iter`          | Number of EP iterations                       |
| `--rho`             | Sparsity level (fraction of retained pixels)  |
| `--subsample-rate`  | Fraction of Fourier coefficients observed     |
| `--size`            | Image size                                    |
| `--image`           | Image name (`camera`, `coins`, etc.)          |
| `--save-graph`      | Save graph HTML                               |

### 💾 Outputs (`examples/scripts/compressed_sensing/results/`)

- `true_sparse.png`
- `reconstructed.png`
- `convergence.png`
- `graph.html`

## 📁 Data

Sample images are downloaded via `skimage.data` into:

```
examples/sample_data/
```

---

## ⚙️ Notes

- Scripts assume NumPy backend
- CuPy support is available but not enabled by default
- Graphs saved using `Bokeh` + `graphviz`
- Figures saved with `matplotlib.pyplot.imsave()`
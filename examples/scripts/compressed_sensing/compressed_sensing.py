import argparse
import numpy as np
import matplotlib.pyplot as plt

from gpie import model, SparsePrior, GaussianMeasurement, fft2, mse
from gpie.core.linalg_utils import random_binary_mask
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from io_utils import load_sample_image

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def sparsify_image(img: np.ndarray, rho: float) -> np.ndarray:
    """
    Sparsify an image by keeping only the top rho fraction of absolute pixel values.
    """
    flat = img.flatten()
    k = max(1, int(rho * flat.size))
    threshold = np.partition(np.abs(flat), -k)[-k]
    sparse_img = np.where(np.abs(img) >= threshold, img, 0.0)
    return sparse_img

def build_compressed_sensing_graph(shape, rho=0.1, var=1e-4, subsample_ratio=0.3, image_name="camera"):
    rng = np.random.default_rng(42)

    # Load and sparsify image
    img = load_sample_image(image_name, shape=shape)
    threshold = np.percentile(img, 100 * (1 - rho))
    sparse_img = np.where(img >= threshold, img, 0).astype(np.complex64)

    # Binary mask for GaussianMeasurement
    mask = random_binary_mask(shape, subsampling_rate=subsample_ratio, rng=rng)

    # Define graph
    @model
    def compressed_sensing(rho, shape, var, mask):
        x = ~SparsePrior(rho=rho, event_shape=shape, label="x", dtype=np.complex64)
        GaussianMeasurement(var=var, with_mask = True) << fft2(x)

    g = compressed_sensing(rho = rho, shape = shape, var = var, mask = mask)

    g.set_sample("x", sparse_img)
    g.generate_observations(rng=np.random.default_rng(seed=9), mask = mask)
    return g, sparse_img


def run_cs(n_iter=100, rho=0.1, size=512, subsample_rate=0.3, image_name="camera", save_graph=False):
    g, true_img = build_compressed_sensing_graph(shape=(size,size), rho=rho, subsample_ratio=subsample_rate, image_name=image_name)
    true_x = g.get_wave("x").get_sample()

    mse_list = []

    def monitor(graph, t):
        if t % 10 == 0 or t == n_iter - 1:
            est = graph["x"]["mean"]
            gt = graph["x"]["sample"]
            err = mse(est, gt)
            mse_list.append(err)
            print(f"[t={t}] MSE = {err:.5e}")

    g.set_init_rng(np.random.default_rng(seed=1))
    g.run(n_iter=n_iter, callback=monitor)

    est_x = g["x"]["mean"][0]
    est_img = est_x.reshape((size, size)).real

    # Save images
    plt.imsave(f"{RESULTS_DIR}/true_sparse.png", true_img.real, cmap="gray")
    plt.imsave(f"{RESULTS_DIR}/reconstructed.png", est_img, cmap="gray")

    # Save convergence
    plt.figure()
    plt.plot(np.arange(0, len(mse_list) * 10, 10), mse_list, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.grid(True)
    plt.title("Convergence Curve")
    plt.savefig(f"{RESULTS_DIR}/convergence.png")
    plt.close()

    if save_graph:
        print(f"Saving factor graph visualization to {RESULTS_DIR}/graph.html ...")
        g.visualize(output_path=os.path.join(RESULTS_DIR, "graph.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compressed Sensing demo using Fourier transform")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of EP iterations")
    parser.add_argument("--rho", type=float, default=0.1, help="Sparsity level")
    parser.add_argument("--subsample-rate", type=float, default=0.3, help="Fraction of observed Fourier coefficients")
    parser.add_argument("--size", type=int, default=512, help="Image size")
    parser.add_argument("--image", type=str, default="camera", help="Image name from skimage.data")
    parser.add_argument("--save-graph", action="store_true", help="Save factor graph visualization")

    args = parser.parse_args()
    run_cs(n_iter=args.n_iter,
           rho=args.rho,
           size=args.size,
           subsample_rate=args.subsample_rate,
           image_name=args.image,
           save_graph=args.save_graph)

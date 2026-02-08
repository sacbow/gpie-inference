import argparse
import numpy as np
import matplotlib.pyplot as plt

from gpie import model, SupportPrior, fft2, AmplitudeMeasurement, mse
from gpie.core.linalg_utils import circular_aperture, masked_random_array

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from io_utils import load_sample_image


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


@model
def holography(support, var, ref_wave):
    """
    A model for inline holography using EP.
    """
    obj = ~SupportPrior(support=support, label="obj", dtype=np.complex64)
    AmplitudeMeasurement(var=var) << (fft2(ref_wave + obj))


def build_holography_graph(H=512, W=512, noise=1e-4,
                            obj_image=None, ref_image=None,
                            obj_radius=0.2, ref_radius=0.2):
    """
    Construct the holography graph from either real image or random data.
    """
    # ref image
    support_x = circular_aperture((H, W), radius=ref_radius, center=(-0.2, -0.2))
    amp_x = load_sample_image(ref_image, shape=(H, W))
    data_x = amp_x.astype(np.complex64) * support_x

    # image to estimate
    support_y = circular_aperture((H, W), radius=obj_radius, center=(0.2, 0.2))
    amp_y = load_sample_image(obj_image, shape=(H, W))
    data_y = amp_y.astype(np.complex64) * support_y

    # Construct graph
    g = holography(support = support_y, var = noise, ref_wave = data_x)

    g.set_sample("obj", data_y)  # Inject ground truth
    g.generate_observations(rng=np.random.default_rng(9)) #RNG for random noise
    return g, data_x, data_y


def run_holography(n_iter=100, obj_image=None, ref_image=None,
                   obj_radius=0.2, ref_radius=0.2, save_graph=False):

    """
    Run EP inference and save result images and convergence curve.
    """
    g, ref_sample, true_obj = build_holography_graph(obj_image=obj_image, ref_image=ref_image, obj_radius=obj_radius, ref_radius=ref_radius)
    pse_list = []

    def monitor(graph, t):
        if t % 10 == 0 or t == n_iter - 1:
            est = graph["obj"]["mean"] #posterior mean at iteration t
            gt = graph["obj"]["sample"] #ground truth
            err = mse(est, gt)
            pse_list.append(err)
            print(f"[t={t}] PSE = {err:.5e}")

    g.set_init_rng(np.random.default_rng(11)) #RNG for message initialization
    g.run(n_iter=n_iter, callback=monitor)

    # Save output images
    est = g["obj"]["mean"][0]
    amp = np.abs(est)
    phase = np.angle(est) * (np.abs(true_obj) > 1e-5)

    # Reconstructed images
    plt.imsave(f"{RESULTS_DIR}/reconstructed_amp.png", amp, cmap="gray")
    plt.imsave(f"{RESULTS_DIR}/reconstructed_phase.png", phase, cmap="twilight")

    # Ground truth (obj)
    true_amp = np.abs(true_obj)
    true_phase = np.angle(true_obj) * (true_amp > 1e-5)

    plt.imsave(f"{RESULTS_DIR}/true_amp.png", true_amp, cmap="gray")
    plt.imsave(f"{RESULTS_DIR}/true_phase.png", true_phase, cmap="twilight")

    # Reference wave (ref_wave)

    if ref_sample is not None:
        ref_amp = np.abs(ref_sample)
        ref_phase = np.angle(ref_sample) * (ref_amp > 1e-5)

        plt.imsave(f"{RESULTS_DIR}/ref_amp.png", ref_amp, cmap="gray")
        plt.imsave(f"{RESULTS_DIR}/ref_phase.png", ref_phase, cmap="twilight")

    # Convergence curve
    plt.figure()
    plt.plot(np.arange(0, len(pse_list) * 5, 5), pse_list, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("PSE (MSE)")
    plt.yscale('log')
    plt.title("Convergence Curve")
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/convergence.png")
    plt.close()

    if save_graph:
        print(f"Saving factor graph visualization to {RESULTS_DIR}/graph.html ...")
        g.visualize(output_path=os.path.join(RESULTS_DIR, "graph.html"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inline holography EP experiment")
    parser.add_argument("--obj-img", type=str, default="camera", help="Name of object image from skimage.data")
    parser.add_argument("--ref-img", type=str, default="camera", help="Name of reference image from skimage.data")
    parser.add_argument("--obj-radius", type=float, default=0.1, help="Radius of object support (if not using image)")
    parser.add_argument("--ref-radius", type=float, default=0.1, help="Radius of reference support (if not using image)")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of EP iterations")
    parser.add_argument("--save-graph", action="store_true", help="Save factor graph visualization as HTML")


    args = parser.parse_args()
    run_holography(n_iter=args.n_iter,
               obj_image=args.obj_img,
               ref_image=args.ref_img,
               obj_radius=args.obj_radius,
               ref_radius=args.ref_radius,
               save_graph=args.save_graph)

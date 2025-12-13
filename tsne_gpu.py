#tsne_gpu.py
"""
t-SNE visualization tools for comparing BP vs DFA neural network representations when on GPU

Example usage:
    python tsne_gpu.py --run_dir final_results/Implementation_final

This script expects a run directory created by experiment_GPU.py that contains:
    - config.json        (with "layer_sizes" and optionally "feedback_scale")
    - bp_final.npz       (BP weights: W0, b0, W1, b1, ...)
    - dfa_final.npz      (DFA weights: W0, b0, W1, b1, ...)

It will:
    - Rebuild BP and DFA networks (torch-based, GPU if available)
    - Load the saved weights into them
    - Run forward passes on MNIST to collect hidden activations
    - Run t-SNE (on CPU via scikit-learn) on each hidden layer
    - Produce a 2×(#hidden layers) grid comparing BP vs DFA representations
"""

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch

from src.bp_GPU import BPNetwork
from src.dfa_GPU import DFANetwork
from data.data import load_mnist  


# ---------------------------------------------------------------------
# 1. Utilities for loading weights into torch-based networks
# ---------------------------------------------------------------------
def load_weights_into(net, path, device: torch.device):
    """
    Load weights and biases from an .npz checkpoint into a BP/ DFA network.

    Expects keys: W0, b0, W1, b1, ...

    This assumes the network uses torch.Tensors for W[i], b[i].
    We convert from numpy -> torch on the given device.
    """
    data = np.load(path)

    for i in range(len(net.W)):
        W_arr = data[f"W{i}"]  # (fan_out, fan_in)
        b_arr = data[f"b{i}"]  # (fan_out,)

        # Infer dtype from existing tensors if possible, else default to float32
        if isinstance(net.W[i], torch.Tensor):
            dtype_W = net.W[i].dtype
        else:
            dtype_W = torch.float32
        if isinstance(net.b[i], torch.Tensor):
            dtype_b = net.b[i].dtype
        else:
            dtype_b = torch.float32

        net.W[i] = torch.from_numpy(W_arr).to(device=device, dtype=dtype_W)
        net.b[i] = torch.from_numpy(b_arr).to(device=device, dtype=dtype_b)


# ---------------------------------------------------------------------
# 2. Activation extraction (torch networks, GPU → CPU → NumPy)
# ---------------------------------------------------------------------
def get_layer_activations(net, X, layer_index: int, batch_size: int = 256):
    """
    Compute activations h^{layer_index} for all samples in X.

    Args:
        net: BPNetwork or DFANetwork (torch-based version)
        X: (N, input_dim) NumPy array of inputs
        layer_index: 0 = input, 1 = first hidden, ..., L = output
        batch_size: minibatch size for forward passes

    Returns:
        activations: (N, layer_sizes[layer_index]) as a NumPy array
    """
    activations = []
    N = X.shape[0]

    # Decide which device this net is on
    if hasattr(net, "device"):
        device = net.device
    else:
        # Fallback: take device from first W tensor
        device = net.W[0].device if isinstance(net.W[0], torch.Tensor) else torch.device("cpu")

    for start in range(0, N, batch_size):
        batch = X[start:start + batch_size].astype(np.float32)
        x_batch = torch.from_numpy(batch).to(device)

        with torch.no_grad():
            _ = net.forward(x_batch)              # net.h[layer_index] is torch.Tensor
            h_l = net.h[layer_index].detach().cpu().numpy()

        activations.append(h_l)

    return np.vstack(activations)


# ---------------------------------------------------------------------
# 3. t-SNE wrapper (CPU via scikit-learn)
# ---------------------------------------------------------------------
def tsne_embed(X, random_state: int = 0):
    """
    Run 2D t-SNE on X.

    Args:
        X: (N, D) NumPy array

    Returns:
        (N, 2) NumPy array of embeddings
    """
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="pca",
        random_state=random_state,
    )
    return tsne.fit_transform(X)


# ---------------------------------------------------------------------
# 4. Main plotting function
# ---------------------------------------------------------------------
def plot_tsne_grid(run_dir, n_samples=2000, use_test=True, random_state=0):
    """
    Make a BP vs DFA t-SNE grid for each hidden layer and save to tsne_layers.png
    inside run_dir.
    """
    # --------------------
    # 1. Load config
    # --------------------
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    layer_sizes = config["layer_sizes"]
    feedback_scale = config.get("feedback_scale", 0.1)
    L = len(layer_sizes) - 1
    n_hidden = L - 1

    # --------------------
    # 2. Rebuild nets on GPU (if available) & load weights
    # --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[t-SNE] Using device: {device}")

    # learning_rate=0.0 because we are not training in this script
    bp_net = BPNetwork(layer_sizes, learning_rate=0.0)
    dfa_net = DFANetwork(
        layer_sizes,
        learning_rate=0.0,
        feedback_scale=feedback_scale,
    )

    # If your BPNetwork / DFANetwork classes track a `device` attribute, set it
    if hasattr(bp_net, "device"):
        bp_net.device = device
    if hasattr(dfa_net, "device"):
        dfa_net.device = device

    bp_ckpt = os.path.join(run_dir, "bp_final.npz")
    dfa_ckpt = os.path.join(run_dir, "dfa_final.npz")

    load_weights_into(bp_net, bp_ckpt, device=device)
    load_weights_into(dfa_net, dfa_ckpt, device=device)

    # --------------------
    # 3. Load data & select subset (NumPy)
    # --------------------
    (X_train, y_train), (X_test, y_test) = load_mnist()

    if use_test:
        X, y = X_test, y_test
        split_name = "test"
    else:
        X, y = X_train, y_train
        split_name = "train"

    n_samples = min(n_samples, X.shape[0])
    X = X[:n_samples].astype(np.float32)
    y = y[:n_samples]
    labels = np.argmax(y, axis=1)

    print(f"[t-SNE] Using {n_samples} samples from {split_name} set.")

    # --------------------
    # 4. Run t-SNE per hidden layer
    # --------------------
    fig, axes = plt.subplots(2, n_hidden, figsize=(4 * n_hidden, 8))

    # Ensure axes is 2D even if n_hidden == 1
    if n_hidden == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for li in range(1, L):  # hidden layers 1..L-1
        print(f"[t-SNE] Processing hidden layer {li}/{n_hidden}")

        # Shape: (N, dim_l)
        h_bp = get_layer_activations(bp_net, X, layer_index=li)
        h_dfa = get_layer_activations(dfa_net, X, layer_index=li)

        emb_bp = tsne_embed(h_bp, random_state=random_state)
        emb_dfa = tsne_embed(h_dfa, random_state=random_state)

        ax_bp = axes[0, li - 1]
        ax_dfa = axes[1, li - 1]

        ax_bp.scatter(
            emb_bp[:, 0],
            emb_bp[:, 1],
            c=labels,
            cmap="tab10",
            s=3,
            alpha=0.7,
        )
        ax_dfa.scatter(
            emb_dfa[:, 0],
            emb_dfa[:, 1],
            c=labels,
            cmap="tab10",
            s=3,
            alpha=0.7,
        )

        ax_bp.set_title(f"BP – layer {li}")
        ax_dfa.set_title(f"DFA – layer {li}")

        for ax in (ax_bp, ax_dfa):
            ax.set_xticks([])
            ax.set_yticks([])

    run_name = os.path.basename(run_dir.rstrip("/"))
    fig.suptitle(f"t-SNE of hidden layers\n{run_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(run_dir, "tsne_layers_gpu.png")
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"[t-SNE] Saved grid to {out_path}")


# ---------------------------------------------------------------------
# 5. CLI entrypoint
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization for BP vs DFA runs")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to a results/<timestamp>_<run_name> directory",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of samples from train/test set to use for t-SNE",
    )
    parser.add_argument(
        "--use_train",
        action="store_true",
        help="Use training set instead of test set for t-SNE",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for t-SNE",
    )
    args = parser.parse_args()

    use_test = not args.use_train

    plot_tsne_grid(
        run_dir=args.run_dir,
        n_samples=args.n_samples,
        use_test=use_test,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()

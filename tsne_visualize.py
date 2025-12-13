# tsne_visualize.py
"""
t-SNE visualization tools for comparing BP vs DFA neural network representations.

How to run in terminal: 
    python tsne_visualize.py --run_dir final_results/Implementation_final

This module:
- Loads trained BP and DFA networks from checkpoint files
- Extracts hidden layer activations from MNIST data
- Applies t-SNE to visualize layer representations
- Produces a BP vs DFA comparison grid per hidden layer

Compatible with the GPU-based BP / DFA implementation + experiment_GPU.py outputs.
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
# 1. Utilities for loading weights into (possibly torch-based) networks
# ---------------------------------------------------------------------
def load_weights_into(net, path, device: str = "cpu"):
    """
    Load weights and biases from an .npz checkpoint into a BP/ DFA network.

    Expects keys W0, b0, W1, b1, ...

    Works for:
      - NumPy-based networks (W[i], b[i] are np.ndarray)
      - Torch-based networks (W[i], b[i] are torch.Tensor)

    For torch nets, weights are moved to `device`.
    """
    data = np.load(path)

    for i in range(len(net.W)):
        W_arr = data[f"W{i}"]
        b_arr = data[f"b{i}"]

        # Torch-based network
        if isinstance(net.W[i], torch.Tensor):
            # Pick device from attribute if present, otherwise use argument
            if hasattr(net, "device"):
                dev = net.device
            else:
                dev = device
            net.W[i] = torch.from_numpy(W_arr).to(dev, dtype=net.W[i].dtype)
            net.b[i] = torch.from_numpy(b_arr).to(dev, dtype=net.b[i].dtype)
        else:
            # NumPy-based network
            net.W[i] = W_arr
            net.b[i] = b_arr


# ---------------------------------------------------------------------
# 2. Activation extraction (works for NumPy or torch networks)
# ---------------------------------------------------------------------
def get_layer_activations(net, X, layer_index: int, batch_size: int = 256):
    """
    Compute activations h^{layer_index} for all samples in X.

    Args:
        net: BPNetwork or DFANetwork (NumPy or torch version)
        X: (N, input_dim) NumPy array
        layer_index: 0 = input, 1 = first hidden, ..., L = output
        batch_size: minibatch size for forward passes

    Returns:
        activations: (N, layer_sizes[layer_index]) as NumPy array
    """
    activations = []
    N = X.shape[0]

    # Heuristic: is this a torch-based network?
    is_torch_net = isinstance(net.W[0], torch.Tensor)

    # If it's torch-based, decide which device to use
    if is_torch_net:
        device = net.device if hasattr(net, "device") else net.W[0].device
        net_device = device
    else:
        net_device = None  # unused

    for start in range(0, N, batch_size):
        batch = X[start:start + batch_size]

        if is_torch_net:
            # Convert batch to torch on the net's device
            x_batch = torch.from_numpy(batch.astype(np.float32)).to(net_device)

            with torch.no_grad():
                _ = net.forward(x_batch)  # fills net.h as torch.Tensors
                h_l = net.h[layer_index].detach().cpu().numpy()
        else:
            # Old pure-NumPy path
            net.forward(batch)          # fills net.h as np.ndarray
            h_l = net.h[layer_index]

        activations.append(h_l)

    return np.vstack(activations)


# ---------------------------------------------------------------------
# 3. t-SNE wrapper
# ---------------------------------------------------------------------
def tsne_embed(X, random_state: int = 0):
    """
    Run 2D t-SNE on X.

    Args:
        X: (N, D) NumPy array

    Returns:
        (N, 2) NumPy array of 2D embeddings.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="pca",
        random_state=random_state,
        # n_iter left at default for your sklearn version
    )
    return tsne.fit_transform(X)


# ---------------------------------------------------------------------
# 4. Main plotting function
# ---------------------------------------------------------------------
def plot_tsne_grid(run_dir, n_samples=2000, use_test=True, random_state=0):
    """
    Make a BP vs DFA t-SNE grid for each hidden layer and save to tsne_layers.png
    inside run_dir.

    Assumes experiment_GPU.py (or experiment.py) produced:
      - config.json    (with "layer_sizes" and optionally "feedback_scale")
      - bp_final.npz   (weights)
      - dfa_final.npz  (weights)
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
    # 2. Rebuild nets & load weights (on CPU for t-SNE)
    # --------------------
    # Note: we force learning_rate=0.0 since we won't train here.
    device = torch.device("cpu")

    bp_net = BPNetwork(layer_sizes, learning_rate=0.0)
    dfa_net = DFANetwork(
        layer_sizes,
        learning_rate=0.0,
        feedback_scale=feedback_scale,
    )

    # If these networks support a 'device' attribute, set them to CPU
    if hasattr(bp_net, "device"):
        bp_net.device = device
    if hasattr(dfa_net, "device"):
        dfa_net.device = device

    load_weights_into(bp_net, os.path.join(run_dir, "bp_final.npz"), device=device)
    load_weights_into(dfa_net, os.path.join(run_dir, "dfa_final.npz"), device=device)

    # --------------------
    # 3. Load data & pick subset (NumPy)
    # --------------------
    (X_train, y_train), (X_test, y_test) = load_mnist()

    if use_test:
        X, y = X_test, y_test
    else:
        X, y = X_train, y_train

    n_samples = min(n_samples, len(X))
    X = X[:n_samples].astype(np.float32)
    y = y[:n_samples]
    labels = np.argmax(y, axis=1)

    # --------------------
    # 4. t-SNE per hidden layer
    # --------------------
    fig, axes = plt.subplots(2, n_hidden, figsize=(4 * n_hidden, 8))

    # Ensure axes is 2D even if n_hidden == 1
    if n_hidden == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for li in range(1, L):  # hidden layers 1..L-1
        print(f"[t-SNE] Processing hidden layer {li}/{n_hidden}")

        # (N, dim_l)
        h_bp = get_layer_activations(bp_net, X, layer_index=li)
        h_dfa = get_layer_activations(dfa_net, X, layer_index=li)

        emb_bp = tsne_embed(h_bp, random_state=random_state)
        emb_dfa = tsne_embed(h_dfa, random_state=random_state)

        ax_bp = axes[0, li - 1]
        ax_dfa = axes[1, li - 1]

        sc_bp = ax_bp.scatter(
            emb_bp[:, 0],
            emb_bp[:, 1],
            c=labels,
            cmap="tab10",
            s=3,
            alpha=0.7,
        )
        sc_dfa = ax_dfa.scatter(
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

    out_path = os.path.join(run_dir, "tsne_layers.png")
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

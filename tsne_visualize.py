# tsne_visualize.py
"""
t-SNE visualization tools for comparing BP vs DFA neural network representations.

How to run in terminal: 
python tsne_visualize.py --run_dir results/20251212_162509_tSNE_run

This module provides functionality to:
- Load trained BP and DFA networks from checkpoint files
- Extract hidden layer activations from MNIST test/train data
- Apply t-SNE dimensionality reduction to visualize layer representations
- Generate comparison plots showing BP vs DFA embeddings for each hidden layer

The visualizations help analyze whether DFA networks learn similar internal 
representations to traditional backpropagation networks.
"""

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.bp import BPNetwork
from src.dfa import DFANetwork
from data.data import load_mnist


def load_weights_into(net, path):
    """
    Load weights and biases from an .npz checkpoint into a BP/ DFA network.

    Expects keys W0, b0, W1, b1, ...
    """
    data = np.load(path)
    for i in range(len(net.W)):
        net.W[i] = data[f"W{i}"]
        net.b[i] = data[f"b{i}"]


def get_layer_activations(net, X, layer_index, batch_size=256):
    """
    Compute activations h^{layer_index} for all samples in X.

    Args:
        net: BPNetwork or DFANetwork
        X: (N, input_dim)
        layer_index: 0 = input, 1 = first hidden, ..., L = output
        batch_size: minibatch size for forward passes

    Returns:
        activations: (N, layer_sizes[layer_index])
    """
    activations = []
    N = len(X)

    for start in range(0, N, batch_size):
        batch = X[start:start + batch_size]
        net.forward(batch)          # fills net.h
        h_l = net.h[layer_index]    # (B, dim_l)
        activations.append(h_l)

    return np.vstack(activations)


def tsne_embed(X, random_state=0):
    """
    Run 2D t-SNE on X.

    Args:
        X: (N, D)
    """
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,   # numeric works across old/new sklearn
        init="pca",
        random_state=random_state,
        # n_iter left at its default for your sklearn version
    )
    return tsne.fit_transform(X)



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
    # 2. Rebuild nets & load weights
    # --------------------
    bp_net = BPNetwork(layer_sizes, learning_rate=0.0)
    dfa_net = DFANetwork(
        layer_sizes,
        learning_rate=0.0,
        feedback_scale=feedback_scale,
    )

    load_weights_into(bp_net, os.path.join(run_dir, "bp_final.npz"))
    load_weights_into(dfa_net, os.path.join(run_dir, "dfa_final.npz"))

    # --------------------
    # 3. Load data & pick subset
    # --------------------
    (X_train, y_train), (X_test, y_test) = load_mnist()

    if use_test:
        X, y = X_test, y_test
    else:
        X, y = X_train, y_train

    n_samples = min(n_samples, len(X))
    X = X[:n_samples]
    y = y[:n_samples]
    labels = np.argmax(y, axis=1)

    # --------------------
    # 4. t-SNE per hidden layer
    # --------------------
    fig, axes = plt.subplots(2, n_hidden, figsize=(4 * n_hidden, 8))
    # handle the n_hidden == 1 case so axes is 2D
    if n_hidden == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for li in range(1, L):  # hidden layers 1..L-1
        print(f"[t-SNE] Processing hidden layer {li}/{n_hidden}")

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

    out_path = os.path.join(run_dir, "tsne_layers.png")
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"[t-SNE] Saved grid to {out_path}")


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

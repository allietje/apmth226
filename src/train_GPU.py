# train_GPU.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from .utils_GPU import binary_cross_entropy


def accuracy(net, X, y):
    """
    Compute classification accuracy using net.forward.
    X: NumPy array or torch.Tensor of shape (N, input_dim)
    y: NumPy one-hot array of shape (N, C)
    Returns:
        scalar float accuracy in [0, 1].
    """
    logits = net.forward(X)  # torch.Tensor or NumPy (we'll treat as torch)
    if isinstance(logits, torch.Tensor):
        pred_classes = torch.argmax(logits, dim=1).cpu().numpy()
    else:
        pred_classes = np.argmax(logits, axis=1)
    true_classes = np.argmax(y, axis=1)
    return float((pred_classes == true_classes).mean())


def eval_bce(net, X, y):
    """
    Compute BCE loss on a dataset.
    Returns a Python float.
    """
    y_pred = net.forward(X)
    loss = binary_cross_entropy(y, y_pred)
    if isinstance(loss, torch.Tensor):
        return float(loss.item())
    return float(loss)

def train_one_epoch(net, X, y, batch_size: int, shuffle: bool = True):
    """
    One training epoch over all samples (in minibatches).
    net: BPNetwork or DFANetwork (must be a torch.nn.Module)
    X, y: NumPy arrays (N, ...)
    Returns:
        avg_loss (float), avg_grad_norms_per_layer (list of floats)
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    if shuffle:
        indices = np.random.permutation(indices)

    total_loss = 0.0
    batch_grad_norms = []  # List of per-batch layer norms

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start:start + batch_size]
        x_batch = X[batch_idx]
        y_batch = y[batch_idx]

        # Compute gradients manually (before train_step which consumes them)
        with torch.no_grad():
            y_hat = net.forward(x_batch)
            grads_W, grads_b = net.backward(y_batch)
            
            # Compute gradient norms from manual gradients
            layer_norms = []
            for i in range(len(grads_W)):
                norm_sq = 0.0
                if grads_W[i] is not None:
                    norm_sq += grads_W[i].norm(2).item() ** 2
                if grads_b[i] is not None:
                    norm_sq += grads_b[i].norm(2).item() ** 2
                layer_norms.append(norm_sq ** 0.5 if norm_sq > 0 else 0.0)
            
            batch_grad_norms.append(layer_norms)

        # Now do the training step (which will update parameters)
        batch_loss = net.train_step(x_batch, y_batch)

        # Handle loss value - extract scalar from tensor if needed  
        if isinstance(batch_loss, torch.Tensor):
            if not torch.isfinite(batch_loss):
                print(f"[WARN] Non-finite loss: {batch_loss.item()}. Stopping epoch early.")
                return float("nan"), [float("nan")] * len(net.W)
            batch_loss_val = batch_loss.item()
        else:
            batch_loss_val = float(batch_loss)
            if not np.isfinite(batch_loss_val):
                print(f"[WARN] Non-finite loss: {batch_loss_val}. Stopping epoch early.")
                return float("nan"), [float("nan")] * len(net.W)

        total_loss += batch_loss_val * len(x_batch)

    avg_loss = total_loss / n_samples

    # Average gradient norms across all batches for this epoch
    if batch_grad_norms:
        avg_grad_norms = [sum(col) / len(batch_grad_norms) for col in zip(*batch_grad_norms)]
    else:
        avg_grad_norms = [0.0] * len(net.W)

    return avg_loss, avg_grad_norms


def train_bp_dfa(
    bp_net,
    dfa_net,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs: int = 50,
    batch_size: int = 64,
    log_weights: bool = False,
    plot: bool = True,
    plot_path: str = "bp_vs_dfa_mnist.png",
    grad_plot_path: str = "grad_norms.png",
):
    """
    Train BP and DFA networks side-by-side and log performance + gradient norms.
    Adds gradient norm tracking and a dedicated plot for per-layer gradient norms.
    """
    # Lists to store metrics over epochs
    train_losses_bp, train_losses_dfa = [], []
    test_losses_bp, test_losses_dfa = [], []
    train_acc_bp, train_acc_dfa = [], []
    test_acc_bp, test_acc_dfa = [], []

    # Gradient norms: [epoch][layer]
    grad_norms_bp = []
    grad_norms_dfa = []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train one epoch and collect gradient norms
        train_loss_bp, epoch_grad_bp = train_one_epoch(bp_net, X_train, y_train, batch_size)
        train_loss_dfa, epoch_grad_dfa = train_one_epoch(dfa_net, X_train, y_train, batch_size)

        grad_norms_bp.append(epoch_grad_bp)
        grad_norms_dfa.append(epoch_grad_dfa)

        # Evaluation
        acc_train_bp = accuracy(bp_net, X_train, y_train)
        acc_train_dfa = accuracy(dfa_net, X_train, y_train)
        acc_test_bp = accuracy(bp_net, X_test, y_test)
        acc_test_dfa = accuracy(dfa_net, X_test, y_test)

        loss_bp = eval_bce(bp_net, X_test, y_test)
        loss_dfa = eval_bce(dfa_net, X_test, y_test)

        # Store metrics
        train_losses_bp.append(train_loss_bp)
        train_losses_dfa.append(train_loss_dfa)
        test_losses_bp.append(loss_bp)
        test_losses_dfa.append(loss_dfa)
        train_acc_bp.append(acc_train_bp)
        train_acc_dfa.append(acc_train_dfa)
        test_acc_bp.append(acc_test_bp)
        test_acc_dfa.append(acc_test_dfa)

        print(
            f"BP → train loss {train_loss_bp:.4f} | "
            f"train acc {acc_train_bp:.3%} | test acc {acc_test_bp:.3%}"
        )
        print(
            f"DFA → train loss {train_loss_dfa:.4f} | "
            f"train acc {acc_train_dfa:.3%} | test acc {acc_test_dfa:.3%}"
        )

    # Convert accuracies to numpy arrays
    train_acc_bp = np.array(train_acc_bp)
    train_acc_dfa = np.array(train_acc_dfa)
    test_acc_bp = np.array(test_acc_bp)
    test_acc_dfa = np.array(test_acc_dfa)

    # Compute error rates (%)
    train_err_bp = 100.0 * (1.0 - train_acc_bp)
    train_err_dfa = 100.0 * (1.0 - train_acc_dfa)
    test_err_bp = 100.0 * (1.0 - test_acc_bp)
    test_err_dfa = 100.0 * (1.0 - test_acc_dfa)

    # Original performance plots
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax1, ax2 = axes

        # Train & test error (%)
        ax1.plot(train_err_bp, label="BP train error")
        ax1.plot(test_err_bp, label="BP test error", linestyle="--")
        ax1.plot(train_err_dfa, label="DFA train error")
        ax1.plot(test_err_dfa, label="DFA test error", linestyle="--")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Error (%)")
        ax1.set_title("Train & Test Error")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Test accuracy
        ax2.plot(test_acc_bp, label="BP")
        ax2.plot(test_acc_dfa, label="DFA")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Test Accuracy")
        ax2.set_title("Test Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

        # New: Gradient norm plots
        if grad_norms_bp and grad_norms_dfa:
            num_layers = len(grad_norms_bp[0])
            fig, (ax_bp, ax_dfa) = plt.subplots(1, 2, figsize=(14, 6))
            epochs_x = np.arange(1, epochs + 1)

            for layer_idx in range(num_layers):
                bp_norms = [epoch_norms[layer_idx] for epoch_norms in grad_norms_bp]
                dfa_norms = [epoch_norms[layer_idx] for epoch_norms in grad_norms_dfa]
                ax_bp.plot(epochs_x, bp_norms, label=f"Layer {layer_idx + 1}")
                ax_dfa.plot(epochs_x, dfa_norms, label=f"Layer {layer_idx + 1}")

            for ax, title in zip([ax_bp, ax_dfa], ["BP Gradient Norms", "DFA Gradient Norms"]):
                ax.set_yscale("log")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Avg L2 Gradient Norm (log scale)")
                ax.set_title(title)
                ax.legend(fontsize="small", ncol=2)
                ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(grad_plot_path, dpi=200)
            plt.close(fig)

    return {
        "train_losses_bp": np.array(train_losses_bp),
        "test_losses_bp": np.array(test_losses_bp),
        "train_losses_dfa": np.array(train_losses_dfa),
        "test_losses_dfa": np.array(test_losses_dfa),
        "train_acc_bp": train_acc_bp,
        "train_acc_dfa": train_acc_dfa,
        "test_acc_bp": test_acc_bp,
        "test_acc_dfa": test_acc_dfa,
        "train_err_bp": train_err_bp,
        "train_err_dfa": train_err_dfa,
        "test_err_bp": test_err_bp,
        "test_err_dfa": test_err_dfa,
        "grad_norms_bp": np.array(grad_norms_bp),   # shape: [epochs, layers]
        "grad_norms_dfa": np.array(grad_norms_dfa),
    }
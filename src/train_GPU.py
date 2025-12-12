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

    net: BPNetwork or DFANetwork
    X, y: NumPy arrays (N, ...) (still fine; conversion happens inside net)
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    if shuffle:
        indices = np.random.permutation(indices)

    total_loss = 0.0

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        x_batch = X[batch_idx]
        y_batch = y[batch_idx]

        batch_loss = net.train_step(x_batch, y_batch)  # torch scalar or float
        if isinstance(batch_loss, torch.Tensor):
            if not torch.isfinite(batch_loss):
                print(f"[WARN] Non-finite loss encountered: {batch_loss.item()}. "
                    f"Stopping epoch early.")
                return float("nan")
            batch_loss_val = batch_loss.item()
        else:
            batch_loss_val = float(batch_loss)
            if not np.isfinite(batch_loss_val):
                print(f"[WARN] Non-finite loss encountered: {batch_loss_val}. "
                    f"Stopping epoch early.")
                return float("nan")

        total_loss += batch_loss_val * len(x_batch)

    return total_loss / n_samples


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
):
    """
    Train BP and DFA networks side by side on MNIST-like data.

    Left plot: train & test error (%) vs epoch
               - train: solid line
               - test : dotted line
    Right plot: test accuracy vs epoch
    """
    train_losses_bp, test_losses_bp = [], []
    train_losses_dfa, test_losses_dfa = [], []

    train_acc_bp, train_acc_dfa = [], []
    test_acc_bp, test_acc_dfa = [], []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train one epoch
        train_loss_bp = train_one_epoch(bp_net, X_train, y_train, batch_size)
        train_loss_dfa = train_one_epoch(dfa_net, X_train, y_train, batch_size)

        # Accuracies (train + test)
        acc_train_bp = accuracy(bp_net, X_train, y_train)
        acc_train_dfa = accuracy(dfa_net, X_train, y_train)
        acc_test_bp = accuracy(bp_net, X_test, y_test)
        acc_test_dfa = accuracy(dfa_net, X_test, y_test)

        # BCE on test
        loss_bp = eval_bce(bp_net, X_test, y_test)
        loss_dfa = eval_bce(dfa_net, X_test, y_test)

        train_losses_bp.append(train_loss_bp)
        train_losses_dfa.append(train_loss_dfa)
        test_losses_bp.append(loss_bp)
        test_losses_dfa.append(loss_dfa)

        train_acc_bp.append(acc_train_bp)
        train_acc_dfa.append(acc_train_dfa)
        test_acc_bp.append(acc_test_bp)
        test_acc_dfa.append(acc_test_dfa)

        print(
            f"BP  \u2192 train loss {train_loss_bp:.4f} | "
            f"train acc {acc_train_bp:.3%} | test acc {acc_test_bp:.3%}"
        )
        print(
            f"DFA \u2192 train loss {train_loss_dfa:.4f} | "
            f"train acc {acc_train_dfa:.3%} | test acc {acc_test_dfa:.3%}"
        )

    # Convert to arrays
    train_acc_bp = np.array(train_acc_bp)
    train_acc_dfa = np.array(train_acc_dfa)
    test_acc_bp = np.array(test_acc_bp)
    test_acc_dfa = np.array(test_acc_dfa)

    # error % = 100 * (1 - accuracy)
    train_err_bp = 100.0 * (1.0 - train_acc_bp)
    train_err_dfa = 100.0 * (1.0 - train_acc_dfa)
    test_err_bp = 100.0 * (1.0 - test_acc_bp)
    test_err_dfa = 100.0 * (1.0 - test_acc_dfa)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax1, ax2 = axes

        # 1) Train & test error (%) vs epoch
        ax1.plot(train_err_bp, label="BP train error")
        ax1.plot(test_err_bp, label="BP test error", linestyle="--")
        ax1.plot(train_err_dfa, label="DFA train error")
        ax1.plot(test_err_dfa, label="DFA test error", linestyle="--")
        ax1.set_xlabel("Epoch")
        ax1.set_ylim(0, 10)   
        ax1.set_ylabel("Error (%)")
        ax1.set_title("Train & Test Error on MNIST")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2) Test accuracy vs epoch
        ax2.plot(test_acc_bp, label="BP")
        ax2.plot(test_acc_dfa, label="DFA")
        ax2.set_ylim(0.0, 1.0)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Test Accuracy")
        ax2.set_title("Test Accuracy on MNIST")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.show()

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
    }

# experiment_GPU.py
import os
import json
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

from .train_GPU import train_bp_dfa, train_one_epoch, accuracy, eval_bce 

def _make_run_dir(base_dir="results", run_name=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = "run"
    folder = f"{timestamp}_{run_name}"
    run_dir = os.path.join(base_dir, folder)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def _save_config(config, run_dir):
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

def _save_metrics_npz(metrics, run_dir):
    path = os.path.join(run_dir, "metrics.npz")
    arrays = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            arrays[k] = np.array(v)
    np.savez(path, **arrays)

def _save_metrics_csv(metrics, run_dir):
    path = os.path.join(run_dir, "history.csv")
    any_key = next(iter(metrics))
    num_epochs = len(metrics[any_key])
    cols = [
        "train_err_bp", "test_err_bp", "train_err_dfa", "test_err_dfa",
        "train_acc_bp", "test_acc_bp", "train_acc_dfa", "test_acc_dfa",
    ]
    cols = [c for c in cols if c in metrics]
    with open(path, "w") as f:
        f.write("epoch," + ",".join(cols) + "\n")
        for i in range(num_epochs):
            row_vals = [str(i + 1)]
            for c in cols:
                row_vals.append(f"{metrics[c][i]:.6f}")
            f.write(",".join(row_vals) + "\n")

def _save_model_npz(net, run_dir, name):
    path = os.path.join(run_dir, f"{name}_final.npz")
    data = {}
    for i, (W, b) in enumerate(zip(net.W, net.b)):
        W_np = W.detach().cpu().numpy() if isinstance(W, torch.Tensor) else np.array(W)
        b_np = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.array(b)
        data[f"W{i}"] = W_np
        data[f"b{i}"] = b_np
    np.savez(path, **data)

def run_bp_dfa_experiment(
    config,
    bp_net,
    dfa_net,
    X_train,
    y_train,
    X_test,
    y_test,
    base_dir="results",
):
    """
    High level helper that:
      - creates a fresh run directory under `base_dir`
      - saves config
      - runs train_bp_dfa with its plot going into the run directory
      - saves metrics (.npz + .csv)
      - saves final BP and DFA weights
    """
    run_name = config.get("run_name", "bp_dfa")
    run_dir = _make_run_dir(base_dir=base_dir, run_name=run_name)
    print(f"[experiment] Logging to: {run_dir}")

    # 1) save config
    _save_config(config, run_dir)

    # 2) train and collect metrics, with plot inside run_dir
    plot_path = os.path.join(run_dir, "curves.png")
    metrics = train_bp_dfa(
        bp_net,
        dfa_net,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 64),
        log_weights=config.get("log_weights", False),
        plot=True,
        plot_path=plot_path,
    )

    # 3) save metrics
    _save_metrics_npz(metrics, run_dir)
    _save_metrics_csv(metrics, run_dir)

    # 4) save final models
    _save_model_npz(bp_net, run_dir, name="bp")
    _save_model_npz(dfa_net, run_dir, name="dfa")

    print(f"[experiment] Saved config, metrics, models, and plot in {run_dir}")
    return run_dir, metrics

def run_bp_dfa_experiment_until_convergence(
    config,
    bp_net,
    dfa_net,
    X_train, y_train,
    X_test, y_test,
    base_dir="results",
    window=5,
    epsilon=0.1,
    max_epochs=300,
):
    run_name = config.get("run_name", "bp_dfa_conv")
    run_dir = _make_run_dir(base_dir=base_dir, run_name=run_name)
    print(f"[experiment] Logging to: {run_dir} (training until convergence)")

    _save_config(config, run_dir)

    # Metric storage
    metrics = {
        "train_losses_bp": [], "train_losses_dfa": [],
        "test_err_bp": [], "test_err_dfa": [],
        "train_acc_bp": [], "train_acc_dfa": [],
        "test_acc_bp": [], "test_acc_dfa": [],
        "grad_norms_bp": [], "grad_norms_dfa": [],
    }

    batch_size = config.get("batch_size", 64)

    actual_epochs = 0
    bp_converged = False
    dfa_converged = False

    for epoch in range(1, max_epochs + 1):
        actual_epochs = epoch
        print(f"\nEpoch {epoch}/{max_epochs}")

        # Train one epoch for each network
        train_loss_bp, grad_bp = train_one_epoch(bp_net, X_train, y_train, batch_size)
        train_loss_dfa, grad_dfa = train_one_epoch(dfa_net, X_train, y_train, batch_size)

        # Evaluation
        acc_train_bp = accuracy(bp_net, X_train, y_train)
        acc_train_dfa = accuracy(dfa_net, X_train, y_train)
        acc_test_bp = accuracy(bp_net, X_test, y_test)
        acc_test_dfa = accuracy(dfa_net, X_test, y_test)

        # Store
        metrics["train_losses_bp"].append(train_loss_bp)
        metrics["train_losses_dfa"].append(train_loss_dfa)
        metrics["train_acc_bp"].append(acc_train_bp)
        metrics["train_acc_dfa"].append(acc_train_dfa)
        metrics["test_acc_bp"].append(acc_test_bp)
        metrics["test_acc_dfa"].append(acc_test_dfa)
        metrics["test_err_bp"].append(100.0 * (1.0 - acc_test_bp))
        metrics["test_err_dfa"].append(100.0 * (1.0 - acc_test_dfa))
        metrics["grad_norms_bp"].append(grad_bp)
        metrics["grad_norms_dfa"].append(grad_dfa)

        # Convergence check
        if len(metrics["test_err_bp"]) >= 2 * window:
            ma_bp = np.convolve(metrics["test_err_bp"], np.ones(window)/window, mode='valid')
            if abs(ma_bp[-1] - ma_bp[-window]) < epsilon:
                bp_converged = True

        if len(metrics["test_err_dfa"]) >= 2 * window:
            ma_dfa = np.convolve(metrics["test_err_dfa"], np.ones(window)/window, mode='valid')
            if abs(ma_dfa[-1] - ma_dfa[-window]) < epsilon:
                dfa_converged = True

        print(f"BP converged: {bp_converged} | DFA converged: {dfa_converged}")
        if bp_converged and dfa_converged:
            print(f"[experiment] Both converged at epoch {epoch}!")
            break

    # Convert lists to arrays for consistency
    for k in metrics:
        metrics[k] = np.array(metrics[k])

    # Save everything
    _save_metrics_npz(metrics, run_dir)
    _save_metrics_csv(metrics, run_dir)
    _save_model_npz(bp_net, run_dir, "bp")
    _save_model_npz(dfa_net, run_dir, "dfa")

    plot_path = os.path.join(run_dir, "curves.png")

    epochs_x = np.arange(1, actual_epochs + 1)

        # Main figure: 3 subplots (Training Loss, Test Error %, Test Accuracy)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Training Loss
    ax = axes[0]
    ax.plot(epochs_x, metrics["train_losses_bp"], label="BP")
    ax.plot(epochs_x, metrics["train_losses_dfa"], label="DFA")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.set_yscale("log")  # Helps visualize convergence
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Test Error (%)
    ax = axes[1]
    ax.plot(epochs_x, metrics["test_err_bp"], label="BP")
    ax.plot(epochs_x, metrics["test_err_dfa"], label="DFA")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Error (%)")
    ax.set_title("Test Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Test Accuracy
    ax = axes[2]
    ax.plot(epochs_x, metrics["test_acc_bp"], label="BP")
    ax.plot(epochs_x, metrics["test_acc_dfa"], label="DFA")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"[experiment] Training finished after {actual_epochs} epochs.")
    print(f"[experiment] Saved config, metrics, models, and plot in {run_dir}")
    return run_dir, metrics, actual_epochs
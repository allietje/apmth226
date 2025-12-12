# experiment.py
import os
import json
from datetime import datetime

import numpy as np

from .train import train_bp_dfa


def _make_run_dir(base_dir="results", run_name=None):
    """
    Create a new run directory like results/20250309_142312_dfa_width200
    and return its path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = "run"
    folder = f"{timestamp}_{run_name}"
    run_dir = os.path.join(base_dir, folder)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _save_config(config, run_dir):
    """Save config dict as JSON."""
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def _save_metrics_npz(metrics, run_dir):
    """Save all metrics arrays into a single .npz file."""
    path = os.path.join(run_dir, "metrics.npz")
    # Convert any lists to arrays for consistency
    arrays = {k: np.array(v) for k, v in metrics.items()}
    np.savez(path, **arrays)


def _save_metrics_csv(metrics, run_dir):
    """
    Save a CSV with one row per epoch.
    Assumes all metric arrays have same length = num_epochs.
    """
    path = os.path.join(run_dir, "history.csv")
    # pick a key to infer number of epochs
    any_key = next(iter(metrics))
    num_epochs = len(metrics[any_key])

    # choose columns to log (you can add/remove)
    cols = [
        "train_err_bp",
        "test_err_bp",
        "train_err_dfa",
        "test_err_dfa",
        "train_acc_bp",
        "test_acc_bp",
        "train_acc_dfa",
        "test_acc_dfa",
    ]
    # filter to only those that actually exist
    cols = [c for c in cols if c in metrics]

    with open(path, "w") as f:
        # header
        f.write("epoch," + ",".join(cols) + "\n")
        # rows
        for i in range(num_epochs):
            row_vals = [str(i + 1)]
            for c in cols:
                row_vals.append(f"{metrics[c][i]:.6f}")
            f.write(",".join(row_vals) + "\n")


def _save_model_npz(net, run_dir, name):
    """
    Save a simple NumPy checkpoint for a network.
    Assumes the network has attributes W (list of arrays) and b (list of arrays).
    """
    path = os.path.join(run_dir, f"{name}_final.npz")
    # flatten into a dict like W0, b0, W1, b1, ...
    data = {}
    for i, (W, b) in enumerate(zip(net.W, net.b)):
        data[f"W{i}"] = W
        data[f"b{i}"] = b
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

    Args:
        config: dict of hyperparameters and notes for this run.
        bp_net, dfa_net: initialized networks.
        X_train, y_train, X_test, y_test: NumPy arrays.
        base_dir: root directory for all runs.

    Returns:
        run_dir (str), metrics (dict)
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

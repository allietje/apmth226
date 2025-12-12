# implementation_GPU.py
"""
implementation_gpu.py

GPU-friendly entry point for running BP vs DFA experiments.

This script:
  - sets a random seed
  - loads MNIST using src.data.load_mnist
  - creates BP and DFA networks on a chosen device (CPU or GPU)
  - runs training via src.experiment.run_bp_dfa_experiment
  - saves config, metrics, plots, and final weights under results/

Usage:
  python implementation_gpu.py

To change hyperparameters or device, edit the block under
`# -------------------------`
`# Hyperparameters`
`# -------------------------`
below.
"""

import torch

from src.utils_GPU import set_seed
from data.data import load_mnist
from src.model_GPU import create_networks
from src.experiment_GPU import run_bp_dfa_experiment


def select_device(device_str: str) -> torch.device:
    """
    Convert a device string into a torch.device, with a safe fallback
    to CPU if CUDA is requested but not available.

    Args:
        device_str: e.g. "cuda", "cuda:0", "cuda:1", or "cpu".

    Returns:
        torch.device
    """
    # If user asks for a CUDA device but CUDA isn't available, fall back.
    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            print("[implementation_gpu] Warning: CUDA requested but not available. "
                  "Falling back to CPU.")
            return torch.device("cpu")
    # "cpu" or anything else valid for torch.device
    return torch.device(device_str)


def main():
    # -------------------------
    # Hyperparameters
    # -------------------------
    run_name       = "stability"   # name prefix for this run in results/
    seed           = 42                     # random seed for reproducibility
    width          = 400                    # hidden layer width of the MLP
    depth          = 3                      # number of hidden layers
    lr_bp          = 0.0005                  # learning rate for BP network
    lr_dfa         = 0.001                   # learning rate for DFA network
    batch_size     = 128                     # batch size
    epochs         = 60                     # number of training epochs
    feedback_scale = 0.1                   # DFA feedback scale (B matrices)
    results_dir    = "results"              # root folder for run outputs

    # Device choice:
    #   "cuda"    -> first visible GPU (cuda:0)
    #   "cuda:1"  -> second GPU, etc.
    #   "cpu"     -> force CPU
    device_str    = "cuda"                  # change to "cpu" if debugging on CPU only

    # -------------------------
    # 1. Select device
    # -------------------------
    device = select_device(device_str)
    print(f"[implementation_gpu] Using device: {device}")

    # -------------------------
    # 2. Set random seed
    # -------------------------
    set_seed(seed)

    # -------------------------
    # 3. Load MNIST data
    # -------------------------
    # Assumes load_mnist(flatten=True, normalize=True) returns:
    #   (X_train, y_train), (X_test, y_test)
    # with:
    #   X_*: float32 NumPy arrays, shape (N, 784)
    #   y_*: one-hot float32 NumPy arrays, shape (N, 10)
    (X_train, y_train), (X_test, y_test) = load_mnist(
        flatten=True,
        normalize=True,
    )

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    print(f"[implementation_gpu] MNIST loaded: "
          f"{X_train.shape[0]} train, {X_test.shape[0]} test "
          f"(input_dim={input_dim}, output_dim={output_dim})")

    # -------------------------
    # 4. Build BP and DFA networks on the chosen device
    # -------------------------
    bp_net, dfa_net, layer_sizes = create_networks(
        width=width,
        depth=depth,
        lr_bp=lr_bp,
        lr_dfa=lr_dfa,
        seed=seed,
        feedback_scale=feedback_scale,
        input_dim=input_dim,
        output_dim=output_dim,
        device=str(device),    # create_networks expects a device string
    )

    print(f"[implementation_gpu] Network architecture: {layer_sizes}")

    # -------------------------
    # 5. Pack config for logging
    # -------------------------
    config = {
        "run_name": run_name,
        "seed": seed,
        "layer_sizes": layer_sizes,
        "width": width,
        "depth": depth,
        "lr_bp": lr_bp,
        "lr_dfa": lr_dfa,
        "batch_size": batch_size,
        "epochs": epochs,
        "feedback_scale": feedback_scale,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "device": str(device),
    }

    # -------------------------
    # 6. Run experiment (training + logging + saving)
    # -------------------------
    run_dir, metrics = run_bp_dfa_experiment(
        config,
        bp_net,
        dfa_net,
        X_train,
        y_train,
        X_test,
        y_test,
        base_dir=results_dir,
    )

    # -------------------------
    # 7. Print final summary
    # -------------------------
    final_bp_err = metrics["test_err_bp"][-1]
    final_dfa_err = metrics["test_err_dfa"][-1]
    print("\n[implementation_gpu] Finished run.")
    print(f"  Run directory        : {run_dir}")
    print(f"  Final BP  test error : {final_bp_err:.2f}%")
    print(f"  Final DFA test error : {final_dfa_err:.2f}%")


if __name__ == "__main__":
    main()

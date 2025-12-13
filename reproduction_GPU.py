# reproduction_.py

# models to reproduce:
# 7x240, 1x800, 2x800, 3x800, 4x800

"""
reproduction.py

GPU-friendly entry point for running BP vs DFA experiments over multiple models.

This script:
  - sets a random seed
  - loads MNIST using src.data.load_mnist
  - creates BP and DFA networks on a chosen device (CPU or GPU)
  - runs training via src.experiment.run_bp_dfa_experiment for several architectures
  - saves config, metrics, plots, and final weights under results/

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
    # Hyperparameters (shared)
    # -------------------------
    base_run_name  = "stability"     # base name prefix for runs in results/
    base_seed      = 42              # base random seed for reproducibility
    lr_bp          = 0.0005          # learning rate for BP network
    lr_dfa         = 0.001           # learning rate for DFA network
    batch_size     = 256             # batch size
    epochs         = 60              # number of training epochs
    feedback_scale = 0.1             # DFA feedback scale (B matrices)
    results_dir    = "results"       # root folder for run outputs

    # Device choice:
    device_str     = "cuda"          # change to "cpu" if debugging on CPU only

    # Models to reproduce: (name, depth, width)
    model_configs = [
        ("7x240", 7, 240),
        ("1x800", 1, 800),
        ("2x800", 2, 800),
        ("3x800", 3, 800),
        ("4x800", 4, 800),
    ]

    # -------------------------
    # 1. Select device
    # -------------------------
    device = select_device(device_str)
    print(f"[implementation_gpu] Using device: {device}")

    # -------------------------
    # 2. Set base random seed
    # -------------------------
    set_seed(base_seed)

    # -------------------------
    # 3. Load MNIST data (once)
    # -------------------------
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
    # 4. Loop over model configs
    # -------------------------
    for idx, (model_name, depth, width) in enumerate(model_configs):
        print(f"\n[implementation_gpu] === Running model {model_name} "
              f"(depth={depth}, width={width}) ===")

        # Per-model seed for reproducibility & diversity
        cur_seed = base_seed + idx
        set_seed(cur_seed)

        # 4a. Build BP and DFA networks on the chosen device
        bp_net, dfa_net, layer_sizes = create_networks(
            width=width,
            depth=depth,
            lr_bp=lr_bp,
            lr_dfa=lr_dfa,
            seed=cur_seed,
            feedback_scale=feedback_scale,
            input_dim=input_dim,
            output_dim=output_dim,
            device=str(device),    # create_networks expects a device string
        )

        print(f"[implementation_gpu] Network architecture for {model_name}: {layer_sizes}")

        # 4b. Pack config for logging
        run_name = f"{model_name}_{base_run_name}"
        config = {
            "run_name": run_name,
            "seed": cur_seed,
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

        # 4c. Run experiment (training + logging + saving)
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

        # 4d. Print per-model summary
        final_bp_err = metrics["test_err_bp"][-1]
        final_dfa_err = metrics["test_err_dfa"][-1]
        print("\n[implementation_gpu] Finished run.")
        print(f"  Model                : {model_name}")
        print(f"  Run directory        : {run_dir}")
        print(f"  Final BP  test error : {final_bp_err:.2f}%")
        print(f"  Final DFA test error : {final_dfa_err:.2f}%")

    print("\n[implementation_gpu] All model runs completed.")


if __name__ == "__main__":
    main()

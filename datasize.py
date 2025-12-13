# datasize.py
"""
datasize.py
Extension 2: Final test error % vs training data size
- Fixed architecture: 10x800 tanh
- Subsample sizes: 1000, 5000, ..., 60000 (full)
- 3 independent random subsamples per size
- Train until convergence (same criterion as Extension 1)
- Plot 1: error% vs epoch (up to 60) with convergence line for each of 3 runs (per size)
- Plot 2: final test error % vs sample size with error bars
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.utils_GPU import set_seed
from data.data import load_mnist
from src.model_GPU import create_networks
from src.experiment_GPU import run_bp_dfa_experiment_until_convergence

# Convergence detection (same as convergence.py)
def moving_average(arr, window=5):
    return np.convolve(arr, np.ones(window) / window, mode='valid')

def check_convergence(test_errors, window=5, epsilon=0.1):
    if len(test_errors) < 2 * window:
        return False
    ma = moving_average(test_errors, window)
    return abs(ma[-1] - ma[-window]) < epsilon

def main():
    # -------------------------
    # Hyperparameters
    # -------------------------
    base_run_name = "datasize"
    base_seed = 42
    num_subsamples = 5  # 5 independent subsamples/runs per size
    lr_bp = 0.0005
    lr_dfa = 0.001
    batch_size = 256
    max_epochs = 300
    convergence_window = 5
    convergence_epsilon = 0.1  # test error % change
    feedback_scale = 0.1
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixed architecture
    depth = 10
    width = 800

    # Sample sizes
    sample_sizes = [1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 60000]

    # Load full MNIST once
    set_seed(base_seed)
    (X_train_full, y_train_full), (X_test, y_test) = load_mnist(flatten=True, normalize=True)
    input_dim = X_train_full.shape[1]
    output_dim = y_train_full.shape[1]

    # Containers for final summary
    final_err_data = {size: {'bp': [], 'dfa': []} for size in sample_sizes}

    total_experiments = len(sample_sizes) * num_subsamples
    experiment_counter = 0

    for sample_size in sample_sizes:
        print(f"\n=== Sample Size: {sample_size} ===")

        # Per-sample-size convergence tracking
        bp_convergence_epochs = []
        dfa_convergence_epochs = []
        bp_error_curves = []  # list of arrays (epochs)
        dfa_error_curves = []

        for subsample_idx in range(num_subsamples):
            experiment_counter += 1
            current_seed = base_seed + subsample_idx 
            set_seed(current_seed)
            print(f"--- Subsample {subsample_idx+1}/{num_subsamples} (seed {current_seed}) ---")

            # Create random subsample
            indices = np.random.permutation(len(X_train_full))[:sample_size]
            X_train = X_train_full[indices]
            y_train = y_train_full[indices]

            bp_net, dfa_net, layer_sizes = create_networks(
                width=width, depth=depth,
                lr_bp=lr_bp, lr_dfa=lr_dfa,
                seed=current_seed, feedback_scale=feedback_scale,
                input_dim=input_dim, output_dim=output_dim,
                device=str(device)
            )

            run_name = f"size{sample_size}_subsample{subsample_idx}_{base_run_name}"
            config = {
                "run_name": run_name, "seed": current_seed, "sample_size": sample_size,
                "lr_bp": lr_bp, "lr_dfa": lr_dfa, "batch_size": batch_size,
                "max_epochs": max_epochs, "feedback_scale": feedback_scale,
                "device": str(device),
            }

            run_dir, metrics, actual_epochs = run_bp_dfa_experiment_until_convergence(
                config=config,
                bp_net=bp_net, dfa_net=dfa_net,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                base_dir=results_dir,
                window=convergence_window,
                epsilon=convergence_epsilon,
                max_epochs=max_epochs
            )

            # -------------------------
            # Determine convergence epochs (in terms of epoch index)
            # -------------------------
            bp_conv_epoch = actual_epochs  # default: last epoch if no convergence
            dfa_conv_epoch = actual_epochs

            for epoch in range(convergence_window * 2, actual_epochs):
                if check_convergence(metrics["test_err_bp"][:epoch + 1],
                                     convergence_window, convergence_epsilon):
                    bp_conv_epoch = epoch + 1  # 1-based for plotting
                    break

            for epoch in range(convergence_window * 2, actual_epochs):
                if check_convergence(metrics["test_err_dfa"][:epoch + 1],
                                     convergence_window, convergence_epsilon):
                    dfa_conv_epoch = epoch + 1
                    break

            bp_convergence_epochs.append(bp_conv_epoch)
            dfa_convergence_epochs.append(dfa_conv_epoch)

            # Store the FULL curves (all epochs), so they keep going after convergence
            bp_error_curves.append(np.array(metrics["test_err_bp"]))
            dfa_error_curves.append(np.array(metrics["test_err_dfa"]))

            # Final errors (still use last recorded)
            final_bp_err = metrics["test_err_bp"][-1]
            final_dfa_err = metrics["test_err_dfa"][-1]
            final_err_data[sample_size]['bp'].append(final_bp_err)
            final_err_data[sample_size]['dfa'].append(final_dfa_err)

            print(f"BP conv: {bp_conv_epoch}, final err: {final_bp_err:.2f}%")
            print(f"DFA conv: {dfa_conv_epoch}, final err: {final_dfa_err:.2f}%")


        # -------------------------
        # Plot: test error% vs epoch for 3 subsamples + convergence lines
        # -------------------------
        fig, (ax_bp, ax_dfa) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot full curves (each run up to however many epochs it actually used)
        for i in range(num_subsamples):
            bp_curve = np.array(bp_error_curves[i])
            dfa_curve = np.array(dfa_error_curves[i])

            epochs_bp = np.arange(1, len(bp_curve) + 1)
            epochs_dfa = np.arange(1, len(dfa_curve) + 1)

            ax_bp.plot(epochs_bp, bp_curve,
                       label=f"BP subsample {i+1}", alpha=0.7)
            ax_dfa.plot(epochs_dfa, dfa_curve,
                        label=f"DFA subsample {i+1}", alpha=0.7)

        # Axis limits: go out to the furthest epoch actually run, per method
        max_bp_epochs = max(len(c) for c in bp_error_curves) if bp_error_curves else 1
        max_dfa_epochs = max(len(c) for c in dfa_error_curves) if dfa_error_curves else 1

        # Convergence lines for each run (no clipping; show true epoch)
        for i, conv_epoch in enumerate(bp_convergence_epochs):
            label = "Convergence" if i == 0 else None  # one legend entry
            ax_bp.axvline(conv_epoch, color="red", linestyle=":",
                          alpha=0.9, label=label)

        for i, conv_epoch in enumerate(dfa_convergence_epochs):
            label = "Convergence" if i == 0 else None
            ax_dfa.axvline(conv_epoch, color="red", linestyle=":",
                           alpha=0.9, label=label)

        ax_bp.set_xlim(1, max_bp_epochs+1)
        ax_dfa.set_xlim(1, max_dfa_epochs+1)

        for ax, title in zip([ax_bp, ax_dfa], ["BP", "DFA"]):
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Test Error %")
            ax.set_title(f"{title} - Sample Size {sample_size}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")  # legends top-right on both

        fig.tight_layout()
        curve_plot_path = os.path.join(results_dir, f"curves_size{sample_size}.png")
        fig.savefig(curve_plot_path, dpi=200)
        plt.close(fig)
        print(f"Curve plot saved: {curve_plot_path}")

    # -------------------------
    # Plot: Final test error % vs sample size with error bars
    # -------------------------
    sample_sizes_list = []
    bp_means = []
    bp_stds = []
    dfa_means = []
    dfa_stds = []

    for size in sample_sizes:
        bp_vals = final_err_data[size]['bp']
        dfa_vals = final_err_data[size]['dfa']
        sample_sizes_list.append(size)
        bp_means.append(np.mean(bp_vals))
        bp_stds.append(np.std(bp_vals) if num_subsamples > 1 else 0.0)
        dfa_means.append(np.mean(dfa_vals))
        dfa_stds.append(np.std(dfa_vals) if num_subsamples > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(sample_sizes_list, bp_means, yerr=bp_stds, label="BP", fmt='o-', capsize=5)
    ax.errorbar(sample_sizes_list, dfa_means, yerr=dfa_stds, label="DFA", fmt='s-', capsize=5)
    ax.set_xscale("log")
    ax.set_xlabel("Training Sample Size (log scale)")
    ax.set_ylabel("Final Test Error % (mean ± std)")
    ax.set_title("Final Test Error vs Training Data Size (10x800)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    final_plot_path = os.path.join(results_dir, "final_error_vs_samples.png")
    fig.savefig(final_plot_path, dpi=200)
    plt.close(fig)
    print(f"Final summary plot saved: {final_plot_path}")

    # Save CSV summary
    csv_path = os.path.join(results_dir, "extension2_summary.csv")
    with open(csv_path, "w") as f:
        f.write("sample_size,bp_mean_err,bp_std_err,dfa_mean_err,dfa_std_err\n")
        for size, bp_m, bp_s, dfa_m, dfa_s in zip(sample_sizes_list, bp_means, bp_stds, dfa_means, dfa_stds):
            f.write(f"{size},{bp_m:.4f},{bp_s:.4f},{dfa_m:.4f},{dfa_s:.4f}\n")
    print(f"CSV summary saved: {csv_path}")

    print("\nExtension 2 completed!")

if __name__ == "__main__":
    main()
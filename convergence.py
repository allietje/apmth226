# convergence.py    
"""
convergence.py
Runs a grid search over widths and depths for BP vs DFA on MNIST.
Trains until convergence (or max 300 epochs), computes convergence epoch,
saves per-run CSV + final aggregated summary, and plots depth vs convergence
for each width.
Now with checkpointing: safe against 6-hour job timeouts.
"""
import csv
import statistics
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle  # for saving completed set

from src.utils_GPU import set_seed
from data.data import load_mnist
from src.model_GPU import create_networks
from src.experiment_GPU import run_bp_dfa_experiment_until_convergence

# ------------------------------------------------------------------
# Convergence detection
# ------------------------------------------------------------------
def moving_average(arr, window=5):
    return np.convolve(arr, np.ones(window) / window, mode='valid')

def check_convergence(test_errors, window=5, epsilon=0.1):
    if len(test_errors) < 2 * window:
        return False
    ma = moving_average(test_errors, window)
    return abs(ma[-1] - ma[-window]) < epsilon

# ------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------
def main():
    # -------------------------
    # Hyperparameters
    # -------------------------
    base_run_name = "convergence"
    base_seed = 42
    num_seeds = 3
    lr_bp = 0.0005
    lr_dfa = 0.001
    batch_size = 256
    max_epochs = 300
    convergence_window = 5
    convergence_epsilon = 0.1
    feedback_scale = 0.1
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    device_str = "cuda"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    widths = [200, 400, 600, 800, 1000]
    depths = [2, 4, 6, 8, 10, 12, 14]

    # -------------------------
    # Checkpointing setup (FULL STATE)
    # -------------------------
    checkpoint_file = os.path.join(results_dir, "convergence_checkpoint.pkl")
    state = {
        'completed_models': set(),
        'summary_rows': [],
        'plot_data': {w: {'depths': [], 'bp': [], 'bp_std': [], 'dfa': [], 'dfa_std': []} for w in widths}
    }

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            state = pickle.load(f)
        print(f"Resuming from checkpoint: {len(state['completed_models'])} models completed.")
    else:
        print("No checkpoint found — starting fresh.")

    completed_models = state['completed_models']
    summary_rows = state['summary_rows']
    plot_data = state['plot_data']

    # Data loading (once)
    set_seed(base_seed)
    (X_train, y_train), (X_test, y_test) = load_mnist(flatten=True, normalize=True)
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # Containers
    summary_rows = []
    plot_data = {w: {'depths': [], 'bp': [], 'bp_std': [], 'dfa': [], 'dfa_std': []} for w in widths}

    total_models = len(widths) * len(depths)
    model_counter = 0

    for width in widths:
        for depth in depths:
            model_name = f"{depth}x{width}"
            model_counter += 1

            # Skip if already completed
            if model_name in completed_models:
                print(f"\n=== [{model_counter}/{total_models}] Skipping completed model {model_name} ===")
                continue

            print(f"\n=== [{model_counter}/{total_models}] Running model {model_name} ===")

            bp_conv_epochs = []
            dfa_conv_epochs = []
            bp_final_err = []
            dfa_final_err = []

            for s in range(num_seeds):
                seed = base_seed + s  # same seeds across models
                set_seed(seed)
                print(f"--- Seed {seed} ---")

                bp_net, dfa_net, layer_sizes = create_networks(
                    width=width, depth=depth,
                    lr_bp=lr_bp, lr_dfa=lr_dfa,
                    seed=seed, feedback_scale=feedback_scale,
                    input_dim=input_dim, output_dim=output_dim,
                    device=str(device)
                )

                run_name = f"{model_name}_seed{seed}_{base_run_name}"
                config = {
                    "run_name": run_name, "seed": seed, "width": width, "depth": depth,
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

                # Per-run CSV
                per_run_csv = os.path.join(run_dir, "epoch_metrics.csv")
                with open(per_run_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch", "bp_test_err", "dfa_test_err",
                                     "bp_train_loss", "dfa_train_loss"])
                    for e in range(actual_epochs):
                        writer.writerow([
                            e+1,
                            metrics["test_err_bp"][e],
                            metrics["test_err_dfa"][e],
                            metrics["train_losses_bp"][e],
                            metrics["train_losses_dfa"][e]
                        ])

                # Convergence detection
                bp_conv = actual_epochs
                dfa_conv = actual_epochs
                for e in range(convergence_window * 2, actual_epochs):
                    if check_convergence(metrics["test_err_bp"][:e+1], convergence_window, convergence_epsilon):
                        bp_conv = e + 1
                        break
                for e in range(convergence_window * 2, actual_epochs):
                    if check_convergence(metrics["test_err_dfa"][:e+1], convergence_window, convergence_epsilon):
                        dfa_conv = e + 1
                        break

                bp_conv_epochs.append(bp_conv)
                dfa_conv_epochs.append(dfa_conv)
                bp_final_err.append(metrics["test_err_bp"][-1])
                dfa_final_err.append(metrics["test_err_dfa"][-1])

                print(f"BP converged at epoch {bp_conv} | final err {metrics['test_err_bp'][-1]:.2f}%")
                print(f"DFA converged at epoch {dfa_conv} | final err {metrics['test_err_dfa'][-1]:.2f}%")

            # Aggregate
            bp_mean_conv = statistics.mean(bp_conv_epochs)
            dfa_mean_conv = statistics.mean(dfa_conv_epochs)
            bp_std_conv = statistics.stdev(bp_conv_epochs) if num_seeds > 1 else 0.0
            dfa_std_conv = statistics.stdev(dfa_conv_epochs) if num_seeds > 1 else 0.0

            print(f"\nSummary {model_name}:")
            print(f" BP conv epoch: {bp_mean_conv:.1f} ± {bp_std_conv:.1f}")
            print(f" DFA conv epoch: {dfa_mean_conv:.1f} ± {dfa_std_conv:.1f}")

            summary_rows.append({
                "model": model_name,
                "depth": depth,
                "width": width,
                "n_seeds": num_seeds,
                "bp_conv_mean": bp_mean_conv,
                "bp_conv_std": bp_std_conv,
                "dfa_conv_mean": dfa_mean_conv,
                "dfa_conv_std": dfa_std_conv,
                "bp_final_err_mean": statistics.mean(bp_final_err),
                "dfa_final_err_mean": statistics.mean(dfa_final_err),
            })

            plot_data[width]['depths'].append(depth)
            plot_data[width]['bp'].append(bp_mean_conv)
            plot_data[width]['bp_std'].append(bp_std_conv)
            plot_data[width]['dfa'].append(dfa_mean_conv)
            plot_data[width]['dfa_std'].append(dfa_std_conv)

            # === Save full checkpoint after each model ===
            state['completed_models'].add(model_name)
            state['summary_rows'] = summary_rows
            state['plot_data'] = plot_data
            with open(checkpoint_file, "wb") as f:
                pickle.dump(state, f)
            print(f"Checkpoint updated: {len(state['completed_models'])} models completed.")

    # -------------------------
    # Save final CSV and plots (same as before)
    # -------------------------
    big_csv = os.path.join(results_dir, "convergence_grid_summary.csv")
    with open(big_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model","depth","width","n_seeds",
            "bp_conv_mean","bp_conv_std","dfa_conv_mean","dfa_conv_std",
            "bp_final_err_mean","dfa_final_err_mean"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nFinal summary saved to {big_csv}")


    # -------------------------
    # Plot: depth vs convergence epoch for each width
    # -------------------------
    for width in widths:
        d = plot_data[width]
        idx = np.argsort(d['depths'])
        depths_sorted = np.array(d['depths'])[idx]
        bp_mean = np.array(d['bp'])[idx]
        bp_std = np.array(d['bp_std'])[idx]
        dfa_mean = np.array(d['dfa'])[idx]
        dfa_std = np.array(d['dfa_std'])[idx]

        plt.figure(figsize=(8, 6))
        plt.errorbar(depths_sorted, bp_mean, yerr=bp_std,
                     label="BP", fmt='o-', capsize=5, markersize=8)
        plt.errorbar(depths_sorted, dfa_mean, yerr=dfa_std,
                     label="DFA", fmt='s-', capsize=5, markersize=8)
        plt.xlabel("Depth")
        plt.ylabel("Convergence Epoch (mean ± std)")
        plt.title(f"Convergence vs Depth (Width = {width})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = os.path.join(results_dir, f"convergence_vs_depth_width{width}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Plot saved: {plot_path}")
    
        # -------------------------
    # 7. Additional Plot: Width vs Convergence Epoch for each Depth and each Width
    # -------------------------
    print("\nGenerating width vs convergence epoch plots...")
    
    # Group data by depth instead of width
    conv_data_by_depth = {depth: {'widths': [], 'bp_mean': [], 'bp_std': [], 'dfa_mean': [], 'dfa_std': []} for depth in depths}
    
    for row in summary_rows:
        d = row['depth']
        w = row['width']
        conv_data_by_depth[d]['widths'].append(w)
        conv_data_by_depth[d]['bp_mean'].append(row['bp_conv_mean'])
        conv_data_by_depth[d]['bp_std'].append(row['bp_conv_std'])
        conv_data_by_depth[d]['dfa_mean'].append(row['dfa_conv_mean'])
        conv_data_by_depth[d]['dfa_std'].append(row['dfa_conv_std'])
    
    for depth in depths:
        data = conv_data_by_depth[depth]
        if not data['widths']:
            continue  # skip if no data (shouldn't happen)
        
        # Sort by width
        idx = np.argsort(data['widths'])
        widths_sorted = np.array(data['widths'])[idx]
        bp_mean = np.array(data['bp_mean'])[idx]
        bp_std = np.array(data['bp_std'])[idx]
        dfa_mean = np.array(data['dfa_mean'])[idx]
        dfa_std = np.array(data['dfa_std'])[idx]
        
        plt.figure(figsize=(8, 6))
        plt.errorbar(widths_sorted, bp_mean, yerr=bp_std,
                     label="BP", fmt='o-', capsize=5, markersize=8, color='tab:blue')
        plt.errorbar(widths_sorted, dfa_mean, yerr=dfa_std,
                     label="DFA", fmt='s-', capsize=5, markersize=8, color='tab:orange')
        
        plt.xlabel("Width (Hidden Units)")
        plt.ylabel("Convergence Epoch (mean ± std)")
        plt.title(f"Convergence Epoch vs Width (Depth = {depth})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path_width = os.path.join(results_dir, f"convergence_vs_width_depth{depth}.png")
        plt.savefig(plot_path_width, dpi=200)
        plt.close()
        print(f"Plot saved: {plot_path_width}")


    print("\nAll done! Full grid search completed safely.")

if __name__ == "__main__":
    main()
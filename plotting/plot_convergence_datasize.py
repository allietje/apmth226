# plot_convergence_datasize.py
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import json

# Convergence detection functions (same as datasize.py)
def moving_average(arr, window=5):
    return np.convolve(arr, np.ones(window) / window, mode='valid')

def check_convergence(test_errors, window=5, epsilon=0.1):
    if len(test_errors) < 2 * window:
        return False
    ma = moving_average(test_errors, window)
    return abs(ma[-1] - ma[-window]) < epsilon

def extract_convergence_epochs(metrics_file):
    """Extract convergence epochs from saved metrics"""
    try:
        data = np.load(metrics_file)
        
        # Get test error arrays
        test_err_bp = data['test_err_bp']
        test_err_dfa = data['test_err_dfa']
        
        # Find convergence epochs
        bp_conv_epoch = len(test_err_bp)  # Default to max
        dfa_conv_epoch = len(test_err_dfa)
        
        convergence_window = 5
        convergence_epsilon = 0.1
        
        for epoch in range(convergence_window * 2, len(test_err_bp)):
            if check_convergence(test_err_bp[:epoch+1], convergence_window, convergence_epsilon):
                bp_conv_epoch = epoch + 1
                break
                
        for epoch in range(convergence_window * 2, len(test_err_dfa)):
            if check_convergence(test_err_dfa[:epoch+1], convergence_window, convergence_epsilon):
                dfa_conv_epoch = epoch + 1
                break
        
        return bp_conv_epoch, dfa_conv_epoch
    except Exception as e:
        print(f"Error loading {metrics_file}: {e}")
        return None, None

def main():
    # Path to results directory (relative to project root)
    results_dir = os.path.join("..", "results")
    
    # Find all result directories
    pattern = os.path.join(results_dir, "*_size*_subsample*_datasize")
    run_dirs = glob.glob(pattern)
    
    if not run_dirs:
        print("No result directories found")
        return
    
    # Group by sample size
    convergence_data = {}
    
    for run_dir in sorted(run_dirs):  # Debug with first 3 only
        print(f"\nProcessing: {run_dir}")
        try:
            # Extract sample size from directory name
            dirname = os.path.basename(run_dir)
            # Find the part that starts with 'size' and extract the number
            import re
            size_match = re.search(r'size(\d+)', dirname)
            if not size_match:
                print(f"Could not parse sample size from: {dirname}")
                continue
            sample_size = int(size_match.group(1))
            print(f"Sample size: {sample_size}")
            
            # Load config to verify
            config_file = os.path.join(run_dir, 'config.json')
            if not os.path.exists(config_file):
                print(f"Config file not found: {config_file}")
                continue
                
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if config.get('sample_size') != sample_size:
                print(f"Config sample_size {config.get('sample_size')} != parsed {sample_size}")
                continue
            
            # Extract convergence epochs
            metrics_file = os.path.join(run_dir, 'metrics.npz')
            if not os.path.exists(metrics_file):
                print(f"Metrics file not found: {metrics_file}")
                continue
                
            print(f"Loading metrics from: {metrics_file}")
            bp_conv, dfa_conv = extract_convergence_epochs(metrics_file)
            print(f"Convergence epochs: BP={bp_conv}, DFA={dfa_conv}")
            
            if bp_conv is not None and dfa_conv is not None:
                if sample_size not in convergence_data:
                    convergence_data[sample_size] = {'bp': [], 'dfa': []}
                
                convergence_data[sample_size]['bp'].append(bp_conv)
                convergence_data[sample_size]['dfa'].append(dfa_conv)
                
                print(f"Size {sample_size}: BP conv {bp_conv}, DFA conv {dfa_conv}")
        
        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nConvergence data collected: {convergence_data}")
    if not convergence_data:
        print("No convergence data found")
        return
    
    # Prepare data for plotting
    sample_sizes = sorted(convergence_data.keys())
    bp_means = []
    bp_stds = []
    dfa_means = []
    dfa_stds = []
    
    for size in sample_sizes:
        bp_vals = convergence_data[size]['bp']
        dfa_vals = convergence_data[size]['dfa']
        
        bp_means.append(np.mean(bp_vals))
        bp_stds.append(np.std(bp_vals) if len(bp_vals) > 1 else 0.0)
        dfa_means.append(np.mean(dfa_vals))
        dfa_stds.append(np.std(dfa_vals) if len(dfa_vals) > 1 else 0.0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(sample_sizes, bp_means, yerr=bp_stds, label="BP", 
                fmt='o-', capsize=5, markersize=6)
    ax.errorbar(sample_sizes, dfa_means, yerr=dfa_stds, label="DFA", 
                fmt='s-', capsize=5, markersize=6)
    
    ax.set_xscale("log")
    ax.set_xlabel("Training Sample Size (log scale)")
    ax.set_ylabel("Convergence Epoch")
    ax.set_title("Average Convergence Speed vs Training Data Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot to results directory
    plot_path = os.path.join(results_dir, "convergence_speed_vs_samples.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Convergence speed plot saved: {plot_path}")
    
    # Save CSV summary to results directory
    csv_path = os.path.join(results_dir, "convergence_speed_summary.csv")
    with open(csv_path, "w") as f:
        f.write("sample_size,bp_mean_epochs,bp_std_epochs,dfa_mean_epochs,dfa_std_epochs\n")
        for size, bp_m, bp_s, dfa_m, dfa_s in zip(sample_sizes, bp_means, bp_stds, dfa_means, dfa_stds):
            f.write(f"{size},{bp_m:.2f},{bp_s:.2f},{dfa_m:.2f},{dfa_s:.2f}\n")
    print(f"Convergence speed CSV saved: {csv_path}")

if __name__ == "__main__":
    main()
# convergence_plotting.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_bp_vs_dfa_error_by_width(csv_path, save_path=None):
    """
    Plot BP vs DFA final error % holding width constant.
    
    Args:
        csv_path: Path to convergence_grid_summary.csv
        save_path: Optional path to save the plot
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Get unique widths and sort them
    widths = sorted(df['width'].unique())
    
    # Create subplots - arrange in a grid
    n_widths = len(widths)
    n_cols = min(3, n_widths)
    n_rows = int(np.ceil(n_widths / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_widths == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, width in enumerate(widths):
        ax = axes[i]
        
        # Filter data for this width
        width_data = df[df['width'] == width].sort_values('depth')
        
        # Plot BP and DFA error rates
        ax.plot(width_data['depth'], width_data['bp_final_err_mean'], 
                'o-', label='BP', color='blue', linewidth=2, markersize=6)
        ax.plot(width_data['depth'], width_data['dfa_final_err_mean'], 
                's-', label='DFA', color='red', linewidth=2, markersize=6)
        
        # Formatting
        ax.set_xlabel('Depth')
        ax.set_ylabel('Final Error (%)')
        ax.set_title(f'Width = {width}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        max_error = max(width_data['bp_final_err_mean'].max(), 
                       width_data['dfa_final_err_mean'].max())
        ax.set_ylim(0, min(max_error * 1.1, 100))  # Cap at 100% if needed
    
    # Hide empty subplots if any
    for i in range(len(widths), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_bp_vs_dfa_error_single_width(csv_path, width, save_path=None):
    """
    Plot BP vs DFA final error % for a single width.
    
    Args:
        csv_path: Path to convergence_grid_summary.csv
        width: Specific width to plot
        save_path: Optional path to save the plot
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Filter for the specific width
    width_data = df[df['width'] == width].sort_values('depth')
    
    if width_data.empty:
        print(f"No data found for width {width}")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(width_data['depth'], width_data['bp_final_err_mean'], 
             'o-', label='BP', color='blue', linewidth=3, markersize=8)
    plt.plot(width_data['depth'], width_data['dfa_final_err_mean'], 
             's-', label='DFA', color='red', linewidth=3, markersize=8)
    
    # Add error bars if available (using std deviation as approximation)
    # Note: CSV has bp_conv_std and dfa_conv_std, but we could add final error std if available
    
    plt.xlabel('Network Depth', fontsize=12)
    plt.ylabel('Final Test Error (%)', fontsize=12)
    plt.title(f'BP vs DFA Final Error (Width = {width})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    max_error = max(width_data['bp_final_err_mean'].max(), 
                   width_data['dfa_final_err_mean'].max())
    plt.ylim(0, min(max_error * 1.2, 100))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
if __name__ == "__main__":
    csv_path = "/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/final_results/convergence/convergence_grid_summary.csv"
    
    # Plot all widths in a grid
    plot_bp_vs_dfa_error_by_width(csv_path, "results/bp_vs_dfa_all_widths.png")
    
    # Plot a specific width
    plot_bp_vs_dfa_error_single_width(csv_path, 800, "results/bp_vs_dfa_width_800.png")
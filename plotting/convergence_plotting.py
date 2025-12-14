# convergence_plotting.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from plotting_style import apply_style, get_bp_dfa_colors, save_plot, set_log_scale

# Apply consistent styling
apply_style()

def plot_bp_vs_dfa_error_by_width(csv_path, save_path=None, subdir='convergence'):
    """
    Plot BP vs DFA final error % holding width constant.
    
    Args:
        csv_path: Path to convergence_grid_summary.csv
        save_path: Optional path to save the plot
        subdir: Subdirectory for saving plots
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
    
    colors = get_bp_dfa_colors()
    
    for i, width in enumerate(widths):
        ax = axes[i]
        
        # Filter data for this width
        width_data = df[df['width'] == width].sort_values('depth')
        
        # Plot BP and DFA error rates
        ax.plot(width_data['depth'], width_data['bp_final_err_mean'], 
                'o-', label='BP', color=colors['bp'], markersize=6)
        ax.plot(width_data['depth'], width_data['dfa_final_err_mean'], 
                's-', label='DFA', color=colors['dfa'], markersize=6)
        
        # Formatting
        ax.set_xlabel('Depth')
        ax.set_ylabel('Final Error (%)')
        ax.set_title(f'Width = {width}')
        ax.legend()
        
        # Set reasonable y-axis limits
        max_error = max(width_data['bp_final_err_mean'].max(), 
                       width_data['dfa_final_err_mean'].max())
        ax.set_ylim(0, min(max_error * 1.1, 100))  # Cap at 100% if needed
    
    # Hide empty subplots if any
    for i in range(len(widths), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        save_plot(fig, save_path, subdir=subdir)
    plt.show()

def plot_bp_vs_dfa_error_single_width(csv_path, width, save_path=None, subdir='convergence'):
    """
    Plot BP vs DFA final error % for a single width.
    
    Args:
        csv_path: Path to convergence_grid_summary.csv
        width: Specific width to plot
        save_path: Optional path to save the plot
        subdir: Subdirectory for saving plots
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Filter for the specific width
    width_data = df[df['width'] == width].sort_values('depth')
    
    if width_data.empty:
        print(f"No data found for width {width}")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    colors = get_bp_dfa_colors()
    
    ax.plot(width_data['depth'], width_data['bp_final_err_mean'], 
             'o-', label='BP', color=colors['bp'], markersize=6)
    ax.plot(width_data['depth'], width_data['dfa_final_err_mean'], 
             's-', label='DFA', color=colors['dfa'], markersize=6)
    
    ax.set_xlabel('Network Depth')
    ax.set_ylabel('Final Test Error (%)')
    ax.set_title(f'BP vs DFA Final Error (Width = {width})')
    ax.legend()
    
    # Set reasonable y-axis limits
    max_error = max(width_data['bp_final_err_mean'].max(), 
                   width_data['dfa_final_err_mean'].max())
    ax.set_ylim(0, min(max_error * 1.2, 100))
    
    plt.tight_layout()
    if save_path:
        save_plot(fig, save_path, subdir=subdir)
    plt.show()

# Example usage:
if __name__ == "__main__":
    csv_path = "../figures/convergence/convergence_grid_summary.csv"
    
    # Plot all widths in a grid
    plot_bp_vs_dfa_error_by_width(csv_path, "bp_vs_dfa_all_widths")
    
    # Plot a specific width
    # plot_bp_vs_dfa_error_single_width(csv_path, 800, "bp_vs_dfa_width_800")
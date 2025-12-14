# epochs_plotting.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from plotting_style import apply_style, get_bp_dfa_colors, save_plot, set_log_scale

# Apply consistent styling
apply_style()

def create_convergence_subplots(subdir='convergence'):
    """
    Create two subplot figures:
    1. All depths with width constant (6 panels, one per width)
    2. All widths with depth constant (8 panels, one per depth)
    """
    
    # Read the data
    csv_path = "../figures/convergence/convergence_grid_summary.csv"
    df = pd.read_csv(csv_path)
    
    # Define the widths and depths
    widths = [200, 400, 600, 800, 1000, 1200]  # 6 widths
    depths = [2, 4, 6, 8, 10, 12, 14, 16]     # 8 depths
    
    colors = get_bp_dfa_colors()
    
    # ============================================================================
    # FIGURE 1: All depths with width constant (6 panels)
    # ============================================================================
    
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))  # 2x3 grid = 6 panels
    axes1 = axes1.flatten()
    
    for i, width in enumerate(widths):
        ax = axes1[i]
        
        # Filter data for this width and sort by depth
        width_data = df[df['width'] == width].sort_values('depth')
        
        # Plot BP and DFA final errors vs depth
        ax.plot(width_data['depth'], width_data['bp_final_err_mean'], 
                'o-', label='BP', color=colors['bp'], markersize=6)
        ax.plot(width_data['depth'], width_data['dfa_final_err_mean'], 
                's-', label='DFA', color=colors['dfa'], markersize=6)
        
        ax.set_xlabel('Depth')
        ax.set_ylabel('Final Error (%)')
        ax.set_title(f'Width = {width}')
        ax.legend()
        
        # Set reasonable y-axis limits
        max_error = max(width_data['bp_final_err_mean'].max(), 
                       width_data['dfa_final_err_mean'].max())
        ax.set_ylim(0, min(max_error * 1.1, 100))
    
    plt.tight_layout()
    save_plot(fig1, 'error_vs_depth_by_width', subdir=subdir)
    plt.close(fig1)
    
    # ============================================================================
    # FIGURE 2: All widths with depth constant (8 panels)
    # ============================================================================
    
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))  # 2x4 grid = 8 panels
    axes2 = axes2.flatten()
    
    for i, depth in enumerate(depths):
        ax = axes2[i]
        
        # Filter data for this depth and sort by width
        depth_data = df[df['depth'] == depth].sort_values('width')
        
        # Plot BP and DFA final errors vs width
        ax.plot(depth_data['width'], depth_data['bp_final_err_mean'], 
                'o-', label='BP', color=colors['bp'], markersize=6)
        ax.plot(depth_data['width'], depth_data['dfa_final_err_mean'], 
                's-', label='DFA', color=colors['dfa'], markersize=6)
        
        ax.set_xlabel('Width')
        ax.set_ylabel('Final Error (%)')
        ax.set_title(f'Depth = {depth}')
        ax.legend()
        
        # Set reasonable y-axis limits
        max_error = max(depth_data['bp_final_err_mean'].max(), 
                       depth_data['dfa_final_err_mean'].max())
        ax.set_ylim(0, min(max_error * 1.1, 100))
    
    plt.tight_layout()
    save_plot(fig2, 'error_vs_width_by_depth', subdir=subdir)
    plt.close(fig2)

def create_convergence_subplots_convergence_epochs(subdir='convergence'):
    """
    Create two subplot figures for convergence epochs:
    1. All depths with width constant (6 panels, one per width)
    2. All widths with depth constant (8 panels, one per depth)
    """
    
    # Read the data
    csv_path = "../figures/convergence/convergence_grid_summary.csv"
    df = pd.read_csv(csv_path)
    
    # Define the widths and depths
    widths = [200, 400, 600, 800, 1000, 1200]  # 6 widths
    depths = [2, 4, 6, 8, 10, 12, 14, 16]     # 8 depths
    
    colors = get_bp_dfa_colors()
    
    # ============================================================================
    # FIGURE 1: Convergence epochs - All depths with width constant (6 panels)
    # ============================================================================
    
    # First pass: find global maximum y-value across all panels
    global_max_epoch_fig1 = 0
    for width in widths:
        width_data = df[df['width'] == width].sort_values('depth')
        max_epoch = max(width_data['bp_conv_mean'].max() + width_data['bp_conv_std'].max(),
                       width_data['dfa_conv_mean'].max() + width_data['dfa_conv_std'].max())
        global_max_epoch_fig1 = max(global_max_epoch_fig1, max_epoch)
    
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))  # 2x3 grid = 6 panels
    axes1 = axes1.flatten()
    
    for i, width in enumerate(widths):
        ax = axes1[i]
        
        # Filter data for this width and sort by depth
        width_data = df[df['width'] == width].sort_values('depth')
        
        # Plot BP and DFA convergence epochs vs depth
        ax.errorbar(width_data['depth'], width_data['bp_conv_mean'], 
                   yerr=width_data['bp_conv_std'],
                   label='BP', fmt='o-', color=colors['bp'])
        ax.errorbar(width_data['depth'], width_data['dfa_conv_mean'], 
                   yerr=width_data['dfa_conv_std'],
                   label='DFA', fmt='s-', color=colors['dfa'])
        
        ax.set_xlabel('Depth')
        ax.set_ylabel('Convergence Epoch')
        ax.set_title(f'Width = {width}')
        ax.legend()
        
        # Set consistent y-axis limits across all panels
        ax.set_ylim(0, global_max_epoch_fig1 * 1.1)
    
    plt.tight_layout()
    save_plot(fig1, 'convergence_vs_depth_by_width', subdir=subdir)
    plt.close(fig1)
    
    # ============================================================================
    # FIGURE 2: Convergence epochs - All widths with depth constant (8 panels)
    # ============================================================================
    
    # First pass: find global maximum y-value across all panels
    global_max_epoch_fig2 = 0
    for depth in depths:
        depth_data = df[df['depth'] == depth].sort_values('width')
        max_epoch = max(depth_data['bp_conv_mean'].max() + depth_data['bp_conv_std'].max(),
                       depth_data['dfa_conv_mean'].max() + depth_data['dfa_conv_std'].max())
        global_max_epoch_fig2 = max(global_max_epoch_fig2, max_epoch)
    
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))  # 2x4 grid = 8 panels
    axes2 = axes2.flatten()
    
    for i, depth in enumerate(depths):
        ax = axes2[i]
        
        # Filter data for this depth and sort by width
        depth_data = df[df['depth'] == depth].sort_values('width')
        
        # Plot BP and DFA convergence epochs vs width
        ax.errorbar(depth_data['width'], depth_data['bp_conv_mean'], 
                   yerr=depth_data['bp_conv_std'],
                   label='BP', fmt='o-', color=colors['bp'])
        ax.errorbar(depth_data['width'], depth_data['dfa_conv_mean'], 
                   yerr=depth_data['dfa_conv_std'],
                   label='DFA', fmt='s-', color=colors['dfa'])
        
        ax.set_xlabel('Width')
        ax.set_ylabel('Convergence Epoch')
        ax.set_title(f'Depth = {depth}')
        ax.legend()
        
        # Set consistent y-axis limits across all panels
        ax.set_ylim(0, global_max_epoch_fig2 * 1.1)
    
    plt.tight_layout()
    save_plot(fig2, 'convergence_vs_width_by_depth', subdir=subdir)
    plt.close(fig2)
    
if __name__ == "__main__":
    create_convergence_subplots()
    create_convergence_subplots_convergence_epochs()
# epochs_plotting.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_convergence_subplots():
    """
    Create two subplot figures:
    1. All depths with width constant (6 panels, one per width)
    2. All widths with depth constant (8 panels, one per depth)
    """
    
    # Read the data
    csv_path = "/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/final_results/convergence/convergence_grid_summary.csv"
    df = pd.read_csv(csv_path)
    
    # Define the widths and depths
    widths = [200, 400, 600, 800, 1000, 1200]  # 6 widths
    depths = [2, 4, 6, 8, 10, 12, 14, 16]     # 8 depths
    
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
                'o-', label='BP', color='blue', linewidth=2, markersize=6)
        ax.plot(width_data['depth'], width_data['dfa_final_err_mean'], 
                's-', label='DFA', color='red', linewidth=2, markersize=6)
        
        ax.set_xlabel('Depth')
        ax.set_ylabel('Final Error (%)')
        ax.set_title(f'Width = {width}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        max_error = max(width_data['bp_final_err_mean'].max(), 
                       width_data['dfa_final_err_mean'].max())
        ax.set_ylim(0, min(max_error * 1.1, 100))
    
    fig1.suptitle('BP vs DFA Final Error: Depth vs Error (Width Constant)', fontsize=16)
    plt.tight_layout()
    fig1.savefig('/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/final_results/convergence/error_vs_depth_by_width.png', 
                 dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("Saved: error_vs_depth_by_width.png")
    
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
                'o-', label='BP', color='blue', linewidth=2, markersize=6)
        ax.plot(depth_data['width'], depth_data['dfa_final_err_mean'], 
                's-', label='DFA', color='red', linewidth=2, markersize=6)
        
        ax.set_xlabel('Width')
        ax.set_ylabel('Final Error (%)')
        ax.set_title(f'Depth = {depth}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        max_error = max(depth_data['bp_final_err_mean'].max(), 
                       depth_data['dfa_final_err_mean'].max())
        ax.set_ylim(0, min(max_error * 1.1, 100))
    
    fig2.suptitle('BP vs DFA Final Error: Width vs Error (Depth Constant)', fontsize=16)
    plt.tight_layout()
    fig2.savefig('/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/final_results/convergence/error_vs_width_by_depth.png', 
                 dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("Saved: error_vs_width_by_depth.png")

def create_convergence_subplots_convergence_epochs():
    """
    Create two subplot figures for convergence epochs:
    1. All depths with width constant (6 panels, one per width)
    2. All widths with depth constant (8 panels, one per depth)
    """
    
    # Read the data
    csv_path = "/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/final_results/convergence/convergence_grid_summary.csv"
    df = pd.read_csv(csv_path)
    
    # Define the widths and depths
    widths = [200, 400, 600, 800, 1000, 1200]  # 6 widths
    depths = [2, 4, 6, 8, 10, 12, 14, 16]     # 8 depths
    
    # ============================================================================
    # FIGURE 1: Convergence epochs - All depths with width constant (6 panels)
    # ============================================================================
    
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))  # 2x3 grid = 6 panels
    axes1 = axes1.flatten()
    
    for i, width in enumerate(widths):
        ax = axes1[i]
        
        # Filter data for this width and sort by depth
        width_data = df[df['width'] == width].sort_values('depth')
        
        # Plot BP and DFA convergence epochs vs depth
        ax.errorbar(width_data['depth'], width_data['bp_conv_mean'], 
                   yerr=width_data['bp_conv_std'],
                   label='BP', fmt='o-', capsize=3, markersize=6, color='blue')
        ax.errorbar(width_data['depth'], width_data['dfa_conv_mean'], 
                   yerr=width_data['dfa_conv_std'],
                   label='DFA', fmt='s-', capsize=3, markersize=6, color='red')
        
        ax.set_xlabel('Depth')
        ax.set_ylabel('Convergence Epoch')
        ax.set_title(f'Width = {width}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        max_epoch = max(width_data['bp_conv_mean'].max() + width_data['bp_conv_std'].max(),
                       width_data['dfa_conv_mean'].max() + width_data['dfa_conv_std'].max())
        ax.set_ylim(0, max_epoch * 1.1)
    
    fig1.suptitle('BP vs DFA Convergence Epochs: Depth vs Convergence (Width Constant)', fontsize=16)
    plt.tight_layout()
    fig1.savefig('/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/final_results/convergence/convergence_vs_depth_by_width.png', 
                 dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("Saved: convergence_vs_depth_by_width.png")
    
    # ============================================================================
    # FIGURE 2: Convergence epochs - All widths with depth constant (8 panels)
    # ============================================================================
    
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))  # 2x4 grid = 8 panels
    axes2 = axes2.flatten()
    
    for i, depth in enumerate(depths):
        ax = axes2[i]
        
        # Filter data for this depth and sort by width
        depth_data = df[df['depth'] == depth].sort_values('width')
        
        # Plot BP and DFA convergence epochs vs width
        ax.errorbar(depth_data['width'], depth_data['bp_conv_mean'], 
                   yerr=depth_data['bp_conv_std'],
                   label='BP', fmt='o-', capsize=3, markersize=6, color='blue')
        ax.errorbar(depth_data['width'], depth_data['dfa_conv_mean'], 
                   yerr=depth_data['dfa_conv_std'],
                   label='DFA', fmt='s-', capsize=3, markersize=6, color='red')
        
        ax.set_xlabel('Width')
        ax.set_ylabel('Convergence Epoch')
        ax.set_title(f'Depth = {depth}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        max_epoch = max(depth_data['bp_conv_mean'].max() + depth_data['bp_conv_std'].max(),
                       depth_data['dfa_conv_mean'].max() + depth_data['dfa_conv_std'].max())
        ax.set_ylim(0, max_epoch * 1.1)
    
    fig2.suptitle('BP vs DFA Convergence Epochs: Width vs Convergence (Depth Constant)', fontsize=16)
    plt.tight_layout()
    fig2.savefig('/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/final_results/convergence/convergence_vs_width_by_depth.png', 
                 dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("Saved: convergence_vs_width_by_depth.png")

if __name__ == "__main__":
    create_convergence_subplots()
    create_convergence_subplots_convergence_epochs()
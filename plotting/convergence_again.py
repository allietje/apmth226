# convergence_again.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def create_paper_quality_barplots():
    """
    Create publication-quality bar plots for convergence analysis.
    """
    
    # Read the data
    csv_path = "/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/final_results/convergence/convergence_grid_summary.csv"
    df = pd.read_csv(csv_path)
    
    # Define the widths and depths
    widths = [200, 400, 600, 800, 1000, 1200]
    depths = [2, 4, 6, 8, 10, 12, 14, 16]
    
    # Set up the plotting style for publication quality
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.grid': False,
        'grid.alpha': 0.3
    })
    
    # ============================================================================
    # BAR PLOT 1: Convergence vs Depth (Width Constant) - 6 panels
    # ============================================================================
    
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    axes1 = axes1.flatten()
    
    # Color scheme
    bp_color = '#2E86AB'  # Professional blue
    dfa_color = '#F24236'  # Professional red
    
    bar_width = 0.35
    x_positions = np.arange(len(depths))
    
    for i, width in enumerate(widths):
        ax = axes1[i]
        
        # Filter data for this width and sort by depth
        width_data = df[df['width'] == width].sort_values('depth')
        
        # Create bars for BP and DFA
        bp_bars = ax.bar(x_positions - bar_width/2, width_data['bp_conv_mean'], 
                        bar_width, yerr=width_data['bp_conv_std'],
                        color=bp_color, alpha=0.8, capsize=3, 
                        error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        dfa_bars = ax.bar(x_positions + bar_width/2, width_data['dfa_conv_mean'], 
                         bar_width, yerr=width_data['dfa_conv_std'],
                         color=dfa_color, alpha=0.8, capsize=3,
                         error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        
        # Formatting
        ax.set_xlabel('Network Depth', fontsize=12, fontweight='bold')
        ax.set_ylabel('Convergence Epoch', fontsize=12, fontweight='bold')
        ax.set_title(f'Width = {width}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(depths)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
    
    # Create custom legend - positioned below the title
    bp_patch = mpatches.Patch(color=bp_color, alpha=0.8, label='Backpropagation')
    dfa_patch = mpatches.Patch(color=dfa_color, alpha=0.8, label='Direct Feedback Alignment')
    fig1.legend(handles=[bp_patch, dfa_patch], loc='upper center', 
                bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, fontsize=14)
    
    fig1.suptitle('Convergence Analysis: Network Depth vs Convergence Epoch', 
                  fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig('/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/final_results/convergence/convergence_barplot_depth_by_width.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print("Saved: convergence_barplot_depth_by_width.png")
    
    # ============================================================================
    # BAR PLOT 2: Convergence vs Width (Depth Constant) - 8 panels
    # ============================================================================
    
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10), dpi=300)
    axes2 = axes2.flatten()
    
    bar_width = 0.35
    x_positions = np.arange(len(widths))
    width_labels = [str(w) for w in widths]
    
    for i, depth in enumerate(depths):
        ax = axes2[i]
        
        # Filter data for this depth and sort by width
        depth_data = df[df['depth'] == depth].sort_values('width')
        
        # Create bars for BP and DFA
        bp_bars = ax.bar(x_positions - bar_width/2, depth_data['bp_conv_mean'], 
                        bar_width, yerr=depth_data['bp_conv_std'],
                        color=bp_color, alpha=0.8, capsize=3,
                        error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        dfa_bars = ax.bar(x_positions + bar_width/2, depth_data['dfa_conv_mean'], 
                         bar_width, yerr=depth_data['dfa_conv_std'],
                         color=dfa_color, alpha=0.8, capsize=3,
                         error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        
        # Formatting
        ax.set_xlabel('Network Width', fontsize=12, fontweight='bold')
        ax.set_ylabel('Convergence Epoch', fontsize=12, fontweight='bold')
        ax.set_title(f'Depth = {depth}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(width_labels, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
    
    # Create custom legend - positioned below the title
    bp_patch = mpatches.Patch(color=bp_color, alpha=0.8, label='Backpropagation')
    dfa_patch = mpatches.Patch(color=dfa_color, alpha=0.8, label='Direct Feedback Alignment')
    fig2.legend(handles=[bp_patch, dfa_patch], loc='upper center', 
                bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, fontsize=14)
    
    fig2.suptitle('Convergence Analysis: Network Width vs Convergence Epoch', 
                  fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig('/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/final_results/convergence/convergence_barplot_width_by_depth.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print("Saved: convergence_barplot_width_by_depth.png")

def create_single_summary_barplot():
    """
    Create a single summary bar plot showing the overall convergence comparison.
    """
    
    csv_path = "/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/final_results/convergence/convergence_grid_summary.csv"
    df = pd.read_csv(csv_path)
    
    # Calculate overall averages across all architectures
    bp_overall_mean = df['bp_conv_mean'].mean()
    bp_overall_std = df['bp_conv_mean'].std()
    dfa_overall_mean = df['dfa_conv_mean'].mean()
    dfa_overall_std = df['dfa_conv_mean'].std()
    
    # Create the summary plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    
    bp_color = '#2E86AB'
    dfa_color = '#F24236'
    
    methods = ['Backpropagation', 'Direct Feedback Alignment']
    means = [bp_overall_mean, dfa_overall_mean]
    stds = [bp_overall_std, dfa_overall_std]
    colors = [bp_color, dfa_color]
    
    bars = ax.bar(methods, means, yerr=stds, color=colors, alpha=0.8, 
                  capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Remove the value labels on bars
    # for bar, mean in zip(bars, means):
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width()/2., height + stds[bars.index(bar)] + 1,
    #             '.1f', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Average Convergence Epoch', fontsize=14, fontweight='bold')
    ax.set_title('Overall Convergence Comparison\n(Across All Architectures)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    fig.savefig('/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/final_results/convergence/convergence_summary_barplot.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Saved: convergence_summary_barplot.png")
    
    print(".1f")
    print(".1f")

if __name__ == "__main__":
    create_paper_quality_barplots()
    create_single_summary_barplot()
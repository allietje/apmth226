# plot_datasize.py
"""
plot_datasize.py - Replot datasize analysis figures with consistent styling
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from plotting_style import apply_style, get_bp_dfa_colors, save_plot

# Apply consistent styling
apply_style()

def replot_final_error_vs_samples():
    """Replot final error vs sample size with new styling"""
    
    # Read the data
    csv_path = "../final_results/datasize/extension2_summary.csv"
    df = pd.read_csv(csv_path)
    
    # Extract data
    sample_sizes = df['sample_size'].values
    bp_means = df['bp_mean_err'].values
    bp_stds = df['bp_std_err'].values
    dfa_means = df['dfa_mean_err'].values
    dfa_stds = df['dfa_std_err'].values
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    colors = get_bp_dfa_colors()
    
    ax.errorbar(sample_sizes, bp_means, yerr=bp_stds, 
                label='BP', color=colors['bp'], fmt='o-', capsize=3)
    ax.errorbar(sample_sizes, dfa_means, yerr=dfa_stds, 
                label='DFA', color=colors['dfa'], fmt='s-', capsize=3)
    
    ax.set_xscale('log')
    ax.set_xlabel('Training Sample Size (log scale)')
    ax.set_ylabel('Final Test Error (%)')
    # No title as requested
    ax.legend()
    
    # Save to datasize directory
    save_plot(fig, 'final_error_vs_samples', subdir='datasize')
    plt.close(fig)

def replot_convergence_speed_vs_samples():
    """Replot convergence speed vs sample size with new styling"""
    
    # Read the data
    csv_path = "../final_results/datasize/convergence_speed_summary.csv"
    df = pd.read_csv(csv_path)
    
    # Extract data
    sample_sizes = df['sample_size'].values
    bp_means = df['bp_mean_epochs'].values
    bp_stds = df['bp_std_epochs'].values
    dfa_means = df['dfa_mean_epochs'].values
    dfa_stds = df['dfa_std_epochs'].values
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    colors = get_bp_dfa_colors()
    
    ax.errorbar(sample_sizes, bp_means, yerr=bp_stds, 
                label='BP', color=colors['bp'], fmt='o-', capsize=3)
    ax.errorbar(sample_sizes, dfa_means, yerr=dfa_stds, 
                label='DFA', color=colors['dfa'], fmt='s-', capsize=3)
    
    ax.set_xscale('log')
    ax.set_xlabel('Training Sample Size (log scale)')
    ax.set_ylabel('Convergence Epoch')
    # No title as requested
    ax.legend()
    
    # Save to datasize directory
    save_plot(fig, 'convergence_speed_vs_samples', subdir='datasize')
    plt.close(fig)

if __name__ == "__main__":
    print("Replotting datasize figures with consistent styling...")
    
    replot_final_error_vs_samples()
    print("✓ Replotted final_error_vs_samples.png")
    
    replot_convergence_speed_vs_samples()
    print("✓ Replotted convergence_speed_vs_samples.png")
    
    print("\nPlots saved to final_results/datasize/ with consistent styling (no titles)")
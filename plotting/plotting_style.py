# plotting_style.py
"""
plotting_style.py - Consistent plotting style and utilities for DFA experiments

This module provides standardized matplotlib/seaborn styling and helper functions
for creating publication-ready plots across all DFA experiments.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

def setup_publication_style():
    """
    Set up matplotlib parameters for publication-quality plots.
    
    Call this at the beginning of any plotting script to ensure consistent styling.
    """
    # Base style parameters
    plt.rcParams.update({
        'font.family': 'Helvetica',  # Or 'Arial' if Helvetica not available
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.figsize': (4, 3),  # Default size
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'axes.linewidth': 0.8,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'errorbar.capsize': 3,
    })
    
    # Try to use Helvetica, fall back to Arial if not available
    try:
        plt.rcParams['font.family'] = 'Arial'
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'  # matplotlib default
    
    # Set color cycle for BP vs DFA comparisons
    bp_dfa_colors = ['#1f77b4', '#ff7f0e']  # Blue for BP, orange for DFA
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=bp_dfa_colors)

def get_bp_dfa_colors():
    """
    Get the standard color scheme for BP vs DFA plots.
    
    Returns:
        dict: Dictionary with 'bp' and 'dfa' color keys
    """
    return {
        'bp': '#1f77b4',   # Professional blue
        'dfa': '#ff7f0e'   # Professional orange
    }

def get_colorblind_palette():
    """
    Get a colorblind-friendly palette for multiple comparisons.
    
    Returns:
        list: List of hex color codes
    """
    return [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
    ]

def save_plot(fig, filename, output_dir='figures', subdir=None, 
              formats=['pdf', 'png'], **kwargs):
    """
    Save a plot in multiple formats with consistent settings.
    
    Args:
        fig: matplotlib figure object
        filename: base filename (without extension)
        output_dir: base output directory
        subdir: subdirectory within output_dir
        formats: list of formats to save (e.g., ['pdf', 'png'])
        **kwargs: additional arguments passed to savefig
    """
    if subdir:
        full_output_dir = os.path.join(output_dir, subdir)
    else:
        full_output_dir = output_dir
    
    os.makedirs(full_output_dir, exist_ok=True)
    
    default_kwargs = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'transparent': False
    }
    default_kwargs.update(kwargs)
    
    for fmt in formats:
        filepath = os.path.join(full_output_dir, f"{filename}.{fmt}")
        fig.savefig(filepath, format=fmt, **default_kwargs)
        print(f"Saved: {filepath}")

def create_bp_dfa_plot(xlabel='', ylabel='', title='', figsize=(4, 3)):
    """
    Create a standardized BP vs DFA comparison plot.
    
    Args:
        xlabel: X-axis label
        ylabel: Y-axis label  
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    return fig, ax

def plot_with_errorbars(x, y_means, y_stds, labels=None, xlabel='', ylabel='', 
                       title='', ax=None, **kwargs):
    """
    Plot data with error bars using consistent styling.
    
    Args:
        x: X-axis values
        y_means: Y-axis mean values (list of arrays for multiple series)
        y_stds: Y-axis standard deviation values
        labels: Labels for each series
        xlabel, ylabel, title: Axis labels and title
        ax: Existing axis to plot on (creates new if None)
        **kwargs: Additional arguments for errorbar
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.get_figure()
    
    if not isinstance(y_means, list):
        y_means = [y_means]
        y_stds = [y_stds]
    
    colors = get_colorblind_palette()
    
    for i, (means, stds) in enumerate(zip(y_means, y_stds)):
        color = colors[i % len(colors)]
        label = labels[i] if labels and i < len(labels) else None
        
        ax.errorbar(x, means, yerr=stds, 
                   fmt='o-', capsize=3, markersize=6, linewidth=2,
                   color=color, label=label, **kwargs)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if labels:
        ax.legend()
    
    return fig, ax

def set_log_scale(ax, axis='x'):
    """
    Set logarithmic scaling on specified axis with nice formatting.
    
    Args:
        ax: matplotlib axis object
        axis: 'x', 'y', or 'both'
    """
    if axis in ['x', 'both']:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    if axis in ['y', 'both']:
        ax.set_yscale('log') 
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

# Convenience function to apply all styling
def apply_style():
    """
    Apply all publication styling. Call this at the start of plotting scripts.
    """
    setup_publication_style()

# Auto-apply style when module is imported (optional)
# Comment out if you want to manually control when styling is applied
# apply_style()
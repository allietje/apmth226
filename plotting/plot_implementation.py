# plot_implementation.py
"""
plot_implementation.py - Replot the Implementation_final curves with 4 colors
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plotting_style import apply_style, save_plot

# Apply consistent styling
apply_style()

def replot_implementation_curves():
    """Replot the Implementation_final curves with 4 colors (single panel, no title)"""
    
    # Read the training history
    csv_path = "../final_results/Implementation_final/history.csv"
    df = pd.read_csv(csv_path)
    
    # Extract the 4 error curves
    epochs = df['epoch'].values
    train_err_bp = df['train_err_bp'].values
    test_err_bp = df['test_err_bp'].values
    train_err_dfa = df['train_err_dfa'].values
    test_err_dfa = df['test_err_dfa'].values
    
    # Create single panel plot (left panel only)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Use 4 distinct colors from the colorblind palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    # Plot the 4 lines
    ax.plot(epochs, train_err_bp, color=colors[0], linewidth=2, 
            label='BP Train Error')
    ax.plot(epochs, test_err_bp, color=colors[1], linewidth=2, linestyle='--',
            label='BP Test Error')
    ax.plot(epochs, train_err_dfa, color=colors[2], linewidth=2,
            label='DFA Train Error')
    ax.plot(epochs, test_err_dfa, color=colors[3], linewidth=2, linestyle='--',
            label='DFA Test Error')
    
    # Labels (no title as requested)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error (%)')
    ax.legend()
    
    # Set reasonable axis limits
    ax.set_xlim(1, len(epochs))
    max_error = max(np.max(train_err_bp), np.max(test_err_bp), 
                   np.max(train_err_dfa), np.max(test_err_dfa))
    ax.set_ylim(0, 10)
    
    fig.tight_layout()
    
    # Save to Implementation_final directory
    save_plot(fig, 'curves_replotted', subdir='Implementation_final')
    plt.close(fig)

if __name__ == "__main__":
    print("Replotting Implementation_final curves with 4 colors...")
    replot_implementation_curves()
    print("✓ Replotted curves_replotted.png/pdf in Implementation_final/")
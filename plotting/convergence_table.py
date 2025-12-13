# convergence_table.py
import pandas as pd
import numpy as np

def create_convergence_table():
    """
    Create a formatted table showing convergence results for all models.
    Mark BP models with final error > 10% as "didn't converge".
    """
    
    # Read the data
    csv_path = "/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/figures/convergence/convergence_grid_summary.csv"
    df = pd.read_csv(csv_path)
    
    # Define convergence threshold
    convergence_threshold = 10.0  # Final error > 10% means didn't converge
    
    print("=" * 90)
    print("Convergence Results Summary")
    print("=" * 90)
    print("<10s")
    print("-" * 90)
    
    # Sort by model name for consistent ordering
    df_sorted = df.sort_values('model')
    
    for _, row in df_sorted.iterrows():
        model = row['model']
        depth = int(row['depth'])
        width = int(row['width'])
        
        # BP convergence info
        bp_final_err = row['bp_final_err_mean']
        if bp_final_err > convergence_threshold:
            bp_conv_str = "— (no conv.)"
        else:
            bp_conv_mean = row['bp_conv_mean']
            bp_conv_std = row['bp_conv_std']
            bp_conv_str = f"{bp_conv_mean:.0f} ± {bp_conv_std:.0f}"
        
        bp_err_str = f"{bp_final_err:.2f} ± 0.00"  # Assuming single seed std=0 for simplicity
        
        # DFA convergence info (all DFA converged based on data)
        dfa_conv_mean = row['dfa_conv_mean']
        dfa_conv_std = row['dfa_conv_std']
        dfa_conv_str = f"{dfa_conv_mean:.0f} ± {dfa_conv_std:.0f}"
        
        dfa_final_err = row['dfa_final_err_mean']
        dfa_err_str = f"{dfa_final_err:.2f} ± 0.00"  # Assuming single seed std=0 for simplicity
        
        print("<10s")

def create_latex_table():
    """
    Create a LaTeX table for the paper.
    """
    
    csv_path = "/orcd/data/zhang_f/001/azong/projects/DFA/apmth226_gpu/figures/convergence/convergence_grid_summary.csv"
    df = pd.read_csv(csv_path)
    
    convergence_threshold = 10.0
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Convergence Results for Different Network Architectures}")
    print("\\label{tab:convergence}")
    print("\\begin{tabular}{@{}lccccccc@{}}")
    print("\\toprule")
    print("Model & Depth & Width & BP Conv. Epoch & BP Final Err (\%) & DFA Conv. Epoch & DFA Final Err (\%) \\\\")
    print("\\midrule")
    
    df_sorted = df.sort_values('model')
    
    for _, row in df_sorted.iterrows():
        model = row['model']
        depth = int(row['depth'])
        width = int(row['width'])
        
        # BP convergence info
        bp_final_err = row['bp_final_err_mean']
        if bp_final_err > convergence_threshold:
            bp_conv_str = "---"
        else:
            bp_conv_mean = row['bp_conv_mean']
            bp_conv_std = row['bp_conv_std']
            bp_conv_str = f"{bp_conv_mean:.0f} $\\pm$ {bp_conv_std:.0f}"
        
        bp_err_str = f"{bp_final_err:.2f}"
        
        # DFA convergence info
        dfa_conv_mean = row['dfa_conv_mean']
        dfa_conv_std = row['dfa_conv_std']
        dfa_conv_str = f"{dfa_conv_mean:.0f} $\\pm$ {dfa_conv_std:.0f}"
        
        dfa_final_err = row['dfa_final_err_mean']
        dfa_err_str = f"{dfa_final_err:.2f}"
        
        print(f"{model} & {depth} & {width} & {bp_conv_str} & {bp_err_str} & {dfa_conv_str} & {dfa_err_str} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    print("Generating text table:")
    create_convergence_table()
    print("\n\nGenerating LaTeX table:")
    create_latex_table()
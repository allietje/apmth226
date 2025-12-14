# tablize_multi_seed.py
"""
tablize_multi_seed.py - Convert Implementation_final multi_seed_summary.csv to LaTeX table
For 30x800 model, put dashes for bp_mean and bp_std
"""

import pandas as pd
import os

def format_number(x, decimals=3):
    """Format number to specified decimal places"""
    try:
        return f"{float(x):.{decimals}f}"
    except (ValueError, TypeError):
        return str(x)

def main():
    """Convert Implementation_final multi_seed_summary.csv to LaTeX table"""
    
    # Read the specific CSV file from Implementation_final
    csv_file = "../final_results/Implementation_final/multi_seed_summary.csv"
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded data from {csv_file}")
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    print("Data:")
    print(df)
    
    # Sort by depth
    df = df.sort_values('depth').reset_index(drop=True)
    
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Multi-seed performance comparison}")
    latex_lines.append("\\label{tab:multi_seed}")
    latex_lines.append("\\begin{tabular}{@{}lcccc@{}}")
    latex_lines.append("\\toprule")
    latex_lines.append("Model & BP Mean & BP Std & DFA Mean & DFA Std \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in df.iterrows():
        model = row['model']
        
        # Format BP values - use dashes for 30x800 model
        if model == '30x800':
            bp_mean = "---"
            bp_std = "---"
        else:
            bp_mean = format_number(row['bp_mean'])
            bp_std = format_number(row['bp_std'])
        
        dfa_mean = format_number(row['dfa_mean'])
        dfa_std = format_number(row['dfa_std'])
        
        # Create table row
        row_str = f"{model} & {bp_mean} & {bp_std} & {dfa_mean} & {dfa_std} \\\\"
        latex_lines.append(row_str)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    # Write to file
    output_file = "../final_results/Implementation_final/multi_seed_table.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"LaTeX table saved to: {output_file}")
    
    # Also print to console
    print("\nGenerated LaTeX table:")
    print('=' * 50)
    for line in latex_lines:
        print(line)
    print('=' * 50)

if __name__ == "__main__":
    main()
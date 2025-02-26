import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import os
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# Set up matplotlib for paper quality with larger fonts
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 18  # Increased base font size
mpl.rcParams['axes.linewidth'] = 2.0  # Thicker axes
mpl.rcParams['axes.labelsize'] = 22  # Larger axis labels
mpl.rcParams['xtick.labelsize'] = 20  # Larger tick labels
mpl.rcParams['ytick.labelsize'] = 20  # Larger tick labels
mpl.rcParams['legend.fontsize'] = 20  # Larger legend
mpl.rcParams['figure.figsize'] = (16, 8)  # Larger figure
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1

def extract_rho_from_filename(filename):
    """Extract rho value from filename like 'rho_10.csv' or 'rho_2-5.csv'"""
    match = re.search(r'rho_(\d+(?:-\d+)?)\.csv', filename)
    if match:
        rho_str = match.group(1)
        if '-' in rho_str:
            # Handle decimal values like '2-5' (meaning 2.5)
            parts = rho_str.split('-')
            return float(parts[0]) + float(parts[1])/10
        else:
            return float(rho_str)
    return None

def read_csv_file(file_path):
    """Read a CSV file and clean it"""
    try:
        # Read the file as text first to remove the END marker
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Filter out lines containing "END" and empty lines
        clean_lines = [line for line in lines if "END" not in line and line.strip()]
        
        # Write to a temporary file
        temp_file = f"{file_path}.temp"
        with open(temp_file, 'w') as f:
            f.writelines(clean_lines)
        
        # Read the clean CSV file
        df = pd.read_csv(temp_file)
        
        # Remove temporary file
        os.remove(temp_file)
        
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def plot_rho_comparison():
    """Plot comparison between fixed and adaptive rho values with grouped bars"""
    # Define files to compare
    fixed_rho_5_file = "fixed/rho_5.csv"
    fixed_rho_40_file = "fixed/rho_40.csv"
    adaptive_rho_5_file = "adapt/adapt_rho_5.csv"
    adaptive_rho_40_file = "adapt/adapt_rho_40.csv"
    
    # Check if files exist
    for file_path in [fixed_rho_5_file, fixed_rho_40_file, adaptive_rho_5_file, adaptive_rho_40_file]:
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found")
            return
    
    # Read data
    fixed_rho_5_df = read_csv_file(fixed_rho_5_file)
    fixed_rho_40_df = read_csv_file(fixed_rho_40_file)
    adaptive_rho_5_df = read_csv_file(adaptive_rho_5_file)
    adaptive_rho_40_df = read_csv_file(adaptive_rho_40_file)
    
    if fixed_rho_5_df is None or fixed_rho_40_df is None or adaptive_rho_5_df is None or adaptive_rho_40_df is None:
        print("Error: Could not read one or more files")
        return
    
    # Ensure column names are consistent
    # For adaptive files, rename columns if needed
    if 'iterations' not in adaptive_rho_5_df.columns:
        # Assuming columns are [rho, trial, iterations, solve_time_us, success]
        if len(adaptive_rho_5_df.columns) >= 5:
            adaptive_rho_5_df.columns = ['rho', 'trial', 'iterations', 'solve_time_us', 'success']
        elif len(adaptive_rho_5_df.columns) == 4:
            adaptive_rho_5_df.columns = ['rho', 'trial', 'iterations', 'solve_time_us']
            
    if 'iterations' not in adaptive_rho_40_df.columns:
        # Assuming columns are [rho, trial, iterations, solve_time_us, success]
        if len(adaptive_rho_40_df.columns) >= 5:
            adaptive_rho_40_df.columns = ['rho', 'trial', 'iterations', 'solve_time_us', 'success']
        elif len(adaptive_rho_40_df.columns) == 4:
            adaptive_rho_40_df.columns = ['rho', 'trial', 'iterations', 'solve_time_us']
    
    # Calculate statistics
    stats = [
        {
            'rho': 5.0,
            'method': 'Fixed',
            'iterations_mean': fixed_rho_5_df['iterations'].mean(),
            'iterations_std': fixed_rho_5_df['iterations'].std(),
            'solve_time_mean': fixed_rho_5_df['solve_time_us'].mean(),
            'solve_time_std': fixed_rho_5_df['solve_time_us'].std(),
        },
        {
            'rho': 5.0,
            'method': 'Adaptive',
            'iterations_mean': adaptive_rho_5_df['iterations'].mean(),
            'iterations_std': adaptive_rho_5_df['iterations'].std(),
            'solve_time_mean': adaptive_rho_5_df['solve_time_us'].mean(),
            'solve_time_std': adaptive_rho_5_df['solve_time_us'].std(),
        },
        {
            'rho': 40.0,
            'method': 'Fixed',
            'iterations_mean': fixed_rho_40_df['iterations'].mean(),
            'iterations_std': fixed_rho_40_df['iterations'].std(),
            'solve_time_mean': fixed_rho_40_df['solve_time_us'].mean(),
            'solve_time_std': fixed_rho_40_df['solve_time_us'].std(),
        },
        {
            'rho': 40.0,
            'method': 'Adaptive',
            'iterations_mean': adaptive_rho_40_df['iterations'].mean(),
            'iterations_std': adaptive_rho_40_df['iterations'].std(),
            'solve_time_mean': adaptive_rho_40_df['solve_time_us'].mean(),
            'solve_time_std': adaptive_rho_40_df['solve_time_us'].std(),
        }
    ]
    
    # Calculate speedups
    stats[1]['speedup'] = stats[0]['solve_time_mean'] / stats[1]['solve_time_mean']
    stats[3]['speedup'] = stats[2]['solve_time_mean'] / stats[3]['solve_time_mean']
    
    # Convert to DataFrame for easier plotting
    stats_df = pd.DataFrame(stats)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Define colors for fixed and adaptive
    fixed_color = 'steelblue'
    adaptive_color = 'firebrick'
    
    # Define bar width and positions
    bar_width = 0.35
    r1 = np.array([0, 2])  # Positions for rho=5.0 and rho=40.0 (fixed)
    r2 = r1 + bar_width    # Positions for rho=5.0 and rho=40.0 (adaptive)
    
    # --- First plot: Iterations ---
    # Extract data for plotting
    fixed_iterations = [stats[0]['iterations_mean'], stats[2]['iterations_mean']]
    adaptive_iterations = [stats[1]['iterations_mean'], stats[3]['iterations_mean']]
    fixed_iterations_std = [stats[0]['iterations_std'], stats[2]['iterations_std']]
    adaptive_iterations_std = [stats[1]['iterations_std'], stats[3]['iterations_std']]
    
    # Create grouped bar chart for iterations
    bars1_fixed = ax1.bar(r1, fixed_iterations, width=bar_width, color=fixed_color, 
                         edgecolor='black', linewidth=2.0, label='Fixed')
    bars1_adaptive = ax1.bar(r2, adaptive_iterations, width=bar_width, color=adaptive_color, 
                            edgecolor='black', linewidth=2.0, label='Adaptive')
    
    # Add error bars
    ax1.errorbar(r1, fixed_iterations, yerr=fixed_iterations_std, fmt='none', 
                ecolor='black', capsize=6, capthick=2.0, elinewidth=2.0)
    ax1.errorbar(r2, adaptive_iterations, yerr=adaptive_iterations_std, fmt='none', 
                ecolor='black', capsize=6, capthick=2.0, elinewidth=2.0)
    
    # Set log scale for y-axis
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=0.1)  # Start from 0.1 to show zero values
    
    # Customize plot
    ax1.set_ylabel('Iterations (log scale)', fontweight='bold')
    ax1.set_xticks(r1 + bar_width/2)
    ax1.set_xticklabels(['ρ=5.0', 'ρ=40.0'])
    # Remove grid lines
    ax1.grid(False)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper left')
    
    # --- Second plot: Solve time ---
    # Extract data for plotting
    fixed_solve_times = [stats[0]['solve_time_mean'], stats[2]['solve_time_mean']]
    adaptive_solve_times = [stats[1]['solve_time_mean'], stats[3]['solve_time_mean']]
    fixed_solve_times_std = [stats[0]['solve_time_std'], stats[2]['solve_time_std']]
    adaptive_solve_times_std = [stats[1]['solve_time_std'], stats[3]['solve_time_std']]
    
    # Create grouped bar chart for solve time
    bars2_fixed = ax2.bar(r1, fixed_solve_times, width=bar_width, color=fixed_color, 
                         edgecolor='black', linewidth=2.0, label='Fixed')
    bars2_adaptive = ax2.bar(r2, adaptive_solve_times, width=bar_width, color=adaptive_color, 
                            edgecolor='black', linewidth=2.0, label='Adaptive')
    
    # Add error bars
    ax2.errorbar(r1, fixed_solve_times, yerr=fixed_solve_times_std, fmt='none', 
                ecolor='black', capsize=6, capthick=2.0, elinewidth=2.0)
    ax2.errorbar(r2, adaptive_solve_times, yerr=adaptive_solve_times_std, fmt='none', 
                ecolor='black', capsize=6, capthick=2.0, elinewidth=2.0)
    
    # Set log scale for y-axis
    ax2.set_yscale('log')
    
    # Customize plot
    ax2.set_ylabel('Solve Time (μs, log scale)', fontweight='bold')
    ax2.set_xticks(r1 + bar_width/2)
    ax2.set_xticklabels(['ρ=5.0', 'ρ=40.0'])
    # Remove grid lines
    ax2.grid(False)
    ax2.set_axisbelow(True)
    # No legend on second plot
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('rho_comparison_grouped.png', dpi=300, bbox_inches='tight')
    plt.savefig('rho_comparison_grouped.pdf', format='pdf', bbox_inches='tight')
    
    print(f"Plot saved as 'rho_comparison_grouped.png' and 'rho_comparison_grouped.pdf'")
    
    # Show plot
    plt.show()
    
    # Print statistics
    print("\nPerformance Statistics:")
    print(stats_df)

if __name__ == "__main__":
    plot_rho_comparison()
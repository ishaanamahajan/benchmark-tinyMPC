import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Let's create a simplified parsing function that's more robust
def parse_admm_file(filepath):
    """Parse the ADMM trial data from file or string"""
    try:
        # Try to read from file
        with open(filepath, 'r') as f:
            content = f.read()
    except (FileNotFoundError, IsADirectoryError, TypeError):
        # If file doesn't exist, assume filepath is the content string
        content = filepath
    
    # Process each line individually to be more robust
    lines = content.split('\n')
    data = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and header lines
        if not line or line.startswith('==='):
            continue
            
        # Check if line contains CSV data with the expected format
        parts = line.split(',')
        if len(parts) >= 7 and (parts[0] == 'Fixed' or parts[0] == 'Adaptive'):
            try:
                data.append({
                    'type': parts[0],
                    'trial': int(parts[1]),
                    'total_time': int(parts[2]),
                    'admm_time': int(parts[3]),
                    'rho_time': int(parts[4]),
                    'iterations': int(parts[5]),
                    'accuracy': float(parts[6])
                })
            except (ValueError, IndexError):
                # Skip lines that don't parse correctly
                continue
    
    return pd.DataFrame(data)

# Set the plotting style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 20,
    'grid.linestyle': ':',
    'grid.alpha': 0.5,
    'lines.linewidth': 2.5,
    'figure.facecolor': 'white'
})


def scientific_notation(x, pos):
    """Format y-axis ticks in scientific notation"""
    return f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'
    #return ''

def create_boxplot_comparison(df):
    """Create an enhanced box and whiskers plot that tells the optimization performance story"""

    # Set the plotting style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 16,
        'axes.labelsize': 20,
        'grid.linestyle': ':',
        'grid.alpha': 0.5,
        'lines.linewidth': 2.5
    })
    
    # Set a more vibrant color palette
    fixed_color = '#e74c3c'    # A deeper red
    adaptive_color = '#3498db'  # A vibrant blue
    
    # Create the figure with larger size for better visibility
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Add a subtle background gradient to enhance visual appeal
    gradient = np.linspace(0, 1, 100).reshape(-1, 1)
    ax.imshow(gradient, aspect='auto', extent=[ax.get_xlim()[0], ax.get_xlim()[1], 
                                              0, max(df['total_time']) * 1.2],
             cmap='Blues', alpha=0.1, zorder=0)
    
    # Create a custom palette for seaborn
    palette = {'Fixed': fixed_color, 'Adaptive': adaptive_color}
    
    # Create the main box plot with thicker lines to emphasize the difference
    ax = sns.boxplot(x='type', y='total_time', data=df, palette=palette, 
                   width=0.6, fliersize=5, linewidth=2.5, ax=ax)
    
    # Calculate statistics we'll need for annotations
    fixed_df = df[df['type'] == 'Fixed']
    adaptive_df = df[df['type'] == 'Adaptive']
    stats = df.groupby('type')['total_time'].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
    
    # Calculate non-convergence statistics
    fixed_non_converged = fixed_df[fixed_df['iterations'] >= 500]
    adaptive_non_converged = adaptive_df[adaptive_df['iterations'] >= 500]
    fixed_non_converged_count = len(fixed_non_converged)
    adaptive_non_converged_count = len(adaptive_non_converged)
    fixed_non_converged_pct = 100 * fixed_non_converged_count / len(fixed_df)
    adaptive_non_converged_pct = 100 * adaptive_non_converged_count / len(adaptive_df)
    
    # Add individual data points with jitter for better visualization
    # Use different marker shapes to distinguish outliers
    sns.stripplot(x='type', y='total_time', data=df[(df['iterations'] < 500)], 
                 marker='o', color='black', size=8, alpha=0.6, jitter=True, ax=ax)
    
    # Highlight trials that didn't converge with different markers
    sns.stripplot(x='type', y='total_time', data=df[(df['iterations'] >= 500)], 
                 marker='X', color='#ff5733', size=10, alpha=0.8, jitter=True, ax=ax)
    
    plt.xlabel('Method', fontsize=20)
    plt.ylabel('Total Solve Time (µs)', fontsize=20)
    
    # Customize x-tick labels with more descriptive names
    plt.xticks([0, 1], ['Fixed', 'First-Order\nAdaptive'], fontsize=18)
    
    # Set y-axis formatter for better readability
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'))
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Calculate convergence rates (percentage of trials that converged)
    fixed_converged = 100 * (fixed_df['iterations'] < 500).mean()
    adaptive_converged = 100 * (adaptive_df['iterations'] < 500).mean()
    
    # Create a callout box for key findings
    findings_text = f"Key Findings:\n" \
                   f"• Fixed ADMM: {stats.loc['Fixed', 'mean']:.0f} µs avg. solve time\n" \
                   f"• Adaptive ADMM: {stats.loc['Adaptive', 'mean']:.0f} µs avg. solve time\n" \
                   f"• Convergence: {fixed_converged:.1f}% vs {adaptive_converged:.1f}%\n" \
                   f"• Iteration reduction: {(fixed_df['iterations'].mean() - adaptive_df['iterations'].mean()) / fixed_df['iterations'].mean() * 100:.1f}%"


    print(findings_text)
    
    # # Add the findings box in the upper right
    # props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    # ax.text(0.97, 0.97, findings_text, transform=ax.transAxes, fontsize=12,
    #        verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Calculate the speedup
    fixed_mean = stats.loc['Fixed', 'mean']
    adaptive_mean = stats.loc['Adaptive', 'mean']
    speedup = (fixed_mean - adaptive_mean) / fixed_mean * 100
    
    # Create a dramatic arrow showing the performance improvement
    plt.annotate(f'{speedup:.1f}% Performance Improvement',
               xy=(0.5, (fixed_mean + adaptive_mean) / 2),
               xytext=(0.5, max(df['total_time']) * 0.75),
               arrowprops=dict(arrowstyle='fancy', facecolor='green', 
                              connectionstyle='arc3,rad=0.3', linewidth=2.5),
               fontsize=20, fontweight='bold', color='green',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', alpha=0.9),
               ha='center')
    
    # Add an annotation for the overhead - INSIDE the graph in the top right
    adaptive_rho_time = adaptive_df['rho_time'].mean()
    adaptive_total_time = adaptive_df['total_time'].mean()
    overhead_pct = (adaptive_rho_time / adaptive_total_time) * 100
    
    # Add callout for the overhead inside the plot area
    plot_y_max = ax.get_ylim()[1]
    plot_x_max = ax.get_xlim()[1]
    
    ax.text(0.85, 0.85, 
          f"Rho Update Overhead:\nOnly {overhead_pct:.1f}% of total\ncomputation time",
          transform=ax.transAxes, fontsize=18, ha='left',
          bbox=dict(boxstyle='round', facecolor='#ffe6cc', alpha=0.9, edgecolor=adaptive_color))
    
    # Add non-converged information directly to the boxes
    # For fixed method
    ax.annotate(f'Non-converged: {fixed_non_converged_pct:.1f}%)',
              xy=(0, ax.get_ylim()[1] * 0.85),
              xytext=(0, ax.get_ylim()[1] * 0.85),
              ha='center', va='center', fontsize=16,
              bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.9, edgecolor=fixed_color))
              
    # For adaptive method
    ax.annotate(f'Non-converged: {adaptive_non_converged_pct:.1f}%)',
              xy=(1, ax.get_ylim()[1] * 0.9),
              xytext=(1, ax.get_ylim()[1] * 0.9),
              ha='center', va='center', fontsize=16,
              bbox=dict(boxstyle='round', facecolor='#ccebff', alpha=0.9, edgecolor=adaptive_color))
    
    # Create custom legend for data points
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, 
              label='Converged Trials'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#ff5733', markersize=12, 
              label='Non-converged Trials (500+ iterations)')
    ]
    
    # Add the legend
    ax.legend(handles=legend_elements, loc='upper left',
            frameon=True, facecolor='white', framealpha=0.9, fontsize=16)
    
    # Add a decorative border to frame the story
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_edgecolor('gray')
    
    # Add subtle annotations on each box
    for i, method in enumerate(['Fixed', 'Adaptive']):
        median_val = stats.loc[method, 'median']
        ax.text(i, stats.loc[method, 'max'] * 1.02, 
               f"Max: {stats.loc[method, 'max']:.0f} µs",
               ha='center', va='bottom', fontsize=14, color='gray')
        
        ax.text(i, stats.loc[method, 'min'] * 0.95, 
               f"Min: {stats.loc[method, 'min']:.0f} µs",
               ha='center', va='top', fontsize=14, color='gray')
    
    plt.tight_layout()
    plt.show()

# Function to create sample data for testing
def create_sample_data():
    """Create sample data for testing if file reading fails"""
    sample_data = """Fixed,3,18742,18742,0,316,85.00
=== Starting Fixed Rho Trial 4 ===
Fixed,4,29567,29567,0,500,85.00
=== Starting Fixed Rho Trial 5 ===
Fixed,5,29566,29566,0,500,85.00
=== Starting Fixed Rho Trial 6 ===
Fixed,6,29569,29569,0,500,85.00
=== Starting Fixed Rho Trial 7 ===
Fixed,7,25487,25486,0,430,85.00
=== Starting Fixed Rho Trial 8 ===
Fixed,8,29567,29567,0,500,85.00
=== Starting Fixed Rho Trial 9 ===
Fixed,9,20045,20045,0,338,85.00
=== Starting Fixed Rho Trial 10 ===
Fixed,10,13720,13720,0,231,85.00
=== Starting Fixed Rho Trial 11 ===
Fixed,11,29570,29570,0,500,85.00
=== Starting Adaptive Rho Trial 1 ===
Adaptive,1,15240,14952,288,252,85.00
=== Starting Adaptive Rho Trial 2 ===
Adaptive,2,18456,18002,454,304,85.00
=== Starting Adaptive Rho Trial 3 ===
Adaptive,3,10980,10690,290,180,85.00
=== Starting Adaptive Rho Trial 4 ===
Adaptive,4,12678,12345,333,213,85.00
=== Starting Adaptive Rho Trial 5 ===
Adaptive,5,14520,14103,417,245,85.00"""
    return sample_data

# Main execution
if __name__ == "__main__":
    # Replace this with the path to your data file
    file_path = "/Users/ishaanmahajan/benchmark-tinyMPC/ICRA_benchmarks/teensy_tinympc_benchmark/benchmark_results.csv"
    
    try:
        # Try to parse from file first
        df = parse_admm_file(file_path)
        if len(df) == 0:
            print("Warning: No valid data found in file. Using sample data.")
            df = parse_admm_file(create_sample_data())
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Using sample data instead.")
        df = parse_admm_file(create_sample_data())
    
    # Create standalone box and whiskers plot
    create_boxplot_comparison(df)
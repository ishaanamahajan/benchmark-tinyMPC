import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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
    'font.size': 12,
    'axes.labelsize': 14,
    'grid.linestyle': ':',
    'grid.alpha': 0.5,
    'lines.linewidth': 2.5
})

def create_separate_plots(df):
    """Create separate CDF and bar chart plots"""
    # Set up for solved problems calculation
    df['solved'] = df['iterations'] < 500
    fixed_df = df[df['type'] == 'Fixed'].sort_values('total_time')
    adaptive_df = df[df['type'] == 'Adaptive'].sort_values('total_time')
    
    fixed_color = '#d62728'
    adapt_color = '#1f77b4'
    
    # 1. Create CDF Plot
    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    
    # Plot CDF with x-axis limit
    def plot_cdf(df, color, label):
        if len(df) == 0:
            return
        df_sorted = df.sort_values('total_time')
        times = df_sorted['total_time'].values
        cumulative_solved = np.cumsum(df_sorted['solved'].values)
        percentages = 100 * cumulative_solved / len(df_sorted)
        
        if len(times) > 0 and times[0] > 0:
            times = np.insert(times, 0, 0)
            percentages = np.insert(percentages, 0, 0)
        
        ax1.plot(times, percentages, '-', color=color, linewidth=2, label=label)
        ax1.fill_between(times, 0, percentages, color=color, alpha=0.2)
        return max(times)  # Return max time for setting limit

    # Plot and get max times
    fixed_max = plot_cdf(fixed_df, fixed_color, 'Fixed Rho')
    _ = plot_cdf(adaptive_df, adapt_color, 'Adaptive Rho')
    
    # Set x-axis limit to fixed maximum
    ax1.set_xlim(0, fixed_max)
    
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (µs)')
    ax1.set_ylabel('Problems Solved (%)')
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=14)
    ax1.set_ylim(0, 105)

        # Set the plotting style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'grid.linestyle': ':',
        'grid.alpha': 0.5,
        'lines.linewidth': 2.5
    })



    
    plt.tight_layout()

    plt.show()
   
    plt.savefig('cdf_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create Bar Chart
    plt.figure(figsize=(6, 5))
    ax2 = plt.gca()
    
    # For the stacked bar chart
    grouped = df.groupby('type').mean().reset_index()
    x = np.arange(len(grouped['type']))
    width = 0.5
    
    # Plot bars with separate labels for legend
    p1_fixed = ax2.bar(x[0], grouped[grouped['type'] == 'Fixed']['admm_time'], 
                       width, label='Fixed ADMM Time', color=fixed_color)
    p1_adaptive = ax2.bar(x[1], grouped[grouped['type'] == 'Adaptive']['admm_time'], 
                         width, label='Adaptive ADMM Time', color=adapt_color)
    p2 = ax2.bar(x[1], grouped[grouped['type'] == 'Adaptive']['rho_time'], 
                 width, bottom=grouped[grouped['type'] == 'Adaptive']['admm_time'],
                 label='Rho Update Time', color=adapt_color, alpha=0.3)
    
    ax2.set_ylabel('Time (µs)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Fixed', 'First-Order\nAdaptive'])
    ax2.legend()
    
    # Add value labels on bars
    for i, p in enumerate([p1_fixed, p1_adaptive]):
        height = p[0].get_height()
        if height > 0:
            ax2.text(p[0].get_x() + p[0].get_width()/2, height/2,
                    f'{int(height):,}',
                    ha='center', va='center', color='black',
                    fontweight='bold', fontsize=11)
    
    if len(p2) > 0:
        height = p2[0].get_height()
        bottom = p2[0].get_y()
        if height > 0:
            ax2.text(p2[0].get_x() + p2[0].get_width()/2, bottom + height/2,
                    f'{int(height):,}',
                    ha='center', va='center', color='black',
                    fontweight='bold', fontsize=11)
    
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    ax2.set_ylim(0, 25000)

    # Set the plotting style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'grid.linestyle': ':',
        'grid.alpha': 0.5,
        'lines.linewidth': 2.5
    })


    
    plt.tight_layout()
    
    
    plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Calculate statistics for reporting
    fixed_total = grouped[grouped['type'] == 'Fixed']['total_time'].values[0]
    adaptive_total = grouped[grouped['type'] == 'Adaptive']['total_time'].values[0]
    speedup = (fixed_total - adaptive_total) / fixed_total * 100
    
    # Print statistics
    print(f"Fixed total time: {fixed_total:.3f} µs")
    print(f"Adaptive total time: {adaptive_total:.3f} µs")
    print(f"Speedup: {speedup:.1f}%")

# Replace the call to create_combined_plots with create_separate_plots in the main execution
if __name__ == "__main__":
    # ... [existing code] ...
    
    # Create visualizations - replace this line
    # create_combined_plots(df)
    # with this:
    file_path = "/Users/ishaanmahajan/benchmark-tinyMPC/ICRA_benchmarks/teensy_tinympc_benchmark/benchmark_results_dynamics_10.csv"
    
    try:
        # Try to parse from file first
        df = parse_admm_file(file_path)
        if len(df) == 0:
            print("Warning: No valid data found in file. Using sample data.")
            df = parse_admm_file(sample_data)
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Using sample data instead.")
        df = parse_admm_file(sample_data)
    
    create_separate_plots(df)
    
    # ... [rest of the code remains the same] ...
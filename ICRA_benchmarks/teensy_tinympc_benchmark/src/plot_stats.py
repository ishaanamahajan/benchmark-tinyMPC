import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter

def parse_admm_file(filepath):
    """Parse the ADMM trial data from file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except:
        print(f"Error reading file: {filepath}")
        return pd.DataFrame()
    
    lines = content.split('\n')
    data = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('==='):
            continue
            
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
            except:
                continue
    
    return pd.DataFrame(data)

def plot_solve_time_distribution(all_data):
    plt.figure(figsize=(10, 6))
    
    # Colors
    fixed_color = '#ff7f0e'    # Orange
    adapt_color = '#1f77b4'    # Blue
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Calculate metrics for both types
    for solver_type in ['Fixed', 'Adaptive']:
        df_type = all_data[all_data['type'] == solver_type]
        solve_times = df_type['total_time'].values
        rho_times = df_type['rho_time'].values
        admm_times = df_type['admm_time'].values
        
        # Calculate metrics
        metrics[solver_type] = {
            'mean': np.mean(solve_times),
            'median': np.median(solve_times),
            'percentile_95': np.percentile(solve_times, 95),
            'std': np.std(solve_times),
            'mean_rho_time': np.mean(rho_times),
            'mean_admm_time': np.mean(admm_times)
        }
        
        # Calculate and plot CDF
        sorted_times = np.sort(solve_times)
        percentages = 100 * np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        
        color = fixed_color if solver_type == 'Fixed' else adapt_color
        label = 'Fixed Rho' if solver_type == 'Fixed' else 'Adaptive Rho'
        
        # Plot main CDF line
        plt.plot(sorted_times, percentages/100, '-', 
                color=color, 
                linewidth=2, 
                label=label)
        plt.fill_between(sorted_times, 0, percentages/100, color=color, alpha=0.1)
        
        # Add iteration limit lines
        last_percentage = percentages[-1]/100
        if solver_type == 'Fixed':
            plt.vlines(x=28000, ymin=last_percentage, ymax=1.0, 
                      color=color, linestyle='-', linewidth=2)
        else:
            plt.plot([27500, 28500], [last_percentage, 1.0], 
                    color=color, linewidth=2)
        
        # Horizontal line at 100%
        plt.hlines(y=1.0, xmin=28000, xmax=30000,
                  color=color, linestyle='-', linewidth=2)

    # Calculate adaptation overhead
    adapt_mean = metrics['Adaptive']['mean']
    fixed_mean = metrics['Fixed']['mean']
    adaptation_overhead = ((adapt_mean - fixed_mean) / fixed_mean) * 100

    plt.xlabel('Solve Time (µs)')
    plt.ylabel('Percentage of Solves')
    plt.grid(True, alpha=0.2)
    plt.legend(loc='upper left')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.xlim(4000, 30000)
    plt.ylim(0, 1.0)
    
    plt.annotate('Iteration\nLimit', 
                xy=(28000, 0.95),
                xytext=(26000, 0.85),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig('solve_time_distribution.png', dpi=300)
    plt.show()

    # Calculate grouped statistics
    grouped = all_data.groupby('type').mean().reset_index()
    
    # Calculate overhead based on rho time
    fixed_admm = grouped[grouped['type'] == 'Fixed']['admm_time'].values[0]
    adaptive_admm = grouped[grouped['type'] == 'Adaptive']['admm_time'].values[0]
    rho_time = grouped[grouped['type'] == 'Adaptive']['rho_time'].values[0]
    
    # Calculate overhead percentage
    overhead_percentage = (rho_time / adaptive_admm) * 100
    
    # Print metrics in milliseconds
    print("\nMetrics:")
    print(f"{'Metric':<25} {'Fixed':<15} {'Adaptive':<15}")
    print("-" * 55)
    print(f"{'Mean Solve Time (ms)':<25} {metrics['Fixed']['mean']/1000:<15.3f} {metrics['Adaptive']['mean']/1000:<15.3f}")
    print(f"{'Median Solve Time (ms)':<25} {metrics['Fixed']['median']/1000:<15.3f} {metrics['Adaptive']['median']/1000:<15.3f}")
    print(f"{'95th Percentile Time (ms)':<25} {metrics['Fixed']['percentile_95']/1000:<15.3f} {metrics['Adaptive']['percentile_95']/1000:<15.3f}")
    print(f"{'Standard Deviation (ms)':<25} {metrics['Fixed']['std']/1000:<15.3f} {metrics['Adaptive']['std']/1000:<15.3f}")
    print(f"{'Mean ADMM Time (ms)':<25} {metrics['Fixed']['mean_admm_time']/1000:<15.3f} {metrics['Adaptive']['mean_admm_time']/1000:<15.3f}")
    print(f"{'Mean Rho Time (ms)':<25} {metrics['Fixed']['mean_rho_time']/1000:<15.3f} {metrics['Adaptive']['mean_rho_time']/1000:<15.3f}")
    print(f"\nRho adaptation overhead: {overhead_percentage:.1f}%")

    # Add print statement to show number of data points
    print(f"\nTotal number of data points: {len(all_data)}")
    print(f"Number of files processed: {len(all_data) // 200}")  # Divided by 200 because each file has both Fixed and Adaptive

def calculate_comparative_stats(all_data):
    """Calculate comparative statistics between Fixed and Adaptive approaches"""
    
    # Get solve times for both methods
    fixed_times = all_data[all_data['type'] == 'Fixed']['total_time'].values
    adaptive_times = all_data[all_data['type'] == 'Adaptive']['total_time'].values
    
    # Get iterations for both methods
    fixed_iters = all_data[all_data['type'] == 'Fixed']['iterations'].values
    adaptive_iters = all_data[all_data['type'] == 'Adaptive']['iterations'].values
    
    # Calculate percentage of adaptive solves slower than fixed
    fixed_sorted = np.sort(fixed_times)
    adaptive_sorted = np.sort(adaptive_times)
    
    # Find fastest and slowest times
    fastest_fixed = np.min(fixed_times)
    fastest_adaptive = np.min(adaptive_times)
    slowest_fixed = np.max(fixed_times)
    slowest_adaptive = np.max(adaptive_times)
    
    # Calculate percentage faster/slower
    fastest_improvement = ((fastest_fixed - fastest_adaptive) / fastest_fixed) * 100
    slowest_comparison = (slowest_adaptive / slowest_fixed) * 100
    
    # Calculate percentage of solves that are slower
    slower_count = sum(adaptive_times > np.median(fixed_times))
    slower_percentage = (slower_count / len(adaptive_times)) * 100
    
    # Calculate percentage under iteration limit (assuming 500 is the limit)
    ITERATION_LIMIT = 500
    fixed_under_limit = (sum(fixed_iters < ITERATION_LIMIT) / len(fixed_iters)) * 100
    adaptive_under_limit = (sum(adaptive_iters < ITERATION_LIMIT) / len(adaptive_iters)) * 100
    
    print("\nComparative Statistics:")
    print(f"{slower_percentage:.1f}% of adaptive solves were slower than the fixed approach")
    print(f"Slowest adaptive solve was {slowest_comparison:.1f}% of the slowest fixed solve")
    print(f"Fastest adaptive solve was {fastest_improvement:.1f}% faster than the fastest fixed solve")
    print(f"Adaptive approach solves {adaptive_under_limit:.1f}% of problems under the iteration limit")
    print(f"Fixed approach solves {fixed_under_limit:.1f}% of problems under the iteration limit")

if __name__ == "__main__":
    # Base path and file pattern
    base_path = "/Users/ishaanmahajan/replicate/benchmark-tinyMPC/ICRA_benchmarks/teensy_tinympc_benchmark/"
    
    # Collect data from all files
    all_data = pd.DataFrame()
    for i in range(1, 101):
        filename = f"benchmark_results_12_{i}.csv"
        filepath = os.path.join(base_path, filename)
        df = parse_admm_file(filepath)
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    if len(all_data) > 0:
        plot_solve_time_distribution(all_data)
        calculate_comparative_stats(all_data)
    else:
        print("No data was loaded. Please check the file paths.")
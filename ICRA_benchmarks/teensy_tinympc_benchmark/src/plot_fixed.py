import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_plots(data_file):
    """
    Create multiple plots for ADMM benchmark comparison:
    1. Enhanced stacked bar chart with error bars
    2. Box plot showing distribution of iteration times
    3. Scatter plot showing correlation between iterations and solve time
    4. Grouped bar chart comparing key metrics
    
    All plots are displayed using plt.show() instead of saving to files.
    """
    
    # Read and process data - skip lines with ===
    lines = []
    with open(data_file, 'r') as f:
        for line in f:
            if '===' not in line and ',' in line:
                lines.append(line)
    
    # Find the header line
    header_line = None
    for i, line in enumerate(lines):
        if "Method" in line and "Trial" in line:
            header_line = i
            break
    
    if header_line is None:
        print("Could not find header line, using first line as header")
        header_line = 0
    
    # Parse CSV data
    data = []
    headers = [h.strip() for h in lines[header_line].split(',')]
    
    for line in lines[header_line+1:]:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == len(headers):
            row = {}
            for i, header in enumerate(headers):
                try:
                    row[header] = float(parts[i]) if i > 0 else parts[i]
                except:
                    row[header] = parts[i]
            
            # Simplify to just "Fixed" or "Adaptive" methods
            if "Fixed" in row['Method']:
                row['Method'] = "Fixed"
            elif "Adaptive" in row['Method']:
                row['Method'] = "Adaptive"
                
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate time per iteration
    df['TimePerIteration'] = df['SolveTime'] / df['Iterations']
    df['ADMMTimePerIteration'] = df['ADMMTime'] / df['Iterations']
    df['RhoTimePerIteration'] = df['RhoTime'] / df['Iterations']
    
    # Split data by method
    fixed_data = df[df['Method'] == 'Fixed']
    adaptive_data = df[df['Method'] == 'Adaptive']
    
    # Set style for all plots - compatible with all matplotlib versions
    try:
        # Try the newer style name format
        plt.style.use('seaborn-whitegrid')
    except:
        try:
            # Try the older style name format
            plt.style.use('seaborn')
        except:
            # Fallback to default style
            plt.style.use('default')
    
    # Configure seaborn context
    sns.set_context("paper", font_scale=1.2)
    
    # Set color scheme
    fixed_color = '#d62728'  # Red
    adapt_color = '#1f77b4'  # Blue
    
    # 1. Create enhanced stacked bar chart with error bars
    create_stacked_bar_with_error_bars(df, fixed_data, adaptive_data, 
                                      fixed_color, adapt_color)
    
    # 2. Create box plot for distribution visualization
    create_boxplot(df, fixed_color, adapt_color)
    
    # 3. Create scatter plot
    create_scatter_plot(df, fixed_color, adapt_color)
    
    # 4. Create grouped bar chart for key metrics
    create_grouped_bar_chart(fixed_data, adaptive_data, 
                            fixed_color, adapt_color)
    
    # Print summary statistics
    print_statistics(fixed_data, adaptive_data)


def create_stacked_bar_with_error_bars(df, fixed_data, adaptive_data, fixed_color, adapt_color):
    """Create an enhanced stacked bar chart with error bars and display it"""
    
    # Calculate statistics for plotting
    fixed_admm_mean = fixed_data['ADMMTimePerIteration'].mean()
    fixed_admm_std = fixed_data['ADMMTimePerIteration'].std()
    
    adaptive_admm_mean = adaptive_data['ADMMTimePerIteration'].mean()
    adaptive_admm_std = adaptive_data['ADMMTimePerIteration'].std()
    
    adaptive_rho_mean = adaptive_data['RhoTimePerIteration'].mean()
    adaptive_rho_std = adaptive_data['RhoTimePerIteration'].std()
    
    # Create figure and get axis object
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define bar positions and width
    x = np.array([0, 1])
    width = 0.5
    
    # Plot bars
    b1 = ax.bar(x[0], fixed_admm_mean, width, label='ADMM Time', color=fixed_color)
    b2 = ax.bar(x[1], adaptive_admm_mean, width, label='ADMM Time', color=adapt_color)
    b3 = ax.bar(x[1], adaptive_rho_mean, width, bottom=adaptive_admm_mean, 
               label='Rho Update Time', color=adapt_color, alpha=0.4)
    
    # Add error bars
    ax.errorbar(x[0], fixed_admm_mean, yerr=fixed_admm_std, fmt='none', color='black', capsize=5)
    ax.errorbar(x[1], adaptive_admm_mean, yerr=adaptive_admm_std, fmt='none', color='black', capsize=5)
    ax.errorbar(x[1], adaptive_admm_mean + adaptive_rho_mean, 
               yerr=adaptive_rho_std, fmt='none', color='black', capsize=5)
    
    # Add value labels to bars
    ax.text(x[0], fixed_admm_mean/2, f'{fixed_admm_mean:.2f}',
            ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    
    ax.text(x[1], adaptive_admm_mean/2, f'{adaptive_admm_mean:.2f}',
            ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    
    ax.text(x[1], adaptive_admm_mean + adaptive_rho_mean/2, f'{adaptive_rho_mean:.2f}',
            ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    
    # Set axis labels and ticks
    ax.set_ylabel('Time per Iteration (ms)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Fixed-Step\nADMM', 'First-Order\nAdaptive ADMM'], fontweight='bold')
    
    # Calculate total time for annotation
    fixed_total = fixed_admm_mean
    adaptive_total = adaptive_admm_mean + adaptive_rho_mean
    
    # Add total time annotation
    ax.text(x[0], fixed_total + 1, f'Total: {fixed_total:.2f} ms',
            ha='center', va='bottom', fontsize=10)
    ax.text(x[1], adaptive_total + 1, f'Total: {adaptive_total:.2f} ms',
            ha='center', va='bottom', fontsize=10)
    
    # Set y-axis limit with some padding
    y_max = max(fixed_total + fixed_admm_std, adaptive_total + adaptive_rho_std) * 1.2
    ax.set_ylim(0, y_max)
    
    # Add title and custom legend
    ax.set_title('Computation Time per Iteration', fontweight='bold', fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = [(handles[0], 'Fixed ADMM Time'), 
                     (handles[1], 'Adaptive ADMM Time'),
                     (handles[2], 'Rho Update Time')]
    ax.legend(*zip(*unique_labels), loc='upper right')
    
    # Add grid for y-axis
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Display the figure
    plt.tight_layout()
    plt.show()


def create_boxplot(df, fixed_color, adapt_color):
    """Create a box plot showing the distribution of iteration times and display it"""
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create color palette
    palette = {'Fixed': fixed_color, 'Adaptive': adapt_color}
    
    # Create the boxplot with individual points
    sns.boxplot(x='Method', y='TimePerIteration', data=df, palette=palette, width=0.5)
    sns.stripplot(x='Method', y='TimePerIteration', data=df, color='black', alpha=0.5, size=4, jitter=True)
    
    # Set labels and title
    plt.ylabel('Time per Iteration (ms)', fontweight='bold')
    plt.xlabel('')
    plt.title('Distribution of Iteration Times', fontweight='bold', fontsize=14)
    
    # Add median values as text
    medians = df.groupby('Method')['TimePerIteration'].median()
    for i, method in enumerate(['Fixed', 'Adaptive']):
        plt.text(i, medians[method] * 1.1, f'Median: {medians[method]:.2f} ms',
                ha='center', va='bottom', fontweight='bold')
    
    # Display the figure
    plt.tight_layout()
    plt.show()


def create_scatter_plot(df, fixed_color, adapt_color):
    """Create a scatter plot showing the relationship between iterations and solve time and display it"""
    
    # Create figure
    plt.figure(figsize=(9, 6))
    
    # Create color palette
    palette = {'Fixed': fixed_color, 'Adaptive': adapt_color}
    
    # Create the scatter plot with linear regression line
    sns.scatterplot(x='Iterations', y='SolveTime', hue='Method', data=df, palette=palette, s=80, alpha=0.7)
    sns.regplot(x='Iterations', y='SolveTime', data=df[df['Method']=='Fixed'], 
               scatter=False, color=fixed_color, line_kws={'linestyle':'--'})
    sns.regplot(x='Iterations', y='SolveTime', data=df[df['Method']=='Adaptive'], 
               scatter=False, color=adapt_color, line_kws={'linestyle':'--'})
    
    # Set labels and title
    plt.ylabel('Total Solve Time (ms)', fontweight='bold')
    plt.xlabel('Number of Iterations', fontweight='bold')
    plt.title('Iterations vs. Total Solve Time', fontweight='bold', fontsize=14)
    
    # Add legend with more descriptive labels
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=['Fixed-Step ADMM', 'First-Order Adaptive ADMM'], 
              title='Method', loc='upper left')
    
    # Add annotations for trends
    fixed_avg_iter = df[df['Method']=='Fixed']['Iterations'].mean()
    adaptive_avg_iter = df[df['Method']=='Adaptive']['Iterations'].mean()
    
    plt.axvline(x=fixed_avg_iter, color=fixed_color, linestyle=':', alpha=0.5)
    plt.axvline(x=adaptive_avg_iter, color=adapt_color, linestyle=':', alpha=0.5)
    
    plt.text(fixed_avg_iter, df['SolveTime'].max() * 0.2, 
            f'Avg Fixed: {fixed_avg_iter:.1f}', 
            ha='right', va='bottom', color=fixed_color, fontweight='bold')
    
    plt.text(adaptive_avg_iter, df['SolveTime'].max() * 0.2, 
            f'Avg Adaptive: {adaptive_avg_iter:.1f}', 
            ha='left', va='bottom', color=adapt_color, fontweight='bold')
    
    # Display the figure
    plt.tight_layout()
    plt.show()


def create_grouped_bar_chart(fixed_data, adaptive_data, fixed_color, adapt_color):
    """Create a grouped bar chart comparing key metrics and display it"""
    
    # Calculate key metrics
    metrics = {
        'Avg Iterations': (fixed_data['Iterations'].mean(), adaptive_data['Iterations'].mean()),
        'Avg Solve Time (ms)': (fixed_data['SolveTime'].mean(), adaptive_data['SolveTime'].mean()),
    }
    
    # Calculate standard deviations for error bars
    errors = {
        'Avg Iterations': (fixed_data['Iterations'].std(), adaptive_data['Iterations'].std()),
        'Avg Solve Time (ms)': (fixed_data['SolveTime'].std(), adaptive_data['SolveTime'].std()),
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Set bar width and positions
    width = 0.35
    x = np.arange(len(metrics))
    
    # Create bars
    rects1 = ax.bar(x - width/2, [m[0] for m in metrics.values()], width, 
                   label='Fixed-Step ADMM', color=fixed_color)
    rects2 = ax.bar(x + width/2, [m[1] for m in metrics.values()], width, 
                   label='First-Order Adaptive ADMM', color=adapt_color)
    
    # Add error bars
    ax.errorbar(x - width/2, [m[0] for m in metrics.values()], 
               yerr=[e[0] for e in errors.values()], fmt='none', color='black', capsize=5)
    ax.errorbar(x + width/2, [m[1] for m in metrics.values()], 
               yerr=[e[1] for e in errors.values()], fmt='none', color='black', capsize=5)
    
    # Add labels and title
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontweight='bold')
    ax.set_title('Performance Comparison', fontweight='bold', fontsize=14)
    ax.legend()
    
    # Add percentage improvement annotations
    for i, (metric, (fixed_val, adaptive_val)) in enumerate(metrics.items()):
        pct_change = (fixed_val - adaptive_val) / fixed_val * 100
        improvement_text = f"{pct_change:.1f}% reduction"
        
        ax.annotate(improvement_text, 
                   xy=(i, min(fixed_val, adaptive_val) / 2),
                   xytext=(0, 0),
                   textcoords="offset points",
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.3))
    
    # Add value labels to bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(rect.get_x() + rect.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def print_statistics(fixed_data, adaptive_data):
    """Print summary statistics"""
    
    # Per-iteration stats
    fixed_admm_per_iter = fixed_data['ADMMTimePerIteration'].mean()
    adaptive_admm_per_iter = adaptive_data['ADMMTimePerIteration'].mean()
    adaptive_rho_per_iter = adaptive_data['RhoTimePerIteration'].mean()
    
    # Total times
    fixed_total_per_iter = fixed_data['TimePerIteration'].mean()
    adaptive_total_per_iter = adaptive_data['TimePerIteration'].mean()
    
    # Solve time stats
    fixed_avg_iterations = fixed_data['Iterations'].mean()
    adaptive_avg_iterations = adaptive_data['Iterations'].mean()
    fixed_avg_solve_time = fixed_data['SolveTime'].mean()
    adaptive_avg_solve_time = adaptive_data['SolveTime'].mean()
    
    # Calculate speedup
    iter_reduction = (fixed_avg_iterations - adaptive_avg_iterations) / fixed_avg_iterations * 100
    time_reduction = (fixed_avg_solve_time - adaptive_avg_solve_time) / fixed_avg_solve_time * 100
    
    # Print statistics
    print("\n==================== ADMM Performance Statistics ====================")
    print("\nTime per Iteration Statistics:")
    print(f"Fixed ADMM time per iteration: {fixed_admm_per_iter:.3f} ms (± {fixed_data['ADMMTimePerIteration'].std():.3f})")
    print(f"Adaptive ADMM time per iteration: {adaptive_admm_per_iter:.3f} ms (± {adaptive_data['ADMMTimePerIteration'].std():.3f})")
    print(f"Adaptive Rho time per iteration: {adaptive_rho_per_iter:.3f} ms (± {adaptive_data['RhoTimePerIteration'].std():.3f})")
    print(f"Fixed total time per iteration: {fixed_total_per_iter:.3f} ms")
    print(f"Adaptive total time per iteration: {adaptive_total_per_iter:.3f} ms")
    
    print("\nOverall Solve Time Statistics:")
    print(f"Fixed average iterations: {fixed_avg_iterations:.1f} (± {fixed_data['Iterations'].std():.1f})")
    print(f"Adaptive average iterations: {adaptive_avg_iterations:.1f} (± {adaptive_data['Iterations'].std():.1f})")
    print(f"Fixed average solve time: {fixed_avg_solve_time:.3f} ms (± {fixed_data['SolveTime'].std():.3f})")
    print(f"Adaptive average solve time: {adaptive_avg_solve_time:.3f} ms (± {adaptive_data['SolveTime'].std():.3f})")
    
    print("\nPerformance Improvements:")
    print(f"Iteration reduction: {iter_reduction:.1f}%")
    print(f"Overall solve time reduction: {time_reduction:.1f}%")
    print(f"Rho calculation overhead: {(adaptive_rho_per_iter / adaptive_total_per_iter * 100):.1f}% of iteration time")
    print("\n====================================================================")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Get file path from command line argument or use default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter path to data file: ")
    
    # Get output folder from command line argument or use default
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    else:
        output_folder = "plots"
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(exist_ok=True)
    
    # Create plots
    create_plots(file_path)
    print(f"Plots saved to '{output_folder}' folder")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# def create_bar_chart(data_file, output_file='iteration_overhead.png'):
#     """
#     Create a simple stacked bar chart from the MPC benchmark data
#     """
#     # Read and process data - skip lines with ===
#     lines = []
#     with open(data_file, 'r') as f:
#         for line in f:
#             if '===' not in line and ',' in line:
#                 lines.append(line)
    
#     # Find the header line
#     header_line = None
#     for i, line in enumerate(lines):
#         if "Method" in line and "Trial" in line:
#             header_line = i
#             break
    
#     if header_line is None:
#         print("Could not find header line, using first line as header")
#         header_line = 0
    
#     # Parse CSV data
#     data = []
#     headers = [h.strip() for h in lines[header_line].split(',')]
    
#     for line in lines[header_line+1:]:
#         parts = [p.strip() for p in line.split(',')]
#         if len(parts) == len(headers):
#             row = {}
#             for i, header in enumerate(headers):
#                 try:
#                     row[header] = float(parts[i]) if i > 0 else parts[i]
#                 except:
#                     row[header] = parts[i]
            
#             # Simplify to just "Fixed" or "Adaptive" methods
#             if "Fixed" in row['Method']:
#                 row['Method'] = "Fixed"
#             elif "Adaptive" in row['Method']:
#                 row['Method'] = "Adaptive"
                
#             data.append(row)
    
#     # Create DataFrame
#     df = pd.DataFrame(data)
    
#     # Calculate time per iteration
#     df['TimePerIteration'] = df['SolveTime'] / df['Iterations']
#     df['ADMMTimePerIteration'] = df['ADMMTime'] / df['Iterations']
#     df['RhoTimePerIteration'] = df['RhoTime'] / df['Iterations']
    
#     # Calculate statistics
#     fixed_data = df[df['Method'] == 'Fixed']
#     adaptive_data = df[df['Method'] == 'Adaptive']
    
#     # Per-iteration stats
#     fixed_admm_per_iter = fixed_data['ADMMTimePerIteration'].mean()
#     adaptive_admm_per_iter = adaptive_data['ADMMTimePerIteration'].mean()
#     adaptive_rho_per_iter = adaptive_data['RhoTimePerIteration'].mean()
    
#     # Total times
#     fixed_total_per_iter = fixed_data['TimePerIteration'].mean()
#     adaptive_total_per_iter = adaptive_data['TimePerIteration'].mean()
    
#     # Solve time stats
#     fixed_avg_iterations = fixed_data['Iterations'].mean()
#     adaptive_avg_iterations = adaptive_data['Iterations'].mean()
#     fixed_avg_solve_time = fixed_data['SolveTime'].mean()
#     adaptive_avg_solve_time = adaptive_data['SolveTime'].mean()
    
#     # Calculate speedup
#     speedup = (fixed_avg_solve_time - adaptive_avg_solve_time) / fixed_avg_solve_time * 100
    
#     # Create the bar chart
#     fixed_color = '#d62728'  # Red
#     adapt_color = '#1f77b4'  # Blue
    
#     # Create figure 
#     plt.figure(figsize=(8, 6))
    
#     # Define bar positions and width
#     x = np.array([0, 1])
#     width = 0.5

#     plt.style.use('default')

#         # Set clean, minimal style
#     plt.rcParams.update({
#         'font.family': 'serif',
#         'font.size': 12,
#         'axes.labelsize': 14,
#         'grid.linestyle': ':',
#         'grid.alpha': 0.5,
#         'lines.linewidth': 2.5,
#         'axes.grid': False,

#     })
    
#     # Create figure and get axis object
#     fig, ax = plt.subplots(figsize=(8, 6))
    
#     # Plot bars
#     ax.bar(x[0], fixed_admm_per_iter, width, label='Fixed ADMM Time', color=fixed_color)
#     ax.bar(x[1], adaptive_admm_per_iter, width, label='Adaptive ADMM Time', color=adapt_color)
#     ax.bar(x[1], adaptive_rho_per_iter, width, bottom=adaptive_admm_per_iter, 
#            label='Rho Update Time', color=adapt_color, alpha=0.3)
    
#     # Add value labels to bars
#     ax.text(x[0], fixed_admm_per_iter/2, f'{fixed_admm_per_iter:.2f}',
#             ha='center', va='center', color='black', fontweight='bold', fontsize=14)
    
#     ax.text(x[1], adaptive_admm_per_iter/2, f'{adaptive_admm_per_iter:.2f}',
#             ha='center', va='center', color='black', fontweight='bold', fontsize=14)
    
#     ax.text(x[1], adaptive_admm_per_iter + adaptive_rho_per_iter/2, f'{adaptive_rho_per_iter:.2f}',
#             ha='center', va='center', color='black', fontweight='bold')
    
#     # Set axis labels and ticks
#     ax.set_ylabel('Time per Iteration (ms)')
#     ax.set_xticks(x)
#     ax.set_xticklabels(['Fixed', 'First-Order\nAdaptive'])
    
#     # Make axes visible
#     ax.spines['left'].set_visible(True)
#     ax.spines['bottom'].set_visible(True)
#     ax.spines['left'].set_color('black')
#     ax.spines['bottom'].set_color('black')
    
#     ax.set_ylim(0, 75)
#     # Add grid only for y-axis
#     #ax.yaxis.grid(False, linestyle='--', alpha=0.7)
    
#     ax.legend()
    
#     # Save the figure
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Print statistics
#     print("\nTime per Iteration Statistics:")
#     print(f"Fixed ADMM time per iteration: {fixed_admm_per_iter:.3f} ms")
#     print(f"Adaptive ADMM time per iteration: {adaptive_admm_per_iter:.3f} ms")
#     print(f"Adaptive Rho time per iteration: {adaptive_rho_per_iter:.3f} ms")
#     print(f"Fixed total time per iteration: {fixed_total_per_iter:.3f} ms")
#     print(f"Adaptive total time per iteration: {adaptive_total_per_iter:.3f} ms")
    
#     print("\nOverall Solve Time Statistics:")
#     print(f"Fixed average iterations: {fixed_avg_iterations:.1f}")
#     print(f"Adaptive average iterations: {adaptive_avg_iterations:.1f}")
#     print(f"Fixed average solve time: {fixed_avg_solve_time:.3f} ms")
#     print(f"Adaptive average solve time: {adaptive_avg_solve_time:.3f} ms")
#     print(f"Overall speedup: {speedup:.1f}%")
    
#     print(f"\nRho calculation overhead: {(adaptive_rho_per_iter / adaptive_total_per_iter * 100):.1f}% of iteration time")

# if __name__ == "__main__":
#     import sys
    
#     # Get file path from command line argument or use default
#     file_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter path to data file: ")
    
#     create_bar_chart(file_path)
#     print(f"Bar chart saved to 'iteration_overhead.png'")
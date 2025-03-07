import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data processing function
def process_data(data_str):
    # Split the data into lines and filter out non-data lines
    lines = [line for line in data_str.split('\n') 
            if line and ',' in line and 'solver' not in line and 'END' not in line]
    
    # Parse into DataFrame
    data = []
    for line in lines:
        solver, run, iters, time = line.split(',')
        data.append({
            'solver': solver,
            'run': int(run),
            'iterations': int(iters),
            'solve_time_us': float(time)
        })
    
    return pd.DataFrame(data)

# Create DataFrame
df = process_data("""
solver,run,iterations,solve_time_us
fixed,0,500,29589.00
fixed,1,500,29595.00
fixed,2,500,29587.00
fixed,3,76,4587.00
fixed,4,64,3877.00
fixed,5,64,3878.00
fixed,6,64,3878.00
fixed,7,64,3877.00
fixed,8,64,3878.00
fixed,9,64,3878.00
fixed,10,64,3877.00
fixed,11,64,3878.00
fixed,12,64,3878.00
fixed,13,64,3877.00
fixed,14,64,3878.00
fixed,15,64,3878.00
fixed,16,64,3877.00
fixed,17,64,3878.00
fixed,18,64,3878.00
fixed,19,64,3877.00
fixed,20,64,3878.00
fixed,21,64,3878.00
fixed,22,64,3877.00
fixed,23,64,3878.00
fixed,24,64,3878.00
fixed,25,64,3878.00
fixed,26,64,3877.00
fixed,27,64,3878.00
fixed,28,64,3878.00
fixed,29,64,3877.00
fixed,30,64,3878.00
fixed,31,64,3878.00
fixed,32,64,3877.00
fixed,33,64,3878.00
fixed,34,64,3878.00
fixed,35,64,3877.00
fixed,36,64,3878.00
fixed,37,64,3878.00
fixed,38,64,3877.00
fixed,39,64,3878.00
fixed,40,64,3878.00
fixed,41,64,3877.00
fixed,42,64,3878.00
fixed,43,64,3878.00
fixed,44,64,3877.00
fixed,45,64,3878.00
fixed,46,64,3878.00
fixed,47,64,3878.00
fixed,48,64,3878.00
fixed,49,64,3878.00
fixed,50,64,3878.00
fixed,51,64,3878.00
fixed,52,64,3878.00
fixed,53,64,3878.00
fixed,54,64,3878.00
fixed,55,64,3878.00
fixed,56,64,3878.00
fixed,57,64,3877.00
fixed,58,64,3878.00
fixed,59,64,3878.00
fixed,60,64,3877.00
fixed,61,64,3878.00
fixed,62,64,3878.00
fixed,63,64,3877.00
fixed,64,64,3878.00
fixed,65,64,3878.00
fixed,66,64,3877.00
fixed,67,64,3878.00
fixed,68,64,3878.00
fixed,69,64,3877.00
fixed,70,64,3878.00
fixed,71,64,3878.00
fixed,72,64,3877.00
fixed,73,64,3878.00
fixed,74,64,3878.00
fixed,75,64,3877.00
fixed,76,64,3878.00
fixed,77,64,3878.00
fixed,78,64,3877.00
fixed,79,64,3878.00
fixed,80,64,3878.00
fixed,81,64,3877.00
fixed,82,64,3877.00
fixed,83,64,3878.00
fixed,84,64,3878.00
fixed,85,64,3877.00
fixed,86,64,3878.00
fixed,87,64,3878.00
fixed,88,64,3877.00
fixed,89,64,3878.00
fixed,90,64,3878.00
fixed,91,64,3877.00
fixed,92,64,3878.00
fixed,93,64,3878.00
fixed,94,64,3877.00
fixed,95,64,3878.00
fixed,96,64,3878.00
fixed,97,64,3877.00
fixed,98,64,3878.00
fixed,99,64,3878.00
END
solver,run,iterations,solve_time_us
adaptive,0,316,20056.00
adaptive,1,146,9315.00
adaptive,2,69,4440.00
adaptive,3,69,4439.00
adaptive,4,69,4439.00
adaptive,5,70,4517.00
adaptive,6,70,4518.00
adaptive,7,70,4517.00
adaptive,8,69,4439.00
adaptive,9,70,4517.00
adaptive,10,69,4439.00
adaptive,11,69,4439.00
adaptive,12,70,4517.00
adaptive,13,70,4518.00
adaptive,14,69,4439.00
adaptive,15,70,4518.00
adaptive,16,70,4518.00
adaptive,17,69,4439.00
adaptive,18,69,4439.00
adaptive,19,70,4517.00
adaptive,20,70,4518.00
adaptive,21,70,4518.00
adaptive,22,69,4440.00
adaptive,23,69,4439.00
adaptive,24,70,4517.00
adaptive,25,69,4440.00
adaptive,26,70,4518.00
adaptive,27,70,4518.00
adaptive,28,70,4517.00
adaptive,29,70,4518.00
adaptive,30,70,4517.00
adaptive,31,69,4439.00
adaptive,32,69,4439.00
adaptive,33,69,4439.00
adaptive,34,70,4518.00
adaptive,35,70,4518.00
adaptive,36,70,4517.00
adaptive,37,70,4518.00
adaptive,38,70,4518.00
adaptive,39,70,4517.00
adaptive,40,69,4440.00
adaptive,41,69,4439.00
adaptive,42,70,4518.00
adaptive,43,69,4439.00
adaptive,44,69,4439.00
adaptive,45,69,4439.00
adaptive,46,70,4517.00
adaptive,47,69,4440.00
adaptive,48,69,4439.00
adaptive,49,69,4440.00
adaptive,50,70,4518.00
adaptive,51,70,4518.00
adaptive,52,70,4518.00
adaptive,53,70,4517.00
adaptive,54,70,4518.00
adaptive,55,69,4440.00
adaptive,56,70,4517.00
adaptive,57,70,4517.00
adaptive,58,69,4440.00
adaptive,59,70,4518.00
adaptive,60,70,4517.00
adaptive,61,69,4439.00
adaptive,62,70,4517.00
adaptive,63,69,4439.00
adaptive,64,70,4518.00
adaptive,65,70,4517.00
adaptive,66,69,4440.00
adaptive,67,69,4439.00
adaptive,68,70,4518.00
adaptive,69,69,4440.00
adaptive,70,69,4440.00
adaptive,71,70,4518.00
adaptive,72,69,4440.00
adaptive,73,69,4440.00
adaptive,74,69,4439.00
adaptive,75,70,4518.00
adaptive,76,70,4517.00
adaptive,77,70,4518.00
adaptive,78,70,4518.00
adaptive,79,69,4439.00
adaptive,80,70,4517.00
adaptive,81,70,4518.00
adaptive,82,70,4518.00
adaptive,83,70,4517.00
adaptive,84,70,4517.00
adaptive,85,69,4439.00
adaptive,86,70,4518.00
adaptive,87,69,4439.00
adaptive,88,70,4518.00
adaptive,89,70,4517.00
adaptive,90,70,4518.00
adaptive,91,69,4439.00
adaptive,92,70,4518.00
adaptive,93,70,4518.00
adaptive,94,70,4518.00
adaptive,95,69,4439.00
adaptive,96,69,4439.00
adaptive,97,69,4440.00
adaptive,98,69,4439.00
adaptive,99,70,4518.00
END
""")  # Paste your data between the triple quotes

# Split by solver type
fixed_data = df[df['solver'] == 'fixed']
adaptive_data = df[df['solver'] == 'adaptive']

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Iterations vs Solve Number (Line Plot)
ax1.plot(fixed_data['run'], fixed_data['iterations'], 'b-', label='Fixed', alpha=0.7)
ax1.plot(adaptive_data['run'], adaptive_data['iterations'], 'r-', label='Adaptive', alpha=0.7)
ax1.set_xlabel('Solve Number')
ax1.set_ylabel('Iterations')
ax1.set_title('Iterations per Solve')
ax1.grid(True)
ax1.legend()

# Add statistics to first plot
stats_text1 = (f'Fixed Avg: {fixed_data["iterations"].mean():.1f}\n'
              f'Adaptive Avg: {adaptive_data["iterations"].mean():.1f}\n'
              f'Fixed Max: {fixed_data["iterations"].max()}\n'
              f'Adaptive Max: {adaptive_data["iterations"].max()}')
ax1.text(0.02, 0.98, stats_text1,
         transform=ax1.transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: CDF of solve times for successful solves
def plot_cdf(data, label, color):
    # Split into successful and failed solves
    successful = data[data['iterations'] < 500]
    failed = data[data['iterations'] >= 500]
    
    # Calculate percentage of successful solves
    success_rate = len(successful) / len(data) * 100
    
    if len(successful) > 0:
        # Sort successful solve times
        times_sorted = np.sort(successful['solve_time_us'])
        # Calculate percentiles as percentage of ALL solves (including failures)
        percentiles = np.arange(len(times_sorted)) / len(data) * 100
        ax2.plot(times_sorted, percentiles, f'{color}-', label=f'{label} ({success_rate:.1f}% success)', alpha=0.7)
    
    # Add a horizontal line at the final percentage to show failed solves
    if len(failed) > 0:
        ax2.axhline(y=success_rate, color=color, linestyle='--', alpha=0.3)

plot_cdf(fixed_data, 'Fixed', 'b')
plot_cdf(adaptive_data, 'Adaptive', 'r')

ax2.set_xlabel('Solve Time (µs)')
ax2.set_ylabel('Percent of Solves (%)')
ax2.set_title('Cumulative Distribution of Successful Solve Times')
ax2.grid(True)
ax2.legend()

# Add statistics to second plot
stats_text2 = (f'Fixed Avg: {fixed_data[fixed_data["iterations"] < 500]["solve_time_us"].mean():.1f}µs\n'
              f'Adaptive Avg: {adaptive_data[adaptive_data["iterations"] < 500]["solve_time_us"].mean():.1f}µs\n'
              f'Fixed Success: {(fixed_data["iterations"] < 500).mean()*100:.1f}%\n'
              f'Adaptive Success: {(adaptive_data["iterations"] < 500).mean()*100:.1f}%')
ax2.text(0.02, 0.98, stats_text2,
         transform=ax2.transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('solver_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from io import StringIO
# import numpy as np

# # Read and process data
# with open('benchmark_results.csv', 'r') as file:
#     lines = file.readlines()

# data_lines = []
# header = None
# for line in lines:
#     if line.startswith('Method,Trial'):
#         header = line
#     elif ',' in line and not line.startswith('===') and not line.startswith('Starting'):
#         data_lines.append(line)

# data_str = header + ''.join(data_lines)
# df = pd.read_csv(StringIO(data_str))
# df['Method'] = df['Method'].str.replace(' Hover', '')

# # Function to identify outliers
# def find_outliers(group):
#     q1 = group['SolveTime'].quantile(0.25)
#     q3 = group['SolveTime'].quantile(0.75)
#     iqr = q3 - q1
#     upper_bound = q3 + 1.5 * iqr
#     lower_bound = q1 - 1.5 * iqr
#     outliers = group[(group['SolveTime'] > upper_bound) | (group['SolveTime'] < lower_bound)]
#     return outliers

# # Find outliers for each method
# fixed_outliers = find_outliers(df[df['Method'] == 'Fixed'])
# adaptive_outliers = find_outliers(df[df['Method'] == 'Adaptive'])

# # 1. Violin Plot
# plt.figure(figsize=(10, 6))
# ax = sns.violinplot(data=df, x='Method', y='SolveTime', 
#                     palette=['lightcoral', 'lightblue'],
#                     inner=None,
#                     width=0.7)

# # Add larger median dots manually
# for i, method in enumerate(['Fixed', 'Adaptive']):
#     median = df[df['Method'] == method]['SolveTime'].median()
#     ax.scatter(i, median, color='white', edgecolor='black', s=200, zorder=3)

# plt.xlabel('Method', fontsize=12, labelpad=10)
# plt.ylabel('Computation Time (µs)', fontsize=12, labelpad=10)
# plt.grid(True, axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# # 2. CDF Plot with shading (corrected)
# plt.figure(figsize=(10, 6))
# for method, color in zip(['Fixed', 'Adaptive'], ['lightcoral', 'lightblue']):
#     data = df[df['Method'] == method]
    
#     # Calculate total number of problems including non-converged ones
#     total_problems = len(data)
    
#     # Filter out non-converged cases (500 iterations)
#     converged_data = data[data['Iterations'] < 500]['SolveTime']
#     num_converged = len(converged_data)
    
#     # Calculate CDF
#     x = np.sort(converged_data)
#     # Adjust percentage calculation to account for non-converged cases
#     y = np.arange(1, len(x) + 1) / total_problems * 100
    
#     # Plot line and filled area
#     plt.plot(x, y, color=color, 
#             label=f'{method} ({num_converged/total_problems*100:.1f}% solved)', 
#             linewidth=2)
#     plt.fill_between(x, y, alpha=0.2, color=color)

# plt.xlabel('Solve Time (µs)', fontsize=12, labelpad=10)
# plt.ylabel('Percentage of Solved Problems (%)', fontsize=12, labelpad=10)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # Print convergence statistics
# print("\nConvergence Statistics:")
# for method in ['Fixed', 'Adaptive']:
#     data = df[df['Method'] == method]
#     total = len(data)
#     converged = len(data[data['Iterations'] < 500])
#     print(f"\n{method} Method:")
#     print(f"Total problems: {total}")
#     print(f"Converged problems: {converged}")
#     print(f"Convergence rate: {converged/total*100:.1f}%")

# # Print statistics (for your reference)
# print("\nKey Statistics:")
# for method in ['Fixed', 'Adaptive']:
#     data = df[df['Method'] == method]
#     print(f"\n{method} Method:")
#     print(f"Median Time: {data['SolveTime'].median():.2f} µs")
#     print(f"95th percentile: {data['SolveTime'].quantile(0.95):.2f} µs")

# # Print outlier analysis
# print("\nOutlier Analysis:")
# print("\nFixed Method Outliers:")
# if len(fixed_outliers) == 0:
#     print("No outliers found")
# else:
#     print(f"Number of outliers: {len(fixed_outliers)}")
#     print("\nOutlier details:")
#     print(fixed_outliers[['Trial', 'SolveTime', 'Iterations']])

# print("\nAdaptive Method Outliers:")
# if len(adaptive_outliers) == 0:
#     print("No outliers found")
# else:
#     print(f"Number of outliers: {len(adaptive_outliers)}")
#     print("\nOutlier details:")
#     print(adaptive_outliers[['Trial', 'SolveTime', 'Iterations']])

# # Print basic statistics for context
# print("\nBasic Statistics:")
# for method in ['Fixed', 'Adaptive']:
#     data = df[df['Method'] == method]
#     q1 = data['SolveTime'].quantile(0.25)
#     q3 = data['SolveTime'].quantile(0.75)
#     iqr = q3 - q1
#     print(f"\n{method} Method:")
#     print(f"Q1: {q1:.2f} µs")
#     print(f"Q3: {q3:.2f} µs")
#     print(f"IQR: {iqr:.2f} µs")
#     print(f"Upper bound for outliers: {(q3 + 1.5*iqr):.2f} µs")
#     print(f"Lower bound for outliers: {(q1 - 1.5*iqr):.2f} µs")

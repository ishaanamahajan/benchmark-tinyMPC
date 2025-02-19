import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np

# Read and process data
with open('benchmark_results.csv', 'r') as file:
    lines = file.readlines()

data_lines = []
header = None
for line in lines:
    if line.startswith('Method,Trial'):
        header = line
    elif ',' in line and not line.startswith('===') and not line.startswith('Starting'):
        data_lines.append(line)

data_str = header + ''.join(data_lines)
df = pd.read_csv(StringIO(data_str))
df['Method'] = df['Method'].str.replace(' Hover', '')

# Function to identify outliers
def find_outliers(group):
    q1 = group['SolveTime'].quantile(0.25)
    q3 = group['SolveTime'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    outliers = group[(group['SolveTime'] > upper_bound) | (group['SolveTime'] < lower_bound)]
    return outliers

# Find outliers for each method
fixed_outliers = find_outliers(df[df['Method'] == 'Fixed'])
adaptive_outliers = find_outliers(df[df['Method'] == 'Adaptive'])

# 1. Violin Plot
plt.figure(figsize=(10, 6))
ax = sns.violinplot(data=df, x='Method', y='SolveTime', 
                    palette=['lightcoral', 'lightblue'],
                    inner=None,
                    width=0.7)

# Add larger median dots manually
for i, method in enumerate(['Fixed', 'Adaptive']):
    median = df[df['Method'] == method]['SolveTime'].median()
    ax.scatter(i, median, color='white', edgecolor='black', s=200, zorder=3)

plt.xlabel('Method', fontsize=12, labelpad=10)
plt.ylabel('Computation Time (µs)', fontsize=12, labelpad=10)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. CDF Plot with shading (corrected)
plt.figure(figsize=(10, 6))
for method, color in zip(['Fixed', 'Adaptive'], ['lightcoral', 'lightblue']):
    data = df[df['Method'] == method]
    
    # Calculate total number of problems including non-converged ones
    total_problems = len(data)
    
    # Filter out non-converged cases (500 iterations)
    converged_data = data[data['Iterations'] < 500]['SolveTime']
    num_converged = len(converged_data)
    
    # Calculate CDF
    x = np.sort(converged_data)
    # Adjust percentage calculation to account for non-converged cases
    y = np.arange(1, len(x) + 1) / total_problems * 100
    
    # Plot line and filled area
    plt.plot(x, y, color=color, 
            label=f'{method} ({num_converged/total_problems*100:.1f}% solved)', 
            linewidth=2)
    plt.fill_between(x, y, alpha=0.2, color=color)

plt.xlabel('Solve Time (µs)', fontsize=12, labelpad=10)
plt.ylabel('Percentage of Solved Problems (%)', fontsize=12, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Print convergence statistics
print("\nConvergence Statistics:")
for method in ['Fixed', 'Adaptive']:
    data = df[df['Method'] == method]
    total = len(data)
    converged = len(data[data['Iterations'] < 500])
    print(f"\n{method} Method:")
    print(f"Total problems: {total}")
    print(f"Converged problems: {converged}")
    print(f"Convergence rate: {converged/total*100:.1f}%")

# Print statistics (for your reference)
print("\nKey Statistics:")
for method in ['Fixed', 'Adaptive']:
    data = df[df['Method'] == method]
    print(f"\n{method} Method:")
    print(f"Median Time: {data['SolveTime'].median():.2f} µs")
    print(f"95th percentile: {data['SolveTime'].quantile(0.95):.2f} µs")

# Print outlier analysis
print("\nOutlier Analysis:")
print("\nFixed Method Outliers:")
if len(fixed_outliers) == 0:
    print("No outliers found")
else:
    print(f"Number of outliers: {len(fixed_outliers)}")
    print("\nOutlier details:")
    print(fixed_outliers[['Trial', 'SolveTime', 'Iterations']])

print("\nAdaptive Method Outliers:")
if len(adaptive_outliers) == 0:
    print("No outliers found")
else:
    print(f"Number of outliers: {len(adaptive_outliers)}")
    print("\nOutlier details:")
    print(adaptive_outliers[['Trial', 'SolveTime', 'Iterations']])

# Print basic statistics for context
print("\nBasic Statistics:")
for method in ['Fixed', 'Adaptive']:
    data = df[df['Method'] == method]
    q1 = data['SolveTime'].quantile(0.25)
    q3 = data['SolveTime'].quantile(0.75)
    iqr = q3 - q1
    print(f"\n{method} Method:")
    print(f"Q1: {q1:.2f} µs")
    print(f"Q3: {q3:.2f} µs")
    print(f"IQR: {iqr:.2f} µs")
    print(f"Upper bound for outliers: {(q3 + 1.5*iqr):.2f} µs")
    print(f"Lower bound for outliers: {(q1 - 1.5*iqr):.2f} µs")

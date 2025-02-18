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

# Create boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Method', y='SolveTime', 
            palette=['lightblue', 'lightcoral'],
            width=0.5,
            fliersize=4)

plt.title('Solve Time Distribution: Fixed vs Adaptive ρ', pad=20, fontsize=14)
plt.xlabel('Method', fontsize=12, labelpad=10)
plt.ylabel('Solve Time (µs)', fontsize=12, labelpad=10)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=11)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

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

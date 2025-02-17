import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np

# Read the raw file first
with open('benchmark_results.csv', 'r') as file:
    lines = file.readlines()

# Filter only the data lines that contain actual data (comma-separated values)
data_lines = []
header = None

for line in lines:
    if line.startswith('Method,Trial'):  # Capture the header
        header = line
    elif ',' in line and not line.startswith('===') and not line.startswith('Starting'):
        # Only include lines with data (contain commas and aren't section markers)
        data_lines.append(line)

# Create a temporary string with header and just the data
data_str = header + ''.join(data_lines)

# Read the filtered data into pandas
df = pd.read_csv(StringIO(data_str))

# Create figure with 3x2 subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

# 1. Paired differences plot (Fixed - Adaptive for same trial)
df_fixed = df[df['Method'] == 'Fixed'].set_index('Trial')
df_adaptive = df[df['Method'] == 'Adaptive'].set_index('Trial')
differences = df_fixed['SolveTime'] - df_adaptive['SolveTime']
sns.histplot(differences, ax=axes[0,0])
axes[0,0].axvline(x=0, color='r', linestyle='--')
axes[0,0].set_title('Distribution of Time Differences (Fixed - Adaptive)')
axes[0,0].set_xlabel('Time Difference (µs)')

# 2. Performance profiles
times_fixed = df[df['Method'] == 'Fixed']['SolveTime'].sort_values()
times_adaptive = df[df['Method'] == 'Adaptive']['SolveTime'].sort_values()
x_range = np.linspace(0, max(times_fixed.max(), times_adaptive.max()), 1000)
profile_fixed = [np.mean(times_fixed <= x) for x in x_range]
profile_adaptive = [np.mean(times_adaptive <= x) for x in x_range]
axes[0,1].plot(x_range, profile_fixed, label='Fixed')
axes[0,1].plot(x_range, profile_adaptive, label='Adaptive')
axes[0,1].set_title('Performance Profile')
axes[0,1].set_xlabel('Time (µs)')
axes[0,1].set_ylabel('Fraction of Problems Solved')
axes[0,1].legend()

# 3. Normalized comparison (relative to mean time per trial)
trial_means = df.groupby('Trial')['SolveTime'].mean()
df['NormalizedTime'] = df.apply(lambda row: row['SolveTime'] / trial_means[row['Trial']], axis=1)
sns.boxplot(data=df, x='Method', y='NormalizedTime', ax=axes[1,0])
axes[1,0].set_title('Normalized Solve Times')
axes[1,0].set_ylabel('Time Relative to Mean')

# 4. Time series with rolling average
window = 50
for method in ['Fixed', 'Adaptive']:
    method_data = df[df['Method'] == method]
    rolling_mean = method_data['SolveTime'].rolling(window=window).mean()
    axes[1,1].plot(method_data['Trial'], rolling_mean, label=method)
axes[1,1].set_title(f'Rolling Average Solve Time (window={window})')
axes[1,1].set_xlabel('Trial Number')
axes[1,1].set_ylabel('Time (µs)')
axes[1,1].legend()

# 5. Scatter plot comparing Fixed vs Adaptive for same trial
axes[2,0].scatter(df_fixed['SolveTime'], df_adaptive['SolveTime'], alpha=0.5)
max_time = max(df['SolveTime'])
axes[2,0].plot([0, max_time], [0, max_time], 'r--')  # diagonal line
axes[2,0].set_title('Fixed vs Adaptive Times for Same Trial')
axes[2,0].set_xlabel('Fixed Time (µs)')
axes[2,0].set_ylabel('Adaptive Time (µs)')

# 6. Difficulty analysis (based on average time per trial)
df['Difficulty'] = df.apply(lambda row: 'Easy' if trial_means[row['Trial']] < trial_means.median() 
                          else 'Hard', axis=1)
sns.boxplot(data=df, x='Difficulty', y='SolveTime', hue='Method', ax=axes[2,1])
axes[2,1].set_title('Performance by Problem Difficulty')
axes[2,1].set_ylabel('Time (µs)')

plt.tight_layout()
plt.show()

# Print some statistical insights
print("\nStatistical Insights:")
print(f"Percentage of trials where Adaptive is faster: {(differences > 0).mean()*100:.1f}%")
print("\nPerformance by difficulty:")
print(df.groupby(['Difficulty', 'Method'])['SolveTime'].agg(['mean', 'std', 'count']))
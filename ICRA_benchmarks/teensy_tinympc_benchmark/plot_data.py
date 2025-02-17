import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

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

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Box plot comparing solve times
sns.boxplot(data=df, x='Method', y='SolveTime', ax=axes[0,0])
axes[0,0].set_title('Solve Time Distribution by Method')
axes[0,0].set_ylabel('Time (µs)')

# 2. Box plot comparing iterations
sns.boxplot(data=df, x='Method', y='Iterations', ax=axes[0,1])
axes[0,1].set_title('Iterations Distribution by Method')

# 3. Time series of solve times
sns.lineplot(data=df, x='Trial', y='SolveTime', hue='Method', ax=axes[1,0])
axes[1,0].set_title('Solve Time vs Trial Number')
axes[1,0].set_ylabel('Time (µs)')

# 4. Histogram of solve times
sns.histplot(data=df, x='SolveTime', hue='Method', multiple="layer", ax=axes[1,1])
axes[1,1].set_title('Distribution of Solve Times')
axes[1,1].set_xlabel('Time (µs)')

plt.tight_layout()
plt.show()
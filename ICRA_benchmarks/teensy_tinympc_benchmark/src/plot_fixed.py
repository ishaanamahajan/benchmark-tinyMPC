import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import os
import matplotlib as mpl

# Set up matplotlib for paper quality with larger fonts and wider plot
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 18  # Increased base font size
mpl.rcParams['axes.linewidth'] = 2.0  # Thicker axes
mpl.rcParams['axes.labelsize'] = 22  # Larger axis labels
mpl.rcParams['xtick.labelsize'] = 20  # Larger tick labels
mpl.rcParams['ytick.labelsize'] = 20  # Larger tick labels
mpl.rcParams['legend.fontsize'] = 20  # Larger legend
mpl.rcParams['figure.figsize'] = (16, 8)  # Wider figure
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

def read_and_combine_data(folder_path='fixed'):
    """Read all rho CSV files from the specified folder and combine them"""
    all_data = []
    
    # Find all rho CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, 'rho_*.csv'))
    
    if not csv_files:
        print(f"No CSV files found matching the pattern 'rho_*.csv' in folder '{folder_path}'")
        return None
    
    print(f"Found {len(csv_files)} CSV files in '{folder_path}':")
    
    # Process each file
    for file in csv_files:
        rho = extract_rho_from_filename(os.path.basename(file))
        if rho is None:
            print(f"Warning: Could not extract rho value from {file}, skipping")
            continue
            
        # Skip rho values less than 5.0
        if rho < 5.0:
            print(f"  Skipping {file} (rho={rho} < 5.0)")
            continue
            
        try:
            # Read the file as text first to remove the END marker
            with open(file, 'r') as f:
                lines = f.readlines()
            
            # Filter out lines containing "END" and empty lines
            clean_lines = [line for line in lines if "END" not in line and line.strip()]
            
            # Write to a temporary file
            temp_file = f"{file}.temp"
            with open(temp_file, 'w') as f:
                f.writelines(clean_lines)
            
            # Read the clean CSV file
            df = pd.read_csv(temp_file)
            
            # Remove temporary file
            os.remove(temp_file)
            
            # Check if file has content
            if df.empty:
                print(f"Warning: {file} is empty after cleaning, skipping")
                continue
            
            # Ensure rho is a float and not NaN
            df['rho'] = rho
            
            all_data.append(df)
            print(f"  Processed {file}: {len(df)} rows, rho={rho}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not all_data:
        print("No data could be processed from the CSV files")
        return None
        
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Drop any rows with NaN values
    combined_df = combined_df.dropna()
    
    return combined_df

def plot_rho_comparison(df):
    """Plot iterations vs rho values as a bar chart"""
    if df is None or len(df) == 0:
        print("No data to plot")
        return
    
    # Drop any rows with NaN values
    df = df.dropna(subset=['rho', 'iterations'])
    
    # Group by rho and calculate statistics
    stats = df.groupby('rho').agg({
        'iterations': ['mean', 'std', 'min', 'max', 'count']
    })
    
    # Sort rho values
    rho_values = sorted(df['rho'].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Bar chart of mean iterations
    mean_iterations = stats['iterations']['mean']
    std_iterations = stats['iterations']['std']
    
    # Find best rho value
    best_rho = mean_iterations.idxmin()
    
    # Create bar colors (green for best, blue for others)
    colors = ['forestgreen' if rho == best_rho else 'steelblue' for rho in rho_values]
    
    # Create equally spaced x positions
    x_positions = np.arange(len(rho_values))
    
    # Create bar chart with wider bars and equally spaced positions
    bars = ax.bar(x_positions, mean_iterations.loc[rho_values], 
            width=0.9,  # Slightly reduced from 0.9 to avoid overlap with equal spacing
            color=colors, 
            edgecolor='black', 
            linewidth=2.0,
            zorder=3)
    
    # Add error bars with thicker lines
    ax.errorbar(x_positions, mean_iterations.loc[rho_values], 
                yerr=std_iterations.loc[rho_values], 
                fmt='none', 
                ecolor='black', 
                capsize=6,
                capthick=2.0,
                elinewidth=2.0,
                zorder=4)
    
    # Customize plot with larger, bolder labels
    ax.set_xlabel('Penalty Parameter (ρ)', fontweight='bold')
    ax.set_ylabel('Average Iterations', fontweight='bold')
    
    # Set equally spaced x-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(rho)) if rho.is_integer() else f"{rho:.1f}" for rho in rho_values])
    
    # Remove grid as requested
    ax.grid(False)
    ax.set_axisbelow(True)
    
    # No title as requested
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('rho_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('rho_comparison.pdf', format='pdf', bbox_inches='tight')
    
    print(f"Plot saved as 'rho_comparison.png' and 'rho_comparison.pdf'")
    
    # Show plot
    plt.show()
    
    # Print statistics
    print("\nStatistics by Rho Value:")
    print(stats)
    print(f"\nBest Rho Value (lowest mean iterations): {best_rho}")

if __name__ == "__main__":
    print("Reading CSV files from 'fixed' folder...")
    data = read_and_combine_data('fixed')
    
    if data is not None:
        print(f"Total data points: {len(data)}")
        plot_rho_comparison(data)
    else:
        print("No data to plot")
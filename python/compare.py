import argparse
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(mode):

    base_path = f"{mode}/data/"
    # File paths for iterations and rho values
    iteration_files = {
        'normal': f"{base_path}iterations/normal_{mode}.txt",
        'adapt': f"{base_path}iterations/adapt_{mode}.txt",
        'normal_wind': f"{base_path}iterations/normal_{mode}_wind.txt", 
        'adapt_wind': f"{base_path}iterations/adapt_{mode}_wind.txt"
    }

    rho_files = {
        'adapt': f"{base_path}rho_vals/adapt_{mode}.txt",
        'adapt_wind': f"{base_path}rho_vals/adapt_{mode}_wind.txt"
    }

    # Initialize data storage
    iterations = {}
    rho_values = {}

    # Read iteration data
    for key, filepath in iteration_files.items():
        try:
            iterations[key] = np.loadtxt(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

    # Read rho data for adaptive cases
    for key, filepath in rho_files.items():
        try:
            rho_values[key] = np.loadtxt(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot iterations for all cases
    for key, iters in iterations.items():
        label = key.replace('_', ' ').title()
        ax1.plot(iters, label=label)

    # Add total iterations to legend labels
    legend_labels = []
    for key, iters in iterations.items():
        total_iters = int(np.sum(iters))
        label = f"{key.replace('_', ' ').title()} (Total: {total_iters})"
        legend_labels.append(label)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Iteration Count')
    ax1.set_title(f'Iteration Comparison - {mode.title()}')
    ax1.grid(True)
    ax1.legend(legend_labels)

    # Plot rho values for adaptive cases
    for key, rho in rho_values.items():
        label = key.replace('_', ' ').title()
        ax2.plot(rho, label=label)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Rho Value')
    ax2.set_title(f'Rho Values Comparison - {mode.title()}')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot comparison data for hover or trajectory cases')
    parser.add_argument('--mode', type=str, choices=['hover', 'traj'], required=True,
                        help='Mode to plot: hover or traj')
    
    args = parser.parse_args()
    plot_comparison(args.mode)

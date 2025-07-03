#!/usr/bin/env python3
"""
Generate all 6 plots and data files for TinyMPC scaling analysis
Replicates the safety filtering paper plots with additional input dimension analysis
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_scaling_data():
    """Load all benchmark data files"""
    data_files = glob.glob("benchmark_data/*.json")
    
    state_scaling_data = []
    horizon_scaling_data = []
    
    for file in data_files:
        with open(file, 'r') as f:
            data = json.load(f)
            
        # Add parsed fields if not present
        if 'nstates' not in data:
            config = data['config']
            parts = config.split('_')
            try:
                data['nstates'] = int(parts[1])
                data['ninputs'] = int(parts[3])
                if 'nhorizon' in config:
                    data['nhorizon'] = int(parts[5])
                else:
                    data['nhorizon'] = 20  # default
            except:
                continue
        
        # Sort into categories
        if 'state_scaling' in file:
            state_scaling_data.append(data)
        elif 'horizon_scaling' in file:
            horizon_scaling_data.append(data)
    
    return state_scaling_data, horizon_scaling_data

def create_plot_and_save_data(x_data, y_data, x_errors, y_errors, 
                             x_label, y_label, title, filename_prefix, 
                             log_scale=False, marker='o', color='blue'):
    """Create a plot and save corresponding data file"""
    
    # Create plot
    plt.figure(figsize=(8, 6))
    
    if x_errors is not None and y_errors is not None:
        plt.errorbar(x_data, y_data, xerr=x_errors, yerr=y_errors, 
                    fmt=marker, markersize=8, capsize=5, color=color, 
                    linewidth=2, markerfacecolor=color, markeredgecolor='black')
    else:
        plt.plot(x_data, y_data, marker=marker, markersize=8, 
                linewidth=2, color=color, markerfacecolor=color, markeredgecolor='black')
    
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    # Style similar to the reference plot
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{filename_prefix}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data file
    data_filename = f"{filename_prefix}.txt"
    with open(data_filename, 'w') as f:
        f.write(f"# {title}\n")
        f.write(f"# {x_label}\t{y_label}\n")
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            if x_errors is not None and y_errors is not None:
                f.write(f"{x:.1f}\t{y:.3f}\t{x_errors[i]:.3f}\t{y_errors[i]:.3f}\n")
            else:
                f.write(f"{x:.1f}\t{y:.3f}\n")
    
    print(f"Generated: {plot_filename} and {data_filename}")
    return plot_filename, data_filename

def main():
    print("Generating all 6 TinyMPC scaling plots and data files...")
    print("=" * 70)
    
    # Load data
    state_data, horizon_data = load_scaling_data()
    
    if not state_data:
        print("No state scaling data found! Run: python3 TRO/generate_comprehensive_data.py")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sort data
    state_data.sort(key=lambda x: x['nstates'])
    horizon_data.sort(key=lambda x: x['nhorizon'])
    
    # 1. State Dimension vs Memory Usage
    print("\n1. State Dimension vs Memory Usage")
    nstates_vals = [d['nstates'] for d in state_data]
    memory_vals = [d['stats']['memory_kb'] for d in state_data]
    
    create_plot_and_save_data(
        nstates_vals, memory_vals, None, None,
        'State Dimension (n)', 'Memory Usage (KB)',
        'Memory Usage vs State Dimension',
        f'plot1_state_vs_memory_{timestamp}',
        marker='o', color='blue'
    )
    
    # 2. State Dimension vs Time per Iteration
    print("2. State Dimension vs Time per Iteration")
    time_vals = [d['stats']['avg_time_us'] for d in state_data]
    time_errors = [d['stats']['std_time_us'] for d in state_data]
    
    create_plot_and_save_data(
        nstates_vals, time_vals, None, time_errors,
        'State Dimension (n)', 'Time per Iteration (μs)',
        'Computation Time vs State Dimension',
        f'plot2_state_vs_time_{timestamp}',
        log_scale=True, marker='o', color='blue'
    )
    
    # 3. Input Dimension vs Memory Usage (from state scaling data)
    print("3. Input Dimension vs Memory Usage")
    ninputs_vals = [d['ninputs'] for d in state_data]
    
    create_plot_and_save_data(
        ninputs_vals, memory_vals, None, None,
        'Input Dimension (m)', 'Memory Usage (KB)',
        'Memory Usage vs Input Dimension',
        f'plot3_input_vs_memory_{timestamp}',
        marker='s', color='green'
    )
    
    # 4. Input Dimension vs Time per Iteration
    print("4. Input Dimension vs Time per Iteration")
    
    create_plot_and_save_data(
        ninputs_vals, time_vals, None, time_errors,
        'Input Dimension (m)', 'Time per Iteration (μs)',
        'Computation Time vs Input Dimension',
        f'plot4_input_vs_time_{timestamp}',
        log_scale=True, marker='s', color='green'
    )
    
    # 5. Time Horizon vs Time per Iteration
    print("5. Time Horizon vs Time per Iteration")
    if horizon_data:
        horizon_vals = [d['nhorizon'] for d in horizon_data]
        horizon_time_vals = [d['stats']['avg_time_us'] for d in horizon_data]
        horizon_time_errors = [d['stats']['std_time_us'] for d in horizon_data]
        
        create_plot_and_save_data(
            horizon_vals, horizon_time_vals, None, horizon_time_errors,
            'Time Horizon (N)', 'Time per Iteration (μs)',
            'Computation Time vs Time Horizon',
            f'plot5_horizon_vs_time_{timestamp}',
            log_scale=True, marker='^', color='red'
        )
    
    # 6. Time Horizon vs Memory Usage
    print("6. Time Horizon vs Memory Usage")
    if horizon_data:
        horizon_memory_vals = [d['stats']['memory_kb'] for d in horizon_data]
        
        create_plot_and_save_data(
            horizon_vals, horizon_memory_vals, None, None,
            'Time Horizon (N)', 'Memory Usage (KB)',
            'Memory Usage vs Time Horizon',
            f'plot6_horizon_vs_memory_{timestamp}',
            marker='^', color='red'
        )
    
    # Create summary plot (2x3 grid like the reference image)
    print("\n7. Creating combined summary plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('TinyMPC Scaling Analysis - Predictive Safety Filtering', fontsize=20, fontweight='bold')
    
    # Plot 1: State vs Memory
    axes[0,0].plot(nstates_vals, memory_vals, 'o-', markersize=8, linewidth=2, color='blue')
    axes[0,0].set_xlabel('State Dimension (n)', fontweight='bold')
    axes[0,0].set_ylabel('Memory Usage (KB)', fontweight='bold')
    axes[0,0].set_title('Memory vs State Dimension', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: State vs Time  
    axes[0,1].errorbar(nstates_vals, time_vals, yerr=time_errors, fmt='o-', 
                       markersize=8, capsize=5, linewidth=2, color='blue')
    axes[0,1].set_xlabel('State Dimension (n)', fontweight='bold')
    axes[0,1].set_ylabel('Time per Iteration (μs)', fontweight='bold')
    axes[0,1].set_title('Time vs State Dimension', fontweight='bold')
    axes[0,1].set_yscale('log')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Input vs Memory
    axes[0,2].plot(ninputs_vals, memory_vals, 's-', markersize=8, linewidth=2, color='green')
    axes[0,2].set_xlabel('Input Dimension (m)', fontweight='bold')
    axes[0,2].set_ylabel('Memory Usage (KB)', fontweight='bold')
    axes[0,2].set_title('Memory vs Input Dimension', fontweight='bold')
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Input vs Time
    axes[1,0].errorbar(ninputs_vals, time_vals, yerr=time_errors, fmt='s-',
                       markersize=8, capsize=5, linewidth=2, color='green')
    axes[1,0].set_xlabel('Input Dimension (m)', fontweight='bold')
    axes[1,0].set_ylabel('Time per Iteration (μs)', fontweight='bold')
    axes[1,0].set_title('Time vs Input Dimension', fontweight='bold')
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 5: Horizon vs Time
    if horizon_data:
        axes[1,1].errorbar(horizon_vals, horizon_time_vals, yerr=horizon_time_errors, 
                          fmt='^-', markersize=8, capsize=5, linewidth=2, color='red')
        axes[1,1].set_xlabel('Time Horizon (N)', fontweight='bold')
        axes[1,1].set_ylabel('Time per Iteration (μs)', fontweight='bold')
        axes[1,1].set_title('Time vs Time Horizon', fontweight='bold')
        axes[1,1].set_yscale('log')
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Horizon vs Memory
        axes[1,2].plot(horizon_vals, horizon_memory_vals, '^-', markersize=8, linewidth=2, color='red')
        axes[1,2].set_xlabel('Time Horizon (N)', fontweight='bold')
        axes[1,2].set_ylabel('Memory Usage (KB)', fontweight='bold')
        axes[1,2].set_title('Memory vs Time Horizon', fontweight='bold')
        axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_filename = f'tinympc_scaling_summary_{timestamp}.png'
    plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated combined plot: {summary_filename}")
    
    print("\n" + "=" * 70)
    print("All plots and data files generated successfully!")
    print(f"\nFiles created:")
    print(f"- 6 individual plots (plot1-6_*.png)")
    print(f"- 6 corresponding data files (plot1-6_*.txt)")
    print(f"- 1 combined summary plot ({summary_filename})")
    
    # List all generated files
    print(f"\nGenerated files:")
    for file in sorted(glob.glob(f"plot*_{timestamp}.*")):
        print(f"  {file}")
    print(f"  {summary_filename}")

if __name__ == "__main__":
    main()
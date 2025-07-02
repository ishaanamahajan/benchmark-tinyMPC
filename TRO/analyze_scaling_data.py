#!/usr/bin/env python3
"""
TinyMPC Scaling Analysis Script
Analyzes collected benchmark data and generates plots
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def load_benchmark_data(data_dir="benchmark_data"):
    """Load all benchmark data files"""
    data_files = glob.glob(f"{data_dir}/*.json")
    all_data = []
    
    for file in data_files:
        with open(file, 'r') as f:
            data = json.load(f)
            # Parse config name
            config = data['config']
            parts = config.split('_')
            if len(parts) >= 4 and parts[0] == 'nstates' and parts[2] == 'ninputs':
                data['nstates'] = int(parts[1])
                data['ninputs'] = int(parts[3])
                all_data.append(data)
    
    return all_data

def estimate_memory(nstates, ninputs, nhorizon=20):
    """Calculate memory usage in KB"""
    float_size = 4  # bytes
    
    # Based on TinyMPC data structures
    memory_bytes = (
        # Workspace matrices
        nstates * nhorizon * float_size * 6 +  # x, q, p, v, vnew, g
        ninputs * (nhorizon-1) * float_size * 6 +  # u, r, d, z, znew, y
        nstates * nstates * float_size * 2 +  # Adyn, AmBKt
        nstates * ninputs * float_size * 2 +  # Bdyn, Kinf
        ninputs * ninputs * float_size * 2 +  # Quu_inv, R
        nstates * float_size +  # Q
        nstates * nhorizon * float_size * 4 +  # Xref, x_min, x_max
        ninputs * (nhorizon-1) * float_size * 3 +  # Uref, u_min, u_max
        nstates * nstates * float_size  # Pinf
    )
    
    return memory_bytes / 1024  # Convert to KB

def analyze_scaling(all_data):
    """Analyze scaling trends"""
    # Separate NSTATES and NINPUTS experiments
    nstates_data = {}
    ninputs_data = {}
    
    for data in all_data:
        if 'stats' not in data or not data['iterations']:
            continue
            
        if data['ninputs'] == 2:  # NSTATES scaling experiment
            if data['nstates'] not in nstates_data:
                nstates_data[data['nstates']] = []
            nstates_data[data['nstates']].append(data)
            
        if data['nstates'] == 10:  # NINPUTS scaling experiment
            if data['ninputs'] not in ninputs_data:
                ninputs_data[data['ninputs']] = []
            ninputs_data[data['ninputs']].append(data)
    
    return nstates_data, ninputs_data

def plot_scaling_results(nstates_data, ninputs_data, output_prefix="scaling_analysis"):
    """Generate scaling plots"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('TinyMPC Scaling Analysis', fontsize=16)
    
    # NSTATES scaling
    if nstates_data:
        nstates_vals = sorted(nstates_data.keys())
        memory_means = []
        memory_stds = []
        time_means = []
        time_stds = []
        
        for ns in nstates_vals:
            # Calculate memory
            mem_kb = estimate_memory(ns, 2)
            memory_means.append(mem_kb)
            memory_stds.append(0)  # No variation in memory calculation
            
            # Calculate timing stats
            times = []
            for run in nstates_data[ns]:
                times.extend(run['times_us'])
            time_means.append(np.mean(times) if times else 0)
            time_stds.append(np.std(times) if times else 0)
        
        # Plot NSTATES vs Memory
        ax = axes[0, 0]
        ax.errorbar(nstates_vals, memory_means, yerr=memory_stds, 
                   marker='o', markersize=8, capsize=5)
        ax.set_xlabel('NSTATES')
        ax.set_ylabel('Memory Usage (KB)')
        ax.set_title('Memory Usage vs Problem Size (NINPUTS=2)')
        ax.grid(True, alpha=0.3)
        
        # Plot NSTATES vs Time
        ax = axes[0, 1]
        ax.errorbar(nstates_vals, time_means, yerr=time_stds,
                   marker='o', markersize=8, capsize=5, color='orange')
        ax.set_xlabel('NSTATES')
        ax.set_ylabel('Time per Iteration (μs)')
        ax.set_title('Computation Time vs Problem Size (NINPUTS=2)')
        ax.grid(True, alpha=0.3)
    
    # NINPUTS scaling
    if ninputs_data:
        ninputs_vals = sorted(ninputs_data.keys())
        memory_means = []
        memory_stds = []
        time_means = []
        time_stds = []
        
        for nu in ninputs_vals:
            # Calculate memory
            mem_kb = estimate_memory(10, nu)
            memory_means.append(mem_kb)
            memory_stds.append(0)
            
            # Calculate timing stats
            times = []
            for run in ninputs_data[nu]:
                times.extend(run['times_us'])
            time_means.append(np.mean(times) if times else 0)
            time_stds.append(np.std(times) if times else 0)
        
        # Plot NINPUTS vs Memory
        ax = axes[1, 0]
        ax.errorbar(ninputs_vals, memory_means, yerr=memory_stds,
                   marker='s', markersize=8, capsize=5, color='green')
        ax.set_xlabel('NINPUTS')
        ax.set_ylabel('Memory Usage (KB)')
        ax.set_title('Memory Usage vs Input Dimension (NSTATES=10)')
        ax.grid(True, alpha=0.3)
        
        # Plot NINPUTS vs Time
        ax = axes[1, 1]
        ax.errorbar(ninputs_vals, time_means, yerr=time_stds,
                   marker='s', markersize=8, capsize=5, color='red')
        ax.set_xlabel('NINPUTS')
        ax.set_ylabel('Time per Iteration (μs)')
        ax.set_title('Computation Time vs Input Dimension (NSTATES=10)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = f"{output_prefix}_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {output_file}")
    
    # Also save a summary CSV
    summary_file = f"analysis_summary_{timestamp}.csv"
    with open(summary_file, 'w') as f:
        f.write("Experiment,Parameter,Value,Memory_KB,Avg_Time_us,Std_Time_us,Samples\n")
        
        for ns in sorted(nstates_data.keys()):
            mem_kb = estimate_memory(ns, 2)
            times = []
            for run in nstates_data[ns]:
                times.extend(run['times_us'])
            avg_time = np.mean(times) if times else 0
            std_time = np.std(times) if times else 0
            samples = len(times)
            f.write(f"NSTATES,{ns},2,{mem_kb:.2f},{avg_time:.2f},{std_time:.2f},{samples}\n")
            
        for nu in sorted(ninputs_data.keys()):
            mem_kb = estimate_memory(10, nu)
            times = []
            for run in ninputs_data[nu]:
                times.extend(run['times_us'])
            avg_time = np.mean(times) if times else 0
            std_time = np.std(times) if times else 0
            samples = len(times)
            f.write(f"NINPUTS,10,{nu},{mem_kb:.2f},{avg_time:.2f},{std_time:.2f},{samples}\n")
    
    print(f"Summary saved to: {summary_file}")
    
    return output_file, summary_file

def main():
    parser = argparse.ArgumentParser(description='Analyze TinyMPC scaling data')
    parser.add_argument('--data-dir', default='benchmark_data', help='Directory with benchmark data')
    parser.add_argument('--output-prefix', default='scaling_analysis', help='Output file prefix')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading benchmark data...")
    all_data = load_benchmark_data(args.data_dir)
    print(f"Found {len(all_data)} data files")
    
    if not all_data:
        print("No data files found!")
        return
    
    # Analyze scaling
    nstates_data, ninputs_data = analyze_scaling(all_data)
    
    print(f"\nNSTATES experiments: {list(nstates_data.keys())}")
    print(f"NINPUTS experiments: {list(ninputs_data.keys())}")
    
    # Generate plots
    if nstates_data or ninputs_data:
        plot_scaling_results(nstates_data, ninputs_data, args.output_prefix)
    else:
        print("\nNo valid data found for plotting!")
        print("Make sure data files contain actual measurements.")

if __name__ == "__main__":
    main()
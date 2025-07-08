#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_benchmark_file(filepath):
    """Parse a benchmark file and extract iterations and time data."""
    iterations = []
    times = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header lines and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Parse lines with format: [timestamp] iterations time_microseconds
            match = re.match(r'\[.*?\]\s+(\d+)\s+(\d+)', line)
            if match:
                iter_count = int(match.group(1))
                time_us = int(match.group(2))
                iterations.append(iter_count)
                times.append(time_us)
    
    return iterations, times

def extract_states_from_filename(filename):
    """Extract number of states from filename pattern benchmark_N{states}_I{inputs}_H{horizon}.txt"""
    match = re.search(r'_N(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def collect_benchmark_data(data_dir):
    """Collect and organize benchmark data from all files."""
    tinympc_data = {}
    osqp_data = {}
    
    data_path = Path(data_dir)
    
    # Process TinyMPC files
    for file_path in data_path.glob('benchmark_N*.txt'):
        states = extract_states_from_filename(file_path.name)
        if states is not None:
            iterations, times = parse_benchmark_file(file_path)
            if iterations and times:
                avg_time = np.mean(times)
                tinympc_data[states] = avg_time
                print(f"TinyMPC N={states}: {len(times)} samples, avg={avg_time:.1f}μs")
    
    # Process OSQP files
    for file_path in data_path.glob('osqp_benchmark_N*.txt'):
        states = extract_states_from_filename(file_path.name)
        if states is not None:
            iterations, times = parse_benchmark_file(file_path)
            if iterations and times:
                avg_time = np.mean(times)
                osqp_data[states] = avg_time
                print(f"OSQP N={states}: {len(times)} samples, avg={avg_time:.1f}μs")
    
    return tinympc_data, osqp_data

def plot_comparison(tinympc_data, osqp_data):
    """Create comparison plot of TinyMPC vs OSQP performance."""
    # Get all state sizes from both datasets
    all_tinympc_states = sorted(tinympc_data.keys())
    all_osqp_states = sorted(osqp_data.keys())
    common_states = sorted(set(tinympc_data.keys()) & set(osqp_data.keys()))
    
    if not tinympc_data and not osqp_data:
        print("No data found for either TinyMPC or OSQP")
        return
    
    # Prepare data for plotting
    tinympc_times = [tinympc_data[n] for n in all_tinympc_states]
    osqp_times = [osqp_data[n] for n in all_osqp_states]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot both datasets
    if tinympc_data:
        plt.plot(all_tinympc_states, tinympc_times, 'o-', label='TinyMPC', linewidth=2, markersize=8)
    if osqp_data:
        plt.plot(all_osqp_states, osqp_times, 's-', label='OSQP', linewidth=2, markersize=8)
    
    # Customize the plot
    plt.xlabel('Number of States', fontsize=12)
    plt.ylabel('Average Iteration Time (μs)', fontsize=12)
    plt.title('TinyMPC vs OSQP Performance Comparison', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show all state values
    all_states = sorted(set(all_tinympc_states + all_osqp_states))
    plt.xticks(all_states)
    
    # Use log scale for y-axis
    plt.yscale('log')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('/home/ishaan/benchmark-tinyMPC/TRO/benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print performance summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"{'States':<8} {'TinyMPC (μs)':<12} {'OSQP (μs)':<12} {'Speedup':<10}")
    print("-"*50)
    for n in common_states:
        speedup = osqp_times[common_states.index(n)] / tinympc_times[common_states.index(n)]
        print(f"{n:<8} {tinympc_data[n]:<12.1f} {osqp_data[n]:<12.1f} {speedup:<10.2f}x")

def main():
    """Main function to run the benchmark analysis."""
    data_dir = '/home/ishaan/benchmark-tinyMPC/TRO/data'
    
    print("Collecting benchmark data...")
    tinympc_data, osqp_data = collect_benchmark_data(data_dir)
    
    if not tinympc_data:
        print("No TinyMPC data found!")
        return
    
    if not osqp_data:
        print("No OSQP data found!")
        return
    
    print("\nCreating comparison plot...")
    plot_comparison(tinympc_data, osqp_data)
    
    print("\nAnalysis complete! Plot saved as 'benchmark_comparison.png'")

if __name__ == "__main__":
    main()
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

def extract_horizon_from_filename(filename):
    """Extract horizon length from filename pattern benchmark_H{horizon}.txt"""
    match = re.search(r'_H(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    return None

def parse_memory_data():
    """Parse memory usage data for different horizon lengths."""
    # STM32 RAM limit in bytes (from the Arduino output: Maximum is 131072 bytes)
    RAM_LIMIT = 131072  # 128KB
    
    # OSQP memory data (in bytes) - optimized benchmark results
    osqp_memory = {
        4: 21348,   # Global variables use 21348 bytes
        8: 38228,   # Global variables use 38228 bytes
        16: 71988,  # Global variables use 71988 bytes
        32: RAM_LIMIT + 9976,  # RAM overflowed by 9976 bytes
        64: RAM_LIMIT + 145016, # RAM overflowed by 145016 bytes
        100: RAM_LIMIT + 296936, # RAM overflowed by 296936 bytes
    }
    
    # TinyMPC memory data (in bytes) - optimized benchmark results  
    tinympc_memory = {
        4: 9004,    # Global variables use 9004 bytes
        8: 11172,   # Global variables use 11172 bytes
        16: 15488,  # Global variables use 15488 bytes
        32: 24128,  # Global variables use 24128 bytes
        64: 41408,  # Global variables use 41408 bytes
        100: 60848, # Global variables use 60848 bytes
    }
    
    return tinympc_memory, osqp_memory, RAM_LIMIT

def collect_benchmark_data(data_dir):
    """Collect and organize benchmark data from all files."""
    tinympc_data = {}
    osqp_data = {}
    
    data_path = Path(data_dir)
    
    # Process TinyMPC files
    for file_path in data_path.glob('benchmark_H*.txt'):
        horizon = extract_horizon_from_filename(file_path.name)
        if horizon is not None:
            iterations, times = parse_benchmark_file(file_path)
            if iterations and times:
                avg_time = np.mean(times)
                tinympc_data[horizon] = avg_time
                print(f"TinyMPC H={horizon}: {len(times)} samples, avg={avg_time:.1f}μs")
    
    # Process OSQP files
    for file_path in data_path.glob('osqp_benchmark_H*.txt'):
        horizon = extract_horizon_from_filename(file_path.name)
        if horizon is not None:
            iterations, times = parse_benchmark_file(file_path)
            if iterations and times:
                avg_time = np.mean(times)
                osqp_data[horizon] = avg_time
                print(f"OSQP H={horizon}: {len(times)} samples, avg={avg_time:.1f}μs")
    
    return tinympc_data, osqp_data

def plot_comparison(tinympc_data, osqp_data):
    """Create comparison plot of TinyMPC vs OSQP performance and memory usage."""
    # Get memory data
    tinympc_memory, osqp_memory, RAM_LIMIT = parse_memory_data()
    
    # Get all horizon lengths from both datasets
    all_tinympc_horizons = sorted(tinympc_data.keys())
    all_osqp_horizons = sorted(osqp_data.keys())
    common_horizons = sorted(set(tinympc_data.keys()) & set(osqp_data.keys()))
    
    if not tinympc_data and not osqp_data:
        print("No data found for either TinyMPC or OSQP")
        return
    
    # Prepare data for plotting
    tinympc_times = [tinympc_data[h] for h in all_tinympc_horizons]
    osqp_times = [osqp_data[h] for h in all_osqp_horizons]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Time comparison
    if tinympc_data:
        ax1.plot(all_tinympc_horizons, tinympc_times, 'o-', label='TinyMPC (Optimized)', linewidth=2, markersize=8)
    if osqp_data:
        ax1.plot(all_osqp_horizons, osqp_times, 's-', label='OSQP (Optimized)', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Horizon Length', fontsize=12)
    ax1.set_ylabel('Average Iteration Time (μs)', fontsize=12)
    ax1.set_title('Optimized Performance Comparison by Horizon Length', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Set x-axis to show all horizon values
    all_horizons = sorted(set(all_tinympc_horizons + all_osqp_horizons))
    ax1.set_xticks(all_horizons)
    
    # Plot 2: Memory comparison
    tinympc_mem_horizons = sorted(tinympc_memory.keys())
    osqp_mem_horizons = sorted(osqp_memory.keys())
    
    # Separate OSQP data into successful compilations and overflows
    osqp_successful_horizons = [h for h in osqp_mem_horizons if osqp_memory[h] <= RAM_LIMIT]
    osqp_overflow_horizons = [h for h in osqp_mem_horizons if osqp_memory[h] > RAM_LIMIT]
    
    tinympc_mem_values = [tinympc_memory[h] / 1024 for h in tinympc_mem_horizons]  # Convert to KB
    osqp_successful_values = [osqp_memory[h] / 1024 for h in osqp_successful_horizons]
    osqp_overflow_values = [osqp_memory[h] / 1024 for h in osqp_overflow_horizons]
    
    # Plot TinyMPC
    ax2.plot(tinympc_mem_horizons, tinympc_mem_values, 'o-', label='TinyMPC (Optimized)', linewidth=2, markersize=8, color='blue')
    
    # Plot OSQP successful compilations
    if osqp_successful_horizons:
        ax2.plot(osqp_successful_horizons, osqp_successful_values, 's-', label='OSQP (Optimized)', linewidth=2, markersize=8, color='green')
    
    # Plot OSQP overflow cases
    if osqp_overflow_horizons:
        ax2.plot(osqp_overflow_horizons, osqp_overflow_values, 'X-', label='OSQP (Overflow)', linewidth=2, markersize=10, color='red')
    
    # Add horizontal line for RAM limit
    ax2.axhline(y=RAM_LIMIT / 1024, color='red', linestyle='--', linewidth=2, alpha=0.7, label='STM32 RAM Limit (128KB)')
    
    ax2.set_xlabel('Horizon Length', fontsize=12)
    ax2.set_ylabel('Global Variable Memory (KB)', fontsize=12)
    ax2.set_title('Optimized Memory Usage Comparison by Horizon Length', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis to show all horizon values for memory plot
    all_mem_horizons = sorted(set(tinympc_mem_horizons + osqp_mem_horizons))
    ax2.set_xticks(all_mem_horizons)
    
    plt.tight_layout()
    
    # Save the plot
    #plt.savefig('/home/ishaan/benchmark-tinyMPC/TRO/horizon_benchmark_comparison_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print performance summary for compiled configurations
    print("\n" + "="*80)
    print("OPTIMIZED HORIZON PERFORMANCE SUMMARY (Successfully Compiled Configurations)")
    print("="*80)
    print(f"{'Horizon':<8} {'TinyMPC (μs)':<12} {'OSQP (μs)':<12} {'Speedup':<10} {'TinyMPC (KB)':<12} {'OSQP (KB)':<12}")
    print("-"*80)
    for h in common_horizons:
        if h in osqp_successful_horizons:  # Only show successfully compiled OSQP configs
            speedup = osqp_data[h] / tinympc_data[h]
            tinympc_mem_kb = tinympc_memory.get(h, 0) / 1024
            osqp_mem_kb = osqp_memory.get(h, 0) / 1024
            print(f"{h:<8} {tinympc_data[h]:<12.1f} {osqp_data[h]:<12.1f} {speedup:<10.2f}x {tinympc_mem_kb:<12.1f} {osqp_mem_kb:<12.1f}")
    
    # Print memory summary for all TinyMPC configurations
    print("\nTinyMPC Optimized Memory Usage by Horizon:")
    print(f"{'Horizon':<8} {'Memory (KB)':<12} {'Status':<15}")
    print("-"*35)
    for h in sorted(tinympc_memory.keys()):
        mem_kb = tinympc_memory[h] / 1024
        status = "✓ Compiled" if mem_kb <= RAM_LIMIT / 1024 else "✗ Would Overflow"
        print(f"{h:<8} {mem_kb:<12.1f} {status:<15}")
    
    # Print OSQP overflow summary
    print("\nOSQP Optimized Memory Summary by Horizon:")
    print(f"{'Horizon':<8} {'Est. Memory (KB)':<15} {'Overflow (KB)':<12} {'Status':<15}")
    print("-"*50)
    for h in sorted(osqp_memory.keys()):
        mem_kb = osqp_memory[h] / 1024
        if h in osqp_overflow_horizons:
            overflow_kb = mem_kb - (RAM_LIMIT / 1024)
            status = "✗ Overflow"
            print(f"{h:<8} {mem_kb:<15.1f} {overflow_kb:<12.1f} {status:<15}")
        else:
            status = "✓ Compiled"
            print(f"{h:<8} {mem_kb:<15.1f} {'0.0':<12} {status:<15}")

    # Calculate and display key aggregate metrics
    # ------------------------------------------------------------------
    # Maximum iteration speed-up (OSQP time / TinyMPC time) where OSQP compiled successfully
    speedup_values = [(h, osqp_data[h] / tinympc_data[h])
                      for h in common_horizons if h in osqp_successful_horizons]
    if speedup_values:
        max_speed_horizon, max_speedup = max(speedup_values, key=lambda x: x[1])
        print(f"\nMaximum iteration speed-up: TinyMPC is {max_speedup:.2f}× faster than OSQP (H={max_speed_horizon}).")

    # Maximum memory usage ratio (OSQP RAM / TinyMPC RAM)
    memory_ratios = [(h, osqp_memory[h] / tinympc_memory[h])
                     for h in tinympc_memory if h in osqp_memory]
    if memory_ratios:
        max_mem_horizon, max_mem_ratio = max(memory_ratios, key=lambda x: x[1])
        print(f"Maximum RAM usage ratio: OSQP uses {max_mem_ratio:.2f}× more global RAM than TinyMPC (H={max_mem_horizon}).")
    print("-"*80)

def main():
    """Main function to run the benchmark analysis."""
    #data_dir = '/home/ishaan/benchmark-tinyMPC/TRO/data_optimized/horizon'
    data_dir = 'data_optimized/horizon'
    
    print("Collecting optimized horizon benchmark data...")
    tinympc_data, osqp_data = collect_benchmark_data(data_dir)
    
    if not tinympc_data:
        print("No TinyMPC data found!")
        return
    
    if not osqp_data:
        print("No OSQP data found!")
        return
    
    print("\nCreating optimized horizon comparison plot...")
    plot_comparison(tinympc_data, osqp_data)
    
    print("\nOptimized horizon analysis complete!")

if __name__ == "__main__":
    main()
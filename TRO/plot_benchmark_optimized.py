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

def parse_memory_data():
    """Parse memory usage data from the provided text."""
    # STM32 RAM limit in bytes (from the Arduino output: Maximum is 131072 bytes)
    RAM_LIMIT = 131072  # 128KB
    
    # OSQP memory data (in bytes) - optimized benchmark results
    osqp_memory = {
        2: 14444,   # Global variables use 14444 bytes
        4: 23580,   # Global variables use 23580 bytes  
        8: 38508,   # Global variables use 38508 bytes
        16: 71148,  # Global variables use 71148 bytes
        32: RAM_LIMIT + 6896,  # RAM overflow by 6896 bytes
    }
    
    # TinyMPC memory data (in bytes) - optimized benchmark results  
    tinympc_memory = {
        2: 6316,    # Global variables use 6316 bytes
        4: 7564,    # Global variables use 7564 bytes
        8: 10532,   # Global variables use 10532 bytes
        16: 18160,  # Global variables use 18160 bytes
        32: 39920,  # Global variables use 39920 bytes
    }
    
    return tinympc_memory, osqp_memory, RAM_LIMIT

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
    """Create comparison plot of TinyMPC vs OSQP performance and memory usage."""
    # Get memory data
    tinympc_memory, osqp_memory, RAM_LIMIT = parse_memory_data()
    
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
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Time comparison
    if tinympc_data:
        ax1.plot(all_tinympc_states, tinympc_times, 'o-', label='TinyMPC (Optimized)', linewidth=2, markersize=8)
    if osqp_data:
        ax1.plot(all_osqp_states, osqp_times, 's-', label='OSQP (Optimized)', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of States', fontsize=12)
    ax1.set_ylabel('Average Iteration Time (μs)', fontsize=12)
    ax1.set_title('Optimized Performance Comparison by Number of States', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Set x-axis to show all state values
    all_states = sorted(set(all_tinympc_states + all_osqp_states))
    ax1.set_xticks(all_states)
    
    # Plot 2: Memory comparison
    tinympc_mem_states = sorted(tinympc_memory.keys())
    osqp_mem_states = sorted(osqp_memory.keys())
    
    # Separate OSQP data into successful compilations and overflows
    osqp_successful_states = [s for s in osqp_mem_states if osqp_memory[s] <= RAM_LIMIT]
    osqp_overflow_states = [s for s in osqp_mem_states if osqp_memory[s] > RAM_LIMIT]
    
    tinympc_mem_values = [tinympc_memory[s] / 1024 for s in tinympc_mem_states]  # Convert to KB
    osqp_successful_values = [osqp_memory[s] / 1024 for s in osqp_successful_states]
    osqp_overflow_values = [osqp_memory[s] / 1024 for s in osqp_overflow_states]
    
    # Plot TinyMPC
    ax2.plot(tinympc_mem_states, tinympc_mem_values, 'o-', label='TinyMPC (Optimized)', linewidth=2, markersize=8, color='blue')
    
    # Plot OSQP successful compilations
    if osqp_successful_states:
        ax2.plot(osqp_successful_states, osqp_successful_values, 's-', label='OSQP (Optimized)', linewidth=2, markersize=8, color='green')
    
    # Plot OSQP overflow cases
    if osqp_overflow_states:
        ax2.plot(osqp_overflow_states, osqp_overflow_values, 'X-', label='OSQP (Overflow)', linewidth=2, markersize=10, color='red')
    
    # Add horizontal line for RAM limit
    ax2.axhline(y=RAM_LIMIT / 1024, color='red', linestyle='--', linewidth=2, alpha=0.7, label='STM32 RAM Limit (128KB)')
    
    ax2.set_xlabel('Number of States', fontsize=12)
    ax2.set_ylabel('Global Variable Memory (KB)', fontsize=12)
    ax2.set_title('Optimized Memory Usage Comparison by Number of States', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Set x-axis to show all state values for memory plot
    all_mem_states = sorted(set(tinympc_mem_states + osqp_mem_states))
    ax2.set_xticks(all_mem_states)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('/home/ishaan/benchmark-tinyMPC/TRO/benchmark_comparison_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print performance summary for compiled configurations
    print("\n" + "="*80)
    print("OPTIMIZED PERFORMANCE SUMMARY (Successfully Compiled Configurations)")
    print("="*80)
    print(f"{'States':<8} {'TinyMPC (μs)':<12} {'OSQP (μs)':<12} {'Speedup':<10} {'TinyMPC (KB)':<12} {'OSQP (KB)':<12}")
    print("-"*80)
    for n in common_states:
        if n in osqp_successful_states:  # Only show successfully compiled OSQP configs
            speedup = osqp_data[n] / tinympc_data[n]
            tinympc_mem_kb = tinympc_memory.get(n, 0) / 1024
            osqp_mem_kb = osqp_memory.get(n, 0) / 1024
            print(f"{n:<8} {tinympc_data[n]:<12.1f} {osqp_data[n]:<12.1f} {speedup:<10.2f}x {tinympc_mem_kb:<12.1f} {osqp_mem_kb:<12.1f}")
    
    # Print memory summary for all TinyMPC configurations
    print("\nTinyMPC Optimized Memory Usage:")
    print(f"{'States':<8} {'Memory (KB)':<12} {'Status':<15}")
    print("-"*35)
    for s in sorted(tinympc_memory.keys()):
        mem_kb = tinympc_memory[s] / 1024
        status = "✓ Compiled" if mem_kb <= RAM_LIMIT / 1024 else "✗ Would Overflow"
        print(f"{s:<8} {mem_kb:<12.1f} {status:<15}")
    
    # Print OSQP overflow summary
    print("\nOSQP Optimized Memory Summary:")
    print(f"{'States':<8} {'Est. Memory (KB)':<15} {'Overflow (KB)':<12} {'Status':<15}")
    print("-"*50)
    for s in sorted(osqp_memory.keys()):
        mem_kb = osqp_memory[s] / 1024
        if s in osqp_overflow_states:
            overflow_kb = mem_kb - (RAM_LIMIT / 1024)
            status = "✗ Overflow"
            print(f"{s:<8} {mem_kb:<15.1f} {overflow_kb:<12.1f} {status:<15}")
        else:
            status = "✓ Compiled"
            print(f"{s:<8} {mem_kb:<15.1f} {'0.0':<12} {status:<15}")

def main():
    """Main function to run the benchmark analysis."""
    data_dir = '/home/ishaan/benchmark-tinyMPC/TRO/data_optimized/states'
    
    print("Collecting optimized benchmark data...")
    tinympc_data, osqp_data = collect_benchmark_data(data_dir)
    
    if not tinympc_data:
        print("No TinyMPC data found!")
        return
    
    if not osqp_data:
        print("No OSQP data found!")
        return
    
    print("\nCreating optimized comparison plot...")
    plot_comparison(tinympc_data, osqp_data)
    
    print("\nOptimized analysis complete!")

if __name__ == "__main__":
    main()

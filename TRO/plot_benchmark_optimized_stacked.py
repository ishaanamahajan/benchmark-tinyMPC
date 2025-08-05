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
    times_per_iter = []
    
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
                times_per_iter.append(time_us / iter_count)  # Calculate time per iteration
    
    return iterations, times, times_per_iter

def extract_states_from_filename(filename):
    """Extract number of states from filename pattern benchmark_N{states}_I{inputs}_H{horizon}.txt"""
    match = re.search(r'_N(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def extract_horizon_from_filename(filename):
    """Extract horizon length from filename pattern benchmark_H{horizon}.txt"""
    match = re.search(r'_H(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    return None

def parse_states_memory_data():
    """Parse memory usage data for different state dimensions."""
    # STM32 RAM limit in bytes
    RAM_LIMIT = 131072  # 128KB
    
    # OSQP memory data (in bytes) - optimized benchmark results
    osqp_memory = {
        2: 14444,   # Global variables use 14444 bytes
        4: 23580,   # Global variables use 23580 bytes  
        8: 38508,   # Global variables use 38508 bytes
        12: 55000,  # Estimated Global variables use ~55000 bytes
        16: 71148,  # Global variables use 71148 bytes
        24: 105000, # Estimated Global variables use ~105000 bytes
        28: 120000, # Estimated Global variables use ~120000 bytes
        32: RAM_LIMIT + 6896,  # RAM overflow by 6896 bytes
    }
    
    # TinyMPC memory data (in bytes) - optimized benchmark results  
    tinympc_memory = {
        2: 6316,    # Global variables use 6316 bytes
        4: 7564,    # Global variables use 7564 bytes
        8: 10532,   # Global variables use 10532 bytes
        12: 13500,  # Estimated Global variables use ~13500 bytes
        16: 18160,  # Global variables use 18160 bytes
        24: 28000,  # Estimated Global variables use ~28000 bytes
        28: 33000,  # Estimated Global variables use ~33000 bytes
        32: 39920,  # Global variables use 39920 bytes
    }
    
    return tinympc_memory, osqp_memory, RAM_LIMIT

def parse_horizon_memory_data():
    """Parse memory usage data for different horizon lengths."""
    # STM32 RAM limit in bytes
    RAM_LIMIT = 131072  # 128KB
    
    # OSQP memory data (in bytes) - optimized benchmark results
    # Stop plotting OSQP memory after first failure at horizon 32
    osqp_memory = {
        4: 21348,   # Global variables use 21348 bytes
        8: 38228,   # Global variables use 38228 bytes
        16: 71988,  # Global variables use 71988 bytes
        32: RAM_LIMIT + 9976,   # First RAM overflow - stop plotting OSQP after this
    }
    
    # TinyMPC memory data (in bytes) - optimized benchmark results  
    tinympc_memory = {
        4: 9004,    # Global variables use 9004 bytes
        8: 11172,   # Global variables use 11172 bytes
        16: 15488,  # Global variables use 15488 bytes
        32: 24128,  # Global variables use 24128 bytes
        40: 30000,  # Estimated Global variables use ~30000 bytes
        50: 37000,  # Estimated Global variables use ~37000 bytes
        64: 41408,  # Global variables use 41408 bytes
        75: 50000,  # Estimated Global variables use ~50000 bytes
        100: 60848, # Global variables use 60848 bytes
    }
    
    return tinympc_memory, osqp_memory, RAM_LIMIT

def collect_states_benchmark_data(data_dir):
    """Collect and organize benchmark data from state dimension files."""
    tinympc_data = {}
    osqp_data = {}
    
    data_path = Path(data_dir)
    
    # Process TinyMPC files
    for file_path in data_path.glob('benchmark_N*.txt'):
        states = extract_states_from_filename(file_path.name)
        if states is not None:
            iterations, times, times_per_iter = parse_benchmark_file(file_path)
            if iterations and times:
                times_per_iter_array = np.array(times_per_iter)
                avg_time = np.mean(times_per_iter_array)
                min_time = np.min(times_per_iter_array)
                max_time = np.max(times_per_iter_array)
                tinympc_data[states] = {
                    'mean': avg_time,
                    'min': min_time,
                    'max': max_time,
                    'std': np.std(times_per_iter_array),
                    'samples': len(times)
                }
                print(f"States TinyMPC N={states}: {len(times)} samples, avg={avg_time:.1f}μs per iteration")
    
    # Process OSQP files
    for file_path in data_path.glob('osqp_benchmark_N*.txt'):
        states = extract_states_from_filename(file_path.name)
        if states is not None:
            iterations, times, times_per_iter = parse_benchmark_file(file_path)
            if iterations and times:
                times_per_iter_array = np.array(times_per_iter)
                avg_time = np.mean(times_per_iter_array)
                min_time = np.min(times_per_iter_array)
                max_time = np.max(times_per_iter_array)
                osqp_data[states] = {
                    'mean': avg_time,
                    'min': min_time,
                    'max': max_time,
                    'std': np.std(times_per_iter_array),
                    'samples': len(times)
                }
                print(f"States OSQP N={states}: {len(times)} samples, avg={avg_time:.1f}μs per iteration")
    
    return tinympc_data, osqp_data

def collect_horizon_benchmark_data(data_dir):
    """Collect and organize benchmark data from horizon files."""
    tinympc_data = {}
    osqp_data = {}
    
    data_path = Path(data_dir)
    
    # Process TinyMPC files
    for file_path in data_path.glob('benchmark_H*.txt'):
        horizon = extract_horizon_from_filename(file_path.name)
        if horizon is not None:
            iterations, times, times_per_iter = parse_benchmark_file(file_path)
            if iterations and times:
                times_per_iter_array = np.array(times_per_iter)
                avg_time = np.mean(times_per_iter_array)
                min_time = np.min(times_per_iter_array)
                max_time = np.max(times_per_iter_array)
                tinympc_data[horizon] = {
                    'mean': avg_time,
                    'min': min_time,
                    'max': max_time,
                    'std': np.std(times_per_iter_array),
                    'samples': len(times)
                }
                print(f"Horizon TinyMPC H={horizon}: {len(times)} samples, avg={avg_time:.1f}μs per iteration")
    
    # Process OSQP files
    for file_path in data_path.glob('osqp_benchmark_H*.txt'):
        horizon = extract_horizon_from_filename(file_path.name)
        if horizon is not None:
            iterations, times, times_per_iter = parse_benchmark_file(file_path)
            if iterations and times:
                times_per_iter_array = np.array(times_per_iter)
                avg_time = np.mean(times_per_iter_array)
                min_time = np.min(times_per_iter_array)
                max_time = np.max(times_per_iter_array)
                osqp_data[horizon] = {
                    'mean': avg_time,
                    'min': min_time,
                    'max': max_time,
                    'std': np.std(times_per_iter_array),
                    'samples': len(times)
                }
                print(f"Horizon OSQP H={horizon}: {len(times)} samples, avg={avg_time:.1f}μs per iteration")
    
    return tinympc_data, osqp_data



def plot_benchmark_comparison_bars(states_data, horizon_data):
    """Create a 2x2 benchmark comparison plot with bar charts for memory (fig_draft2)."""
    # Colors matching TikZ style: mycolor1=RGB(0,0,0.6), mycolor2=RGB(1,0,0), mycolor3=RGB(0.46667,0.67451,0.18824)
    TINYMPC_COLOR = (0, 0, 0.6)     # mycolor1 - Dark blue for TinyMPC
    OSQP_COLOR = (1.0, 0.0, 0.0)    # mycolor2 - Red for OSQP
    RAM_LIMIT_COLOR = 'black'  # Black for limit line
    
    # Define offset for x-values to prevent marker overlap
    STATES_OFFSET = 0.15  # Smaller offset for states (smaller numbers)
    HORIZON_OFFSET = 0.2  # Larger offset for horizon (larger numbers)
    SPECIAL_OFFSET = 1.0  # Special offset for N=16
    
    # Unpack data
    states_tinympc_timing, states_osqp_timing = states_data['timing']
    horizon_tinympc_timing, horizon_osqp_timing = horizon_data['timing']
    states_tinympc_memory, states_osqp_memory, states_ram_limit = states_data['memory']
    horizon_tinympc_memory, horizon_osqp_memory, horizon_ram_limit = horizon_data['memory']
    
    # Set high quality plotting parameters
    plt.rcParams.update({
        'font.size': 14,
        'axes.linewidth': 1.5,
        'grid.alpha': 0.3,
        'grid.linewidth': 1.0,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'legend.fontsize': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    # Create 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10), layout='constrained')
    
    # Top left: States Time (mean values only)
    states_timing_x = sorted(states_tinympc_timing.keys())
    states_tinympc_timing_y = [states_tinympc_timing[s]['mean'] for s in states_timing_x]
    states_tinympc_timing_min = [states_tinympc_timing[s]['min'] for s in states_timing_x]
    states_tinympc_timing_max = [states_tinympc_timing[s]['max'] for s in states_timing_x]

    states_osqp_timing_x = sorted(states_osqp_timing.keys())
    states_osqp_timing_y = [states_osqp_timing[s]['mean'] for s in states_osqp_timing_x]
    states_osqp_timing_min = [states_osqp_timing[s]['min'] for s in states_osqp_timing_x]
    states_osqp_timing_max = [states_osqp_timing[s]['max'] for s in states_osqp_timing_x]
    
    # Use uniform x positions instead of actual values for even spacing - no offsets
    states_x_positions = list(range(len(states_timing_x)))
    states_osqp_x_positions = [states_timing_x.index(x) for x in states_osqp_timing_x if x in states_timing_x]
    
    # Calculate asymmetric error bars from mean to min/max
    states_tinympc_yerr = np.array([
        [y - ymin for y, ymin in zip(states_tinympc_timing_y, states_tinympc_timing_min)],
        [ymax - y for y, ymax in zip(states_tinympc_timing_y, states_tinympc_timing_max)]
    ])
    states_osqp_yerr = np.array([
        [y - ymin for y, ymin in zip(states_osqp_timing_y, states_osqp_timing_min)],
        [ymax - y for y, ymax in zip(states_osqp_timing_y, states_osqp_timing_max)]
    ])
    
    ax1.errorbar(
        states_x_positions, states_tinympc_timing_y, yerr=states_tinympc_yerr,
        fmt='D', color=TINYMPC_COLOR, markersize=10,
        capsize=15, capthick=1, elinewidth=1, linewidth=0, markeredgecolor='black'
    )
    ax1.errorbar(
        states_osqp_x_positions, states_osqp_timing_y, yerr=states_osqp_yerr,
        fmt='o', color=OSQP_COLOR, markersize=10,
        capsize=8, capthick=2, elinewidth=2, linewidth=0, markeredgecolor='black'
    )
    
    ax1.set_xlabel('State dimension (n)', fontweight = 'bold')
    ax1.set_ylabel('Time per Iteration (μs)', fontweight = 'bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(states_x_positions)
    ax1.set_xticklabels(states_timing_x)
    # Add solver legend to top left plot
    handles = [
        plt.Rectangle((0,0),1,1, color=TINYMPC_COLOR, label='TinyMPC'),
        plt.Rectangle((0,0),1,1, color=OSQP_COLOR, label='OSQP')
    ]
    ax1.legend(handles=handles, loc='upper left', fontsize=14)
    
    # Top right: Horizon Time (mean values only) - exclude horizon 30
    horizon_timing_x = sorted([h for h in horizon_tinympc_timing.keys() if h != 30])
    horizon_tinympc_timing_y = [horizon_tinympc_timing[h]['mean'] for h in horizon_timing_x]
    horizon_tinympc_timing_min = [horizon_tinympc_timing[h]['min'] for h in horizon_timing_x]
    horizon_tinympc_timing_max = [horizon_tinympc_timing[h]['max'] for h in horizon_timing_x]

    horizon_osqp_timing_x = sorted([h for h in horizon_osqp_timing.keys() if h != 30])
    horizon_osqp_timing_y = [horizon_osqp_timing[h]['mean'] for h in horizon_osqp_timing_x]
    horizon_osqp_timing_min = [horizon_osqp_timing[h]['min'] for h in horizon_osqp_timing_x]
    horizon_osqp_timing_max = [horizon_osqp_timing[h]['max'] for h in horizon_osqp_timing_x]
    
    # Use uniform x positions instead of actual values for even spacing - no offsets
    horizon_x_positions = list(range(len(horizon_timing_x)))
    horizon_osqp_x_positions = [horizon_timing_x.index(x) for x in horizon_osqp_timing_x if x in horizon_timing_x]
    
    # Calculate asymmetric error bars from mean to min/max
    horizon_tinympc_yerr = np.array([
        [y - ymin for y, ymin in zip(horizon_tinympc_timing_y, horizon_tinympc_timing_min)],
        [ymax - y for y, ymax in zip(horizon_tinympc_timing_y, horizon_tinympc_timing_max)]
    ])
    horizon_osqp_yerr = np.array([
        [y - ymin for y, ymin in zip(horizon_osqp_timing_y, horizon_osqp_timing_min)],
        [ymax - y for y, ymax in zip(horizon_osqp_timing_y, horizon_osqp_timing_max)]
    ])
    
    ax2.errorbar(
        horizon_x_positions, horizon_tinympc_timing_y, yerr=horizon_tinympc_yerr,
        fmt='D', color=TINYMPC_COLOR, markersize=10,
        capsize=15, capthick=1, elinewidth=1, linewidth=0, markeredgecolor='black'
    )
    ax2.errorbar(
        horizon_osqp_x_positions, horizon_osqp_timing_y, yerr=horizon_osqp_yerr,
        fmt='o', color=OSQP_COLOR, markersize=10,
        capsize=8, capthick=2, elinewidth=2, linewidth=0, markeredgecolor='black'
    )
    
    ax2.set_xlabel('Time horizon (N)', fontweight = 'bold')
    ax2.set_ylabel('Time per Iteration (μs)', fontweight = 'bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(horizon_x_positions)
    ax2.set_xticklabels(horizon_timing_x)
    
    # Bottom left: States Memory with bar charts (only plot up to 32 states)
    states_x = [s for s in sorted(states_tinympc_memory.keys()) if s <= 32]
    states_tinympc_mem_y = [states_tinympc_memory[s] / 1024 for s in states_x]
    states_osqp_mem_y = [states_osqp_memory.get(s, 0) / 1024 for s in states_x if s <= 32]
    
    # Create bar chart with same x positions as timing plot
    x_pos = np.arange(len(states_x))
    width = 0.3  # Increased width to match timing plot offset
    ax3.bar(x_pos - width/2, states_tinympc_mem_y, width, color=TINYMPC_COLOR, alpha=0.8, edgecolor='black')
    ax3.bar(x_pos + width/2, states_osqp_mem_y, width, color=OSQP_COLOR, alpha=0.8, edgecolor='black')
    
    ax3.axhline(y=states_ram_limit / 1024, color=RAM_LIMIT_COLOR, linestyle='--', 
                linewidth=2, alpha=0.8)
    ax3.set_xlabel('State dimension (n)', fontweight = 'bold')
    ax3.set_ylabel('Memory Usage (kB)', fontweight= 'bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(states_x)
    ax3.set_ylim(0, 160)  # Extend y-axis to give room for memory limit text
    ax3.grid(True, alpha=0.3)
    # Add memory limit text label to the leftmost memory plot - right above the line, extreme left
    ax3.text(0, (states_ram_limit / 1024) + 5, 'MEMORY LIMIT', 
             ha='left', va='bottom', fontweight='bold', fontsize=14, 
             color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Bottom right: Horizon Memory with bar charts - exclude horizon 30
    horizon_x = sorted([h for h in horizon_tinympc_memory.keys() if h != 30])
    horizon_tinympc_mem_y = [horizon_tinympc_memory[h] / 1024 for h in horizon_x]
    horizon_osqp_mem_y = [horizon_osqp_memory.get(h, 0) / 1024 for h in horizon_x]
    
    # Create bar chart with same x positions as timing plot
    x_pos_h = np.arange(len(horizon_x))
    ax4.bar(x_pos_h - width/2, horizon_tinympc_mem_y, width, color=TINYMPC_COLOR, alpha=0.8, edgecolor='black')
    ax4.bar(x_pos_h + width/2, horizon_osqp_mem_y, width, color=OSQP_COLOR, alpha=0.8, edgecolor='black')
    
    ax4.axhline(y=horizon_ram_limit / 1024, color=RAM_LIMIT_COLOR, linestyle='--', 
                linewidth=2, alpha=0.8)
    ax4.set_xlabel('Time horizon (N)', fontweight = 'bold')
    ax4.set_ylabel('Memory Usage (kB)', fontweight = 'bold')
    ax4.set_xticks(x_pos_h)
    ax4.set_xticklabels(horizon_x)
    ax4.set_ylim(0, 160)  # Extend y-axis to give room for memory limit text
    ax4.grid(True, alpha=0.3)
    
    # Legend removed from memory plot (moved to top left timing plot)
    
    # Memory limit text moved to leftmost plot (ax3)
    
    plt.savefig('fig_draft2.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def main():
    """Main function to run the benchmark analysis."""
    states_data_dir = 'data_optimized/states'
    horizon_data_dir = 'data_optimized/horizon'
    
    print("Collecting optimized states benchmark data...")
    states_tinympc_timing, states_osqp_timing = collect_states_benchmark_data(states_data_dir)
    
    print("Collecting optimized horizon benchmark data...")
    horizon_tinympc_timing, horizon_osqp_timing = collect_horizon_benchmark_data(horizon_data_dir)
    
    if not states_tinympc_timing or not horizon_tinympc_timing:
        print("Missing TinyMPC data!")
        return
    
    if not states_osqp_timing or not horizon_osqp_timing:
        print("Missing OSQP data!")
        return
    
    # Get memory data
    states_tinympc_memory, states_osqp_memory, states_ram_limit = parse_states_memory_data()
    horizon_tinympc_memory, horizon_osqp_memory, horizon_ram_limit = parse_horizon_memory_data()
    
    # Package data for plotting function
    states_data = {
        'timing': (states_tinympc_timing, states_osqp_timing),
        'memory': (states_tinympc_memory, states_osqp_memory, states_ram_limit)
    }
    
    horizon_data = {
        'timing': (horizon_tinympc_timing, horizon_osqp_timing),
        'memory': (horizon_tinympc_memory, horizon_osqp_memory, horizon_ram_limit)
    }
    
    print("\nCreating final benchmark comparison plot...")
    # plot_benchmark_comparison_lines(states_data, horizon_data)
    plot_benchmark_comparison_bars(states_data, horizon_data)
    
    print("\nBenchmark analysis complete! Saved as 'fig_draft1.png' and 'fig_draft2.pdf'")

if __name__ == "__main__":
    main() 
import matplotlib.pyplot as plt
import numpy as np
import re

def parse_scs_log(file_path):
    """Parse SCS benchmark log file"""
    horizon_data = {}  # horizon -> (static_mem, dynamic_mem, times_list)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    current_horizon = None
    
    for line in lines:
        line = line.strip()
        
        # Parse horizon and memory info
        if "Horizon:" in line:
            # Example: "Horizon: 2, 39.8 KB, Dynamic = 16.76 KB"
            # The second number is static/RAM, not total
            match = re.search(r'Horizon: (\d+), ([\d.]+) KB, Dynamic = ([\d.]+) KB', line)
            if match:
                current_horizon = int(match.group(1))
                static_kb = float(match.group(2))  # This is actually static, not total
                dynamic_kb = float(match.group(3))
                
                
                if current_horizon not in horizon_data:
                    horizon_data[current_horizon] = {'static': static_kb, 'dynamic': dynamic_kb, 'times': []}
        
        # Parse timing data - format is just "iterations total_time"
        elif current_horizon is not None and re.match(r'^\d+\s+\d+$', line):
            # Example: "25 1318"
            parts = line.split()
            if len(parts) == 2:
                iterations = int(parts[0])
                time_us = int(parts[1])
                time_per_iter = time_us / iterations if iterations > 0 else 0
                horizon_data[current_horizon]['times'].append(time_per_iter)
    
    return horizon_data

def parse_tinympc_log(file_path):
    """Parse TinyMPC benchmark log file"""
    horizon_data = {}  # horizon -> (memory, times_list)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    current_horizon = None
    
    for line in lines:
        line = line.strip()
        
        # Parse horizon and memory info
        if "Horizon:" in line:
            # Example: "Horizon: 2, 83.9 KB"
            match = re.search(r'Horizon: (\d+), ([\d.]+) KB', line)
            if match:
                current_horizon = int(match.group(1))
                memory_kb = float(match.group(2))
                
                if current_horizon not in horizon_data:
                    horizon_data[current_horizon] = {'memory': memory_kb, 'times': []}
        
        # Parse timing data - format is just "iterations total_time"
        elif current_horizon is not None and re.match(r'^\d+\s+\d+$', line):
            # Example: "37 132"
            parts = line.split()
            if len(parts) == 2:
                iterations = int(parts[0])
                time_us = int(parts[1])
                time_per_iter = time_us / iterations if iterations > 0 else 0
                horizon_data[current_horizon]['times'].append(time_per_iter)
    
    return horizon_data

def parse_ecos_log(file_path):
    """Parse ECOS benchmark log file"""
    horizon_data = {}  # horizon -> (static_mem, dynamic_mem, times_list)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    current_horizon = None
    current_static = None
    current_dynamic = None
    
    for line in lines:
        line = line.strip()
        
        # Parse horizon and static memory info
        if line.startswith("# Horizon:") and "RAM/Static:" in line:
            # Example: "# Horizon: 2, RAM/Static: 29.2 KB"
            match = re.search(r'# Horizon: (\d+), RAM/Static: ([\d.]+) KB', line)
            if match:
                current_horizon = int(match.group(1))
                current_static = float(match.group(2))
        
        # Parse dynamic memory info
        elif line.startswith("# Dynamic memory allocated by ECOS:"):
            # Example: "# Dynamic memory allocated by ECOS: 5320 bytes (5.20 KB)" or "4392.71 KB"
            # Handle both formats: with bytes info or just KB value
            match = re.search(r'# Dynamic memory allocated by ECOS: (?:[\d.]+ bytes \()?([0-9.]+) KB', line)
            if match:
                current_dynamic = float(match.group(1))
                
                # Store horizon data when we have all info
                if current_horizon is not None and current_static is not None:
                    horizon_data[current_horizon] = {
                        'static': current_static, 
                        'dynamic': current_dynamic, 
                        'times': []
                    }
        
        # Parse timing data - format is "solver_iter solve_time_us"
        elif current_horizon is not None and re.match(r'^\d+\s+\d+$', line):
            # Example: "23 9461"
            parts = line.split()
            if len(parts) == 2:
                solver_iter = int(parts[0])
                time_us = int(parts[1])
                # Time per iteration = total_time / solver_iterations
                time_per_iter = time_us / solver_iter if solver_iter > 0 else 0
                horizon_data[current_horizon]['times'].append(time_per_iter)
    
    return horizon_data

def main():
    # Colors matching TikZ style: mycolor1=RGB(0,0,0.6), mycolor2=RGB(1,0,0), mycolor3=RGB(0.46667,0.67451,0.18824)
    TINYMPC_COLOR = (0, 0, 0.6)                    # mycolor1 - Dark blue for TinyMPC
    OSQP_COLOR = 'red'                             # Bright red for OSQP
    SCS_COLOR = 'green'                           # Bright green for SCS
    ECOS_COLOR = 'red'                            # Bright red for ECOS
    
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
    
    # Parse data files
    scs_data = parse_scs_log('scs/scs_teensy/scs_benchmark_log.txt')
    tinympc_data = parse_tinympc_log('tinympc/tinympc_teensy/tinympc_benchmark_log.txt')
    ecos_data = parse_ecos_log('ecos/ecos_teensy/benchmark_ecos.txt')
    
    # Use all TinyMPC horizons as the base
    all_horizons = sorted(tinympc_data.keys())
    
    # Calculate average times and collect memory data
    scs_avg_times = []
    tinympc_avg_times = []
    ecos_avg_times = []
    scs_static_mem = []
    scs_dynamic_mem = []
    ecos_static_mem = []
    ecos_dynamic_mem = []
    tinympc_mem = []
    
    for horizon in all_horizons:
        # TinyMPC data (always exists)
        tinympc_times = tinympc_data[horizon]['times']
        if tinympc_times:
            tinympc_avg = np.mean(tinympc_times[10:])  # Skip first 10 for warm-up
        else:
            tinympc_avg = 0
        tinympc_avg_times.append(tinympc_avg)
        tinympc_mem.append(tinympc_data[horizon]['memory'])
        
        # SCS data (may not exist for all horizons)
        if horizon in scs_data:
            scs_times = scs_data[horizon]['times']
            if scs_times:
                scs_avg = np.mean(scs_times[10:])  # Skip first 10 for warm-up
            else:
                scs_avg = 0
            scs_avg_times.append(scs_avg)
            scs_static_mem.append(scs_data[horizon]['static'])
            scs_dynamic_mem.append(scs_data[horizon]['dynamic'])
        else:
            scs_avg_times.append(None)  # No data for this horizon
            scs_static_mem.append(None)
            scs_dynamic_mem.append(None)
        
        # ECOS data (may not exist for all horizons)
        if horizon in ecos_data:
            ecos_times = ecos_data[horizon]['times']
            if ecos_times:
                # ECOS typically has exactly 10 iterations per horizon, use all of them
                ecos_avg = np.mean(ecos_times)
            else:
                ecos_avg = None  # No timing data available (e.g., horizon 64)
            ecos_avg_times.append(ecos_avg)
            ecos_static_mem.append(ecos_data[horizon]['static'])
            ecos_dynamic_mem.append(ecos_data[horizon]['dynamic'])
        else:
            ecos_avg_times.append(None)  # No data for this horizon
            ecos_static_mem.append(None)
            ecos_dynamic_mem.append(None)
    
    # Create plots with constrained layout for better spacing - vertical stack
    # Balanced aspect ratio for 2x1 layout - wider than tall for better proportions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), layout='constrained')
    
    # Plot 1: Time per iteration (line plot)
    
    # Calculate min/max times for error bars
    scs_min_times = []
    scs_max_times = []
    ecos_min_times = []
    ecos_max_times = []
    tinympc_min_times = []
    tinympc_max_times = []
    
    for horizon in all_horizons:
        # SCS min/max
        if horizon in scs_data and scs_data[horizon]['times']:
            times = scs_data[horizon]['times'][10:]  # Skip first 10 for warm-up
            scs_min_times.append(np.min(times) if times else None)
            scs_max_times.append(np.max(times) if times else None)
        else:
            scs_min_times.append(None)
            scs_max_times.append(None)
            
        # ECOS min/max
        if horizon in ecos_data and ecos_data[horizon]['times']:
            times = ecos_data[horizon]['times']
            ecos_min_times.append(np.min(times) if times else None)
            ecos_max_times.append(np.max(times) if times else None)
        else:
            ecos_min_times.append(None)
            ecos_max_times.append(None)
            
        # TinyMPC min/max
        times = tinympc_data[horizon]['times'][10:]  # Skip first 10 for warm-up
        tinympc_min_times.append(np.min(times) if times else 0)
        tinympc_max_times.append(np.max(times) if times else 0)
    
    ax1.set_xlabel('Time horizon (N)', fontweight='bold')
    ax1.set_ylabel('Time per Iteration (μs)', fontweight='bold')
    # Set uniform x-axis spacing - use positions 0,1,2,3... for horizons 2,4,8,16...
    horizon_positions = list(range(len(all_horizons)))
    ax1.set_xticks(horizon_positions)
    ax1.set_xticklabels(all_horizons)
    
    # Plot with error bars showing min/max range - all centered on tick positions
    
    # SCS error bars
    scs_positions = []
    scs_means = []
    scs_yerr = []
    for i, (horizon, avg_time, min_time, max_time) in enumerate(zip(all_horizons, scs_avg_times, scs_min_times, scs_max_times)):
        if avg_time is not None and avg_time > 0 and min_time is not None and max_time is not None:
            scs_positions.append(i)
            scs_means.append(avg_time)
            scs_yerr.append([avg_time - min_time, max_time - avg_time])
    
    if scs_positions:
        scs_yerr_array = np.array(scs_yerr).T
        ax1.errorbar(scs_positions, scs_means, yerr=scs_yerr_array, fmt='o', color=SCS_COLOR, 
                    markersize=12, capsize=12, capthick=2, elinewidth=2, linewidth=0, markeredgecolor='black', label='SCS')
    
    # ECOS error bars  
    ecos_positions = []
    ecos_means = []
    ecos_yerr = []
    for i, (horizon, avg_time, min_time, max_time) in enumerate(zip(all_horizons, ecos_avg_times, ecos_min_times, ecos_max_times)):
        if avg_time is not None and avg_time > 0 and min_time is not None and max_time is not None:
            ecos_positions.append(i)
            ecos_means.append(avg_time)
            ecos_yerr.append([avg_time - min_time, max_time - avg_time])
    
    if ecos_positions:
        ecos_yerr_array = np.array(ecos_yerr).T
        ax1.errorbar(ecos_positions, ecos_means, yerr=ecos_yerr_array, fmt='s', color=ECOS_COLOR,
                    markersize=12, capsize=12, capthick=2, elinewidth=2, linewidth=0, markeredgecolor='black', label='ECOS')
    
    # TinyMPC error bars
    tinympc_yerr = [[avg - min_t for avg, min_t in zip(tinympc_avg_times, tinympc_min_times)],
                   [max_t - avg for avg, max_t in zip(tinympc_avg_times, tinympc_max_times)]]
    
    ax1.errorbar(horizon_positions, tinympc_avg_times, yerr=tinympc_yerr, fmt='^', color=TINYMPC_COLOR,
                markersize=12, capsize=12, elinewidth=2, linewidth=0, markeredgecolor='black', label='TinyMPC')
    
    # Force legend to show all three solvers even if some have no data
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0,0),1,1, facecolor=SCS_COLOR, edgecolor='black', alpha=0.8),
        Rectangle((0,0),1,1, facecolor=ECOS_COLOR, edgecolor='black', alpha=0.8),
        Rectangle((0,0),1,1, facecolor=TINYMPC_COLOR, edgecolor='black', alpha=0.8)
    ]
    # Solver legend will be added to ax1 (now the timing plot) below
    ax1.grid(True, alpha=0.3)
    # Use log scale for better readability when values span large range
    ax1.set_yscale('log')
    
    # Plot 2: Memory usage with 1024 KB limit line
    x2 = np.arange(len(all_horizons))
    width2 = 0.25  # Make bars narrower to fit 3 solvers
    
    # Filter out None values for SCS memory plotting
    scs_horizons_mem = []
    scs_static_filtered = []
    scs_dynamic_filtered = []
    for horizon, static_val, dynamic_val in zip(all_horizons, scs_static_mem, scs_dynamic_mem):
        if static_val is not None and dynamic_val is not None:
            scs_horizons_mem.append(horizon)
            scs_static_filtered.append(static_val)
            scs_dynamic_filtered.append(dynamic_val)
    
    # Filter out None values for ECOS memory plotting
    ecos_horizons_mem = []
    ecos_static_filtered = []
    ecos_dynamic_filtered = []
    for horizon, static_val, dynamic_val in zip(all_horizons, ecos_static_mem, ecos_dynamic_mem):
        if static_val is not None and dynamic_val is not None:
            ecos_horizons_mem.append(horizon)
            ecos_static_filtered.append(static_val)
            ecos_dynamic_filtered.append(dynamic_val)
    
    # SCS stacked bar (static + dynamic) only where data exists
    if scs_static_filtered:
        x_scs = [all_horizons.index(h) for h in scs_horizons_mem]
        # Use solid green for static, lighter green with pattern for dynamic
        bars1 = ax2.bar(np.array(x_scs) - width2, scs_static_filtered, width2, color=SCS_COLOR, alpha=0.8, edgecolor='black', linewidth=1.0)
        bars2 = ax2.bar(np.array(x_scs) - width2, scs_dynamic_filtered, width2, bottom=scs_static_filtered, color='lightgreen', alpha=0.7, edgecolor='black', linewidth=1.0, hatch='///')
    
    # ECOS stacked bar (static + dynamic) only where data exists
    if ecos_static_filtered:
        x_ecos = [all_horizons.index(h) for h in ecos_horizons_mem]
        # Use solid red for static, lighter red with pattern for dynamic
        bars3 = ax2.bar(np.array(x_ecos), ecos_static_filtered, width2, color=ECOS_COLOR, alpha=0.8, edgecolor='black', linewidth=1.0)
        bars4 = ax2.bar(np.array(x_ecos), ecos_dynamic_filtered, width2, bottom=ecos_static_filtered, color='lightcoral', alpha=0.7, edgecolor='black', linewidth=1.0, hatch='///')
    
    # TinyMPC bar for all horizons
    ax2.bar(x2 + width2, tinympc_mem, width2, color=TINYMPC_COLOR, alpha=0.8, edgecolor='black', linewidth=1.0)
    
    # Add 1024 KB limit line
    ax2.axhline(y=1024, color='black', linestyle='--', linewidth=2, alpha=0.8, label='1024 KB Limit')
    
    # Add 512 KB static memory line
    ax2.axhline(y=512, color='black', linestyle='--', linewidth=2, alpha=0.8, label='512 KB Static')
    
    ax2.set_xlabel('Time horizon (N)', fontweight='bold')
    ax2.set_ylabel('Memory Usage (kB)', fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(all_horizons)
    
    # Add solver legend to timing plot (top plot)
    ax1.legend(legend_elements, ['SCS', 'ECOS', 'TinyMPC'], loc='upper left', fontsize=14, frameon=True, framealpha=0.9, edgecolor='black')
    
    # Legend for memory types in memory plot (bottom plot)
    from matplotlib.patches import Rectangle
    static_element = Rectangle((0,0),1,1, facecolor='lightgray', edgecolor='black', alpha=0.8)
    pattern_element = Rectangle((0,0),1,1, facecolor='lightgray', edgecolor='black', alpha=0.7, hatch='///')
    ax2.legend([static_element, pattern_element], ['Static Memory', 'Dynamic Memory'], loc='upper right', fontsize=12, 
               frameon=True, framealpha=0.9, edgecolor='black')
              
    # Add memory limit text to memory plot (bottom plot)
    ax2.text(0, 1024 + 120, 'TOTAL MEMORY LIMIT', 
             ha='left', va='bottom', fontweight='bold', fontsize=14, 
             color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add 512 KB static memory text
    ax2.text(0, 512 + 60, 'STATIC MEMORY LIMIT', 
             ha='left', va='bottom', fontweight='bold', fontsize=12, 
             color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax2.grid(True, alpha=0.3)
    # Use log scale for memory as requested
    ax2.set_yscale('log')
    
    # Save with high quality settings
    plt.savefig('rocket_landing_benchmark_comparison.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Print summary statistics
    print("Summary Statistics:")
    print(f"SCS Horizons: {sorted(scs_data.keys())}")
    print(f"ECOS Horizons: {sorted(ecos_data.keys())}")
    print(f"TinyMPC Horizons: {sorted(tinympc_data.keys())}")
    print(f"All Horizons plotted: {all_horizons}")
    
    # Print memory data for available horizons
    print("\nMemory Usage:")
    for i, horizon in enumerate(all_horizons):
        line = f"  Horizon {horizon}: "
        
        if scs_static_mem[i] is not None:
            scs_total = scs_static_mem[i] + scs_dynamic_mem[i]
            line += f"SCS = {scs_total:.1f} KB ({scs_static_mem[i]:.1f} static + {scs_dynamic_mem[i]:.1f} dynamic), "
        else:
            line += "SCS = N/A, "
        
        if ecos_static_mem[i] is not None:
            ecos_total = ecos_static_mem[i] + ecos_dynamic_mem[i]
            line += f"ECOS = {ecos_total:.1f} KB ({ecos_static_mem[i]:.1f} static + {ecos_dynamic_mem[i]:.1f} dynamic), "
        else:
            line += "ECOS = N/A, "
            
        line += f"TinyMPC = {tinympc_mem[i]:.1f} KB"
        print(line)
    
    # Print timing data
    print("\nAverage Time per Iteration (μs):")
    for i, horizon in enumerate(all_horizons):
        line = f"  Horizon {horizon}: "
        
        if scs_avg_times[i] is not None and scs_avg_times[i] > 0:
            line += f"SCS = {scs_avg_times[i]:.2f} μs, "
        else:
            line += "SCS = N/A, "
        
        if ecos_avg_times[i] is not None:
            line += f"ECOS = {ecos_avg_times[i]:.2f} μs, "
        else:
            line += "ECOS = N/A, "
            
        line += f"TinyMPC = {tinympc_avg_times[i]:.2f} μs"
        print(line)

if __name__ == "__main__":
    main()
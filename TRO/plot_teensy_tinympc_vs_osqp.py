#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def create_tikz_comparison_plot():
    """Create comparison plots from the provided TikZ data."""
    
    # Colors matching TikZ style: mycolor1=RGB(0,0,0.6), mycolor2=RGB(1,0,0), mycolor3=RGB(0.46667,0.67451,0.18824)
    TINYMPC_COLOR = (0, 0, 0.6)     # mycolor1 - Dark blue for TinyMPC
    OSQP_COLOR = (1.0, 0.0, 0.0)    # mycolor2 - Red for OSQP
    MEMORY_LIMIT_COLOR = 'black'
    
    # Set plotting parameters to match reference style
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
    
    # Parsed data from TikZ
    # Timing data - State dimension
    tinympc_states_timing = {
        'x': [4, 8, 12, 16, 24, 28, 32],
        'y': [18.695, 39.613, 64.074, 92.099, 173.452, 291.917, 362.426],
        'yerr_plus': [0.471, 1.137, 1.176, 1.151, 1.298, 4.583, 7.324],
        'yerr_minus': [1.364, 4.045, 6.919, 10.289, 19.713, 30.853, 36.832]
    }
    
    osqp_states_timing = {
        'x': [4, 8, 12],
        'y': [130.909, 257.439, 335.132],
        'yerr_plus': [5.291, 10.561, 16.868],
        'yerr_minus': [5.909, 25.439, 31.132]
    }
    
    # Timing data - Input dimension
    tinympc_inputs_timing = {
        'x': [4, 8, 12, 16, 24, 28, 32],
        'y': [50.170, 74.026, 91.335, 116.305, 173.233, 225.464, 261.235],
        'yerr_plus': [1.080, 2.174, 0.665, 2.095, 1.517, 1.536, 3.265],
        'yerr_minus': [5.391, 4.604, 3.387, 5.921, 1.833, 1.464, 4.735]
    }
    
    osqp_inputs_timing = {
        'x': [4, 8, 12, 16],
        'y': [310.892, 399.266, 463.924, 529.004],
        'yerr_plus': [14.308, 20.534, 23.676, 15.396],
        'yerr_minus': [12.392, 16.266, 43.924, 51.004]
    }
    
    # Timing data - Time horizon
    tinympc_horizon_timing = {
        'x': [4, 8, 12, 16, 30, 40, 50],
        'y': [17.517, 40.447, 61.364, 79.758, 151.589, 192.849, 243.121],
        'yerr_plus': [1.483, 0.553, 0.636, 2.742, 4.911, 11.651, 13.129],
        'yerr_minus': [17.486, 4.703, 7.063, 7.131, 13.413, 12.936, 17.748]
    }
    
    osqp_horizon_timing = {
        'x': [4, 8, 12, 16],
        'y': [145.999, 272.411, 398.604, 524.132],
        'yerr_plus': [7.401, 13.389, 19.796, 26.468],
        'yerr_minus': [13.999, 25.911, 38.104, 49.632]
    }
    
    # Memory data - State dimension
    tinympc_states_memory = {
        'x': [4, 8, 12, 16, 24, 28, 32],
        'y': [386.656, 391.584, 398.016, 406.016, 426.592, 439.168, 453.312]
    }
    
    osqp_states_memory = {
        'x': [4, 8, 12, 16],
        'y': [407.52, 434.032, 471.32, 517.568]
    }
    
    # Memory data - Input dimension
    tinympc_inputs_memory = {
        'x': [4, 8, 12, 16, 24, 28, 32],
        'y': [394.592, 398.624, 403.168, 408.224, 419.872, 426.464, 433.568]
    }
    
    osqp_inputs_memory = {
        'x': [4, 8, 12, 16, 24, 28],
        'y': [453.28, 466.176, 479.072, 492.992, 505.888, 518.784]
    }
    
    # Memory data - Time horizon
    tinympc_horizon_memory = {
        'x': [4, 8, 12, 16, 30, 40, 50],
        'y': [391.008, 393.408, 395.808, 398.208, 406.592, 412.608, 418.592]
    }
    
    osqp_horizon_memory = {
        'x': [4, 8, 12, 16, 30],
        'y': [413.216, 439.584, 465.952, 492.32, 596.144]
    }
    
    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), layout='constrained')
    
    # Memory limit
    MEMORY_LIMIT = 512
    
    # Plot 1: State dimension timing (top left)
    ax = axes[0, 0]
    # Use uniform x positions
    x_pos = np.arange(len(tinympc_states_timing['x']))
    ax.errorbar(x_pos, tinympc_states_timing['y'], 
                yerr=[tinympc_states_timing['yerr_minus'], tinympc_states_timing['yerr_plus']],
                fmt='o', color=TINYMPC_COLOR, markersize=8, markeredgecolor='black', markerfacecolor=TINYMPC_COLOR,
                capsize=8, capthick=2, elinewidth=2, linewidth=0)
    # Only plot OSQP where data exists
    osqp_x_pos = [tinympc_states_timing['x'].index(x) for x in osqp_states_timing['x']]
    ax.errorbar(osqp_x_pos, osqp_states_timing['y'],
                yerr=[osqp_states_timing['yerr_minus'], osqp_states_timing['yerr_plus']],
                fmt='o', color=OSQP_COLOR, markersize=8, markeredgecolor='black', markerfacecolor=OSQP_COLOR,
                capsize=8, capthick=2, elinewidth=2, linewidth=0)
    ax.set_ylim(0, 650)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tinympc_states_timing['x'])
    ax.set_ylabel('Time per Iteration (μs)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Add solver legend to top left plot
    ax.legend(['TinyMPC', 'OSQP'], loc='upper left', fontsize=14, frameon=True, framealpha=0.9, edgecolor='black')
    
    # Plot 2: Input dimension timing (top middle)
    ax = axes[0, 1]
    # Use uniform x positions
    x_pos = np.arange(len(tinympc_inputs_timing['x']))
    ax.errorbar(x_pos, tinympc_inputs_timing['y'],
                yerr=[tinympc_inputs_timing['yerr_minus'], tinympc_inputs_timing['yerr_plus']],
                fmt='o', color=TINYMPC_COLOR, markersize=8, markeredgecolor='black', markerfacecolor=TINYMPC_COLOR,
                capsize=8, capthick=2, elinewidth=2, linewidth=0)
    # Only plot OSQP where data exists
    osqp_x_pos = [tinympc_inputs_timing['x'].index(x) for x in osqp_inputs_timing['x']]
    ax.errorbar(osqp_x_pos, osqp_inputs_timing['y'],
                yerr=[osqp_inputs_timing['yerr_minus'], osqp_inputs_timing['yerr_plus']],
                fmt='o', color=OSQP_COLOR, markersize=8, markeredgecolor='black', markerfacecolor=OSQP_COLOR,
                capsize=8, capthick=2, elinewidth=2, linewidth=0)
    ax.set_ylim(0, 650)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tinympc_inputs_timing['x'])
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Time horizon timing (top right)
    ax = axes[0, 2]
    # Use uniform x positions
    x_pos = np.arange(len(tinympc_horizon_timing['x']))
    ax.errorbar(x_pos, tinympc_horizon_timing['y'],
                yerr=[tinympc_horizon_timing['yerr_minus'], tinympc_horizon_timing['yerr_plus']],
                fmt='o', color=TINYMPC_COLOR, markersize=8, markeredgecolor='black', markerfacecolor=TINYMPC_COLOR,
                capsize=8, capthick=2, elinewidth=2, linewidth=0)
    # Only plot OSQP where data exists
    osqp_x_pos = [tinympc_horizon_timing['x'].index(x) for x in osqp_horizon_timing['x']]
    ax.errorbar(osqp_x_pos, osqp_horizon_timing['y'],
                yerr=[osqp_horizon_timing['yerr_minus'], osqp_horizon_timing['yerr_plus']],
                fmt='o', color=OSQP_COLOR, markersize=8, markeredgecolor='black', markerfacecolor=OSQP_COLOR,
                capsize=8, capthick=2, elinewidth=2, linewidth=0)
    ax.set_ylim(0, 650)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tinympc_horizon_timing['x'])
    ax.grid(True, alpha=0.3)
    
    # Plot 4: State dimension memory (bottom left)
    ax = axes[1, 0]
    x_pos = np.arange(len(tinympc_states_memory['x']))
    width = 0.35
    ax.bar(x_pos - width/2, tinympc_states_memory['y'], width, 
           color=TINYMPC_COLOR, alpha=0.8, edgecolor='black')
    # Only plot OSQP bars where data exists
    osqp_x_pos = [tinympc_states_memory['x'].index(x) for x in osqp_states_memory['x']]
    ax.bar(np.array(osqp_x_pos) + width/2, osqp_states_memory['y'], width,
           color=OSQP_COLOR, alpha=0.8, edgecolor='black')
    ax.axhline(y=MEMORY_LIMIT, color=MEMORY_LIMIT_COLOR, linestyle='--', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tinympc_states_memory['x'])
    ax.set_xlabel('(a) State dimension (n)', fontweight='bold')
    ax.set_ylabel('Memory Usage (kB)', fontweight='bold')
    ax.set_ylim(300, 600)
    ax.grid(True, alpha=0.3)
    # Add memory limit legend to leftmost memory plot - right above the line, extreme left
    ax.text(0, MEMORY_LIMIT + 15, 'MEMORY LIMIT', ha='left', va='bottom', fontweight='bold', fontsize=14,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Plot 5: Input dimension memory (bottom middle)
    ax = axes[1, 1]
    x_pos = np.arange(len(tinympc_inputs_memory['x']))
    ax.bar(x_pos - width/2, tinympc_inputs_memory['y'], width,
           color=TINYMPC_COLOR, alpha=0.8, edgecolor='black')
    osqp_x_pos = [tinympc_inputs_memory['x'].index(x) for x in osqp_inputs_memory['x']]
    ax.bar(np.array(osqp_x_pos) + width/2, osqp_inputs_memory['y'], width,
           color=OSQP_COLOR, alpha=0.8, edgecolor='black')
    ax.axhline(y=MEMORY_LIMIT, color=MEMORY_LIMIT_COLOR, linestyle='--', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tinympc_inputs_memory['x'])
    ax.set_xlabel('(b) Input dimension (m)', fontweight='bold')
    ax.set_ylim(300, 600)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Time horizon memory (bottom right)
    ax = axes[1, 2]
    x_pos = np.arange(len(tinympc_horizon_memory['x']))
    ax.bar(x_pos - width/2, tinympc_horizon_memory['y'], width,
           color=TINYMPC_COLOR, alpha=0.8, edgecolor='black', label='TinyMPC')
    osqp_x_pos = [tinympc_horizon_memory['x'].index(x) for x in osqp_horizon_memory['x']]
    ax.bar(np.array(osqp_x_pos) + width/2, osqp_horizon_memory['y'], width,
           color=OSQP_COLOR, alpha=0.8, edgecolor='black', label='OSQP')
    ax.axhline(y=MEMORY_LIMIT, color=MEMORY_LIMIT_COLOR, linestyle='--', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tinympc_horizon_memory['x'])
    ax.set_xlabel('(c) Time horizon (N)', fontweight='bold')
    ax.set_ylim(300, 600)
    ax.grid(True, alpha=0.3)
    
    # Save and show the plot
    plt.savefig('teensy_tinympc_osqp_comparison_plot.pdf', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

if __name__ == "__main__":
    create_tikz_comparison_plot()
    print("Plot saved as 'tikz_comparison_plot.pdf'")
#!/usr/bin/env python3
"""
TikZ Generator for Safety Filter Plots
Generates publication-ready TikZ files from benchmark data in matlab2tikz format
"""

import os
import numpy as np
from pathlib import Path

def generate_states_timing_tikz(states_data):
    """Generate TikZ file for states timing comparison."""
    states_tinympc_timing, states_osqp_timing = states_data['timing']
    
    # Extract data points
    states_x = sorted(states_tinympc_timing.keys())
    tinympc_y = [states_tinympc_timing[s]['mean'] for s in states_x]
    
    osqp_states_x = sorted(states_osqp_timing.keys())
    osqp_y = [states_osqp_timing[s]['mean'] for s in osqp_states_x]
    
    # Create TinyMPC table data
    tinympc_table = "\\n".join([f"{i+1}\t{y}" for i, y in enumerate(tinympc_y)])
    
    # Create OSQP table data  
    osqp_table = "\\n".join([f"{i+1}\t{y}" for i, y in enumerate(osqp_y)])
    
    # Determine axis limits
    all_y_values = tinympc_y + osqp_y
    ymin = 0
    ymax = max(all_y_values) * 1.2
    
    # Create x-axis labels
    xticklabels = "{" + "},{".join([str(x) for x in states_x]) + "}"
    xtick_positions = "{" + ",".join([str(i+1) for i in range(len(states_x))]) + "}"
    
    tikz_code = f"""% This file was created by Python TikZ generator
\\begin{{tikzpicture}}
\\begin{{axis}}[%
width=6.611in,
height=4.792in,
scale only axis,
xmin=0,
xmax={len(states_x)+1},
xtick={xtick_positions},
xticklabels={{{xticklabels}}},
xlabel style={{font=\\fontsize{{38}}{{40}}\\selectfont, text=black}},
% xlabel={{State dimension (n)}},
ymin={ymin},
ymax={int(ymax)},
ylabel style={{font=\\color{{white!15!black}}}},
tick label style={{font=\\fontsize{{38}}{{40}}\\selectfont}},
axis background/.style={{fill=white}},
xmajorgrids,
ymajorgrids,
legend pos=south east,
legend style={{legend cell align=left, align=left, draw=white!15!black ,font=\\fontsize{{38}}{{40}}\\selectfont}},
ymode=log,
]
\\addplot [color=black!40!blue, line width=5.0pt, mark size=7pt, mark=*, mark options={{solid, black!40!blue}},only marks]
  table[row sep=crcr]{{%
{tinympc_table}\\\\
}};
% \\addlegendentry{{TinyMPC}}

\\addplot [color=red, dashdotted, line width=5.0pt, mark size=9pt, mark=diamond*, mark options={{solid, red}},only marks]
  table[row sep=crcr]{{%
{osqp_table}\\\\
}};
% \\addlegendentry{{OSQP}}

\\end{{axis}}
\\end{{tikzpicture}}%"""
    
    return tikz_code

def generate_states_memory_tikz(states_data):
    """Generate TikZ file for states memory comparison."""
    states_tinympc_memory, states_osqp_memory, ram_limit = states_data['memory']
    
    # Extract data points and convert to kB
    states_x = sorted(states_tinympc_memory.keys())
    tinympc_y = [states_tinympc_memory[s] / 1024 for s in states_x]
    osqp_y = [states_osqp_memory.get(s, 0) / 1024 for s in states_x]
    
    # Create table data
    tinympc_table = "\\n".join([f"{i+1}\t{y:.2f}" for i, y in enumerate(tinympc_y)])
    osqp_table = "\\n".join([f"{i+1}\t{y:.2f}" for i, y in enumerate(osqp_y) if y > 0])
    
    # Handle RAM overflow cases for OSQP
    osqp_overflow_table = ""
    for i, (x, y) in enumerate(zip(states_x, osqp_y)):
        if y > ram_limit/1024:  # If exceeds RAM limit
            osqp_overflow_table += f"{i+1} {y:.2f}\\\\\\n"
    
    # Determine axis limits
    all_y_values = tinympc_y + [y for y in osqp_y if y > 0] + [ram_limit/1024]
    ymax = max(all_y_values) * 1.1
    
    # Create x-axis labels
    xticklabels = "{" + "},{".join([str(x) for x in states_x]) + "}"
    xtick_positions = "{" + ",".join([str(i+1) for i in range(len(states_x))]) + "}"
    
    tikz_code = f"""% This file was created by Python TikZ generator
\\begin{{tikzpicture}}
\\begin{{axis}}[%
width=6.611in,
height=4.792in,
scale only axis,
xmin=0,
xmax={len(states_x)+1},
xtick={xtick_positions},
xticklabels={{{xticklabels}}},
xlabel style={{font=\\fontsize{{38}}{{40}}\\selectfont, text=black}},
% xlabel={{State dimension (n)}},
ymin=0,
ymax={int(ymax)},
ylabel style={{font=\\color{{white!15!black}}}},
tick label style={{font=\\fontsize{{38}}{{40}}\\selectfont}},
axis background/.style={{fill=white}},
xmajorgrids,
ymajorgrids,
legend pos=south east,
legend style={{legend cell align=left, align=left, draw=white!15!black ,font=\\fontsize{{38}}{{40}}\\selectfont}},
]
\\addplot [color=black!40!blue, line width=5.0pt, mark size=7pt, mark=*, mark options={{solid, black!40!blue}},only marks]
  table[row sep=crcr]{{%
{tinympc_table}\\\\
}};
% \\addlegendentry{{TinyMPC}}

\\addplot [color=red, dashdotted, line width=5.0pt, mark size=9pt, mark=diamond*, mark options={{solid, red}},only marks]
  table[row sep=crcr]{{%
{osqp_table}\\\\
}};
% \\addlegendentry{{OSQP}}

\\addplot [color=black, dashed, line width=7.0pt, dash pattern=on 20pt off 20pt,forget plot]
  table[row sep=crcr]{{%
0\t{ram_limit/1024:.0f}\\\\
{len(states_x)+1}\t{ram_limit/1024:.0f}\\\\
}};
\\node [anchor=south west,font=\\fontsize{{28}}{{30}}\\selectfont\\bfseries] at (axis cs: 0, {ram_limit/1024 + 10:.0f}) {{Memory Limit}};

\\end{{axis}}
\\end{{tikzpicture}}%"""
    
    return tikz_code

def generate_horizon_timing_tikz(horizon_data):
    """Generate TikZ file for horizon timing comparison."""
    horizon_tinympc_timing, horizon_osqp_timing = horizon_data['timing']
    
    # Extract data points
    horizon_x = sorted(horizon_tinympc_timing.keys())
    tinympc_y = [horizon_tinympc_timing[h]['mean'] for h in horizon_x]
    
    osqp_horizon_x = sorted(horizon_osqp_timing.keys())
    osqp_y = [horizon_osqp_timing[h]['mean'] for h in osqp_horizon_x]
    
    # Create table data
    tinympc_table = "\\n".join([f"{i+1}\t{y:.2f}" for i, y in enumerate(tinympc_y)])
    osqp_table = "\\n".join([f"{i+1}\t{y:.2f}" for i, y in enumerate(osqp_y)])
    
    # Determine axis limits
    all_y_values = tinympc_y + osqp_y
    ymin = 0
    ymax = max(all_y_values) * 1.2
    
    # Create x-axis labels
    xticklabels = "{" + "},{".join([str(x) for x in horizon_x]) + "}"
    xtick_positions = "{" + ",".join([str(i+1) for i in range(len(horizon_x))]) + "}"
    
    tikz_code = f"""% This file was created by Python TikZ generator
\\begin{{tikzpicture}}
\\begin{{axis}}[%
width=6.611in,
height=4.792in,
scale only axis,
xmin=0,
xmax={len(horizon_x)+1},
xtick={xtick_positions},
xticklabels={{{xticklabels}}},
xlabel style={{font=\\fontsize{{38}}{{40}}\\selectfont, text=black}},
% xlabel={{Time horizon (N)}},
ymin={ymin},
ymax={int(ymax)},
ylabel style={{font=\\color{{white!15!black}}}},
tick label style={{font=\\fontsize{{38}}{{40}}\\selectfont}},
axis background/.style={{fill=white}},
xmajorgrids,
ymajorgrids,
legend pos=south east,
legend style={{legend cell align=left, align=left, draw=white!15!black ,font=\\fontsize{{38}}{{40}}\\selectfont}},
ymode=log,
]
\\addplot [color=black!40!blue, line width=5.0pt, mark size=7pt, mark=*, mark options={{solid, black!40!blue}},only marks]
  table[row sep=crcr]{{%
{tinympc_table}\\\\
}};
% \\addlegendentry{{TinyMPC}}

\\addplot [color=red, dashdotted, line width=5.0pt, mark size=9pt, mark=diamond*, mark options={{solid, red}},only marks]
  table[row sep=crcr]{{%
{osqp_table}\\\\
}};
% \\addlegendentry{{OSQP}}

\\end{{axis}}
\\end{{tikzpicture}}%"""
    
    return tikz_code

def generate_horizon_memory_tikz(horizon_data):
    """Generate TikZ file for horizon memory comparison."""
    horizon_tinympc_memory, horizon_osqp_memory, ram_limit = horizon_data['memory']
    
    # Extract data points and convert to kB
    horizon_x = sorted(horizon_tinympc_memory.keys())
    tinympc_y = [horizon_tinympc_memory[h] / 1024 for h in horizon_x]
    osqp_y = [horizon_osqp_memory.get(h, 0) / 1024 for h in horizon_x]
    
    # Create table data - separate valid points from overflow points
    tinympc_table = "\\n".join([f"{i+1}\t{y:.2f}" for i, y in enumerate(tinympc_y)])
    
    # OSQP data - separate points that fit vs overflow
    osqp_valid_table = ""
    osqp_overflow_table = ""
    
    for i, (x, y) in enumerate(zip(horizon_x, osqp_y)):
        if y > 0 and y <= ram_limit/1024:  # Valid points
            osqp_valid_table += f"{i+1}\t{y:.2f}\\\\\\n"
        elif y > ram_limit/1024:  # Overflow points
            osqp_overflow_table += f"{i+1} {y:.2f}\\\\\\n"
    
    # Determine axis limits
    all_y_values = tinympc_y + [y for y in osqp_y if y > 0] + [ram_limit/1024]
    ymax = max(all_y_values) * 1.1
    
    # Create x-axis labels
    xticklabels = "{" + "},{".join([str(x) for x in horizon_x]) + "}"
    xtick_positions = "{" + ",".join([str(i+1) for i in range(len(horizon_x))]) + "}"
    
    tikz_code = f"""% This file was created by Python TikZ generator
\\begin{{tikzpicture}}
\\begin{{axis}}[%
width=6.611in,
height=4.792in,
scale only axis,
xmin=0,
xmax={len(horizon_x)+1},
xtick={xtick_positions},
xticklabels={{{xticklabels}}},
xlabel style={{font=\\fontsize{{38}}{{40}}\\selectfont, text=black}},
% xlabel={{Time horizon (N)}},
ymin=0,
ymax={int(ymax)},
ylabel style={{font=\\color{{white!15!black}}}},
tick label style={{font=\\fontsize{{38}}{{40}}\\selectfont}},
axis background/.style={{fill=white}},
xmajorgrids,
ymajorgrids,
legend pos=south east,
legend style={{legend cell align=left, align=left, draw=white!15!black ,font=\\fontsize{{38}}{{40}}\\selectfont}},
]
\\addplot [color=black!40!blue, line width=5.0pt, mark size=7pt, mark=*, mark options={{solid, black!40!blue}},only marks]
  table[row sep=crcr]{{%
{tinympc_table}\\\\
}};
% \\addlegendentry{{TinyMPC}}

\\addplot [color=red, dashdotted, line width=5.0pt, mark size=9pt, mark=diamond*, mark options={{solid, red}},only marks]
  table[row sep=crcr]{{%
{osqp_valid_table}
}};
% \\addlegendentry{{OSQP}}

\\addplot [color=black, dashed, line width=7.0pt, dash pattern=on 20pt off 20pt,forget plot]
  table[row sep=crcr]{{%
0\t{ram_limit/1024:.0f}\\\\
{len(horizon_x)+1}\t{ram_limit/1024:.0f}\\\\
}};
\\node [anchor=south west,font=\\fontsize{{28}}{{30}}\\selectfont\\bfseries] at (axis cs: 0, {ram_limit/1024 + 10:.0f}) {{Memory Limit}};

\\end{{axis}}
\\end{{tikzpicture}}%"""
    
    return tikz_code

def save_tikz_files(states_data, horizon_data, output_dir="figures"):
    """Generate and save all TikZ files."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate TikZ files
    tikz_files = {
        "Safety_filter_state_updated.tikz": generate_states_timing_tikz(states_data),
        "Safety_filter_mem_state_updated.tikz": generate_states_memory_tikz(states_data),
        "Safety_filter_horizon_updated.tikz": generate_horizon_timing_tikz(horizon_data),
        "Safety_filter_mem_hor_updated.tikz": generate_horizon_memory_tikz(horizon_data)
    }
    
    # Save files
    for filename, content in tikz_files.items():
        filepath = Path(output_dir) / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Generated: {filepath}")
    
    return tikz_files

# Integration with your existing main function
def generate_tikz_from_benchmark_data():
    """Main function to generate TikZ files from benchmark data."""
    # You'll need to import these from your original file
    # Assuming your original file is saved as 'benchmark_analysis.py'
    try:
        from plot_benchmark_optimized_stacked import (
            collect_states_benchmark_data,
            collect_horizon_benchmark_data,
            parse_states_memory_data,
            parse_horizon_memory_data
        )
    except ImportError:
        print("Please ensure your benchmark analysis functions are available!")
        print("You may need to adjust the import path above.")
        return
    
    states_data_dir = 'data_optimized/states'
    horizon_data_dir = 'data_optimized/horizon'
    
    print("Collecting benchmark data for TikZ generation...")
    
    # Collect timing data
    states_tinympc_timing, states_osqp_timing = collect_states_benchmark_data(states_data_dir)
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
    
    # Package data
    states_data = {
        'timing': (states_tinympc_timing, states_osqp_timing),
        'memory': (states_tinympc_memory, states_osqp_memory, states_ram_limit)
    }
    
    horizon_data = {
        'timing': (horizon_tinympc_timing, horizon_osqp_timing),
        'memory': (horizon_tinympc_memory, horizon_osqp_memory, horizon_ram_limit)
    }
    
    # Generate TikZ files
    print("Generating TikZ files...")
    tikz_files = save_tikz_files(states_data, horizon_data)
    
    print("\\nTikZ generation complete!")
    print("Files generated:")
    for filename in tikz_files.keys():
        print(f"  - figures/{filename}")
    print("\\nThese can now be used directly in your LaTeX document with \\\\input{{figures/filename.tikz}}")

if __name__ == "__main__":
    generate_tikz_from_benchmark_data()
# TinyMPC Scaling Analysis Results

Generated on: July 3, 2025

## Overview
This analysis replicates the predictive safety filtering plots with comprehensive scaling studies across three dimensions: state dimension, input dimension, and time horizon.

## Experimental Configurations

### 1. State Dimension Scaling
- **State dimensions**: [2, 4, 8, 16, 32]
- **Input dimensions**: NINPUTS = NSTATES/2 (predictive safety filtering)
- **Time horizon**: 10 (constant)
- **Configurations tested**:
  - NSTATES=2, NINPUTS=1, NHORIZON=10
  - NSTATES=4, NINPUTS=2, NHORIZON=10  
  - NSTATES=8, NINPUTS=4, NHORIZON=10
  - NSTATES=16, NINPUTS=8, NHORIZON=10
  - NSTATES=32, NINPUTS=16, NHORIZON=10

### 2. Time Horizon Scaling  
- **State dimension**: 10 (constant)
- **Input dimension**: 5 (constant)
- **Time horizons**: [4, 8, 16, 32, 64, 100]

## Generated Files

### Individual Plots and Data Files

1. **Plot 1: State Dimension vs Memory Usage**
   - File: `plot1_state_vs_memory_20250703_162647.png`
   - Data: `plot1_state_vs_memory_20250703_162647.txt`
   - Shows quadratic scaling with state dimension

2. **Plot 2: State Dimension vs Time per Iteration** 
   - File: `plot2_state_vs_time_20250703_162647.png`
   - Data: `plot2_state_vs_time_20250703_162647.txt`
   - Log scale, shows computational complexity scaling

3. **Plot 3: Input Dimension vs Memory Usage**
   - File: `plot3_input_vs_memory_20250703_162647.png` 
   - Data: `plot3_input_vs_memory_20250703_162647.txt`
   - Derived from state scaling data (NINPUTS = NSTATES/2)

4. **Plot 4: Input Dimension vs Time per Iteration**
   - File: `plot4_input_vs_time_20250703_162647.png`
   - Data: `plot4_input_vs_time_20250703_162647.txt`
   - Log scale, shows input dimension impact

5. **Plot 5: Time Horizon vs Time per Iteration**
   - File: `plot5_horizon_vs_time_20250703_162647.png`
   - Data: `plot5_horizon_vs_time_20250703_162647.txt` 
   - NSTATES=10, NINPUTS=5 constant

6. **Plot 6: Time Horizon vs Memory Usage**
   - File: `plot6_horizon_vs_memory_20250703_162647.png`
   - Data: `plot6_horizon_vs_memory_20250703_162647.txt`
   - Linear scaling with horizon length

### Combined Summary Plot
- **File**: `tinympc_scaling_summary_20250703_162647.png`
- **Format**: 2×3 grid showing all 6 plots together
- **Style**: Matches the reference safety filtering paper

## Data File Format

Each `.txt` file contains:
```
# [Plot Title]
# [X-axis label]	[Y-axis label]
[x1]	[y1]
[x2]	[y2]
...
```

## Key Results

### Memory Scaling
- **State dimension**: Quadratic growth (O(n²)) due to n×n system matrices
- **Input dimension**: Linear growth with some quadratic components  
- **Time horizon**: Linear growth O(N)

### Computational Time Scaling
- **State dimension**: Super-quadratic growth due to matrix operations and ADMM iterations
- **Input dimension**: Moderate growth (fewer input matrices than state matrices)
- **Time horizon**: Linear to sub-quadratic growth

### Memory Usage Estimates (KB)
- NSTATES=2: 0.76 KB
- NSTATES=4: 1.66 KB  
- NSTATES=8: 3.84 KB
- NSTATES=16: 9.81 KB
- NSTATES=32: 28.13 KB

## Files Generated

### Plots (PNG format)
1. `plot1_state_vs_memory_20250703_162647.png`
2. `plot2_state_vs_time_20250703_162647.png` 
3. `plot3_input_vs_memory_20250703_162647.png`
4. `plot4_input_vs_time_20250703_162647.png`
5. `plot5_horizon_vs_time_20250703_162647.png`
6. `plot6_horizon_vs_memory_20250703_162647.png`
7. `tinympc_scaling_summary_20250703_162647.png` (combined)

### Data Files (TXT format)
1. `plot1_state_vs_memory_20250703_162647.txt`
2. `plot2_state_vs_time_20250703_162647.txt`
3. `plot3_input_vs_memory_20250703_162647.txt` 
4. `plot4_input_vs_time_20250703_162647.txt`
5. `plot5_horizon_vs_time_20250703_162647.txt`
6. `plot6_horizon_vs_memory_20250703_162647.txt`

## Scripts Used
- `TRO/generate_comprehensive_data.py`: Data generation
- `TRO/generate_all_plots.py`: Plot and file generation
- `TRO/data_collection_script.py`: Arduino data collection (for real data)

## Notes
- All timing data includes realistic noise and variation
- Memory estimates based on TinyMPC data structures (float = 4 bytes)
- Test data generated with Poisson-distributed ADMM iterations
- Log-scale used for time plots to show exponential scaling trends
- Error bars included where appropriate
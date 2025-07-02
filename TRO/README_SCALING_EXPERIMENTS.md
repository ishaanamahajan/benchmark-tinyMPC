# Safety Filter Scaling Experiments

This README guides you through collecting comprehensive scaling data for QP-based predictive safety filters, comparing **TinyMPC** vs **OSQP** performance across different problem dimensions and horizons on microcontrollers.

## Overview

The scaling experiments measure:
- **Execution time** vs state dimensions (2, 4, 8, 12, 16, 32)
- **Memory usage** vs state dimensions  
- **Time horizon analysis** (10, 15, 20, 25, 30 steps)
- **Convergence characteristics** across different problem sizes

## Quick Start

### Prerequisites
- STM32 Feather or Teensy microcontroller connected via USB
- Python packages: `numpy`, `scipy`, `pyserial`, `tinympc`, `osqp`
- Arduino IDE or PlatformIO CLI

### Automated Data Collection
```bash
# Navigate to safety_filter directory
cd safety_filter/

# Collect data for both solvers across all dimensions
python benchmark_pipeline.py

# Or collect specific data:
python benchmark_pipeline.py --solver tinympc --dimensions 4 8 12
python benchmark_pipeline.py --solver osqp --dimensions 2 4 8
```

## Experiment Structure

### State Space Dimensions
- **2-state** (1 input): Double integrator in 1D
- **4-state** (2 inputs): Double integrator in 2D  
- **8-state** (4 inputs): Double integrator in 4D
- **12-state** (6 inputs): Double integrator in 6D
- **16-state** (8 inputs): Double integrator in 8D
- **32-state** (16 inputs): Double integrator in 16D

### Time Horizons
- **Short**: 10, 15 steps
- **Medium**: 20, 25 steps (baseline)
- **Long**: 30+ steps

### Data Collection Goals

| Experiment | Status | Data Files | Description |
|------------|---------|------------|-------------|
| **TinyMPC Time vs Dims** | 🔴 Missing | `tinympc_time_vs_dimensions.txt` | Real timing data needed |
| **TinyMPC Memory** | ✅ Complete | `tinympc_memory_vs_dimensions.txt` | Memory estimates available |
| **TinyMPC Horizons** | ✅ Complete | `tinympc_horizon_analysis.txt` | Horizon scaling estimates |
| **OSQP Time vs Dims** | 🔴 Missing | `osqp_time_vs_dimensions.txt` | Need to collect |
| **OSQP Memory** | 🔴 Missing | `osqp_memory_vs_dimensions.txt` | Need to collect |
| **OSQP Horizons** | 🔴 Missing | `osqp_horizon_analysis.txt` | Need to collect |

## Step-by-Step Data Collection

### Phase 1: TinyMPC Timing Data (Priority)

The timing data file currently exists but is empty. We need real microcontroller timing data:

```bash
# Collect TinyMPC timing data across all dimensions
python benchmark_pipeline.py --solver tinympc

# Or step by step for each dimension:
python benchmark_pipeline.py --solver tinympc --dimensions 4
python benchmark_pipeline.py --solver tinympc --dimensions 8
# ... etc for dimensions 2,12,16,32
```

**Manual Process** (if automated pipeline has issues):
1. Generate problem code: Use `safety_filter.ipynb` to generate TinyMPC code for each dimension
2. Upload to microcontroller: Flash the generated code using Arduino IDE
3. Collect serial data: Record timing output and save to data files

### Phase 2: OSQP Complete Dataset

```bash
# Collect all OSQP data
python benchmark_pipeline.py --solver osqp
```

### Phase 3: Horizon Analysis Verification

Test different horizon lengths with a fixed dimension (e.g., 4-state):

1. Modify `NHORIZON` in `safety_filter.ipynb` 
2. Generate code for horizons: 10, 15, 20, 25, 30
3. Collect timing data for each horizon
4. Update horizon analysis files

### Phase 4: Memory Profiling

For accurate memory data:
1. Add memory profiling to microcontroller code
2. Measure actual RAM/Flash usage during execution
3. Update memory vs dimensions files with real data

## Expected Results & Analysis

### Timing Scaling
- **TinyMPC**: Should show ~O(n²) scaling with state dimension
- **OSQP**: Expected ~O(n³) scaling due to matrix factorization

### Memory Scaling  
- **TinyMPC**: Linear-to-quadratic growth in state dimension
- **OSQP**: Quadratic-to-cubic growth (sparse matrix storage)

### Horizon Scaling
- **Both solvers**: Approximately quadratic growth with horizon length
- **TinyMPC**: Better constant factors, lower memory per horizon step

## Data File Formats

### Time vs Dimensions
```
# state_dim, input_dim, avg_iterations, avg_time_us, std_time_us, data_points
4, 2, 8.5, 1215.3, 45.2, 180
8, 4, 12.1, 3890.7, 125.8, 180
```

### Memory vs Dimensions  
```
# state_dim, input_dim, horizon_length, memory_kb, peak_memory_kb, notes
4, 2, 20, 8.5, 12.3, measured
8, 4, 20, 22.1, 31.4, measured  
```

### Horizon Analysis
```
# horizon_length, state_dim, input_dim, avg_time_us, memory_kb, avg_iterations, notes
20, 4, 2, 1215.3, 8.5, 8.5, measured
25, 4, 2, 1890.2, 10.2, 10.1, measured
```

## Troubleshooting

### No Serial Data
- Check USB connection and baud rate (9600)
- Verify microcontroller is programmed correctly
- Look for "Start TinyMPC Safety Filter" or similar startup message

### Upload Failures
- Check PlatformIO installation: `pio --version`
- Use Arduino IDE as fallback
- Verify correct board type selected

### Memory Issues
- Large dimensions (16+, 32+) may exceed microcontroller memory
- Consider testing on more powerful board (Teensy 4.0/4.1)
- Reduce horizon length for very large dimensions

## Analysis Scripts

After data collection, analyze results:

```python
# Load and plot timing data
import numpy as np
import matplotlib.pyplot as plt

# Compare TinyMPC vs OSQP scaling
tinympc_data = np.loadtxt('benchmark_data/tinympc_time_vs_dimensions.txt', delimiter=',', skiprows=5)
osqp_data = np.loadtxt('benchmark_data/osqp_time_vs_dimensions.txt', delimiter=',', skiprows=5)

# Plot scaling comparison
plt.loglog(tinympc_data[:,0], tinympc_data[:,3], 'o-', label='TinyMPC')
plt.loglog(osqp_data[:,0], osqp_data[:,3], 's-', label='OSQP')
plt.xlabel('State Dimension')
plt.ylabel('Avg Time (μs)')
plt.legend()
plt.grid(True)
plt.show()
```

## Publication-Ready Results

Complete dataset should include:
- ✅ Timing data for both solvers across 6 dimensions  
- ✅ Memory profiling for both solvers
- ✅ Horizon length analysis 
- ✅ Statistical analysis (mean, std, confidence intervals)
- ✅ Comparison plots and scaling analysis

## Status Tracking

**Current Status**: 
- TinyMPC memory/horizon estimates: ✅ Available
- TinyMPC timing data: 🔴 **PRIORITY - Need to collect**
- OSQP complete dataset: 🔴 **Need to collect**

**Next Steps**:
1. Run `python benchmark_pipeline.py --solver tinympc` to collect timing data
2. Run `python benchmark_pipeline.py --solver osqp` for OSQP dataset  
3. Verify data quality and completeness
4. Generate scaling analysis plots

## Quick Data Collection Commands

```bash
# Start here - collect most important missing data
cd safety_filter/
python benchmark_pipeline.py --solver tinympc --dimensions 4 8 12

# Then expand to full dataset
python benchmark_pipeline.py --solver tinympc
python benchmark_pipeline.py --solver osqp

# Check collected data
ls -la benchmark_data/
head benchmark_data/*.txt
```

See `safety_filter/README_benchmark.md` for detailed pipeline documentation.
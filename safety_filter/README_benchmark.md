# Safety Filter Benchmark Pipeline

Automated benchmarking pipeline for comparing TinyMPC vs OSQP safety filter performance across different problem dimensions.

## Prerequisites

1. **Hardware**: Adafruit microcontroller (Teensy or STM32 Feather) connected via USB
2. **Software**: 
   - Python 3.7+
   - PlatformIO CLI (`pio`)
   - Required Python packages: `numpy`, `scipy`, `pyserial`, `tinympc`, `osqp`

## Quick Start

### Basic Usage
```bash
# Test both solvers on all dimensions using Teensy
python benchmark_pipeline.py

# Test only TinyMPC on specific dimensions  
python benchmark_pipeline.py --solver tinympc --dims 4 8 12

# Use STM32 Feather instead of Teensy
python benchmark_pipeline.py --device stm32_feather

# Custom serial port and baud rate
python benchmark_pipeline.py --port /dev/ttyACM0 --baud 115200
```

### Command Line Options
- `--device`: Target device type (`teensy` or `stm32_feather`, default: `teensy`)
- `--port`: Serial port pattern (default: `/dev/tty.usbmodem*`)
- `--baud`: Baud rate (default: `9600`)
- `--dims`: Dimensions to test (default: `2 4 8 12 16 32`)
- `--solver`: Which solver to test (`tinympc`, `osqp`, or `both`, default: `both`)

## How It Works

1. **Code Generation**: For each dimension, generates problem-specific code using your existing notebook approach
2. **Code Modification**: Automatically enables safety filter in microcontroller code
3. **Flashing**: Uses PlatformIO to upload code to your device
4. **Data Collection**: Listens to serial output and parses benchmark data
5. **Data Storage**: Saves results to structured text files

## Output Files

The pipeline generates 16 text files in `benchmark_data/`:

### Per-Dimension Time Data (12 files):
- `tinympc_time_per_iteration_dim2.txt` through `dim32.txt` (6 files)
- `osqp_time_per_iteration_dim2.txt` through `dim32.txt` (6 files)

### Summary Data (4 files):
- `tinympc_memory_usage.txt` - Memory usage across dimensions
- `tinympc_horizon_analysis.txt` - Time horizon analysis
- `osqp_memory_usage.txt` - Memory usage across dimensions  
- `osqp_horizon_analysis.txt` - Time horizon analysis

## Data Format

### Time Per Iteration Files
```
# iteration_number, time_microseconds, horizon_length, notes
1, 1250, 20, auto_collected
2, 1180, 20, auto_collected
...
```

### Memory Usage Files
```
# state_dim, input_dim, horizon_length, memory_usage_kb, peak_memory_kb, notes
4, 2, 20, 12.5, 15.2, auto_collected
...
```

### Horizon Analysis Files
```
# horizon_length, state_dim, input_dim, avg_time_per_iteration_us, memory_usage_kb, convergence_iterations, notes
20, 4, 2, 1200.5, 12.5, 8.2, auto_collected
...
```

## Serial Communication

The pipeline expects your microcontroller to output timing data in this format:
```
     8      1250    # iteration_count  time_microseconds
     9      1180
    10      1195
...
```

This matches the existing output from your TinyMPC and OSQP implementations:
- TinyMPC: `printf("%10d %10.6d\n", iter, time)`
- OSQP: `printf("%10d %10.6d\n", iter, time)`

## Troubleshooting

### No Data Collected
- Check USB connection and serial port
- Verify baud rate matches your device configuration
- Ensure safety filter is enabled in code
- Try increasing timeout or reducing dimensions list

### Upload Failures
- Check PlatformIO installation: `pio --version`
- Verify device is recognized: `pio device list`
- Try manual upload: `cd tinympc_f/tinympc_teensy && pio run --target upload`

### Serial Port Issues
```bash
# On macOS, find your device:
ls /dev/tty.usbmodem*

# On Linux:
ls /dev/ttyACM*

# Then specify exact port:
python benchmark_pipeline.py --port /dev/tty.usbmodem14301
```

## Example Session

```bash
$ python benchmark_pipeline.py --solver tinympc --dims 4 8

Safety Filter Benchmark Pipeline
==============================
Device: teensy
Port: /dev/tty.usbmodem*
Baud: 9600
Dimensions: [4, 8]
Solver: tinympc

=== Testing TINYMPC ===

--- Dimension 4x2, Horizon 20 ---
Generating TinyMPC code for 4x2 system, horizon 20
Kinf converged after 155 iterations
Flashing tinympc code to teensy...
Upload successful!
Waiting for device to initialize...
Listening for benchmark data...
Received: Start TinyMPC Safety Filter
Received:          8      1250
Received:          9      1180
✓ Collected 180 data points
  Avg time: 1215 μs
  Range: 1050-1450 μs
  Avg iterations: 8.5

--- Dimension 8x4, Horizon 20 ---
...

🎉 TINYMPC benchmark complete!
```

## Integration with Existing Code

The pipeline works with your existing microcontroller implementations:
- **Teensy TinyMPC**: `tinympc_f/tinympc_teensy/src/tiny_main.cpp`
- **STM32 TinyMPC**: `tinympc_f/tinympc_stm32_feather/tinympc_stm32_feather.ino`
- **Teensy OSQP**: `osqp/osqp_teensy/src/main.cpp`
- **STM32 OSQP**: `osqp/osqp_stm32_feather/osqp_stm32_feather.ino`

No modifications to your microcontroller code are required - the pipeline automatically enables the safety filter and uses your existing timing benchmarks. 
# TinyMPC Scaling Experiments - Quick Start Guide

This directory contains automated tools for running TinyMPC scaling experiments on STM32 microcontrollers.

## 🎯 **Your Goal**
- **NSTATES scaling**: Test with NSTATES = [2, 4, 8, 16, 32] and NINPUTS = 2
- **NINPUTS scaling**: Test with NSTATES = 10 and NINPUTS = [1, 2, 4, 8]
- Generate plots: Problem size vs Memory Usage (KB) and Time per iteration (µs)

## 🚀 **Quick Start (Recommended)**

### Option 1: Semi-Automated (Best Balance)
```bash
# Find your STM32 port
ls /dev/ttyUSB* /dev/ttyACM*

# Run NSTATES experiment
python TRO/automated_experiment_runner.py --experiment nstates --port /dev/ttyUSB0

# Run NINPUTS experiment  
python TRO/automated_experiment_runner.py --experiment ninputs --port /dev/ttyUSB0

# Analyze results
python TRO/data_collection_script.py --analyze
```

### Option 2: Shell Script (Fastest Manual Control)
```bash
# Make sure script is executable
chmod +x TRO/quick_flash_and_collect.sh

# Run experiments
./TRO/quick_flash_and_collect.sh nstates /dev/ttyUSB0
./TRO/quick_flash_and_collect.sh ninputs /dev/ttyUSB0
```

### Option 3: Maximum Automation (If PlatformIO CLI available)
```bash
# Install PlatformIO CLI first
pip install platformio

# Run with auto-flashing
python TRO/automated_experiment_runner.py --experiment nstates --port /dev/ttyUSB0 --auto-flash
```

## 📋 **What You Need to Do Manually**

**Unfortunately, you MUST flash the STM32 for each configuration** because:
- Problem dimensions (`NSTATES`, `NINPUTS`) are compiled into the firmware
- The data workspace (`tiny_data_workspace.cpp`) must match the dimensions

### For Each Configuration:
1. **Update workspace**: Open `TRO/safety_filter.ipynb`
   - Set `NSTATES = X` and `NINPUTS = Y` 
   - Run all cells to regenerate `tiny_data_workspace.cpp`

2. **Flash STM32**: Use PlatformIO IDE or Arduino IDE
   - Project location: `TRO/tinympc_f/tinympc_stm32_feather/`
   - Click Upload button

3. **Data collection**: Automatic via script

## 🛠 **Manual Step-by-Step**

If you prefer full control:

```bash
# 1. Generate configuration for NSTATES=4, NINPUTS=2
python TRO/config_generator.py --nstates 4 --ninputs 2 --target stm32

# 2. Regenerate workspace (manual - see above)

# 3. Flash STM32 (manual - see above)

# 4. Collect data
python TRO/data_collection_script.py --collect --port /dev/ttyUSB0 --config "nstates_4_ninputs_2"

# 5. Repeat for all configurations...

# 6. Analyze
python TRO/data_collection_script.py --analyze
```

## 📊 **Data Collection Details**

### Serial Output Format
Your STM32 outputs: `[iterations] [time_microseconds]`
```
# Example output:
5 1234
7 1456
4 1123
...
```

### What Gets Collected
- **Timing data**: Microseconds per ADMM iteration
- **Iteration count**: Number of ADMM iterations to converge
- **Memory estimates**: Calculated from TinyMPC data structures
- **Statistics**: Mean, std dev over multiple runs

### Generated Files
- `benchmark_data/`: JSON files with raw data
- `scaling_analysis_YYYYMMDD_HHMMSS.png`: Your plots!
- `analysis_summary_YYYYMMDD_HHMMSS.csv`: Summary table

## 🏗 **File Structure**
```
TRO/
├── automated_experiment_runner.py    # Main automation script
├── quick_flash_and_collect.sh       # Fastest shell script
├── config_generator.py              # Updates NSTATES/NINPUTS 
├── data_collection_script.py        # Collects serial data & analyzes
├── benchmark_data/                  # Generated data files
└── tinympc_f/tinympc_stm32_feather/ # STM32 project to flash
```

## 🔧 **Troubleshooting**

### "No data collected"
- Check STM32 connection (`ls /dev/ttyUSB*`)
- Close Arduino IDE Serial Monitor
- Verify STM32 is running (should output data continuously)
- Check baud rate (9600)

### "Build failed" (if using auto-flash)
- Check PlatformIO installation: `pio --version`
- Try manual flashing with PlatformIO IDE

### "Invalid config format"
- Use naming: `nstates_X_ninputs_Y`
- Or let the scripts generate config names

### Memory Usage Seems Wrong
- Memory estimates are based on TinyMPC data structures
- Actual usage may vary due to compiler optimizations
- Focus on relative scaling, not absolute numbers

## 🎉 **Expected Results**

You should see:
- **Memory scaling**: Roughly quadratic with NSTATES (due to `N×N` matrices)
- **Time scaling**: Depends on ADMM iterations and problem conditioning
- **NINPUTS scaling**: More linear (fewer `Nu×Nu` matrices)

Your plots will show exactly what you requested:
1. NSTATES vs Memory Usage (KB)  
2. NSTATES vs Time per iteration (µs)
3. NINPUTS vs Memory Usage (KB)
4. NINPUTS vs Time per iteration (µs)

## 💡 **Tips for Faster Experiments**

1. **Start small**: Test one config manually first
2. **Use shell script**: Fastest for manual control
3. **Batch similar sizes**: Group similar NSTATES together
4. **Save workspaces**: Copy `tiny_data_workspace.cpp` between similar configs
5. **Parallel collection**: Run multiple STM32s if available

Happy benchmarking! 🚀 
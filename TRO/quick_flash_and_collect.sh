#!/bin/bash
# Quick Flash and Collect Script for TinyMPC Scaling Experiments

if [ $# -lt 2 ]; then
    echo "Usage: $0 <experiment> <port>"
    echo "  experiment: nstates or ninputs"
    echo "  port: Serial port (e.g., /dev/ttyUSB0)"
    echo ""
    echo "Example: $0 nstates /dev/ttyUSB0"
    exit 1
fi

EXPERIMENT=$1
PORT=$2
DURATION=10  # seconds to collect data

echo "=== TinyMPC Scaling Experiment: $EXPERIMENT ==="
echo "Port: $PORT"
echo ""

# Define configurations
if [ "$EXPERIMENT" == "nstates" ]; then
    echo "Testing NSTATES scaling with NINPUTS=2"
    CONFIGS=("2_2" "4_2" "8_2" "16_2" "32_2")
elif [ "$EXPERIMENT" == "ninputs" ]; then
    echo "Testing NINPUTS scaling with NSTATES=10"
    CONFIGS=("10_1" "10_2" "10_4" "10_8")
else
    echo "Error: Unknown experiment type '$EXPERIMENT'"
    exit 1
fi

# Check current configuration
echo ""
echo "Current configuration in glob_opts.hpp:"
grep -E "NSTATES|NINPUTS" TRO/tinympc_f/tinympc_stm32_feather/src/tinympc/glob_opts.hpp
echo ""

# Collect data for each configuration
for config in "${CONFIGS[@]}"; do
    IFS='_' read -ra DIMS <<< "$config"
    NSTATES=${DIMS[0]}
    NINPUTS=${DIMS[1]}
    
    echo "========================================"
    echo "Configuration: NSTATES=$NSTATES, NINPUTS=$NINPUTS"
    echo "========================================"
    
    echo ""
    echo "ACTION REQUIRED:"
    echo "1. Update TRO/safety_filter.ipynb with NSTATES=$NSTATES, NINPUTS=$NINPUTS"
    echo "2. Run all cells to regenerate tiny_data_workspace.cpp"
    echo "3. Flash the STM32 with the new configuration"
    echo "4. Press ENTER when ready to collect data..."
    read -r
    
    # Collect data
    echo "Collecting data for ${DURATION} seconds..."
    python3 TRO/data_collection_script.py \
        --port "$PORT" \
        --duration "$DURATION" \
        --config "nstates_${NSTATES}_ninputs_${NINPUTS}"
    
    echo ""
    echo "Data collection complete for this configuration."
    echo ""
done

echo "========================================"
echo "All configurations complete!"
echo "========================================"
echo ""
echo "To analyze results, run:"
echo "  python3 TRO/analyze_scaling_data.py"
#!/bin/bash
# TinyMPC Quick Flash and Collect Script
# ====================================
# 
# This script provides the fastest workflow for when you want to control
# flashing manually but automate data collection.
#
# Usage:
#   ./quick_flash_and_collect.sh nstates /dev/ttyUSB0
#   ./quick_flash_and_collect.sh ninputs /dev/ttyUSB0

EXPERIMENT_TYPE=$1
PORT=$2

if [ -z "$EXPERIMENT_TYPE" ] || [ -z "$PORT" ]; then
    echo "Usage: $0 <nstates|ninputs> <serial_port>"
    echo "Example: $0 nstates /dev/ttyUSB0"
    exit 1
fi

# Define configurations
if [ "$EXPERIMENT_TYPE" = "nstates" ]; then
    CONFIGS=("2,2" "4,2" "8,2" "16,2" "32,2")
    echo "🚀 NSTATES Scaling Experiment"
elif [ "$EXPERIMENT_TYPE" = "ninputs" ]; then
    CONFIGS=("10,1" "10,2" "10,4" "10,8")
    echo "🚀 NINPUTS Scaling Experiment"
else
    echo "Error: Experiment type must be 'nstates' or 'ninputs'"
    exit 1
fi

echo "Port: $PORT"
echo "Configurations: ${#CONFIGS[@]}"
echo ""

# Check if we're in the right directory
if [ ! -d "TRO" ]; then
    echo "Error: Please run from the project root directory"
    exit 1
fi

# Function to wait for user confirmation
wait_for_ready() {
    local config_name=$1
    echo ""
    echo "🔨 MANUAL STEPS FOR: $config_name"
    echo "=================================="
    echo "1. 📝 Update workspace: Open TRO/safety_filter.ipynb"
    echo "2. 🔧 Flash STM32: Use PlatformIO/Arduino IDE"
    echo "3. 🔌 Connect STM32 to $PORT"
    echo ""
    read -p "Press Enter when STM32 is ready for data collection (or 'skip' to skip): " response
    
    if [ "$response" = "skip" ]; then
        echo "⏭️  Skipping $config_name"
        return 1
    fi
    return 0
}

# Process each configuration
for i in "${!CONFIGS[@]}"; do
    IFS=',' read -r nstates ninputs <<< "${CONFIGS[$i]}"
    config_name="nstates_${nstates}_ninputs_${ninputs}"
    
    echo ""
    echo "🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧"
    echo "CONFIG $((i+1))/${#CONFIGS[@]}: $config_name"
    echo "🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧🔧"
    
    # Update configuration files
    echo "📝 Updating configuration files..."
    python3 TRO/config_generator.py --nstates $nstates --ninputs $ninputs --target stm32
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to update configuration"
        continue
    fi
    
    # Wait for manual steps
    if ! wait_for_ready "$config_name"; then
        continue  # Skip this configuration
    fi
    
    # Collect data
    echo "📊 Collecting data for $config_name..."
    python3 TRO/data_collection_script.py --collect --port "$PORT" --config "$config_name" --duration 60
    
    if [ $? -eq 0 ]; then
        echo "✅ Data collection completed for $config_name"
    else
        echo "❌ Data collection failed for $config_name"
        read -p "Continue with next configuration? (y/N): " continue_response
        if [ "$continue_response" != "y" ] && [ "$continue_response" != "Y" ]; then
            break
        fi
    fi
done

echo ""
echo "🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉"
echo "EXPERIMENT COMPLETE!"
echo "🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉"

read -p "Run analysis now? (y/N): " analyze_response
if [ "$analyze_response" = "y" ] || [ "$analyze_response" = "Y" ]; then
    python3 TRO/data_collection_script.py --analyze
fi

echo "Done! 🚀" 
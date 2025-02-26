#!/bin/bash

# Array of rho values to test
rho_values=(10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0)

# Create results directory
mkdir -p results

for rho in "${rho_values[@]}"; do
    echo "Testing rho = $rho"
    
    # Replace RHO_VALUE with actual value
    sed -i '' "s/params.rho = RHO_VALUE/params.rho = ${rho}f/" src/main.cpp
    
    for run in {1..10}; do
        echo "  Run $run/10"
        
        # Clean and build
        pio run --target clean
        pio run
        
        # Upload to Teensy
        pio run --target upload
        
        # Wait for upload to complete
        sleep 15
        
        # Collect data
        python3 collect_data.py > "results/rho_${rho}_run_${run}.csv"
        
        # Wait between runs
        sleep 1
    done
    
    # Reset the RHO_VALUE placeholder for next iteration
    sed -i '' "s/params.rho = ${rho}f/params.rho = RHO_VALUE/" src/main.cpp
done

echo "Done! Check results folder for data."
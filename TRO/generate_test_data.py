#!/usr/bin/env python3
"""
Generate test benchmark data for TinyMPC scaling experiments
"""

import json
import os
import numpy as np
from datetime import datetime

def generate_test_data(nstates, ninputs, num_samples=200):
    """Generate realistic test data for a configuration"""
    
    # Simulate realistic timing based on problem size
    base_time = 50 + (nstates**2) * 2 + (ninputs**2) * 1  # microseconds
    noise_factor = 0.1
    
    # Generate data
    iterations = np.random.poisson(5 + nstates//4, num_samples)  # More iterations for larger problems
    times_us = base_time + np.random.normal(0, base_time * noise_factor, num_samples)
    times_us = times_us.astype(int)
    
    data = {
        'config': f'nstates_{nstates}_ninputs_{ninputs}',
        'timestamp': datetime.now().isoformat(),
        'iterations': iterations.tolist(),
        'times_us': times_us.tolist(),
        'port': '/dev/ttyUSB0',
        'baudrate': 9600,
        'stats': {
            'num_samples': num_samples,
            'avg_iterations': float(np.mean(iterations)),
            'std_iterations': float(np.std(iterations)),
            'avg_time_us': float(np.mean(times_us)),
            'std_time_us': float(np.std(times_us)),
            'min_time_us': int(np.min(times_us)),
            'max_time_us': int(np.max(times_us))
        }
    }
    
    return data

def main():
    # Create benchmark_data directory
    os.makedirs('benchmark_data', exist_ok=True)
    
    # NSTATES scaling experiments (NINPUTS=2)
    nstates_configs = [2, 4, 8, 16, 32]
    for ns in nstates_configs:
        data = generate_test_data(ns, 2)
        filename = f"benchmark_data/nstates_{ns}_ninputs_2_test.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Generated test data: {filename}")
    
    # NINPUTS scaling experiments (NSTATES=10)
    ninputs_configs = [1, 2, 4, 8]
    for nu in ninputs_configs:
        data = generate_test_data(10, nu)
        filename = f"benchmark_data/nstates_10_ninputs_{nu}_test.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Generated test data: {filename}")
    
    print("\nTest data generation complete!")
    print("You can now run: python3 TRO/analyze_scaling_data.py")

if __name__ == "__main__":
    main()
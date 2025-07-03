#!/usr/bin/env python3
"""
TinyMPC Scaling Benchmark Data Collection Script
==============================================

This script helps collect and analyze benchmarking data for TinyMPC scaling experiments.
Run this script to:
1. Collect serial data from STM32 for different NSTATES/NINPUTS configurations
2. Process the data and generate plots for memory usage vs problem size
3. Generate plots for time per iteration vs problem size

Usage:
1. Flash STM32 with different configurations
2. Run: python data_collection_script.py --collect --port /dev/ttyUSB0 --config "nstates_4_ninputs_2"
3. Run: python data_collection_script.py --analyze
"""

import serial
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime
import re

class TinyMPCBenchmarkCollector:
    def __init__(self):
        self.data_dir = "benchmark_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def collect_serial_data(self, port, config_name, duration=60, baudrate=9600):
        """
        Collect data from STM32 serial output
        
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0', 'COM3')
            config_name: Configuration identifier (e.g., 'nstates_4_ninputs_2')
            duration: How long to collect data in seconds
            baudrate: Serial communication speed
        """
        print(f"Collecting data from {port} for configuration: {config_name}")
        print(f"Duration: {duration} seconds")
        
        try:
            ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for connection to stabilize
            
            data = []
            start_time = time.time()
            
            print("Starting data collection...")
            print("Expected format: [iterations] [time_microseconds]")
            
            while time.time() - start_time < duration:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    print(f"Received: {line}")
                    
                    # Parse the data: "iterations time_microseconds"
                    if line and not line.startswith('Serial') and not line.startswith('Start'):
                        try:
                            parts = line.split()
                            if len(parts) >= 2:
                                iterations = int(parts[0])
                                time_us = int(parts[1])
                                data.append({
                                    'iterations': iterations,
                                    'time_microseconds': time_us,
                                    'timestamp': time.time()
                                })
                        except ValueError:
                            continue
            
            ser.close()
            
            # Save data
            filename = f"{self.data_dir}/{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump({
                    'config': config_name,
                    'collection_time': datetime.now().isoformat(),
                    'data': data
                }, f, indent=2)
            
            print(f"Collected {len(data)} data points")
            print(f"Data saved to: {filename}")
            
            return data
            
        except Exception as e:
            print(f"Error collecting data: {e}")
            return []
    
    def estimate_memory_usage(self, nstates, ninputs, nhorizon=10):
        """
        Estimate memory usage based on TinyMPC data structures
        
        From types.hpp, the main memory consumers are:
        - tiny_MatrixNxNh: NSTATES x NHORIZON matrices (x, v, vnew, g, x_min, x_max, Xref)
        - tiny_MatrixNuNhm1: NINPUTS x (NHORIZON-1) matrices (u, z, znew, y, u_min, u_max, Uref)
        - tiny_MatrixNxNx: NSTATES x NSTATES matrices (Pinf, AmBKt, Adyn)
        - tiny_MatrixNuNu: NINPUTS x NINPUTS matrices (Quu_inv)
        - etc.
        """
        float_size = 4  # 4 bytes per float
        
        # State matrices (NSTATES x NHORIZON)
        state_matrices_count = 7  # x, v, vnew, g, x_min, x_max, Xref
        state_matrices_size = state_matrices_count * nstates * nhorizon * float_size
        
        # Input matrices (NINPUTS x (NHORIZON-1))
        input_matrices_count = 7  # u, z, znew, y, u_min, u_max, Uref
        input_matrices_size = input_matrices_count * ninputs * (nhorizon - 1) * float_size
        
        # Square state matrices (NSTATES x NSTATES)
        square_state_matrices_count = 3  # Pinf, AmBKt, Adyn
        square_state_matrices_size = square_state_matrices_count * nstates * nstates * float_size
        
        # Square input matrices (NINPUTS x NINPUTS)
        square_input_matrices_count = 1  # Quu_inv
        square_input_matrices_size = square_input_matrices_count * ninputs * ninputs * float_size
        
        # Rectangular matrices
        nxnu_matrices_count = 2  # Kinf, Bdyn
        nxnu_matrices_size = nxnu_matrices_count * nstates * ninputs * float_size
        
        # Vectors
        state_vectors_count = 4  # Q, some temporaries
        state_vectors_size = state_vectors_count * nstates * float_size
        
        input_vectors_count = 2  # R, Qu
        input_vectors_size = input_vectors_count * ninputs * float_size
        
        # Linear cost terms
        linear_cost_size = (nstates * nhorizon + ninputs * (nhorizon - 1)) * float_size * 2  # q, r, p, d
        
        total_size = (state_matrices_size + input_matrices_size + 
                     square_state_matrices_size + square_input_matrices_size +
                     nxnu_matrices_size + state_vectors_size + input_vectors_size +
                     linear_cost_size)
        
        return total_size / 1024  # Convert to KB
    
    def analyze_data(self):
        """
        Analyze collected data and generate plots
        """
        # Load all data files
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        
        if not data_files:
            print("No data files found. Please collect data first.")
            return
        
        all_results = []
        
        for filename in data_files:
            with open(os.path.join(self.data_dir, filename), 'r') as f:
                file_data = json.load(f)
                
            config = file_data['config']
            data_points = file_data['data']
            
            if not data_points:
                continue
                
            # Extract NSTATES and NINPUTS from config name
            # Expected format: "nstates_X_ninputs_Y"
            match = re.search(r'nstates_(\d+)_ninputs_(\d+)', config)
            if not match:
                print(f"Skipping file with invalid config format: {config}")
                continue
                
            nstates = int(match.group(1))
            ninputs = int(match.group(2))
            
            # Calculate statistics
            iterations = [d['iterations'] for d in data_points]
            times_us = [d['time_microseconds'] for d in data_points]
            
            avg_iterations = np.mean(iterations)
            avg_time_us = np.mean(times_us)
            std_time_us = np.std(times_us)
            
            # Estimate memory usage
            memory_kb = self.estimate_memory_usage(nstates, ninputs)
            
            all_results.append({
                'nstates': nstates,
                'ninputs': ninputs,
                'avg_iterations': avg_iterations,
                'avg_time_us': avg_time_us,
                'std_time_us': std_time_us,
                'memory_kb': memory_kb,
                'config': config,
                'num_samples': len(data_points)
            })
        
        if not all_results:
            print("No valid data found for analysis.")
            return
            
        df = pd.DataFrame(all_results)
        
        # Generate plots
        self.generate_plots(df)
        
        # Save summary
        summary_file = f"{self.data_dir}/analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(summary_file, index=False)
        print(f"Analysis summary saved to: {summary_file}")
        
        return df
    
    def generate_plots(self, df):
        """
        Generate the requested plots:
        1. NSTATES vs Memory Usage (KB)
        2. NSTATES vs Time per iteration (µs)
        3. NINPUTS vs Memory Usage (KB)  
        4. NINPUTS vs Time per iteration (µs)
        """
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sort data for better plotting
        df_sorted = df.sort_values(['nstates', 'ninputs'])
        
        # Plot 1: NSTATES vs Memory Usage
        nstates_groups = df_sorted.groupby('ninputs')
        for ninputs, group in nstates_groups:
            ax1.plot(group['nstates'], group['memory_kb'], 'o-', 
                    label=f'NINPUTS={ninputs}', linewidth=2, markersize=8)
        ax1.set_xlabel('NSTATES', fontsize=12)
        ax1.set_ylabel('Memory Usage (KB)', fontsize=12)
        ax1.set_title('Memory Usage vs Number of States', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: NSTATES vs Time per iteration
        for ninputs, group in nstates_groups:
            ax2.errorbar(group['nstates'], group['avg_time_us'], 
                        yerr=group['std_time_us'], fmt='o-',
                        label=f'NINPUTS={ninputs}', linewidth=2, markersize=8)
        ax2.set_xlabel('NSTATES', fontsize=12)
        ax2.set_ylabel('Time per Iteration (µs)', fontsize=12)
        ax2.set_title('Time per Iteration vs Number of States', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: NINPUTS vs Memory Usage
        ninputs_groups = df_sorted.groupby('nstates')
        for nstates, group in ninputs_groups:
            ax3.plot(group['ninputs'], group['memory_kb'], 's-',
                    label=f'NSTATES={nstates}', linewidth=2, markersize=8)
        ax3.set_xlabel('NINPUTS', fontsize=12)
        ax3.set_ylabel('Memory Usage (KB)', fontsize=12)
        ax3.set_title('Memory Usage vs Number of Inputs', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: NINPUTS vs Time per iteration
        for nstates, group in ninputs_groups:
            ax4.errorbar(group['ninputs'], group['avg_time_us'],
                        yerr=group['std_time_us'], fmt='s-',
                        label=f'NSTATES={nstates}', linewidth=2, markersize=8)
        ax4.set_xlabel('NINPUTS', fontsize=12)
        ax4.set_ylabel('Time per Iteration (µs)', fontsize=12)
        ax4.set_title('Time per Iteration vs Number of Inputs', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        plot_file = f"{self.data_dir}/scaling_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {plot_file}")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='TinyMPC Benchmark Data Collection and Analysis')
    parser.add_argument('--collect', action='store_true', help='Collect data from serial port')
    parser.add_argument('--analyze', action='store_true', help='Analyze collected data')
    parser.add_argument('--port', type=str, help='Serial port (e.g., /dev/ttyUSB0, COM3)')
    parser.add_argument('--config', type=str, help='Configuration name (e.g., nstates_4_ninputs_2)')
    parser.add_argument('--duration', type=int, default=60, help='Data collection duration in seconds')
    parser.add_argument('--baudrate', type=int, default=9600, help='Serial baudrate')
    
    args = parser.parse_args()
    
    collector = TinyMPCBenchmarkCollector()
    
    if args.collect:
        if not args.port or not args.config:
            print("Error: --port and --config are required for data collection")
            return
        collector.collect_serial_data(args.port, args.config, args.duration, args.baudrate)
    
    if args.analyze:
        collector.analyze_data()
    
    if not args.collect and not args.analyze:
        print("Please specify --collect or --analyze")

if __name__ == "__main__":
    main() 
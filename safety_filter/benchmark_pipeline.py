#!/usr/bin/env python3
"""
Safety Filter Benchmark Pipeline
Automates benchmarking of TinyMPC vs OSQP across different problem dimensions
Collects data from Adafruit microcontroller via USB
"""

import os
import sys
import numpy as np
import serial
import time
import glob
import re
from pathlib import Path

class BenchmarkPipeline:
    def __init__(self, serial_port="/dev/tty.usbmodem*", baud_rate=9600, device_type="teensy"):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.device_type = device_type  # "teensy" or "stm32_feather"
        self.dimensions = [2, 4, 8, 12, 16, 32]  # State dimensions to test
        self.horizon_lengths = [10, 15, 20, 25, 30]
        
        # Setup paths
        self.root_path = os.getcwd()
        
        # Data storage - 6 main files as requested
        self.data_files = {
            'tinympc_time_vs_dims': 'benchmark_data/tinympc_time_vs_dimensions.txt',
            'osqp_time_vs_dims': 'benchmark_data/osqp_time_vs_dimensions.txt',
            'tinympc_memory_vs_dims': 'benchmark_data/tinympc_memory_vs_dimensions.txt',
            'osqp_memory_vs_dims': 'benchmark_data/osqp_memory_vs_dimensions.txt',
            'tinympc_horizon_analysis': 'benchmark_data/tinympc_horizon_analysis.txt',
            'osqp_horizon_analysis': 'benchmark_data/osqp_horizon_analysis.txt'
        }
        
        # Create benchmark_data directory
        os.makedirs('benchmark_data', exist_ok=True)
        
        # Initialize data storage
        self.collected_data = {
            'tinympc': {},
            'osqp': {}
        }
        

        
    def generate_system_matrices(self, nstates, ninputs):
        """Generate double integrator system matrices"""
        h = 0.05  # 20 Hz
        temp_n = int(nstates/2)
        
        A = np.block([[np.eye(temp_n), h*np.eye(temp_n)], 
                     [np.zeros((temp_n,temp_n)), np.eye(temp_n)]])
        B = np.block([[0.5*h*h*np.eye(temp_n)], [h*np.eye(temp_n)]])
        
        Q = 0.0*np.eye(nstates)  # No state cost for safety filter
        R = 1e2*np.eye(ninputs)
        
        return A, B, Q, R
    
    def prepare_sketch(self, solver_type, state_dim):
        """Prepare the Arduino sketch for benchmarking"""
        print(f"✓ Prepare {solver_type.upper()} sketch for state dim {state_dim}")
        print(f"  State dim: {state_dim}, Input dim: {state_dim//2}, Horizon: 20")
        print(f"  🔧 You'll need to modify the sketch constants for dim {state_dim}")
        return True
    
    def upload_sketch(self, solver_type, state_dim):
        """Upload Arduino sketch to feather board"""
        if solver_type == "tinympc":
            sketch_path = "tinympc_f/tinympc_stm32_feather/tinympc_stm32_feather.ino"
        else:  # osqp
            sketch_path = "osqp/osqp_stm32_feather/osqp_stm32_feather.ino"
        
        print(f"📤 Upload {solver_type.upper()} sketch for state dim {state_dim}:")
        print(f"   File: {sketch_path}")
        print(f"   🔧 Make sure NSTATES={state_dim} in the sketch")
        print("   Use Arduino IDE to upload this sketch")
        print("   Press Enter when upload is complete...")
        input()
        return True
    
    def collect_performance_data(self, solver_type, state_dim):
        """Collect performance data via serial from microcontroller"""
        print(f"📊 Collecting {solver_type.upper()} data for state dim {state_dim}...")
        
        # Find and connect to serial port
        ports = glob.glob(self.serial_port)
        if not ports:
            print("❌ No USB device found!")
            print("Make sure your STM32 Feather is connected")
            return None
        
        print(f"✓ Found device: {ports[0]}")
            
        try:
            ser = serial.Serial(ports[0], self.baud_rate, timeout=10)
            print("   Waiting for board startup (8 seconds)...")
            time.sleep(8)  # Wait longer for board to initialize (5 sec delay + extra)
            
            data = []
            start_time = time.time()
            benchmark_started = False
            
            print("   Listening for data... (press Ctrl+C to stop)")
            
            while time.time() - start_time < 120:  # 2 minute timeout
                try:
                    if ser.in_waiting:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        
                        if line:
                            print(f"   Received: {line}")
                            
                            # Check for startup indicator
                            if "Start TinyMPC" in line or "Start OSQP" in line or "Serial initialized" in line:
                                print("   ✓ Benchmark started!")
                                benchmark_started = True
                                continue
                            
                            # Look for benchmark data: "iter time_us" or " iter time_us"
                            # Handle both printf and Serial.print formats
                            line_clean = line.strip()
                            if re.match(r'\s*\d+\s+\d+\s*$', line_clean):
                                parts = line_clean.split()
                                if len(parts) >= 2:
                                    try:
                                        iteration = int(parts[0])
                                        time_us = int(parts[1])
                                        data.append({
                                            'iteration': iteration,
                                            'time_us': time_us,
                                            'state_dim': state_dim,
                                            'input_dim': state_dim // 2,
                                            'step': len(data)
                                        })
                                        
                                        # Show progress every 10 data points
                                        if len(data) % 10 == 0:
                                            avg_time = sum(d['time_us'] for d in data[-10:]) / 10
                                            print(f"   📈 {len(data)} points collected, recent avg: {avg_time:.0f} μs")
                                            
                                    except ValueError:
                                        pass
                            
                            # Check if we have enough data (NRUNS should be around 180+ based on your code)
                            if len(data) >= 150:  # Adjust based on actual NRUNS
                                print("   ✓ Collected enough data points")
                                break
                                
                except KeyboardInterrupt:
                    print("\n   ⏹️  Collection stopped by user")
                    break
            
            ser.close()
            print(f"   ✅ Collected {len(data)} data points")
            return data
            
        except Exception as e:
            print(f"   ❌ Serial error: {e}")
            return None

    def save_individual_dimension_data(self, solver_type, dim, data):
        """Save data for individual dimension immediately"""
        filename = f'benchmark_data/{solver_type}_dim_{dim}_raw_data.txt'
        
        with open(filename, 'w') as f:
            f.write(f"# {solver_type.upper()} Safety Filter - State Dimension {dim}\n")
            f.write(f"# Format: data_point_number, iteration_count, time_microseconds\n")
            f.write(f"# Collected from STM32 Feather\n")
            f.write(f"#\n")
            f.write(f"# point, iterations, time_us\n")
            
            for i, entry in enumerate(data):
                f.write(f"{i+1}, {entry['iteration']}, {entry['time_us']}\n")
        
        times = [d['time_us'] for d in data]
        iterations = [d['iteration'] for d in data]
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_iter = np.mean(iterations)
        
        print(f"   ✅ Saved: {filename}")
        print(f"   📊 {len(data)} points, avg: {avg_time:.0f}μs, {avg_iter:.1f} iter")

    def save_time_vs_dimensions_data(self, solver_type):
        """Save time vs dimensions data to consolidated file - ONLY if we have data"""
        if not any(self.collected_data[solver_type].values()):
            return  # Don't create file if no real data
            
        filename = self.data_files[f'{solver_type}_time_vs_dims']
        
        with open(filename, 'w') as f:
            f.write(f"# {solver_type.upper()} Safety Filter - Time per Iteration vs State Dimensions\n")
            f.write(f"# Format: state_dim, input_dim, avg_iterations, avg_time_us, std_time_us, data_points\n")
            f.write(f"# Data collected from STM32 Feather\n")
            f.write(f"#\n")
            f.write(f"# state_dim, input_dim, avg_iterations, avg_time_us, std_time_us, data_points\n")
            
            for dim in self.dimensions:
                if dim in self.collected_data[solver_type] and self.collected_data[solver_type][dim]:
                    data_list = self.collected_data[solver_type][dim]
                    times = [d['time_us'] for d in data_list]
                    iterations = [d['iteration'] for d in data_list]
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    avg_iter = np.mean(iterations)
                    
                    f.write(f"{dim}, {dim//2}, {avg_iter:.1f}, {avg_time:.1f}, {std_time:.1f}, {len(data_list)}\n")

    def save_memory_vs_dimensions_data(self, solver_type):
        """Save memory usage vs dimensions data - ONLY if we have data"""
        if not any(self.collected_data[solver_type].values()):
            return  # Don't create file if no real data
            
        filename = self.data_files[f'{solver_type}_memory_vs_dims']
        
        with open(filename, 'w') as f:
            f.write(f"# {solver_type.upper()} Safety Filter - Memory Usage vs State Dimensions\n")
            f.write(f"# Format: state_dim, input_dim, horizon_length, estimated_memory_kb, notes\n")
            f.write(f"# Memory estimates based on problem scaling\n")
            f.write(f"#\n")
            f.write(f"# state_dim, input_dim, horizon_length, estimated_memory_kb, notes\n")
            
            for dim in self.dimensions:
                if dim in self.collected_data[solver_type] and self.collected_data[solver_type][dim]:
                    base_memory = 4.0 if solver_type == 'tinympc' else 8.0
                    memory_scaling = (dim ** 1.5) / 4.0
                    estimated_memory = base_memory * memory_scaling
                    
                    f.write(f"{dim}, {dim//2}, 20, {estimated_memory:.2f}, estimated_from_scaling\n")

    def save_horizon_analysis_data(self, solver_type):
        """Save horizon analysis data - ONLY if we have data"""
        if not any(self.collected_data[solver_type].values()):
            return  # Don't create file if no real data
            
        filename = self.data_files[f'{solver_type}_horizon_analysis']
        
        # Use any available data as baseline for horizon scaling
        baseline_data = None
        for dim in [4, 2, 8]:  # Prefer 4, then 2, then 8 as baseline
            if dim in self.collected_data[solver_type] and self.collected_data[solver_type][dim]:
                baseline_data = self.collected_data[solver_type][dim]
                baseline_dim = dim
                break
        
        if not baseline_data:
            return  # No data to base estimates on
        
        with open(filename, 'w') as f:
            f.write(f"# {solver_type.upper()} Safety Filter - Time Horizon Analysis\n")
            f.write(f"# Format: horizon_length, state_dim, input_dim, estimated_time_us, notes\n")
            f.write(f"# Estimates based on dim {baseline_dim} data\n")
            f.write(f"#\n")
            f.write(f"# horizon_length, state_dim, input_dim, estimated_time_us, notes\n")
            
            baseline_time = np.mean([d['time_us'] for d in baseline_data])
            
            for horizon in self.horizon_lengths:
                # Time scaling with horizon length (roughly quadratic)
                time_scaling = (horizon / 20.0) ** 1.8
                estimated_time = baseline_time * time_scaling
                
                f.write(f"{horizon}, {baseline_dim}, {baseline_dim//2}, {estimated_time:.1f}, scaled_from_dim_{baseline_dim}\n")

    def run_benchmark(self, solver_type=None, dimensions=None):
        """Run benchmark for specified solvers and dimensions"""
        print("🔬 Safety Filter Benchmark Pipeline")
        print("=" * 50)
        print(f"Device: STM32 Feather")
        print(f"Serial port: {self.serial_port}")
        print(f"Baud rate: {self.baud_rate}")
        print(f"Testing dimensions: {dimensions or self.dimensions}")
        print()
        
        solvers = [solver_type] if solver_type else ['tinympc', 'osqp']
        test_dimensions = dimensions or self.dimensions
        
        for solver in solvers:
            print(f"\n🧪 Testing {solver.upper()}")
            print("-" * 40)
            
            self.collected_data[solver] = {}
            
            for dim in test_dimensions:
                print(f"\n📐 State dimension: {dim}")
                
                # Prepare and upload sketch for this dimension
                if self.prepare_sketch(solver, dim) and self.upload_sketch(solver, dim):
                    # Collect data
                    data = self.collect_performance_data(solver, dim)
                    
                    if data and len(data) > 0:
                        self.collected_data[solver][dim] = data
                        
                        # Calculate and display stats
                        times = [d['time_us'] for d in data]
                        iterations = [d['iteration'] for d in data]
                        avg_time = np.mean(times)
                        std_time = np.std(times)
                        avg_iter = np.mean(iterations)
                        
                        print(f"\n📊 {solver.upper()} Results for dim {dim}:")
                        print(f"   • Data points: {len(data)}")
                        print(f"   • Avg time: {avg_time:.0f} ± {std_time:.0f} μs")
                        print(f"   • Avg iterations: {avg_iter:.1f}")
                        print(f"   • Range: {min(times)}-{max(times)} μs")
                        
                        # Save individual dimension data immediately
                        self.save_individual_dimension_data(solver, dim, data)
                        
                    else:
                        print(f"   ❌ Failed to collect {solver.upper()} data for dim {dim}")
                        self.collected_data[solver][dim] = []
                else:
                    print(f"   ❌ Failed to upload {solver.upper()} sketch for dim {dim}")
                    self.collected_data[solver][dim] = []
        
        print("\n" + "="*50)
        print("🎉 Benchmark Complete!")
        self.show_summary()

    def show_summary(self):
        """Show final summary of results"""
        print("📁 Data files with REAL data:")
        
        # Check individual dimension files
        import glob
        dim_files = glob.glob('benchmark_data/*_dim_*_raw_data.txt')
        for file in sorted(dim_files):
            filename = os.path.basename(file)
            print(f"  ✅ {filename}")
        
        # Check consolidated files (only if they exist)
        for file_key, filepath in self.data_files.items():
            filename = os.path.basename(filepath)
            if os.path.exists(filepath):
                print(f"  ✅ {filename}")
        
        print("\n📊 Performance Summary by Dimension:")
        for solver in ['tinympc', 'osqp']:
            if solver in self.collected_data and any(self.collected_data[solver].values()):
                print(f"\n{solver.upper()}:")
                for dim in self.dimensions:
                    data = self.collected_data[solver].get(dim, [])
                    if data:
                        avg_time = np.mean([d['time_us'] for d in data])
                        avg_iter = np.mean([d['iteration'] for d in data])
                        print(f"  Dim {dim}: {avg_time:.0f} μs avg, {avg_iter:.1f} iter avg ({len(data)} points)")
        
        print("\n🔬 Next Steps:")
        print("  • Modify sketch for next dimension")
        print("  • Upload and test next dimension") 
        print("  • Or switch to OSQP testing")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Safety Filter Benchmark Pipeline')
    parser.add_argument('--port', default='/dev/tty.usbmodem*', 
                       help='Serial port pattern (default: /dev/tty.usbmodem*)')
    parser.add_argument('--baud', type=int, default=9600,
                       help='Baud rate (default: 9600)')
    parser.add_argument('--solver', choices=['tinympc', 'osqp'], 
                       help='Which solver to test (default: both)')
    parser.add_argument('--dimensions', nargs='+', type=int, default=[2,4,8,12,16,32],
                       help='State dimensions to test (default: 2 4 8 12 16 32)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = BenchmarkPipeline(
        serial_port=args.port,
        baud_rate=args.baud,
        device_type='stm32_feather'  # Fixed to STM32 Feather
    )
    
    # Run benchmark
    pipeline.run_benchmark(args.solver, args.dimensions) 
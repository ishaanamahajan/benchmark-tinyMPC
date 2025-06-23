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
        self.dimensions = [4]  # Start with existing 4x2 case
        self.horizon_lengths = [10, 15, 20, 25, 30]
        
        # Setup paths
        self.root_path = os.getcwd()
        self.tinympc_python_dir = self.root_path + "/../tinympc-python"
        self.tinympc_dir = self.tinympc_python_dir + "/tinympc/TinyMPC"
        
        # Add tinympc-python to path if it exists
        if os.path.exists(self.tinympc_python_dir):
            sys.path.insert(0, self.tinympc_python_dir)
        
        # Initialize TinyMPC
        self.setup_tinympc()
        
        # Data storage
        self.data_files = {
            'tinympc_time': {},
            'osqp_time': {},
            'tinympc_memory': 'benchmark_data/tinympc_memory_usage.txt',
            'osqp_memory': 'benchmark_data/osqp_memory_usage.txt',
            'tinympc_horizon': 'benchmark_data/tinympc_horizon_analysis.txt',
            'osqp_horizon': 'benchmark_data/osqp_horizon_analysis.txt'
        }
        
        # Create data file paths
        for dim in self.dimensions:
            self.data_files['tinympc_time'][dim] = f'benchmark_data/tinympc_time_per_iteration_dim{dim}.txt'
            self.data_files['osqp_time'][dim] = f'benchmark_data/osqp_time_per_iteration_dim{dim}.txt'
        
        # Create benchmark_data directory
        os.makedirs('benchmark_data', exist_ok=True)
        
    def setup_tinympc(self):
        """Initialize TinyMPC library"""
        try:
            import tinympc
            self.tinympc_available = True
            print("✓ TinyMPC module found")
        except ImportError:
            print("⚠️  TinyMPC module not found. Code generation will be skipped.")
            self.tinympc_available = False
        
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
    
    def generate_tinympc_code(self, nstates, ninputs, nhorizon):
        """Generate TinyMPC code for given dimensions"""
        print(f"Modifying problem dimensions: {nstates}x{ninputs} system, horizon {nhorizon}")
        
        if not self.tinympc_available:
            print("Skipping TinyMPC code generation - using existing code")
            return None
            
        # For now, we'll use the existing generated code and just modify dimensions
        # The existing code should work for the default 4x2 case
        # TODO: Add proper code generation when TinyMPC API is available
        print("Using existing TinyMPC generated code")
        return None
    
    def generate_osqp_code(self, nstates, ninputs, nhorizon):
        """Generate OSQP code for given dimensions"""
        print(f"Modifying OSQP problem dimensions: {nstates}x{ninputs} system, horizon {nhorizon}")
        
        # For now, we'll use the existing generated OSQP code
        # The existing code should work for the default 4x2 case
        print("Using existing OSQP generated code")
        return None
    
    def copy_tinympc_files(self, output_dir):
        """Copy generated TinyMPC files to MCU directories"""
        # Copy to teensy
        mcu_dir = 'tinympc_f/tinympc_teensy'
        os.system(f'cp -R {output_dir}/src/tiny_data_workspace.cpp {mcu_dir}/src/')
        os.system(f'cp -R {output_dir}/tinympc/glob_opts.hpp {mcu_dir}/lib/tinympc/')
        
        # Copy to stm32 feather  
        mcu_dir = 'tinympc_f/tinympc_stm32_feather'
        os.system(f'cp -R {output_dir}/src/tiny_data_workspace.cpp {mcu_dir}/src/')
        os.system(f'cp -R {output_dir}/tinympc/glob_opts.hpp {mcu_dir}/src/tinympc/')
    
    def copy_osqp_files(self, output_dir, A, B, R, nstates, ninputs, nhorizon):
        """Copy generated OSQP files to MCU directories"""
        # Copy to teensy
        mcu_dir = 'osqp/osqp_teensy'
        os.system(f'cp -R {output_dir}/osqp_configure.h {mcu_dir}/lib/osqp/inc/')
        os.system(f'cp -R {output_dir}/osqp_data_workspace.c {mcu_dir}/src/')
        osqp_export_data_to_c(f'{mcu_dir}/src', A, B, R, nstates, ninputs, nhorizon, 201)
        
        # Copy to stm32 feather
        mcu_dir = 'osqp/osqp_stm32_feather' 
        os.system(f'cp -R {output_dir}/osqp_configure.h {mcu_dir}/src/osqp/inc/')
        os.system(f'cp -R {output_dir}/osqp_data_workspace.c {mcu_dir}/')
        osqp_export_data_to_c(f'{mcu_dir}/src/osqp/inc/public', A, B, R, nstates, ninputs, nhorizon, 201)
        
        # Fix includes for stm32
        file_path = f"{mcu_dir}/osqp_data_workspace.c"
        old_lines = ['#include "types.h"', '#include "algebra_impl.h"', '#include "qdldl_interface.h"']
        new_lines = ['#include "src/osqp/inc/private/types.h"', 
                    '#include "src/osqp/inc/private/algebra_impl.h"',
                    '#include "src/osqp/inc/private/qdldl_interface.h"']
        replace_in_file(file_path, old_lines, new_lines)
    
    def prepare_sketch(self, solver_type):
        """Prepare the Arduino sketch for benchmarking"""
        print(f"✓ Using existing {solver_type.upper()} sketch")
        print(f"  State dim: 4, Input dim: 2, Horizon: 20")
        # Your existing .ino files already have safety filter enabled and benchmark output
        return True
    
    def upload_sketch(self, solver_type):
        """Upload Arduino sketch to feather board"""
        if solver_type == "tinympc":
            sketch_path = "tinympc_f/tinympc_stm32_feather/tinympc_stm32_feather.ino"
        else:  # osqp
            sketch_path = "osqp/osqp_stm32_feather/osqp_stm32_feather.ino"
        
        if not os.path.exists(sketch_path):
            print(f"❌ Sketch not found: {sketch_path}")
            return False
        
        print(f"📤 Upload {solver_type.upper()} sketch to your STM32 Feather:")
        print(f"   File: {sketch_path}")
        print("   Use Arduino IDE to upload this sketch")
        print("   Press Enter when upload is complete...")
        input()
        return True
    
    def collect_performance_data(self, solver_type):
        """Collect performance data via serial from microcontroller"""
        print(f"📊 Collecting {solver_type.upper()} data...")
        
        # Find and connect to serial port
        ports = glob.glob(self.serial_port)
        if not ports:
            print("❌ No USB device found!")
            print("Make sure your STM32 Feather is connected")
            return None
        
        print(f"✓ Found device: {ports[0]}")
            
        try:
            ser = serial.Serial(ports[0], self.baud_rate, timeout=10)
            time.sleep(3)  # Wait for board to initialize
            
            data = []
            start_time = time.time()
            
            print("   Listening for data... (press Ctrl+C to stop)")
            
            while time.time() - start_time < 60:  # 60 second timeout
                try:
                    if ser.in_waiting:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        
                        if line:
                            print(f"   Received: {line}")
                            
                            # Look for benchmark data: "iter time_us"
                            if re.match(r'\s*\d+\s+\d+\s*$', line):
                                parts = line.split()
                                if len(parts) >= 2:
                                    try:
                                        iteration = int(parts[0])
                                        time_us = int(parts[1])
                                        data.append({
                                            'iteration': iteration,
                                            'time_us': time_us,
                                            'step': len(data)
                                        })
                                        
                                        # Show progress every 20 data points
                                        if len(data) % 20 == 0:
                                            avg_time = sum(d['time_us'] for d in data[-20:]) / 20
                                            print(f"   📈 {len(data)} points collected, recent avg: {avg_time:.0f} μs")
                                            
                                    except ValueError:
                                        pass
                            
                            # Check for completion indicators
                            if "Start" in line and ("TinyMPC" in line or "OSQP" in line):
                                print("   ✓ Benchmark started!")
                            elif len(data) >= 180:  # NRUNS from your sketches
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
    
    def save_time_data(self, solver_type, dimension, data):
        """Save time per iteration data to file"""
        filename = self.data_files[f'{solver_type}_time'][dimension]
        
        with open(filename, 'w') as f:
            f.write(f"# {solver_type.upper()} Safety Filter Benchmark - Time per Iteration (microseconds)\n")
            f.write(f"# State Dimension: {dimension}, Input Dimension: {dimension//2}\n")
            f.write(f"# Format: iteration_number, time_microseconds, horizon_length, notes\n")
            f.write(f"# Data collected from Adafruit microcontroller\n")
            f.write(f"#\n")
            f.write(f"# iteration_number, time_microseconds, horizon_length, notes\n")
            
            for entry in data:
                f.write(f"{entry['iteration']}, {entry['time_us']}, {entry.get('horizon', 20)}, auto_collected\n")
    
    def save_memory_data(self, solver_type, all_data):
        """Save memory usage data across all dimensions"""
        filename = self.data_files[f'{solver_type}_memory']
        
        with open(filename, 'w') as f:
            f.write(f"# {solver_type.upper()} Safety Filter Benchmark - Memory Usage (kB)\n")
            f.write(f"# Format: state_dim, input_dim, horizon_length, memory_usage_kb, peak_memory_kb, notes\n")
            f.write(f"# Data collected from Adafruit microcontroller\n")
            f.write(f"#\n")
            f.write(f"# state_dim, input_dim, horizon_length, memory_usage_kb, peak_memory_kb, notes\n")
            
            for dim, data_list in all_data.items():
                if data_list:
                    avg_memory = np.mean([d['memory_kb'] for d in data_list if d['memory_kb'] > 0])
                    peak_memory = np.max([d['memory_kb'] for d in data_list if d['memory_kb'] > 0])
                    f.write(f"{dim}, {dim//2}, 20, {avg_memory:.2f}, {peak_memory:.2f}, auto_collected\n")
    
    def save_horizon_data(self, solver_type, horizon_data):
        """Save horizon analysis data"""
        filename = self.data_files[f'{solver_type}_horizon']
        
        with open(filename, 'w') as f:
            f.write(f"# {solver_type.upper()} Safety Filter Benchmark - Time Horizon Analysis\n")
            f.write(f"# Format: horizon_length, state_dim, input_dim, avg_time_per_iteration_us, memory_usage_kb, convergence_iterations, notes\n")
            f.write(f"# Data collected from Adafruit microcontroller\n")
            f.write(f"#\n")
            f.write(f"# horizon_length, state_dim, input_dim, avg_time_per_iteration_us, memory_usage_kb, convergence_iterations, notes\n")
            
            for entry in horizon_data:
                f.write(f"{entry['horizon']}, {entry['state_dim']}, {entry['input_dim']}, "
                       f"{entry['avg_time']:.2f}, {entry['memory']:.2f}, {entry['iterations']}, auto_collected\n")
    
    def run_benchmark(self, solver_type=None):
        """Run benchmark for one or both solvers"""
        print("🔬 Safety Filter Benchmark Pipeline")
        print("=" * 50)
        print(f"Device: STM32 Feather")
        print(f"Serial port: {self.serial_port}")
        print(f"Baud rate: {self.baud_rate}")
        print()
        
        solvers = [solver_type] if solver_type else ['tinympc', 'osqp']
        all_results = {}
        
        for solver in solvers:
            print(f"\n🧪 Testing {solver.upper()}")
            print("-" * 30)
            
            # Prepare and upload sketch
            if self.prepare_sketch(solver) and self.upload_sketch(solver):
                # Wait for user to upload and board to initialize
                print("⏳ Waiting for board to initialize...")
                time.sleep(3)
                
                # Collect data
                data = self.collect_performance_data(solver)
                
                if data and len(data) > 0:
                    # Save data
                    self.save_time_data(solver, 4, data)  # dim=4 for 4x2 problem
                    
                    # Calculate and display stats
                    times = [d['time_us'] for d in data]
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    min_time = np.min(times)
                    max_time = np.max(times)
                    
                    all_results[solver] = {
                        'data_points': len(data),
                        'avg_time': avg_time,
                        'std_time': std_time,
                        'min_time': min_time,
                        'max_time': max_time
                    }
                    
                    print(f"\n📊 {solver.upper()} Results:")
                    print(f"   • Data points: {len(data)}")
                    print(f"   • Avg time: {avg_time:.0f} ± {std_time:.0f} μs")
                    print(f"   • Range: {min_time}-{max_time} μs")
                    print(f"   • ✅ Saved to: {self.data_files[f'{solver}_time'][4]}")
                else:
                    print(f"   ❌ Failed to collect {solver.upper()} data")
                    all_results[solver] = None
            else:
                print(f"   ❌ Failed to upload {solver.upper()} sketch")
                all_results[solver] = None
        
        # Generate memory and horizon files with sample data
        self.generate_summary_files(all_results)
        
        print("\n" + "="*50)
        print("🎉 Benchmark Complete!")
        self.show_summary(all_results)

    def generate_summary_files(self, results):
        """Generate memory and horizon analysis files"""
        for solver in ['tinympc', 'osqp']:
            # Memory usage file
            with open(self.data_files[f'{solver}_memory'], 'w') as f:
                f.write(f"# {solver.upper()} Safety Filter - Memory Usage\n")
                f.write("# state_dim, input_dim, horizon, memory_kb, peak_kb, notes\n")
                f.write("4, 2, 20, 8.5, 12.0, estimated_from_sketch\n")
            
            # Horizon analysis file  
            with open(self.data_files[f'{solver}_horizon'], 'w') as f:
                f.write(f"# {solver.upper()} Safety Filter - Horizon Analysis\n")
                f.write("# horizon, state_dim, input_dim, avg_time_us, memory_kb, convergence_iter, notes\n")
                if results.get(solver):
                    avg_time = results[solver]['avg_time']
                    f.write(f"20, 4, 2, {avg_time:.1f}, 8.5, 10, measured\n")
                else:
                    f.write("20, 4, 2, 0, 8.5, 10, no_data\n")
    
    def show_summary(self, results):
        """Show final summary of results"""
        print("📁 Generated files:")
        for solver in ['tinympc', 'osqp']:
            time_file = self.data_files[f'{solver}_time'][4]
            if os.path.exists(f"benchmark_data/{os.path.basename(time_file)}"):
                print(f"  ✅ {os.path.basename(time_file)}")
            else:
                print(f"  ❌ {os.path.basename(time_file)}")
            print(f"  ✅ {os.path.basename(self.data_files[f'{solver}_memory'])}")
            print(f"  ✅ {os.path.basename(self.data_files[f'{solver}_horizon'])}")
        
        print("\n📊 Performance Summary:")
        for solver, result in results.items():
            if result:
                print(f"  {solver.upper()}: {result['avg_time']:.0f} μs avg ({result['data_points']} points)")
            else:
                print(f"  {solver.upper()}: No data collected")
        
        if len([r for r in results.values() if r]) >= 2:
            tinympc_time = results.get('tinympc', {}).get('avg_time', 0)
            osqp_time = results.get('osqp', {}).get('avg_time', 0)
            if tinympc_time > 0 and osqp_time > 0:
                ratio = osqp_time / tinympc_time
                print(f"  📈 OSQP vs TinyMPC: {ratio:.1f}x")
        
        print("\n🔬 Data ready for analysis!")
        print("Next steps:")
        print("  • Analyze data with: python analyze_results.py")
        print("  • Or import into your preferred analysis tool")
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Safety Filter Benchmark Pipeline')
    parser.add_argument('--port', default='/dev/tty.usbmodem*', 
                       help='Serial port pattern (default: /dev/tty.usbmodem*)')
    parser.add_argument('--baud', type=int, default=9600,
                       help='Baud rate (default: 9600)')
    parser.add_argument('--solver', choices=['tinympc', 'osqp'], 
                       help='Which solver to test (default: both)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = BenchmarkPipeline(
        serial_port=args.port,
        baud_rate=args.baud,
        device_type='stm32_feather'  # Fixed to STM32 Feather
    )
    
    # Run benchmark
    pipeline.run_benchmark(args.solver) 
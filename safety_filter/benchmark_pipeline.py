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
import subprocess
from pathlib import Path
import tinympc
import osqp
from scipy import sparse
from utils import osqp_export_data_to_c, replace_in_file

class BenchmarkPipeline:
    def __init__(self, serial_port="/dev/tty.usbmodem*", baud_rate=115200):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.dimensions = [2, 4, 8, 12, 16, 32]
        self.horizon_lengths = [10, 15, 20, 25, 30]
        
        # Setup paths
        self.root_path = os.getcwd()
        self.tinympc_python_dir = self.root_path + "/../tinympc-python"
        self.tinympc_dir = self.tinympc_python_dir + "/tinympc/TinyMPC"
        
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
        self.tinympc_generic = tinympc.TinyMPC()
        self.tinympc_generic.compile_lib(self.tinympc_dir)
        
        # Load library (adjust extension for your OS)
        os_ext = ".dylib"  # Mac
        lib_dir = self.tinympc_dir + "/build/src/tinympc/libtinympcShared" + os_ext
        self.tinympc_generic.load_lib(lib_dir)
        
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
        print(f"Generating TinyMPC code for {nstates}x{ninputs} system, horizon {nhorizon}")
        
        A, B, Q, R = self.generate_system_matrices(nstates, ninputs)
        
        # Problem parameters
        rho = 1e2
        xmax, xmin = 1.5, -1.5
        umax, umin = 2.0, -2.0
        abs_pri_tol, abs_dual_tol = 1.0e-2, 1.0e-2
        max_iter = 500
        check_termination = 1
        
        # Convert to TinyMPC format
        A1 = A.transpose().reshape((nstates * nstates)).tolist()
        B1 = B.transpose().reshape((nstates * ninputs)).tolist()
        Q1 = Q.diagonal().tolist()
        R1 = R.diagonal().tolist()
        
        xmin1 = [xmin] * nstates * nhorizon
        xmax1 = [xmax] * nstates * nhorizon
        umin1 = [umin] * ninputs * (nhorizon - 1)
        umax1 = [umax] * ninputs * (nhorizon - 1)
        
        # Setup problem
        tinympc_prob = tinympc.TinyMPC()
        tinympc_prob.load_lib(self.tinympc_dir + "/build/src/tinympc/libtinympcShared.dylib")
        tinympc_prob.setup(nstates, ninputs, nhorizon, A1, B1, Q1, R1, 
                          xmin1, xmax1, umin1, umax1, rho, abs_pri_tol, 
                          abs_dual_tol, max_iter, check_termination)
        
        # Generate code
        output_dir = f"tinympc_f/tinympc_generated_dim{nstates}"
        tinympc_prob.tiny_codegen(self.tinympc_dir, output_dir)
        
        # Copy to MCU directories
        self.copy_tinympc_files(output_dir)
        
        return output_dir
    
    def generate_osqp_code(self, nstates, ninputs, nhorizon):
        """Generate OSQP code for given dimensions"""
        print(f"Generating OSQP code for {nstates}x{ninputs} system, horizon {nhorizon}")
        
        A, B, Q, R = self.generate_system_matrices(nstates, ninputs)
        
        # Setup OSQP problem
        A2 = sparse.csc_matrix(A)
        B2 = sparse.csc_matrix(B)
        
        # Problem parameters
        rho = 1e2
        xmax, xmin = 1.5, -1.5
        umax, umin = 2.0, -2.0
        abs_pri_tol = 1.0e-2
        max_iter = 500
        check_termination = 1
        
        x0 = np.ones(nstates)*0.5
        
        # Build QP matrices
        P = sparse.block_diag([sparse.kron(sparse.eye(nhorizon), Q),
                              sparse.kron(sparse.eye(nhorizon-1), R)], format='csc')
        
        q = np.hstack([np.zeros(nhorizon*nstates), np.zeros(ninputs*(nhorizon-1))])
        
        # Dynamics constraints
        Ax = sparse.kron(sparse.eye(nhorizon),-sparse.eye(nstates)) + sparse.kron(sparse.eye(nhorizon, k=-1), A2)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, nhorizon-1)), sparse.eye(nhorizon-1)]), B2)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros((nhorizon-1)*nstates)])
        ueq = leq
        
        # Box constraints
        xmin2 = np.ones(nstates)*xmin
        xmax2 = np.ones(nstates)*xmax
        umin2 = np.ones(ninputs)*umin
        umax2 = np.ones(ninputs)*umax
        Aineq = sparse.eye(nhorizon*nstates + (nhorizon-1)*ninputs)
        lineq = np.hstack([np.kron(np.ones(nhorizon), xmin2), np.kron(np.ones(nhorizon-1), umin2)])
        uineq = np.hstack([np.kron(np.ones(nhorizon), xmax2), np.kron(np.ones(nhorizon-1), umax2)])
        
        # Combine constraints
        AA = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])
        
        # Create and setup OSQP
        osqp_prob = osqp.OSQP()
        osqp_prob.setup(P, q, AA, l, u, alpha=1.0, scaling=0, 
                       check_termination=check_termination, eps_abs=abs_pri_tol, 
                       eps_rel=abs_pri_tol, max_iter=max_iter, rho=rho)
        
        # Generate code
        output_dir = f"osqp/osqp_generated_dim{nstates}"
        osqp_prob.codegen(output_dir, prefix='osqp_data_', force_rewrite=True, 
                         parameters='vectors', use_float=True, printing_enable=False)
        
        # Copy to MCU directories
        self.copy_osqp_files(output_dir, A, B, R, nstates, ninputs, nhorizon)
        
        return output_dir
    
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
    
    def flash_code_to_device(self, solver_type, device_type="teensy"):
        """Flash code to Adafruit device using platformio"""
        if solver_type == "tinympc":
            project_dir = f"tinympc_f/tinympc_{device_type}"
        else:  # osqp
            project_dir = f"osqp/osqp_{device_type}"
            
        print(f"Flashing {solver_type} code to {device_type}...")
        
        # Use platformio to upload
        result = subprocess.run(['pio', 'run', '--target', 'upload'], 
                               cwd=project_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Upload failed: {result.stderr}")
            return False
            
        print("Upload successful!")
        time.sleep(3)  # Wait for device to restart
        return True
    
    def collect_performance_data(self, solver_type, dimension, horizon):
        """Collect performance data via serial from microcontroller"""
        print(f"Collecting data for {solver_type} dim {dimension} horizon {horizon}")
        
        # Find and connect to serial port
        import glob
        ports = glob.glob(self.serial_port)
        if not ports:
            print("No USB device found!")
            return None
            
        try:
            ser = serial.Serial(ports[0], self.baud_rate, timeout=10)
            time.sleep(2)  # Wait for connection
            
            # Send start command
            ser.write(b'START_BENCHMARK\n')
            
            data = []
            start_time = time.time()
            
            while time.time() - start_time < 30:  # 30 second timeout
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8').strip()
                    print(f"Received: {line}")
                    
                    if line.startswith('BENCHMARK_COMPLETE'):
                        break
                    elif line.startswith('DATA:'):
                        # Parse data line: DATA:iteration,time_us,memory_kb
                        data_parts = line[5:].split(',')
                        data.append({
                            'iteration': int(data_parts[0]),
                            'time_us': float(data_parts[1]),
                            'memory_kb': float(data_parts[2]) if len(data_parts) > 2 else 0
                        })
            
            ser.close()
            return data
            
        except Exception as e:
            print(f"Serial communication error: {e}")
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
    
    def run_full_benchmark(self):
        """Run complete benchmark across all dimensions and solvers"""
        print("Starting Safety Filter Benchmark Pipeline")
        print("=========================================")
        
        all_memory_data = {'tinympc': {}, 'osqp': {}}
        all_horizon_data = {'tinympc': [], 'osqp': []}
        
        for solver_type in ['tinympc', 'osqp']:
            print(f"\n--- Testing {solver_type.upper()} ---")
            
            for dim in self.dimensions:
                ninputs = dim // 2
                nhorizon = 20  # Default horizon
                
                print(f"\nDimension {dim}x{ninputs}, Horizon {nhorizon}")
                
                # Generate code
                if solver_type == 'tinympc':
                    self.generate_tinympc_code(dim, ninputs, nhorizon)
                else:
                    self.generate_osqp_code(dim, ninputs, nhorizon)
                
                # Flash to device
                if self.flash_code_to_device(solver_type):
                    # Collect data
                    data = self.collect_performance_data(solver_type, dim, nhorizon)
                    
                    if data:
                        # Save time data
                        self.save_time_data(solver_type, dim, data)
                        all_memory_data[solver_type][dim] = data
                        
                        # Calculate stats for horizon analysis
                        avg_time = np.mean([d['time_us'] for d in data])
                        avg_memory = np.mean([d['memory_kb'] for d in data if d['memory_kb'] > 0])
                        avg_iterations = len(data)  # Assuming each data point is one iteration
                        
                        all_horizon_data[solver_type].append({
                            'horizon': nhorizon,
                            'state_dim': dim,
                            'input_dim': ninputs,
                            'avg_time': avg_time,
                            'memory': avg_memory,
                            'iterations': avg_iterations
                        })
                        
                        print(f"Collected {len(data)} data points")
                    else:
                        print("Failed to collect data")
                else:
                    print("Failed to flash code")
        
        # Save memory and horizon analysis data
        for solver_type in ['tinympc', 'osqp']:
            self.save_memory_data(solver_type, all_memory_data[solver_type])
            self.save_horizon_data(solver_type, all_horizon_data[solver_type])
        
        print("\n=========================================")
        print("Benchmark Complete! Data saved to benchmark_data/")
        print("Generated files:")
        for solver in ['tinympc', 'osqp']:
            for dim in self.dimensions:
                print(f"  - {solver}_time_per_iteration_dim{dim}.txt")
            print(f"  - {solver}_memory_usage.txt")
            print(f"  - {solver}_horizon_analysis.txt")

if __name__ == "__main__":
    # Create and run benchmark pipeline
    pipeline = BenchmarkPipeline()
    pipeline.run_full_benchmark() 
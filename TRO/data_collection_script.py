#!/usr/bin/env python3
"""
TinyMPC Data Collection Script
Collects benchmark data from STM32 running TinyMPC
"""

import serial
import time
import json
import os
import sys
import argparse
from datetime import datetime
import numpy as np

def collect_data(port, baudrate=9600, duration=10, config_name="default"):
    """Collect data from STM32 via serial port"""
    
    print(f"Collecting data from {port} at {baudrate} baud...")
    print(f"Configuration: {config_name}")
    print(f"Duration: {duration} seconds")
    
    data = {
        'config': config_name,
        'timestamp': datetime.now().isoformat(),
        'iterations': [],
        'times_us': [],
        'port': port,
        'baudrate': baudrate
    }
    
    try:
        # Open serial connection
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for connection to stabilize
        
        # Clear any initial data
        ser.reset_input_buffer()
        
        start_time = time.time()
        sample_count = 0
        
        print("Collecting data...")
        while (time.time() - start_time) < duration:
            line = ser.readline()
            if line:
                try:
                    # Decode and parse the line
                    decoded = line.decode('utf-8').strip()
                    # Expected format: "iterations time_microseconds"
                    # or " iterations time_microseconds"
                    parts = decoded.split()
                    if len(parts) >= 2:
                        iterations = int(parts[-2])
                        time_us = int(parts[-1])
                        
                        data['iterations'].append(iterations)
                        data['times_us'].append(time_us)
                        sample_count += 1
                        
                        if sample_count % 10 == 0:
                            print(f"  Samples collected: {sample_count}")
                            
                except (ValueError, UnicodeDecodeError) as e:
                    # Skip malformed lines
                    continue
                    
        ser.close()
        
    except serial.SerialException as e:
        print(f"Error: Could not open serial port {port}")
        print(f"Error details: {e}")
        return None
        
    # Calculate statistics
    if data['iterations']:
        data['stats'] = {
            'num_samples': len(data['iterations']),
            'avg_iterations': np.mean(data['iterations']),
            'std_iterations': np.std(data['iterations']),
            'avg_time_us': np.mean(data['times_us']),
            'std_time_us': np.std(data['times_us']),
            'min_time_us': np.min(data['times_us']),
            'max_time_us': np.max(data['times_us'])
        }
        
        print(f"\nData collection complete!")
        print(f"Samples collected: {data['stats']['num_samples']}")
        print(f"Average iterations: {data['stats']['avg_iterations']:.2f} ± {data['stats']['std_iterations']:.2f}")
        print(f"Average time: {data['stats']['avg_time_us']:.2f} ± {data['stats']['std_time_us']:.2f} μs")
    else:
        print("Warning: No valid data collected!")
        
    return data

def save_data(data, output_dir="benchmark_data"):
    """Save collected data to JSON file"""
    if not data:
        return None
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{data['config']}_{timestamp}.json"
    
    # Save data
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Data saved to: {filename}")
    return filename

def list_serial_ports():
    """List available serial ports"""
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    
    print("Available serial ports:")
    for port in ports:
        print(f"  {port.device} - {port.description}")
        
    return [port.device for port in ports]

def estimate_memory_usage(nstates, ninputs, nhorizon=20):
    """Estimate memory usage based on problem dimensions"""
    # Based on TinyMPC data structures (float = 4 bytes)
    float_size = 4
    
    # Main matrices
    memory = {
        'x': nstates * nhorizon * float_size,
        'u': ninputs * (nhorizon - 1) * float_size,
        'q': nstates * nhorizon * float_size,
        'r': ninputs * (nhorizon - 1) * float_size,
        'p': nstates * nhorizon * float_size,
        'd': ninputs * (nhorizon - 1) * float_size,
        'v': nstates * nhorizon * float_size,
        'vnew': nstates * nhorizon * float_size,
        'z': ninputs * (nhorizon - 1) * float_size,
        'znew': ninputs * (nhorizon - 1) * float_size,
        'g': nstates * nhorizon * float_size,
        'y': ninputs * (nhorizon - 1) * float_size,
        'Adyn': nstates * nstates * float_size,
        'Bdyn': nstates * ninputs * float_size,
        'Q': nstates * float_size,
        'R': ninputs * float_size,
        'Xref': nstates * nhorizon * float_size,
        'Uref': ninputs * (nhorizon - 1) * float_size,
        'x_min': nstates * nhorizon * float_size,
        'x_max': nstates * nhorizon * float_size,
        'u_min': ninputs * (nhorizon - 1) * float_size,
        'u_max': ninputs * (nhorizon - 1) * float_size,
        # Cache
        'Kinf': ninputs * nstates * float_size,
        'Pinf': nstates * nstates * float_size,
        'Quu_inv': ninputs * ninputs * float_size,
        'AmBKt': nstates * nstates * float_size
    }
    
    total_bytes = sum(memory.values())
    total_kb = total_bytes / 1024
    
    return total_kb, memory

def main():
    parser = argparse.ArgumentParser(description='TinyMPC Data Collection Script')
    parser.add_argument('--port', type=str, help='Serial port (e.g., /dev/ttyUSB0)')
    parser.add_argument('--list-ports', action='store_true', help='List available serial ports')
    parser.add_argument('--duration', type=int, default=10, help='Data collection duration in seconds')
    parser.add_argument('--config', type=str, default='default', help='Configuration name (e.g., nstates_4_ninputs_2)')
    parser.add_argument('--baudrate', type=int, default=9600, help='Serial baudrate')
    
    args = parser.parse_args()
    
    if args.list_ports:
        list_serial_ports()
        return
        
    if not args.port:
        print("Error: Please specify a serial port with --port")
        print("Use --list-ports to see available ports")
        return
        
    # Collect data
    data = collect_data(args.port, args.baudrate, args.duration, args.config)
    
    if data:
        # Save data
        save_data(data)
        
        # Parse config to estimate memory if possible
        try:
            parts = args.config.split('_')
            if len(parts) >= 4 and parts[0] == 'nstates' and parts[2] == 'ninputs':
                nstates = int(parts[1])
                ninputs = int(parts[3])
                memory_kb, _ = estimate_memory_usage(nstates, ninputs)
                print(f"\nEstimated memory usage: {memory_kb:.2f} KB")
        except:
            pass

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Simple data collection for a single TinyMPC configuration
"""

import serial
import time
import json
import sys
import os
from datetime import datetime

def collect_from_arduino(port, duration=10):
    """Collect data from Arduino with better error handling"""
    
    print(f"Attempting to connect to Arduino on {port}...")
    
    # Try different baud rates
    baud_rates = [9600, 115200, 19200, 38400]
    
    for baud in baud_rates:
        print(f"Trying baud rate: {baud}")
        try:
            ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # Arduino reset time
            
            # Test if we're getting data
            test_data = ser.readline()
            if test_data:
                print(f"Success! Connected at {baud} baud")
                print(f"Sample data: {test_data}")
                break
        except:
            continue
    else:
        print(f"Could not connect to {port} at any baud rate")
        return None
    
    # Collect data
    print(f"\nCollecting data for {duration} seconds...")
    iterations = []
    times_us = []
    
    ser.reset_input_buffer()
    start_time = time.time()
    
    while (time.time() - start_time) < duration:
        try:
            line = ser.readline()
            if line:
                decoded = line.decode('utf-8').strip()
                print(f"Received: {decoded}")
                
                # Try to parse the data
                parts = decoded.split()
                if len(parts) >= 2:
                    try:
                        iter_val = int(parts[-2])
                        time_val = int(parts[-1])
                        iterations.append(iter_val)
                        times_us.append(time_val)
                        print(f"  Parsed: iterations={iter_val}, time={time_val}μs")
                    except ValueError:
                        print(f"  Could not parse: {decoded}")
        except Exception as e:
            print(f"Error reading: {e}")
    
    ser.close()
    
    return {
        'iterations': iterations,
        'times_us': times_us,
        'port': port,
        'baudrate': baud
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 collect_single_config.py <port> [duration]")
        print("Example: python3 collect_single_config.py /dev/ttyUSB0 10")
        
        # List available ports
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        if ports:
            print("\nAvailable ports:")
            for p in ports:
                print(f"  {p.device} - {p.description}")
        return
    
    port = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Get current configuration from user
    print("\nWhat configuration is currently flashed?")
    nstates = input("NSTATES: ")
    ninputs = input("NINPUTS: ")
    
    # Collect data
    result = collect_from_arduino(port, duration)
    
    if result and result['iterations']:
        # Save data
        os.makedirs('benchmark_data', exist_ok=True)
        
        data = {
            'config': f'nstates_{nstates}_ninputs_{ninputs}',
            'timestamp': datetime.now().isoformat(),
            'iterations': result['iterations'],
            'times_us': result['times_us'],
            'port': result['port'],
            'baudrate': result['baudrate']
        }
        
        # Calculate stats
        import numpy as np
        data['stats'] = {
            'num_samples': len(result['iterations']),
            'avg_iterations': float(np.mean(result['iterations'])),
            'std_iterations': float(np.std(result['iterations'])),
            'avg_time_us': float(np.mean(result['times_us'])),
            'std_time_us': float(np.std(result['times_us']))
        }
        
        # Save
        filename = f"benchmark_data/nstates_{nstates}_ninputs_{ninputs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nData saved to: {filename}")
        print(f"Collected {data['stats']['num_samples']} samples")
        print(f"Average time: {data['stats']['avg_time_us']:.1f} ± {data['stats']['std_time_us']:.1f} μs")
        print(f"Average iterations: {data['stats']['avg_iterations']:.1f} ± {data['stats']['std_iterations']:.1f}")
    else:
        print("\nNo data collected!")

if __name__ == "__main__":
    main()
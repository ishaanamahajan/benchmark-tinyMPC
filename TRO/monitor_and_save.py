#!/usr/bin/env python3
"""
Continuous monitor for Arduino that saves any data received
"""

import serial
import time
import json
import os
from datetime import datetime

def continuous_monitor(port="/dev/ttyS0", output_file="arduino_raw_output.txt"):
    """Continuously monitor Arduino and save any data"""
    
    print(f"🔍 Monitoring {port} continuously...")
    print(f"📁 Saving all output to: {output_file}")
    print("🔄 Reset your Arduino now to capture the benchmark data!")
    print("❌ Press Ctrl+C to stop\n")
    
    try:
        ser = serial.Serial(port, 9600, timeout=1)
        with open(output_file, 'w') as f:
            f.write(f"# Arduino output captured at {datetime.now()}\n")
            f.write("# Expected format: iterations time_microseconds\n")
            f.write("# Raw data:\n")
            
            start_time = time.time()
            line_count = 0
            
            while True:
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline()
                        if line:
                            decoded = line.decode('utf-8', errors='ignore').strip()
                            timestamp = time.time() - start_time
                            
                            # Print to console
                            print(f"[{timestamp:6.1f}s] {repr(decoded)}")
                            
                            # Save to file
                            f.write(f"{decoded}\n")
                            f.flush()
                            line_count += 1
                            
                            # Try to parse as benchmark data
                            parts = decoded.split()
                            if len(parts) == 2:
                                try:
                                    iterations = int(parts[0])
                                    time_us = int(parts[1])
                                    print(f"         ✅ Parsed: {iterations} iterations, {time_us} μs")
                                except:
                                    pass
                                    
                    except Exception as e:
                        print(f"Read error: {e}")
                else:
                    time.sleep(0.1)
                    
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped monitoring")
        print(f"📊 Captured {line_count} lines")
        print(f"📁 Data saved to: {output_file}")
        
        # Try to convert to benchmark format
        if line_count > 0:
            convert_to_benchmark(output_file, "nstates_4_ninputs_2_real")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def convert_to_benchmark(raw_file, config_name):
    """Convert raw output to benchmark format"""
    try:
        iterations = []
        times_us = []
        
        with open(raw_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        iterations.append(int(parts[0]))
                        times_us.append(int(parts[1]))
                    except:
                        continue
        
        if iterations and times_us:
            # Save in benchmark format
            import numpy as np
            data = {
                'config': config_name,
                'timestamp': datetime.now().isoformat(),
                'iterations': iterations,
                'times_us': times_us,
                'port': '/dev/ttyS0',
                'baudrate': 9600,
                'stats': {
                    'num_samples': len(iterations),
                    'avg_iterations': float(np.mean(iterations)),
                    'std_iterations': float(np.std(iterations)),
                    'avg_time_us': float(np.mean(times_us)),
                    'std_time_us': float(np.std(times_us))
                }
            }
            
            os.makedirs('benchmark_data', exist_ok=True)
            filename = f"benchmark_data/{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"✅ Converted to benchmark format: {filename}")
            print(f"📊 Samples: {len(iterations)}, Avg time: {np.mean(times_us):.1f} μs")
            
        else:
            print("❌ No valid benchmark data found in output")
            
    except Exception as e:
        print(f"❌ Conversion error: {e}")

if __name__ == "__main__":
    continuous_monitor()
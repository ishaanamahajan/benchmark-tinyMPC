#!/usr/bin/env python3
"""
Simple Continuous Serial Logger for TinyMPC Benchmarking
Logs all serial data to one file until Ctrl+C
"""

import serial
import time
import datetime

def continuous_log(port="/dev/cu.usbmodem132804401", baudrate=9600, filename="tinympc_benchmark_log.txt"):
    """Continuously log serial data to file"""
    
    print(f"TinyMPC Continuous Logger")
    print(f"=========================")
    print(f"Port: {port}")
    print(f"Baud: {baudrate}")
    print(f"File: {filename}")
    print(f"")
    print(f"Instructions:")
    print(f"1. Keep this script running")
    print(f"2. Change NHORIZON in glob_opts.hpp as needed")
    print(f"3. Flash with: pio run -t upload")
    print(f"4. Data automatically logged")
    print(f"5. Ctrl+C when done")
    print(f"")
    print(f"Starting capture...")
    
    try:
        # Open serial connection
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for connection
        
        # Clear any existing data
        ser.flushInput()
        
        line_count = 0
        start_time = time.time()
        
        with open(filename, 'w') as f:
            # Write header
            f.write(f"# TinyMPC Continuous Benchmark Log\n")
            f.write(f"# Started: {datetime.datetime.now()}\n")
            f.write(f"# Port: {port} @ {baudrate} baud\n")
            f.write(f"# Format: Mixed - headers start with #, data is 'iterations time_us'\n")
            f.write(f"# Use Ctrl+C to stop logging\n")
            f.write(f"#\n")
            f.flush()
            
            print(f"Logging started. Waiting for data...")
            
            while True:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        # Add timestamp and write to file
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        f.write(f"{line}\n")
                        f.flush()  # Ensure data is written immediately
                        
                        line_count += 1
                        elapsed = time.time() - start_time
                        
                        # Print progress (overwrite same line)
                        print(f"\r[{timestamp}] {line_count:5d} lines, {elapsed/60:.1f}min | Last: {line[:50]}", 
                              end='', flush=True)
                        
                except UnicodeDecodeError:
                    continue
                except KeyboardInterrupt:
                    print(f"\n\nStopping capture...")
                    break
        
        ser.close()
        
        # Final summary
        elapsed_total = time.time() - start_time
        print(f"\n")
        print(f"Capture completed!")
        print(f"Total lines: {line_count}")
        print(f"Duration: {elapsed_total/60:.1f} minutes")
        print(f"File: {filename}")
        
        # Add summary to end of file
        with open(filename, 'a') as f:
            f.write(f"\n# Logging stopped: {datetime.datetime.now()}\n")
            f.write(f"# Total lines captured: {line_count}\n")
            f.write(f"# Total duration: {elapsed_total/60:.1f} minutes\n")
        
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except FileNotFoundError:
        print(f"Error: Could not create file {filename}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    continuous_log() 
#!/usr/bin/env python3
import serial
import time
import sys
import os

# Configuration
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
OUTPUT_FILE = 'data_optimized/horizon/osqp_benchmark_H4.txt'

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# Configuration header
header = f"""# Safety Filter Benchmark Results
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
# Hardware: Adafruit Feather STM32
# 
# Serial Monitor Output:
# ======================
"""

print("Starting serial monitor...")
print(f"Port: {SERIAL_PORT}")
print(f"Baud Rate: {BAUD_RATE}")
print(f"Output File: {OUTPUT_FILE}")
print("Waiting for device connection...")

ser = None
try:
    # Wait for port to become available
    while True:
        try:
            if os.path.exists(SERIAL_PORT):
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                print("Connected to device!")
                break
        except (serial.SerialException, OSError):
            pass
        time.sleep(1)
    
    # Open output file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(header)
        f.flush()
        
        print("Monitoring serial data... Press Ctrl+C to stop")
        
        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        timestamp = time.strftime('%H:%M:%S')
                        output = f"[{timestamp}] {line}"
                        print(output)
                        f.write(output + '\n')
                        f.flush()
                time.sleep(0.01)
            except serial.SerialException:
                print("Device disconnected, waiting for reconnection...")
                ser.close()
                # Wait for reconnection
                while True:
                    try:
                        if os.path.exists(SERIAL_PORT):
                            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                            print("Reconnected to device!")
                            break
                    except (serial.SerialException, OSError):
                        pass
                    time.sleep(1)
            
except KeyboardInterrupt:
    print("\nSerial monitoring stopped.")
except Exception as e:
    print(f"Error: {e}")
finally:
    if ser and ser.is_open:
        ser.close()
#!/usr/bin/env python3
import serial
import time
import sys
import os

# Configuration  
SERIAL_PORT = '/dev/ttyACM0'  # Change this for your system
BAUD_RATE = 115200
OUTPUT_FILE = 'data/tinympc_h2.txt'  # Change this filename for each test

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_FILE) if os.path.dirname(OUTPUT_FILE) else '.', exist_ok=True)

# Header for the data file
header = f"""# Rocket Landing Horizon Sweep Data
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
# File: {OUTPUT_FILE}
# Serial Port: {SERIAL_PORT}
# 
# Data Format:
# [timestamp] tracking error: X.XXXX
# [timestamp] time step: N
# [timestamp] iterations: N  
# [timestamp] controls: X.XXXX Y.YYYY Z.ZZZZ
# [timestamp] solve time: NNNN us
# 
# Raw Serial Output:
# ==================
"""

print("Rocket Landing Data Logger")
print("=========================")
print(f"Output file: {OUTPUT_FILE}")
print(f"Serial port: {SERIAL_PORT}")
print(f"Baud rate: {BAUD_RATE}")
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
    
    # Open output file and start logging
    with open(OUTPUT_FILE, 'w') as f:
        f.write(header)
        f.flush()
        
        print("Logging rocket landing data... Press Ctrl+C to stop")
        
        while True:
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        timestamp = time.strftime('%H:%M:%S.%f')[:-3]  # Include milliseconds
                        output = f"[{timestamp}] {line}"
                        print(output)
                        f.write(output + '\n')
                        f.flush()
                time.sleep(0.001)  # Faster polling for better timing accuracy
                
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
    print(f"\nData collection stopped.")
    print(f"Results saved to: {OUTPUT_FILE}")
except Exception as e:
    print(f"Error: {e}")
finally:
    if ser and ser.is_open:
        ser.close()
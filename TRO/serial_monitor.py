#!/usr/bin/env python3
"""
Simple serial monitor to test Arduino connection
"""

import serial
import sys
import time

def monitor_serial(port, baudrate=9600):
    """Monitor serial port and display output"""
    print(f"Monitoring {port} at {baudrate} baud...")
    print("Press Ctrl+C to stop\n")
    
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for connection
        
        while True:
            if ser.in_waiting:
                line = ser.readline()
                try:
                    decoded = line.decode('utf-8').strip()
                    print(decoded)
                except:
                    print(f"Raw: {line}")
                    
    except serial.SerialException as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nStopped.")
        ser.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 serial_monitor.py <port>")
        print("Example: python3 serial_monitor.py /dev/ttyUSB0")
        sys.exit(1)
    
    port = sys.argv[1]
    baudrate = int(sys.argv[2]) if len(sys.argv) > 2 else 9600
    
    monitor_serial(port, baudrate)
#!/usr/bin/env python3
"""
Test Arduino connection with different baud rates and settings
"""

import serial
import time
import sys

def test_port(port, duration=5):
    """Test a port with different baud rates"""
    baud_rates = [9600, 115200, 19200, 38400, 57600, 1200, 2400, 4800]
    
    print(f"Testing port: {port}")
    
    for baud in baud_rates:
        print(f"\nTrying baud rate: {baud}")
        try:
            # Try to open the port
            ser = serial.Serial(port, baud, timeout=1, bytesize=8, parity='N', stopbits=1)
            print(f"  Port opened successfully at {baud} baud")
            
            # Wait a moment for Arduino to reset/stabilize
            time.sleep(2)
            
            # Try to read some data
            print(f"  Reading data for {duration} seconds...")
            start_time = time.time()
            data_count = 0
            
            while (time.time() - start_time) < duration:
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline()
                        if line:
                            decoded = line.decode('utf-8', errors='ignore').strip()
                            print(f"    Received: {repr(decoded)}")
                            data_count += 1
                            
                            if data_count >= 5:  # Stop after a few lines
                                break
                    except Exception as e:
                        print(f"    Read error: {e}")
                else:
                    time.sleep(0.1)
            
            ser.close()
            
            if data_count > 0:
                print(f"  SUCCESS: Received {data_count} lines at {baud} baud")
                return baud
            else:
                print(f"  No data received at {baud} baud")
                
        except Exception as e:
            print(f"  Failed to open port at {baud} baud: {e}")
    
    print(f"\nNo successful connection found for {port}")
    return None

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyS0"
    successful_baud = test_port(port)
    
    if successful_baud:
        print(f"\n✅ Arduino is connected at {port} with {successful_baud} baud")
        print(f"Use this command to collect data:")
        print(f"python3 TRO/data_collection_script.py --port {port} --baudrate {successful_baud} --config nstates_4_ninputs_2")
    else:
        print(f"\n❌ Could not establish connection to Arduino at {port}")
        print("Check:")
        print("1. Arduino is powered on and running")
        print("2. USB cable is connected") 
        print("3. Arduino sketch is running and outputting data")
        print("4. Try a different port (e.g., /dev/ttyS1, /dev/ttyACM0)")
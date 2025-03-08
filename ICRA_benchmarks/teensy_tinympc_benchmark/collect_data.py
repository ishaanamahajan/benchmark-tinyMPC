

import serial
import time

# Open serial port with your Teensy's port
ser = serial.Serial(
    port='/dev/cu.usbmodem132804901',  # Your Teensy's port
    baudrate=115200,
    timeout=1
)

# Open CSV file
with open('benchmark_results_5.csv', 'w') as f:
    while True:
        try:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8').strip()
                print(line)  # Show in terminal
                f.write(line + '\n')  # Write to file
                f.flush()  # Make sure it's written
        except KeyboardInterrupt:
            print("\nStopping data collection...")
            break

ser.close()
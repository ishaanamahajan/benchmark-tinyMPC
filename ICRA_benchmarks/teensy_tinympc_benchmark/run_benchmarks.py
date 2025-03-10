import os
import subprocess
import time
import serial
import shutil
import re

# Configuration
TEENSY_PORT = '/dev/cu.usbmodem132804901'  # Update with your Teensy's port
PLATFORMIO_PATH = 'platformio'  # Use 'pio' if you have it in PATH
PROJECT_DIR = '.'  # Current directory or specify your project directory
TYPES_HPP_PATH = os.path.join(PROJECT_DIR, 'include/types.hpp')
DATA_DIR = os.path.join(PROJECT_DIR, 'include/problem_data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def find_all_source_files():
    """Find all source files that might include problem data"""
    source_files = []
    
    # Search in include directory
    for root, dirs, files in os.walk(os.path.join(PROJECT_DIR, 'include')):
        for file in files:
            if file.endswith('.hpp') or file.endswith('.h') or file.endswith('.cpp'):
                source_files.append(os.path.join(root, file))
    
    # Search in src directory
    for root, dirs, files in os.walk(os.path.join(PROJECT_DIR, 'src')):
        for file in files:
            if file.endswith('.cpp') or file.endswith('.c'):
                source_files.append(os.path.join(root, file))
    
    return source_files

def update_include_files(system_number):
    """Update all source files to include the specific system"""
    source_files = find_all_source_files()
    
    # Pattern to match any problem data include
    pattern = r'#include\s+"problem_data/rand_.*\.hpp"'
    new_include = f'#include "problem_data/rand_system_{system_number}.hpp"'
    
    for file_path in source_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check if file contains problem data include
            if re.search(pattern, content):
                # Replace with the new system include
                new_content = re.sub(pattern, new_include, content)
                
                with open(file_path, 'w') as f:
                    f.write(new_content)
                
                print(f"Updated include in {file_path}")
        except Exception as e:
            print(f"Error updating {file_path}: {e}")
    
    print(f"Updated all include files to use system {system_number}")

def clean_build():
    """Clean the build directory"""
    print("Cleaning build directory...")
    result = subprocess.run([PLATFORMIO_PATH, 'run', '--target', 'clean'], 
                           cwd=PROJECT_DIR, 
                           capture_output=True, 
                           text=True)
    
    if result.returncode != 0:
        print("Warning: Clean failed, but continuing...")
    
    # Also remove .pio directory to ensure complete clean
    pio_dir = os.path.join(PROJECT_DIR, '.pio')
    if os.path.exists(pio_dir):
        try:
            shutil.rmtree(pio_dir)
            print("Removed .pio directory")
        except Exception as e:
            print(f"Warning: Could not remove .pio directory: {e}")

def compile_and_upload():
    """Compile and upload the code to Teensy"""
    print("Compiling and uploading...")
    result = subprocess.run([PLATFORMIO_PATH, 'run', '--target', 'upload'], 
                           cwd=PROJECT_DIR, 
                           capture_output=True, 
                           text=True)
    
    if result.returncode != 0:
        print("Error during compilation/upload:")
        print(result.stderr)
        return False
    
    print("Upload successful")
    return True

def collect_data(system_number):
    """Collect data from the Teensy and save to CSV"""
    csv_filename = os.path.join(RESULTS_DIR, f'benchmark_system_{system_number}.csv')
    
    print(f"Collecting data for system {system_number}...")
    
    # Open serial connection
    ser = serial.Serial(port=TEENSY_PORT, baudrate=115200, timeout=10)
    
    # Wait for device to reset after upload
    time.sleep(10)
    
    # Clear any initial data
    ser.reset_input_buffer()
    
    # Open CSV file
    with open(csv_filename, 'w') as f:
        # Write header if needed
        f.write("iteration,fixed_time,adaptive_time,fixed_iterations,adaptive_iterations\n")
        
        # Set timeout for data collection (adjust as needed)
        start_time = time.time()
        timeout = 60  # 1 minute timeout
        
        data_received = False
        
        while time.time() - start_time < timeout:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8').strip()
                print(line)  # Show in terminal
                f.write(line + '\n')  # Write to file
                f.flush()  # Make sure it's written
                data_received = True
                
                # If we see a specific end marker, we can break early
                if "BENCHMARK COMPLETE" in line:
                    break
            
            time.sleep(0.1)
        
        if not data_received:
            print(f"WARNING: No data received for system {system_number}")
    
    ser.close()
    print(f"Data collection complete for system {system_number}")
    return True

def main():
    # Get list of all system files
    system_files = [f for f in os.listdir(DATA_DIR) if f.startswith('rand_system_') and f.endswith('.hpp')]
    
    # Extract system numbers
    system_numbers = [int(f.split('_')[2].split('.')[0]) for f in system_files]
    system_numbers.sort()
    
    print(f"Found {len(system_numbers)} systems to test")
    
    # Create a summary file
    summary_file = os.path.join(RESULTS_DIR, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Testing {len(system_numbers)} systems\n")
        f.write("System,Status,Notes\n")
    
    # Process each system
    for system_number in system_numbers:
        print(f"\n=== Processing system {system_number} ===")
        
        try:
            # Clean build directory first
            clean_build()
            
            # Update all include files
            update_include_files(system_number)
            
            # Compile and upload
            if not compile_and_upload():
                with open(summary_file, 'a') as f:
                    f.write(f"{system_number},FAILED,Compilation or upload error\n")
                continue
            
            # Collect data
            if not collect_data(system_number):
                with open(summary_file, 'a') as f:
                    f.write(f"{system_number},FAILED,Data collection error\n")
                continue
            
            # Success
            with open(summary_file, 'a') as f:
                f.write(f"{system_number},SUCCESS,\n")
                
        except Exception as e:
            print(f"Error processing system {system_number}: {e}")
            with open(summary_file, 'a') as f:
                f.write(f"{system_number},ERROR,{str(e)}\n")
    
    print("\n=== Testing complete ===")
    print(f"Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
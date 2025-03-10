import os
import subprocess
import time
import serial
import shutil

# Configuration
TEENSY_PORT = '/dev/cu.usbmodem132804901'  # Update with your Teensy's port
PLATFORMIO_PATH = 'platformio'  # Use 'pio' if you have it in PATH
PROJECT_DIR = '.'  # Current directory or specify your project directory
TYPES_HPP_PATH = os.path.join(PROJECT_DIR, 'include/types.hpp')
DATA_DIR = os.path.join(PROJECT_DIR, 'include/problem_data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def update_include_file(system_number):
    """Update the types.hpp file to include the specific system"""
    with open(TYPES_HPP_PATH, 'r') as f:
        content = f.read()
    
    # Replace the include line for problem data
    new_content = content.replace(
        'problem_data/rand_prob_tinympc_params.hpp',
        f'problem_data/rand_system_{system_number}.hpp'
    )
    
    with open(TYPES_HPP_PATH, 'w') as f:
        f.write(new_content)
    
    print(f"Updated types.hpp to use system {system_number}")

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
    time.sleep(2)
    
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
            # Update the include file
            update_include_file(system_number)
            
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
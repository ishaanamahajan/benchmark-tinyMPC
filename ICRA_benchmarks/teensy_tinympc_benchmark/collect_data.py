# import serial
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# def collect_and_plot_data(port='/dev/cu.usbmodem132804901', baudrate=115200):
#     # Initialize serial connection
#     ser = serial.Serial(port=port, baudrate=baudrate, timeout=1)
    
#     # Lists to store data
#     data = []
#     current_solver = None
    
#     print("Collecting data...")
    
#     while True:
#         line = ser.readline().decode('utf-8').strip()
#         if not line:
#             continue
            
#         if line == "END":
#             break
            
#         if line != "solver,run,iterations,solve_time_us":  # Skip header
#             solver, run, iters, time = line.split(',')
#             data.append({
#                 'solver': solver,
#                 'run': int(run),
#                 'iterations': int(iters),
#                 'solve_time_us': float(time)
#             })
    
#     ser.close()
    
#     # Convert to DataFrame
#     df = pd.DataFrame(data)
    
#     # Split by solver type
#     fixed_data = df[df['solver'] == 'fixed']
#     adaptive_data = df[df['solver'] == 'adaptive']
    
#     # Create figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
#     # Plot 1: Iterations vs Solve Number
#     ax1.plot(fixed_data['run'], fixed_data['iterations'], 'b.-', label='Fixed', alpha=0.7)
#     ax1.plot(adaptive_data['run'], adaptive_data['iterations'], 'r.-', label='Adaptive', alpha=0.7)
#     ax1.set_xlabel('Solve Number')
#     ax1.set_ylabel('Iterations')
#     ax1.set_title('Iterations per Solve')
#     ax1.grid(True)
#     ax1.legend()
    
#     # Add statistics to first plot
#     stats_text1 = (f'Fixed Avg: {fixed_data["iterations"].mean():.1f}\n'
#                   f'Adaptive Avg: {adaptive_data["iterations"].mean():.1f}\n'
#                   f'Fixed Max: {fixed_data["iterations"].max()}\n'
#                   f'Adaptive Max: {adaptive_data["iterations"].max()}')
#     ax1.text(0.02, 0.98, stats_text1,
#              transform=ax1.transAxes,
#              verticalalignment='top',
#              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
#     # Plot 2: Percentage of Solves vs Time (CDF)
#     def plot_cdf(data, label, color):
#         # Consider only successful solves (iter < 500)
#         successful = data[data['iterations'] < 500]
#         times_sorted = np.sort(successful['solve_time_us'])
#         percentiles = np.arange(len(times_sorted)) / len(data) * 100
#         ax2.plot(times_sorted, percentiles, f'{color}.-', label=f'{label}', alpha=0.7)
    
#     plot_cdf(fixed_data, 'Fixed', 'b')
#     plot_cdf(adaptive_data, 'Adaptive', 'r')
    
#     ax2.set_xlabel('Solve Time (µs)')
#     ax2.set_ylabel('Percent of Successful Solves (%)')
#     ax2.set_title('Cumulative Distribution of Solve Times')
#     ax2.grid(True)
#     ax2.legend()
    
#     # Add statistics to second plot
#     stats_text2 = (f'Fixed Avg: {fixed_data["solve_time_us"].mean():.1f}µs\n'
#                   f'Adaptive Avg: {adaptive_data["solve_time_us"].mean():.1f}µs\n'
#                   f'Fixed Success: {(fixed_data["iterations"] < 500).mean()*100:.1f}%\n'
#                   f'Adaptive Success: {(adaptive_data["iterations"] < 500).mean()*100:.1f}%')
#     ax2.text(0.02, 0.98, stats_text2,
#              transform=ax2.transAxes,
#              verticalalignment='top',
#              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
#     plt.tight_layout()
    
#     # Save data to CSV
#     df.to_csv('solver_results.csv', index=False)
    
#     # Save plot
#     plt.savefig('solver_comparison.png', dpi=300, bbox_inches='tight')
#     plt.show()

# if __name__ == "__main__":
#     collect_and_plot_data()  # Adjust port if needed

import serial
import time

# Open serial port with your Teensy's port
ser = serial.Serial(
    port='/dev/cu.usbmodem132804901',  # Your Teensy's port
    baudrate=115200,
    timeout=1
)

# Open CSV file
with open('benchmark_results_dynamics_10.csv', 'w') as f:
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
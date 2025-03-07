import re
import csv
import sys

def extract_data(input_file, output_file):
    """
    Extract MPC benchmark data from log files and convert to clean CSV format.
    
    Args:
        input_file (str): Path to the input log file
        output_file (str): Path to the output CSV file
    """
    # Data structure to hold all results
    results = []
    
    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Define patterns to match the data lines
    # This pattern matches lines like: Fixed,0,29569,29569,0,500,85.00
    # Or lines like: Fixed Hover,-1,29568,29567,0,500,85.00
    data_pattern = re.compile(r'(Fixed|Adaptive)(?:\s+Hover)?,(\-?\d+),(\d+),(\d+),(\d+),(\d+),(\d+\.\d+)')
    
    # Find all matches
    matches = data_pattern.findall(content)
    
    # Process matches and add to results
    for match in matches:
        if len(match) == 7:
            method, trial, solve_time, admm_time, rho_time, iterations, final_rho = match
            
            # Remove "Hover" from method name if present - treat Fixed Hover as Fixed and Adaptive Hover as Adaptive
            method = method.split()[0]  # This will get just "Fixed" or "Adaptive" part
                
            results.append({
                'Method': method,
                'Trial': int(trial),
                'SolveTime': int(solve_time),
                'ADMMTime': int(admm_time),
                'RhoTime': int(rho_time),
                'Iterations': int(iterations),
                'FinalRho': float(final_rho)
            })
    
    # Write the data to a CSV file
    if results:
        fieldnames = ['Method', 'Trial', 'SolveTime', 'ADMMTime', 'RhoTime', 'Iterations', 'FinalRho']
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Extracted {len(results)} data points to {output_file}")
    else:
        print("No data found in the input file.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 clean_data.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    extract_data(input_file, output_file)
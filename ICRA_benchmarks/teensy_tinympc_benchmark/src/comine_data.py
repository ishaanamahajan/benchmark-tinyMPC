import pandas as pd
import os

def parse_admm_file(filepath):
    """Parse the ADMM trial data from file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except:
        print(f"Error reading file: {filepath}")
        return pd.DataFrame()
    
    lines = content.split('\n')
    data = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('==='):
            continue
            
        parts = line.split(',')
        if len(parts) >= 7 and (parts[0] == 'Fixed' or parts[0] == 'Adaptive'):
            try:
                data.append({
                    'Method': parts[0],
                    'Trial': int(parts[1]),
                    'SolveTime': int(parts[2]),
                    'ADMMTime': int(parts[3]),
                    'RhoTime': int(parts[4]),
                    'Iterations': int(parts[5]),
                    'FinalRho': float(parts[6])
                })
            except:
                continue
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Base path
    base_path = "/Users/ishaanmahajan/replicate/benchmark-tinyMPC/ICRA_benchmarks/teensy_tinympc_benchmark/"
    
    # Collect data from all files
    all_data = pd.DataFrame()
    for i in range(1, 101):
        filename = f"benchmark_results_12_{i}.csv"
        filepath = os.path.join(base_path, filename)
        df = parse_admm_file(filepath)
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    if len(all_data) > 0:
        # Sort all Fixed first, then all Adaptive
        fixed_data = all_data[all_data['Method'] == 'Fixed'].sort_values('SolveTime')
        adaptive_data = all_data[all_data['Method'] == 'Adaptive'].sort_values('SolveTime')
        
        # Combine them with Fixed first, then Adaptive
        sorted_data = pd.concat([fixed_data, adaptive_data], ignore_index=True)
        
        # Save combined and sorted data
        output_file = os.path.join(base_path, "combined_results.csv")
        sorted_data.to_csv(output_file, index=False)
        print(f"Combined data saved to: {output_file}")
        print(f"Total rows: {len(sorted_data)}")
        print(f"Fixed rows: {len(fixed_data)}")
        print(f"Adaptive rows: {len(adaptive_data)}")
    else:
        print("No data was loaded. Please check the file paths.")
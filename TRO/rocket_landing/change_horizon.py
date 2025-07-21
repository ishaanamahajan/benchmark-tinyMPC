#!/usr/bin/env python3
"""
Script to easily change horizon for all solvers (SCS, ECOS, TinyMPC)
Usage: python change_horizon.py <new_horizon>
"""

import sys
import os
import re

def update_file_horizon(file_path, new_horizon, pattern, replacement_template):
    """Update horizon in a file using regex pattern"""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    new_content = re.sub(pattern, replacement_template.format(new_horizon), content)
    
    if content != new_content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"Updated {file_path}: NHORIZON = {new_horizon}")
        return True
    else:
        print(f"No change needed in {file_path}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python change_horizon.py <new_horizon>")
        sys.exit(1)
    
    try:
        new_horizon = int(sys.argv[1])
    except ValueError:
        print("Error: horizon must be an integer")
        sys.exit(1)
    
    if new_horizon < 2 or new_horizon > 100:
        print("Warning: horizon should be reasonable (2-100)")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Files to update
    files_to_update = [
        # Python files
        (os.path.join(base_dir, "gen_rocket.py"), 
         r'NHORIZON\s*=\s*\d+', 'NHORIZON = {}'),
        (os.path.join(base_dir, "run_ecos.py"), 
         r'NHORIZON\s*=\s*\d+', 'NHORIZON = {}'),
        (os.path.join(base_dir, "gen_tinympc_rocket_clean.py"), 
         r'NHORIZON\s*=\s*\d+', 'NHORIZON = {}'),
        
        # Generated headers
        (os.path.join(base_dir, "tinympc/tinympc_generated/tinympc/glob_opts.hpp"),
         r'#define NHORIZON\s+\d+', '#define NHORIZON {}'),
        (os.path.join(base_dir, "tinympc/tinympc_teensy/lib/tinympc/glob_opts.hpp"),
         r'#define NHORIZON\s+\d+', '#define NHORIZON {}'),
        (os.path.join(base_dir, "tinympc/tinympc_stm32/src/tinympc/glob_opts.hpp"),
         r'#define NHORIZON\s+\d+', '#define NHORIZON {}'),
    ]
    
    updated_count = 0
    for file_path, pattern, template in files_to_update:
        if update_file_horizon(file_path, new_horizon, pattern, template):
            updated_count += 1
    
    print(f"\n=== Horizon Update Complete ===")
    print(f"Updated {updated_count} files with NHORIZON = {new_horizon}")
    print("Now run:")
    print("  python gen_rocket.py          (for SCS)")
    print("  python run_ecos.py            (for ECOS)")  
    print("  python gen_tinympc_rocket_clean.py  (for TinyMPC)")
    print("Then compile and test each solver.")

if __name__ == "__main__":
    main()
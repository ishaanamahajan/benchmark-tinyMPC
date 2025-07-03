#!/usr/bin/env python3
"""
TinyMPC Configuration Generator
==============================

This script automatically generates different configurations for TinyMPC scaling experiments
by modifying NSTATES and NINPUTS in the relevant header files.

Usage:
python config_generator.py --nstates 4 --ninputs 2 --target stm32
python config_generator.py --nstates 8 --ninputs 2 --target teensy
"""

import argparse
import os
import shutil
from datetime import datetime

class TinyMPCConfigGenerator:
    def __init__(self):
        self.base_dir = "TRO"
        
    def backup_file(self, filepath):
        """Create a backup of the original file"""
        backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"Backup created: {backup_path}")
        return backup_path
    
    def update_glob_opts(self, target, nstates, ninputs, nhorizon=10):
        """Update glob_opts.hpp with new dimensions"""
        if target == "stm32":
            glob_opts_path = f"{self.base_dir}/tinympc_f/tinympc_stm32_feather/src/tinympc/glob_opts.hpp"
        elif target == "teensy":
            glob_opts_path = f"{self.base_dir}/tinympc_f/tinympc_teensy/lib/tinympc/glob_opts.hpp"
        else:
            raise ValueError("Target must be 'stm32' or 'teensy'")
        
        if not os.path.exists(glob_opts_path):
            print(f"Error: {glob_opts_path} not found")
            return False
        
        # Backup original file
        self.backup_file(glob_opts_path)
        
        # Read the current file
        with open(glob_opts_path, 'r') as f:
            content = f.read()
        
        # Update the defines
        content = self.update_defines(content, nstates, ninputs, nhorizon)
        
        # Write back
        with open(glob_opts_path, 'w') as f:
            f.write(content)
        
        print(f"Updated {glob_opts_path}")
        print(f"  NSTATES: {nstates}")
        print(f"  NINPUTS: {ninputs}")
        print(f"  NHORIZON: {nhorizon}")
        
        return True
    
    def update_defines(self, content, nstates, ninputs, nhorizon):
        """Update #define statements in the content"""
        import re
        
        # Update NSTATES
        content = re.sub(r'#define\s+NSTATES\s+\d+', f'#define NSTATES {nstates}', content)
        
        # Update NINPUTS  
        content = re.sub(r'#define\s+NINPUTS\s+\d+', f'#define NINPUTS {ninputs}', content)
        
        # Update NHORIZON
        content = re.sub(r'#define\s+NHORIZON\s+\d+', f'#define NHORIZON {nhorizon}', content)
        
        # Calculate and update derived values
        ntotal = nstates * nhorizon + ninputs * (nhorizon - 1)
        content = re.sub(r'#define\s+NTOTAL\s+\d+', f'#define NTOTAL {ntotal}', content)
        
        return content
    
    def regenerate_data_workspace(self, target, nstates, ninputs, nhorizon=10):
        """
        Regenerate the data workspace using the safety filter notebook approach
        This would typically involve running the Python code generation
        """
        print("\n" + "="*50)
        print("IMPORTANT: Data Workspace Regeneration Required")
        print("="*50)
        print(f"You need to regenerate the data workspace for the new dimensions:")
        print(f"  NSTATES: {nstates}")
        print(f"  NINPUTS: {ninputs}")
        print(f"  NHORIZON: {nhorizon}")
        print()
        print("Steps to regenerate:")
        print("1. Open TRO/safety_filter.ipynb")
        print("2. Update the problem dimensions in the notebook:")
        print(f"   NSTATES = {nstates}")
        print(f"   NINPUTS = {ninputs}")
        print(f"   NHORIZON = {nhorizon}")
        print("3. Run all cells to regenerate the data workspace")
        print("4. The notebook will generate new tiny_data_workspace.cpp")
        print("5. Flash the updated code to your STM32")
        print()
        print("Alternative: Use the utils.py approach if you prefer Python scripts")
        print("="*50)
    
    def update_osqp_config(self, target, nstates, ninputs, nhorizon=10):
        """Update OSQP configuration for comparison"""
        if target == "stm32":
            osqp_path = f"{self.base_dir}/osqp/osqp_stm32_feather/src/osqp/inc/public/osqp_problem.h"
        elif target == "teensy":
            osqp_path = f"{self.base_dir}/osqp/osqp_teensy/src/osqp_problem.h"
        else:
            return
        
        if not os.path.exists(osqp_path):
            print(f"OSQP config not found: {osqp_path}")
            return
        
        # Backup and update OSQP config
        self.backup_file(osqp_path)
        
        with open(osqp_path, 'r') as f:
            content = f.read()
        
        content = self.update_defines(content, nstates, ninputs, nhorizon)
        
        with open(osqp_path, 'w') as f:
            f.write(content)
        
        print(f"Updated OSQP config: {osqp_path}")
    
    def generate_config(self, nstates, ninputs, nhorizon=10, target="stm32", update_osqp=False):
        """Generate a complete configuration"""
        print(f"Generating configuration for {target.upper()}:")
        print(f"  NSTATES: {nstates}")
        print(f"  NINPUTS: {ninputs}") 
        print(f"  NHORIZON: {nhorizon}")
        print()
        
        # Update TinyMPC configuration
        success = self.update_glob_opts(target, nstates, ninputs, nhorizon)
        
        if not success:
            return False
        
        # Update OSQP configuration if requested
        if update_osqp:
            self.update_osqp_config(target, nstates, ninputs, nhorizon)
        
        # Print data workspace regeneration instructions
        self.regenerate_data_workspace(target, nstates, ninputs, nhorizon)
        
        return True
    
    def generate_batch_configs(self, nstates_list, ninputs_list, target="stm32"):
        """Generate multiple configurations for batch experiments"""
        configs = []
        
        for nstates in nstates_list:
            for ninputs in ninputs_list:
                config_name = f"nstates_{nstates}_ninputs_{ninputs}"
                configs.append({
                    'name': config_name,
                    'nstates': nstates,
                    'ninputs': ninputs
                })
        
        print(f"Generated {len(configs)} configurations:")
        for config in configs:
            print(f"  - {config['name']}")
        
        print("\nTo run batch experiments:")
        print("1. For each configuration:")
        for config in configs:
            print(f"   python config_generator.py --nstates {config['nstates']} --ninputs {config['ninputs']} --target {target}")
            print(f"   # Regenerate data workspace and flash to STM32")
            print(f"   python data_collection_script.py --collect --port /dev/ttyUSB0 --config {config['name']}")
            print()
        print("2. Analyze all data:")
        print("   python data_collection_script.py --analyze")
        
        return configs

def main():
    parser = argparse.ArgumentParser(description='TinyMPC Configuration Generator')
    parser.add_argument('--nstates', type=int, help='Number of states')
    parser.add_argument('--ninputs', type=int, help='Number of inputs')
    parser.add_argument('--nhorizon', type=int, default=10, help='Horizon length (default: 10)')
    parser.add_argument('--target', choices=['stm32', 'teensy'], default='stm32', help='Target platform')
    parser.add_argument('--update-osqp', action='store_true', help='Also update OSQP configuration')
    
    # Batch generation options
    parser.add_argument('--batch-nstates', action='store_true', help='Generate batch configs for NSTATES scaling')
    parser.add_argument('--batch-ninputs', action='store_true', help='Generate batch configs for NINPUTS scaling')
    
    args = parser.parse_args()
    
    generator = TinyMPCConfigGenerator()
    
    if args.batch_nstates:
        # Your experiment: NSTATES = [2, 4, 8, 16, 32] with NINPUTS = 2
        nstates_list = [2, 4, 8, 16, 32]
        ninputs_list = [2]
        generator.generate_batch_configs(nstates_list, ninputs_list, args.target)
    elif args.batch_ninputs:
        # Your experiment: NINPUTS = [1, 2, 4, 8] with NSTATES = 10
        nstates_list = [10]
        ninputs_list = [1, 2, 4, 8]
        generator.generate_batch_configs(nstates_list, ninputs_list, args.target)
    else:
        if args.nstates is None or args.ninputs is None:
            print("Error: --nstates and --ninputs are required for single config generation")
            print("Or use --batch-nstates or --batch-ninputs for batch generation")
            return
        
        generator.generate_config(args.nstates, args.ninputs, args.nhorizon, args.target, args.update_osqp)

if __name__ == "__main__":
    main() 
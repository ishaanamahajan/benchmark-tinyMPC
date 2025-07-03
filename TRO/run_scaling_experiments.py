#!/usr/bin/env python3
"""
TinyMPC Scaling Experiments - Complete Workflow
==============================================

This script automates the complete workflow for TinyMPC scaling experiments:
1. Generate configurations for different problem sizes
2. Provide instructions for data workspace regeneration
3. Collect data from STM32 
4. Analyze data and generate plots

Usage:
python run_scaling_experiments.py --experiment nstates --port /dev/ttyUSB0
python run_scaling_experiments.py --experiment ninputs --port /dev/ttyUSB0
python run_scaling_experiments.py --analyze-only
"""

import argparse
import os
import sys
import time
import subprocess
from config_generator import TinyMPCConfigGenerator
from data_collection_script import TinyMPCBenchmarkCollector

class TinyMPCScalingExperiments:
    def __init__(self):
        self.config_gen = TinyMPCConfigGenerator()
        self.data_collector = TinyMPCBenchmarkCollector()
        
        # Experiment configurations
        self.nstates_experiment = {
            'name': 'NSTATES Scaling',
            'nstates_list': [2, 4, 8, 16, 32],
            'ninputs_list': [2],
            'description': 'Varies NSTATES from 2 to 32 with NINPUTS=2'
        }
        
        self.ninputs_experiment = {
            'name': 'NINPUTS Scaling', 
            'nstates_list': [10],
            'ninputs_list': [1, 2, 4, 8],
            'description': 'Varies NINPUTS from 1 to 8 with NSTATES=10'
        }
    
    def print_experiment_plan(self, experiment):
        """Print the experiment plan"""
        print("="*60)
        print(f"EXPERIMENT: {experiment['name']}")
        print("="*60)
        print(f"Description: {experiment['description']}")
        print()
        
        configs = []
        for nstates in experiment['nstates_list']:
            for ninputs in experiment['ninputs_list']:
                config_name = f"nstates_{nstates}_ninputs_{ninputs}"
                configs.append({
                    'name': config_name,
                    'nstates': nstates,
                    'ninputs': ninputs
                })
        
        print("Configurations to test:")
        for i, config in enumerate(configs, 1):
            memory_kb = self.data_collector.estimate_memory_usage(config['nstates'], config['ninputs'])
            print(f"  {i}. {config['name']}: {memory_kb:.2f} KB estimated memory")
        
        print()
        return configs
    
    def run_experiment(self, experiment_type, port, target="stm32", collection_duration=60):
        """Run a complete scaling experiment"""
        
        if experiment_type == "nstates":
            experiment = self.nstates_experiment
        elif experiment_type == "ninputs":
            experiment = self.ninputs_experiment
        else:
            raise ValueError("experiment_type must be 'nstates' or 'ninputs'")
        
        configs = self.print_experiment_plan(experiment)
        
        print("WORKFLOW:")
        print("1. Configure TinyMPC for each problem size")
        print("2. Regenerate data workspace (manual step)")
        print("3. Flash STM32 with updated code")
        print("4. Collect benchmarking data via serial")
        print("5. Repeat for all configurations")
        print("6. Analyze data and generate plots")
        print()
        
        # Ask for confirmation
        response = input("Do you want to proceed with this experiment? (y/N): ")
        if response.lower() != 'y':
            print("Experiment cancelled.")
            return
        
        for i, config in enumerate(configs, 1):
            print("\n" + "="*50)
            print(f"CONFIGURATION {i}/{len(configs)}: {config['name']}")
            print("="*50)
            
            # Step 1: Generate configuration
            print(f"Step 1: Generating configuration...")
            success = self.config_gen.generate_config(
                config['nstates'], config['ninputs'], target=target
            )
            
            if not success:
                print(f"Failed to generate configuration for {config['name']}")
                continue
            
            # Step 2: Wait for user to regenerate data workspace and flash
            print(f"\nStep 2: Manual Steps Required")
            print("You need to:")
            print("1. Regenerate the data workspace (see instructions above)")
            print("2. Flash the updated code to your STM32")
            print("3. Ensure STM32 is connected and running")
            
            response = input("\nPress Enter when STM32 is ready for data collection (or 'skip' to skip this config): ")
            if response.lower() == 'skip':
                print(f"Skipping {config['name']}")
                continue
            
            # Step 3: Collect data
            print(f"\nStep 3: Collecting data for {config['name']}...")
            try:
                data = self.data_collector.collect_serial_data(
                    port, config['name'], collection_duration
                )
                
                if len(data) > 0:
                    print(f"✓ Successfully collected {len(data)} data points")
                else:
                    print("⚠ No data collected. Check STM32 connection and serial output.")
                
            except Exception as e:
                print(f"✗ Error collecting data: {e}")
                response = input("Continue with next configuration? (y/N): ")
                if response.lower() != 'y':
                    break
        
        print("\n" + "="*50)
        print("DATA COLLECTION COMPLETE")
        print("="*50)
        print("Next steps:")
        print("1. Run analysis: python run_scaling_experiments.py --analyze-only")
        print("2. Or continue with analysis now...")
        
        response = input("Run analysis now? (y/N): ")
        if response.lower() == 'y':
            self.analyze_data()
    
    def analyze_data(self):
        """Analyze all collected data and generate plots"""
        print("\n" + "="*50)
        print("ANALYZING DATA")
        print("="*50)
        
        df = self.data_collector.analyze_data()
        
        if df is not None and len(df) > 0:
            print("\nData Summary:")
            print(df[['nstates', 'ninputs', 'avg_time_us', 'memory_kb', 'num_samples']].to_string(index=False))
            
            print("\nKey Insights:")
            
            # NSTATES scaling analysis
            nstates_data = df[df['ninputs'] == 2]  # Assuming NINPUTS=2 for NSTATES experiment
            if len(nstates_data) > 1:
                print("\nNSTATES Scaling:")
                for _, row in nstates_data.iterrows():
                    print(f"  NSTATES={row['nstates']}: {row['avg_time_us']:.1f}µs, {row['memory_kb']:.2f}KB")
                
                # Calculate scaling factors
                if len(nstates_data) >= 2:
                    time_scaling = nstates_data['avg_time_us'].iloc[-1] / nstates_data['avg_time_us'].iloc[0]
                    memory_scaling = nstates_data['memory_kb'].iloc[-1] / nstates_data['memory_kb'].iloc[0]
                    nstates_scaling = nstates_data['nstates'].iloc[-1] / nstates_data['nstates'].iloc[0]
                    
                    print(f"  Time scaling factor: {time_scaling:.2f}x for {nstates_scaling:.1f}x NSTATES increase")
                    print(f"  Memory scaling factor: {memory_scaling:.2f}x for {nstates_scaling:.1f}x NSTATES increase")
            
            # NINPUTS scaling analysis  
            ninputs_data = df[df['nstates'] == 10]  # Assuming NSTATES=10 for NINPUTS experiment
            if len(ninputs_data) > 1:
                print("\nNINPUTS Scaling:")
                for _, row in ninputs_data.iterrows():
                    print(f"  NINPUTS={row['ninputs']}: {row['avg_time_us']:.1f}µs, {row['memory_kb']:.2f}KB")
                
                if len(ninputs_data) >= 2:
                    time_scaling = ninputs_data['avg_time_us'].iloc[-1] / ninputs_data['avg_time_us'].iloc[0]
                    memory_scaling = ninputs_data['memory_kb'].iloc[-1] / ninputs_data['memory_kb'].iloc[0]
                    ninputs_scaling = ninputs_data['ninputs'].iloc[-1] / ninputs_data['ninputs'].iloc[0]
                    
                    print(f"  Time scaling factor: {time_scaling:.2f}x for {ninputs_scaling:.1f}x NINPUTS increase")
                    print(f"  Memory scaling factor: {memory_scaling:.2f}x for {ninputs_scaling:.1f}x NINPUTS increase")
            
            print("\n✓ Analysis complete! Check the generated plots and CSV files.")
        else:
            print("No data found for analysis. Please collect data first.")
    
    def install_dependencies(self):
        """Install required Python packages"""
        required_packages = [
            'pyserial',
            'pandas', 
            'matplotlib',
            'numpy'
        ]
        
        print("Checking required packages...")
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✓ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"✗ {package}")
        
        if missing_packages:
            print(f"\nMissing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("✓ All required packages are installed")
        return True

def main():
    parser = argparse.ArgumentParser(description='TinyMPC Scaling Experiments - Complete Workflow')
    parser.add_argument('--experiment', choices=['nstates', 'ninputs'], 
                       help='Type of scaling experiment to run')
    parser.add_argument('--port', type=str, help='Serial port (e.g., /dev/ttyUSB0, COM3)')
    parser.add_argument('--target', choices=['stm32', 'teensy'], default='stm32',
                       help='Target platform (default: stm32)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Data collection duration per config in seconds (default: 60)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only run analysis on existing data')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check if required dependencies are installed')
    
    args = parser.parse_args()
    
    experiments = TinyMPCScalingExperiments()
    
    if args.check_deps:
        experiments.install_dependencies()
        return
    
    if args.analyze_only:
        experiments.analyze_data()
        return
    
    if not args.experiment or not args.port:
        print("Error: --experiment and --port are required for data collection")
        print("Use --analyze-only to analyze existing data")
        print("Use --check-deps to check dependencies")
        return
    
    # Check dependencies before starting
    if not experiments.install_dependencies():
        print("Please install missing dependencies before running experiments.")
        return
    
    # Run the experiment
    experiments.run_experiment(args.experiment, args.port, args.target, args.duration)

if __name__ == "__main__":
    main() 
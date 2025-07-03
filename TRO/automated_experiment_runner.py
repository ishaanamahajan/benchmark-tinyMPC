#!/usr/bin/env python3
"""
TinyMPC Automated Experiment Runner
==================================

This script provides the most automated workflow possible for TinyMPC scaling experiments.
It handles everything that can be automated and provides clear instructions for manual steps.

Usage:
python automated_experiment_runner.py --experiment nstates --port /dev/ttyUSB0
"""

import argparse
import os
import sys
import time
import subprocess
import json
from datetime import datetime
from config_generator import TinyMPCConfigGenerator
from data_collection_script import TinyMPCBenchmarkCollector

class AutomatedExperimentRunner:
    def __init__(self):
        self.config_gen = TinyMPCConfigGenerator()
        self.data_collector = TinyMPCBenchmarkCollector()
        
        # Check if we're in the right directory
        if not os.path.exists("TRO"):
            print("Error: Please run this script from the project root directory")
            sys.exit(1)
    
    def find_platformio_project(self):
        """Find PlatformIO project for STM32"""
        stm32_path = "TRO/tinympc_f/tinympc_stm32_feather"
        if os.path.exists(f"{stm32_path}/platformio.ini"):
            return stm32_path
        elif os.path.exists(f"{stm32_path}/tinympc_stm32_feather.ino"):
            return stm32_path
        return None
    
    def check_platformio_available(self):
        """Check if PlatformIO CLI is available"""
        try:
            result = subprocess.run(['pio', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ PlatformIO CLI found: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        print("⚠ PlatformIO CLI not found")
        print("Install: pip install platformio")
        print("Or use Arduino IDE/PlatformIO IDE manually")
        return False
    
    def generate_workspace_regeneration_script(self, nstates, ninputs, nhorizon=10):
        """Generate a Python script to regenerate the data workspace"""
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated workspace regeneration script
Generated for: NSTATES={nstates}, NINPUTS={ninputs}, NHORIZON={nhorizon}
"""

import sys
import os
sys.path.append('TRO')

# Import the utils function
from utils import tiny_export_data_to_c

def regenerate_workspace():
    print("Regenerating TinyMPC data workspace...")
    print(f"Dimensions: NSTATES={nstates}, NINPUTS={ninputs}, NHORIZON={nhorizon}")
    
    # Set problem dimensions
    NSTATES = {nstates}
    NINPUTS = {ninputs}
    NHORIZON = {nhorizon}
    
    # This would call the workspace generation
    # You'll need to adapt this based on your specific utils.py implementation
    print("\\nNOTE: You still need to manually run the workspace regeneration")
    print("Options:")
    print("1. Open TRO/safety_filter.ipynb and run all cells")
    print("2. Or use TRO/utils.py functions directly")
    print("3. Or copy the generated matrices from a previous run")
    
    return True

if __name__ == "__main__":
    regenerate_workspace()
'''
        
        script_path = f"TRO/regenerate_workspace_nstates_{nstates}_ninputs_{ninputs}.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)  # Make executable
        return script_path
    
    def flash_with_platformio(self, project_path, port):
        """Attempt to flash using PlatformIO CLI"""
        try:
            print("Attempting to flash with PlatformIO...")
            
            # Build the project
            print("Building project...")
            build_result = subprocess.run(
                ['pio', 'run'], 
                cwd=project_path, 
                capture_output=True, 
                text=True
            )
            
            if build_result.returncode != 0:
                print(f"Build failed: {build_result.stderr}")
                return False
            
            # Upload to device
            print(f"Uploading to {port}...")
            upload_result = subprocess.run(
                ['pio', 'run', '--target', 'upload', '--upload-port', port],
                cwd=project_path,
                capture_output=True,
                text=True
            )
            
            if upload_result.returncode == 0:
                print("✓ Successfully flashed with PlatformIO!")
                return True
            else:
                print(f"Upload failed: {upload_result.stderr}")
                return False
                
        except Exception as e:
            print(f"PlatformIO flash failed: {e}")
            return False
    
    def run_streamlined_experiment(self, experiment_type, port, auto_flash=False):
        """Run experiment with maximum automation"""
        
        # Define experiments
        if experiment_type == "nstates":
            configs = [
                {'nstates': 2, 'ninputs': 2},
                {'nstates': 4, 'ninputs': 2},
                {'nstates': 8, 'ninputs': 2},
                {'nstates': 16, 'ninputs': 2},
                {'nstates': 32, 'ninputs': 2},
            ]
        elif experiment_type == "ninputs":
            configs = [
                {'nstates': 10, 'ninputs': 1},
                {'nstates': 10, 'ninputs': 2},
                {'nstates': 10, 'ninputs': 4},
                {'nstates': 10, 'ninputs': 8},
            ]
        else:
            raise ValueError("experiment_type must be 'nstates' or 'ninputs'")
        
        print(f"🚀 AUTOMATED {experiment_type.upper()} SCALING EXPERIMENT")
        print("=" * 60)
        print(f"Configurations: {len(configs)}")
        for i, config in enumerate(configs, 1):
            memory_kb = self.data_collector.estimate_memory_usage(config['nstates'], config['ninputs'])
            print(f"  {i}. NSTATES={config['nstates']}, NINPUTS={config['ninputs']} ({memory_kb:.2f} KB)")
        
        # Check tooling
        pio_available = self.check_platformio_available()
        project_path = self.find_platformio_project()
        
        if project_path:
            print(f"✓ Found STM32 project: {project_path}")
        else:
            print("⚠ STM32 project not found")
        
        print("\\n" + "=" * 60)
        print("AUTOMATION LEVEL:")
        if pio_available and auto_flash:
            print("🤖 MAXIMUM AUTOMATION - Will attempt automatic flashing")
        else:
            print("🔧 SEMI-AUTOMATED - Manual flashing required")
        print("=" * 60)
        
        # Confirm start
        response = input("\\nStart experiment? (y/N): ")
        if response.lower() != 'y':
            print("Experiment cancelled.")
            return
        
        # Process each configuration
        for i, config in enumerate(configs, 1):
            config_name = f"nstates_{config['nstates']}_ninputs_{config['ninputs']}"
            
            print("\\n" + "🔧" * 25)
            print(f"CONFIGURATION {i}/{len(configs)}: {config_name}")
            print("🔧" * 25)
            
            # Step 1: Update configuration files
            print("Step 1: Updating configuration files...")
            success = self.config_gen.generate_config(
                config['nstates'], config['ninputs'], target="stm32"
            )
            
            if not success:
                print(f"❌ Failed to update config for {config_name}")
                continue
            
            # Step 2: Generate workspace regeneration helper
            print("\\nStep 2: Generating workspace regeneration helper...")
            regen_script = self.generate_workspace_regeneration_script(
                config['nstates'], config['ninputs']
            )
            print(f"✓ Created: {regen_script}")
            
            # Step 3: Manual workspace regeneration
            print("\\n" + "⚠" * 30)
            print("MANUAL STEP REQUIRED:")
            print("⚠" * 30)
            print("You must regenerate the data workspace!")
            print("\\nQuick options:")
            print("1. 📓 Jupyter: Open TRO/safety_filter.ipynb")
            print(f"   - Set NSTATES = {config['nstates']}")
            print(f"   - Set NINPUTS = {config['ninputs']}")
            print("   - Run all cells")
            print("\\n2. 🐍 Python: Run the workspace generation script")
            print("\\n3. 🔄 Reuse: Copy workspace from previous similar config")
            
            input("\\nPress Enter when workspace is regenerated...")
            
            # Step 4: Flash firmware
            print("\\nStep 4: Flashing firmware...")
            
            flash_success = False
            if pio_available and auto_flash and project_path:
                flash_success = self.flash_with_platformio(project_path, port)
            
            if not flash_success:
                print("\\n" + "🔨" * 30)
                print("MANUAL FLASHING REQUIRED:")
                print("🔨" * 30)
                print("Flash the STM32 using:")
                print("1. 🎯 PlatformIO IDE: Upload button")
                print("2. 🔧 Arduino IDE: Upload button") 
                print("3. 📟 CLI: pio run --target upload")
                print(f"\\nProject path: {project_path or 'TRO/tinympc_f/tinympc_stm32_feather'}")
                
                input("\\nPress Enter when STM32 is flashed and ready...")
            
            # Step 5: Collect data
            print("\\nStep 5: Collecting data...")
            print("🔌 Make sure STM32 is connected and serial monitor is closed")
            time.sleep(2)  # Give time for STM32 to restart
            
            try:
                data = self.data_collector.collect_serial_data(
                    port, config_name, duration=60
                )
                
                if len(data) > 0:
                    print(f"✅ Successfully collected {len(data)} data points")
                else:
                    print("⚠️ No data collected. Check connections.")
                    
            except Exception as e:
                print(f"❌ Data collection failed: {e}")
                response = input("Continue with next config? (y/N): ")
                if response.lower() != 'y':
                    break
        
        # Final analysis
        print("\\n" + "🎉" * 30)
        print("EXPERIMENT COMPLETE!")
        print("🎉" * 30)
        
        response = input("Run analysis now? (y/N): ")
        if response.lower() == 'y':
            self.data_collector.analyze_data()
    
    def quick_collect_only(self, port, config_name, duration=60):
        """Quick data collection for when STM32 is already flashed"""
        print(f"🔌 Quick data collection for: {config_name}")
        
        try:
            data = self.data_collector.collect_serial_data(port, config_name, duration)
            print(f"✅ Collected {len(data)} data points")
            return len(data) > 0
        except Exception as e:
            print(f"❌ Collection failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Automated TinyMPC Experiment Runner')
    parser.add_argument('--experiment', choices=['nstates', 'ninputs'], 
                       help='Type of scaling experiment')
    parser.add_argument('--port', type=str, 
                       help='Serial port (e.g., /dev/ttyUSB0)')
    parser.add_argument('--auto-flash', action='store_true',
                       help='Attempt automatic flashing with PlatformIO')
    parser.add_argument('--quick-collect', type=str,
                       help='Quick data collection only (specify config name)')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing data only')
    
    args = parser.parse_args()
    
    runner = AutomatedExperimentRunner()
    
    if args.analyze:
        runner.data_collector.analyze_data()
        return
    
    if args.quick_collect:
        if not args.port:
            print("Error: --port required for data collection")
            return
        runner.quick_collect_only(args.port, args.quick_collect)
        return
    
    if not args.experiment or not args.port:
        print("Error: --experiment and --port required")
        print("\\nUsage examples:")
        print("  python automated_experiment_runner.py --experiment nstates --port /dev/ttyUSB0")
        print("  python automated_experiment_runner.py --experiment nstates --port /dev/ttyUSB0 --auto-flash")
        print("  python automated_experiment_runner.py --quick-collect nstates_8_ninputs_2 --port /dev/ttyUSB0")
        print("  python automated_experiment_runner.py --analyze")
        return
    
    runner.run_streamlined_experiment(args.experiment, args.port, args.auto_flash)

if __name__ == "__main__":
    main() 
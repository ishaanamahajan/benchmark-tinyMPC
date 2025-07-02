#!/usr/bin/env python3
"""
Scaling Experiments Data Collection Helper
Simplified script to collect the most critical missing data for TRO paper
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Check if we're in the right directory
    if not os.path.exists('safety_filter'):
        print("❌ ERROR: Run this script from the TRO directory")
        print("   Current dir should contain 'safety_filter/' folder")
        return False
    
    # Check if benchmark pipeline exists
    pipeline_path = "safety_filter/benchmark_pipeline.py"
    if not os.path.exists(pipeline_path):
        print(f"❌ ERROR: Missing {pipeline_path}")
        return False
    
    # Check Python packages
    try:
        import numpy, scipy, serial
        print("✅ Python packages (numpy, scipy, pyserial) found")
    except ImportError as e:
        print(f"❌ ERROR: Missing Python package: {e}")
        print("   Install with: pip install numpy scipy pyserial")
        return False
    
    # Check for USB device
    import glob
    usb_devices = glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/ttyACM*')
    if usb_devices:
        print(f"✅ USB device found: {usb_devices[0]}")
    else:
        print("⚠️  WARNING: No USB device detected")
        print("   Make sure your STM32 Feather/Teensy is connected")
    
    return True

def check_current_data():
    """Check what data has already been collected"""
    print("\n📊 Checking current data status...")
    
    data_dir = "safety_filter/benchmark_data"
    if not os.path.exists(data_dir):
        print(f"❌ No benchmark data directory found")
        return
    
    # Check key data files
    files_to_check = [
        ("tinympc_time_vs_dimensions.txt", "TinyMPC timing data"),
        ("tinympc_memory_vs_dimensions.txt", "TinyMPC memory data"),
        ("tinympc_horizon_analysis.txt", "TinyMPC horizon analysis"),
        ("osqp_time_vs_dimensions.txt", "OSQP timing data"),
        ("osqp_memory_vs_dimensions.txt", "OSQP memory data"),
        ("osqp_horizon_analysis.txt", "OSQP horizon analysis")
    ]
    
    missing_data = []
    for filename, description in files_to_check:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            # Check if file has real data (not just headers)
            with open(filepath, 'r') as f:
                lines = f.readlines()
                data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
                
            if len(data_lines) > 0:
                print(f"✅ {description}: {len(data_lines)} data points")
            else:
                print(f"🔴 {description}: File exists but empty")
                missing_data.append((filename, description))
        else:
            print(f"🔴 {description}: Missing")
            missing_data.append((filename, description))
    
    return missing_data

def run_data_collection(phase="priority"):
    """Run the data collection pipeline"""
    print(f"\n🚀 Starting data collection - {phase} phase...")
    
    os.chdir("safety_filter")
    
    if phase == "priority":
        # Collect most critical missing data first
        print("📋 Phase 1: Collecting TinyMPC timing data (priority)")
        dimensions = [4, 8, 12]  # Start with manageable dimensions
        cmd = ["python", "benchmark_pipeline.py", "--solver", "tinympc", "--dimensions"] + [str(d) for d in dimensions]
        
    elif phase == "tinympc_full":
        print("📋 Phase 2: Collecting full TinyMPC dataset")
        cmd = ["python", "benchmark_pipeline.py", "--solver", "tinympc"]
        
    elif phase == "osqp_full":
        print("📋 Phase 3: Collecting full OSQP dataset")
        cmd = ["python", "benchmark_pipeline.py", "--solver", "osqp"]
        
    elif phase == "all":
        print("📋 Phase 4: Collecting all remaining data")
        cmd = ["python", "benchmark_pipeline.py"]
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    print("📱 Make sure your microcontroller is connected and ready!")
    print("⏳ This will take some time as you'll need to upload sketches manually...")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Data collection completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Data collection failed: {e}")
        return False
    except KeyboardInterrupt:
        print("⏹️  Data collection interrupted by user")
        return False
    finally:
        os.chdir("..")

def show_summary():
    """Show summary of collected data"""
    print("\n📈 Data Collection Summary")
    print("=" * 50)
    
    # Check what data we have now
    missing_data = check_current_data()
    
    if not missing_data:
        print("🎉 All scaling experiment data collected!")
        print("\n📁 Ready for analysis - data files in safety_filter/benchmark_data/")
    else:
        print(f"\n🔄 Still missing {len(missing_data)} datasets:")
        for filename, description in missing_data:
            print(f"   • {description}")
        
        print("\n📋 Next steps:")
        print("   1. Continue with remaining data collection phases")
        print("   2. Check hardware connections if data collection failed")
        print("   3. Use manual process via safety_filter.ipynb if needed")

def main():
    """Main data collection workflow"""
    print("🔬 Safety Filter Scaling Experiments - Data Collection Helper")
    print("=" * 65)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please fix issues and try again.")
        return 1
    
    # Check current data status
    missing_data = check_current_data()
    
    if not missing_data:
        print("\n🎉 All data already collected!")
        show_summary()
        return 0
    
    print(f"\n📋 Found {len(missing_data)} missing datasets")
    print("\n🎯 Recommended data collection strategy:")
    print("   1. Priority: TinyMPC timing data (most critical for paper)")
    print("   2. Full TinyMPC dataset")
    print("   3. OSQP complete dataset")
    print("   4. Verification and analysis")
    
    # Ask user what to do
    print("\n❓ What would you like to do?")
    print("   1. Start with priority data (TinyMPC timing, dims 4,8,12)")
    print("   2. Collect full TinyMPC dataset")
    print("   3. Collect OSQP dataset") 
    print("   4. Collect all remaining data")
    print("   5. Just show me what's missing (no collection)")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            success = run_data_collection("priority")
        elif choice == "2":
            success = run_data_collection("tinympc_full")
        elif choice == "3":
            success = run_data_collection("osqp_full")
        elif choice == "4":
            success = run_data_collection("all")
        elif choice == "5":
            success = True  # Just show summary
        else:
            print("❌ Invalid choice")
            return 1
        
        # Show final summary
        show_summary()
        
        if success:
            print("\n✅ Data collection session completed!")
            print("📖 See TRO/README_SCALING_EXPERIMENTS.md for detailed instructions")
            return 0
        else:
            print("\n⚠️  Data collection had issues - check output above")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Data collection cancelled by user")
        return 1

if __name__ == "__main__":
    exit(main())
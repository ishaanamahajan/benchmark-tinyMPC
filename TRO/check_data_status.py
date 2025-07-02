#!/usr/bin/env python3
"""
Quick status check for scaling experiments data collection
Shows what data is available and what's missing
"""

import os
import glob
from pathlib import Path

def check_data_files():
    """Check status of all data files"""
    print("🔬 Safety Filter Scaling Experiments - Data Status")
    print("=" * 55)
    
    data_dir = "safety_filter/benchmark_data"
    
    if not os.path.exists(data_dir):
        print("❌ No benchmark data directory found")
        print("   Run data collection first!")
        return 0, 0, 6  # completed, empty, missing
    
    # Key data files we expect
    expected_files = {
        'tinympc_time_vs_dimensions.txt': 'TinyMPC Timing vs Dimensions',
        'tinympc_memory_vs_dimensions.txt': 'TinyMPC Memory vs Dimensions', 
        'tinympc_horizon_analysis.txt': 'TinyMPC Horizon Analysis',
        'osqp_time_vs_dimensions.txt': 'OSQP Timing vs Dimensions',
        'osqp_memory_vs_dimensions.txt': 'OSQP Memory vs Dimensions',
        'osqp_horizon_analysis.txt': 'OSQP Horizon Analysis'
    }
    
    print(f"📁 Checking {data_dir}/")
    print()
    
    completed = 0
    missing = 0
    empty = 0
    
    for filename, description in expected_files.items():
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            # Check if file has real data
            with open(filepath, 'r') as f:
                lines = f.readlines()
                data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
            
            if len(data_lines) > 0:
                print(f"✅ {description}")
                print(f"   📊 {len(data_lines)} data points in {filename}")
                completed += 1
            else:
                print(f"🟡 {description}")
                print(f"   📝 File exists but empty: {filename}")
                empty += 1
        else:
            print(f"🔴 {description}")
            print(f"   ❌ Missing: {filename}")
            missing += 1
        print()
    
    # Summary
    total_files = len(expected_files)
    print("📋 Summary:")
    print(f"   ✅ Complete: {completed}/{total_files}")
    print(f"   🟡 Empty: {empty}/{total_files}")
    print(f"   🔴 Missing: {missing}/{total_files}")
    print()
    
    # Progress percentage
    progress = (completed / total_files) * 100
    print(f"📈 Overall Progress: {progress:.1f}%")
    
    # Check for individual dimension files (more detailed data)
    print("\n🔍 Individual Dimension Data:")
    dim_files = sorted(glob.glob(os.path.join(data_dir, "*_dim_*_raw_data.txt")))
    if dim_files:
        print(f"   Found {len(dim_files)} individual dimension datasets:")
        for dim_file in dim_files:
            filename = os.path.basename(dim_file)
            # Count data points
            with open(dim_file, 'r') as f:
                lines = f.readlines()
                data_lines = [l for l in lines if not l.startswith('#') and ',' in l]
            print(f"   📊 {filename}: {len(data_lines)} points")
    else:
        print("   No individual dimension files found")
    
    print()
    return completed, empty, missing

def show_next_steps(completed, empty, missing):
    """Show recommended next steps based on current status"""
    if completed == 6:  # All 6 main files complete
        print("🎉 All scaling experiment data collected!")
        print("\n📈 Ready for analysis! You can:")
        print("   • Generate scaling comparison plots")
        print("   • Compute performance metrics")
        print("   • Prepare publication figures")
        
    elif completed >= 3:  # More than half done
        print("🔄 Good progress! Missing data:")
        if missing > 0:
            print("   • Run remaining data collection phases")
        if empty > 0:
            print("   • Re-run collections for empty files")
            
    elif completed >= 1:  # Some data collected
        print("🚀 Getting started! Recommendations:")
        print("   1. Focus on TinyMPC timing data (highest priority)")
        print("   2. Then collect OSQP dataset for comparison")
        print("   3. Verify data quality and completeness")
        
    else:  # No data yet
        print("🔧 Ready to start data collection!")
        print("   Run: python TRO/collect_scaling_data.py")
        print("   Or: cd safety_filter && python benchmark_pipeline.py")
    
    print(f"\n📖 See TRO/README_SCALING_EXPERIMENTS.md for detailed instructions")

def main():
    """Main status check"""
    completed, empty, missing = check_data_files()
    show_next_steps(completed, empty, missing)

if __name__ == "__main__":
    main()
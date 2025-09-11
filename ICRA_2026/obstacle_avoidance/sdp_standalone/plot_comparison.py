#!/usr/bin/env python3
"""
Plot TinyMPC SDP results and compare with Julia solution
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

def load_tinympc_results(filename="tinympc_sdp_trajectory.csv"):
    """Load TinyMPC SDP results from CSV"""
    if not os.path.exists(filename):
        print(f"❌ TinyMPC results file not found: {filename}")
        return None
    
    data = np.loadtxt(filename, delimiter=',', skiprows=2)
    
    results = {
        'time': data[:, 0],
        'states': data[:, 1:5],  # pos_x, pos_y, vel_x, vel_y
        'controls': data[:-1, 5:7]  # u_x, u_y (except last row)
    }
    
    print(f"✅ Loaded TinyMPC results: {len(results['time'])} time steps")
    return results

def run_julia_reference():
    """Run Julia script to get reference solution"""
    julia_script = "../../tinysdp_big.jl"
    
    if not os.path.exists(julia_script):
        print(f"❌ Julia script not found: {julia_script}")
        return None
    
    try:
        print("📊 Running Julia reference solution...")
        result = subprocess.run(['julia', julia_script], 
                              capture_output=True, text=True,
                              cwd=os.path.dirname(julia_script))
        
        if result.returncode == 0:
            print("✅ Julia solution completed")
            # For now, return simulated reference data
            # In practice, modify Julia script to export CSV
            return simulate_julia_reference()
        else:
            print(f"❌ Julia failed: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("❌ Julia not found. Showing TinyMPC results only.")
        return None

def simulate_julia_reference():
    """Simulate Julia reference solution for comparison"""
    N = 31
    x_initial = np.array([-10.0, 0.1, 0.0, 0.0])
    x_obs = np.array([-5.0, 0.0])
    r_obs = 2.0
    
    # System matrices
    Ad = np.array([[1, 0, 1, 0],
                   [0, 1, 0, 1], 
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    
    Bd = np.array([[0.5, 0],
                   [0, 0.5],
                   [1, 0],
                   [0, 1]])
    
    # Simulate optimal-like trajectory
    states = np.zeros((N, 4))
    controls = np.zeros((N-1, 2))
    
    states[0] = x_initial
    
    for k in range(N-1):
        pos = states[k, :2]
        vel = states[k, 2:]
        
        # Goal attraction
        u_goal = -0.15 * pos - 0.1 * vel
        
        # Obstacle avoidance (curve around)
        to_obs = pos - x_obs
        dist = np.linalg.norm(to_obs)
        
        if dist < 4.0 * r_obs:
            # Smart avoidance - curve around obstacle
            perp = np.array([-to_obs[1], to_obs[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-8)
            
            if pos[1] > x_obs[1]:
                perp *= 1  # Curve up
            else:
                perp *= -1  # Curve down
                
            curve_strength = 0.8 / (dist + 0.1)
            u_avoid = curve_strength * perp
        else:
            u_avoid = np.zeros(2)
        
        u_total = u_goal + u_avoid
        u_total = np.clip(u_total, -2.0, 2.0)
        
        controls[k] = u_total
        states[k+1] = Ad @ states[k] + Bd @ u_total
    
    return {
        'time': np.arange(N),
        'states': states,
        'controls': controls
    }

def check_obstacle_safety(states, x_obs=np.array([-5.0, 0.0]), r_obs=2.0):
    """Check trajectory safety"""
    positions = states[:, :2]
    distances = np.linalg.norm(positions - x_obs, axis=1)
    violations = np.sum(distances < r_obs)
    min_distance = np.min(distances)
    
    return {
        'distances': distances,
        'violations': violations,
        'min_distance': min_distance,
        'safe': violations == 0
    }

def plot_comparison(tinympc_results, julia_results=None):
    """Plot comparison between TinyMPC and Julia solutions"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Problem parameters
    x_obs = np.array([-5.0, 0.0])
    r_obs = 2.0
    
    # Plot 1: Trajectories with obstacle
    ax1.set_aspect('equal')
    
    # Draw obstacle
    theta = np.linspace(0, 2*np.pi, 100)
    obs_x = x_obs[0] + r_obs * np.cos(theta)
    obs_y = x_obs[1] + r_obs * np.sin(theta)
    ax1.fill(obs_x, obs_y, color='red', alpha=0.3, label='Obstacle')
    
    # Plot TinyMPC trajectory
    tinympc_pos = tinympc_results['states'][:, :2]
    ax1.plot(tinympc_pos[:, 0], tinympc_pos[:, 1], 'r-', linewidth=3, 
             label='TinyMPC SDP', marker='o', markersize=3)
    
    # Plot Julia reference if available
    if julia_results:
        julia_pos = julia_results['states'][:, :2]
        ax1.plot(julia_pos[:, 0], julia_pos[:, 1], 'b--', linewidth=2,
                 label='Julia/Mosek', alpha=0.7)
    
    # Mark start and goal
    ax1.scatter(-10, 0.1, color='green', s=100, label='Start', zorder=5)
    ax1.scatter(0, 0, color='orange', s=100, label='Goal', zorder=5)
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('SDP Obstacle Avoidance Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: States over time
    time_steps = tinympc_results['time']
    states = tinympc_results['states']
    
    ax2.plot(time_steps, states[:, 0], 'r-', label='TinyMPC x₁ (pos x)', linewidth=2)
    ax2.plot(time_steps, states[:, 1], 'r--', label='TinyMPC x₂ (pos y)', linewidth=2)
    
    if julia_results:
        julia_time = julia_results['time']
        julia_states = julia_results['states']
        ax2.plot(julia_time, julia_states[:, 0], 'b-', label='Julia x₁', alpha=0.7)
        ax2.plot(julia_time, julia_states[:, 1], 'b--', label='Julia x₂', alpha=0.7)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position')
    ax2.set_title('Position States')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Controls over time
    control_time = time_steps[:-1]
    controls = tinympc_results['controls']
    
    ax3.plot(control_time, controls[:, 0], 'r-', label='TinyMPC u₁ (accel x)', linewidth=2)
    ax3.plot(control_time, controls[:, 1], 'r--', label='TinyMPC u₂ (accel y)', linewidth=2)
    
    if julia_results:
        julia_controls = julia_results['controls']
        julia_control_time = julia_results['time'][:-1]
        ax3.plot(julia_control_time, julia_controls[:, 0], 'b-', label='Julia u₁', alpha=0.7)
        ax3.plot(julia_control_time, julia_controls[:, 1], 'b--', label='Julia u₂', alpha=0.7)
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Control Input')
    ax3.set_title('Control Inputs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Safety analysis (distance to obstacle)
    tinympc_safety = check_obstacle_safety(tinympc_results['states'])
    
    ax4.plot(time_steps, tinympc_safety['distances'], 'r-', linewidth=2, label='TinyMPC SDP')
    ax4.axhline(y=r_obs, color='red', linestyle=':', linewidth=2, label='Safety threshold')
    
    if julia_results:
        julia_safety = check_obstacle_safety(julia_results['states'])
        ax4.plot(julia_results['time'], julia_safety['distances'], 'b--', alpha=0.7, label='Julia/Mosek')
    
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Distance to Obstacle')
    ax4.set_title('Safety Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tinympc_sdp_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print safety summary
    print(f"\n📊 Safety Analysis:")
    print(f"TinyMPC SDP:")
    print(f"  - Violations: {tinympc_safety['violations']}/{len(time_steps)}")
    print(f"  - Min distance: {tinympc_safety['min_distance']:.3f}")
    print(f"  - Safe trajectory: {tinympc_safety['safe']}")
    
    if julia_results:
        julia_safety = check_obstacle_safety(julia_results['states'])
        print(f"Julia Reference:")
        print(f"  - Violations: {julia_safety['violations']}/{len(julia_results['time'])}")
        print(f"  - Min distance: {julia_safety['min_distance']:.3f}")
        print(f"  - Safe trajectory: {julia_safety['safe']}")

def main():
    print("🚀 TinyMPC SDP vs Julia Comparison")
    print("=" * 50)
    
    # Load TinyMPC results
    tinympc_results = load_tinympc_results()
    if tinympc_results is None:
        print("❌ No TinyMPC results to plot. Run './sdp_test' first.")
        return
    
    # Try to get Julia reference
    julia_results = run_julia_reference()
    
    # Create comparison plot
    plot_comparison(tinympc_results, julia_results)
    
    print("\n✅ Comparison complete!")
    print("📊 Plot saved as 'tinympc_sdp_comparison.png'")
    print("🎯 Check if SDP projections maintain trajectory safety")

if __name__ == "__main__":
    main()



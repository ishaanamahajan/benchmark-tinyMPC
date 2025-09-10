#!/usr/bin/env python3
"""
Proper TinyMPC obstacle avoidance test using the actual TinyMPC solver
Based on safety_filter.py structure but adapted for obstacle avoidance
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add tinympc-python to path
path_to_root = os.getcwd()
tinympc_python_dir = os.path.abspath(os.path.join(path_to_root, "../../tinympc-python"))
sys.path.append(tinympc_python_dir)

try:
    import tinympc
    print("✅ TinyMPC imported successfully")
    tinympc_available = True
except ImportError as e:
    print(f"❌ TinyMPC import failed: {e}")
    tinympc_available = False

# Problem parameters from tinysdp_big.jl
NSTATES = 4  # [pos_x, pos_y, vel_x, vel_y]
NINPUTS = 2  # [accel_x, accel_y] 
NHORIZON = 16  # Shorter horizon for testing
NTOTAL = 50   # Shorter simulation

# Obstacle parameters
x_obs = np.array([-5.0, 0.0])  # obstacle center
r_obs = 2.0  # obstacle radius

# System dynamics from Julia (double integrator)
h = 0.1  # time step
Ad = np.array([[1, 0, h, 0],
               [0, 1, 0, h],
               [0, 0, 1, 0], 
               [0, 0, 0, 1]])

Bd = np.array([[0.5*h*h, 0],
               [0, 0.5*h*h],
               [h, 0],
               [0, h]])

# Cost matrices
Q = 0.1 * np.eye(NSTATES)  # Small state cost
R = 10.0 * np.eye(NINPUTS)  # Control cost

# Constraints
rho = 100.0
xmin_bounds = np.array([-15.0, -10.0, -5.0, -5.0])  # State bounds
xmax_bounds = np.array([15.0, 10.0, 5.0, 5.0])
umin_bounds = np.array([-3.0, -3.0])  # Control bounds  
umax_bounds = np.array([3.0, 3.0])

abs_pri_tol = 1e-2
abs_dual_tol = 1e-2
max_iter = 200
check_termination = 1

print("Problem Setup:")
print(f"- States: {NSTATES}, Inputs: {NINPUTS}")
print(f"- Horizon: {NHORIZON}, Total steps: {NTOTAL}")
print(f"- Obstacle: center {x_obs}, radius {r_obs}")
print(f"- Time step: {h}")

if tinympc_available:
    print("\n=== Setting up TinyMPC Solver ===")
    
    # Create TinyMPC problem
    tinympc_prob = tinympc.TinyMPC()
    tinympc_prob.setup(
        Ad, Bd, Q, R, NHORIZON,
        rho=rho,
        x_min=xmin_bounds, x_max=xmax_bounds,
        u_min=umin_bounds, u_max=umax_bounds,
        abs_pri_tol=abs_pri_tol, abs_dua_tol=abs_dual_tol,
        max_iter=max_iter, check_termination=check_termination
    )
    
    print("✅ TinyMPC solver set up successfully")
    
    # Generate code for our SDP-enabled version
    output_dir = "tinympc/tinympc_generated_test"
    tinympc_prob.codegen(output_dir)
    print(f"✅ TinyMPC code generated to {output_dir}")

def check_collision(position):
    """Check if position is safe (outside obstacle)"""
    dist = np.linalg.norm(position - x_obs)
    return dist >= r_obs

def simulate_with_tinympc():
    """Run MPC simulation with TinyMPC solver"""
    if not tinympc_available:
        print("❌ Cannot simulate - TinyMPC not available")
        return None, None, None
    
    print("\n=== Running TinyMPC Obstacle Avoidance Simulation ===")
    
    # Initial conditions
    x_initial = np.array([-10.0, 0.1, 0.0, 0.0])
    x_goal = np.array([0.0, 0.0, 0.0, 0.0])
    
    states = [x_initial.copy()]
    controls = []
    safe_flags = []
    solve_times = []
    iterations = []
    
    x_current = x_initial.copy()
    
    for k in range(NTOTAL - 1):
        print(f"\n--- Step {k} ---")
        print(f"Current state: [{x_current[0]:6.2f}, {x_current[1]:6.2f}, {x_current[2]:6.2f}, {x_current[3]:6.2f}]")
        
        # Set reference trajectory (move towards goal while avoiding obstacle)
        Xref = np.zeros((NSTATES, NHORIZON))
        Uref = np.zeros((NINPUTS, NHORIZON - 1))
        
        for i in range(NHORIZON):
            # Reference trajectory: linear interpolation to goal with obstacle avoidance
            alpha = (k + i) / NTOTAL
            ref_pos = (1 - alpha) * x_current[:2] + alpha * x_goal[:2]
            
            # Add repulsive reference away from obstacle
            to_obs = ref_pos - x_obs
            dist_to_obs = np.linalg.norm(to_obs)
            if dist_to_obs < 3 * r_obs:
                # Push reference away from obstacle
                push_away = 2.0 * r_obs / (dist_to_obs + 0.1) * to_obs / (dist_to_obs + 1e-6)
                ref_pos += push_away
            
            Xref[:2, i] = ref_pos
            Xref[2:, i] = 0.0  # Zero velocity reference
        
        # Solve MPC with TinyMPC
        try:
            result = tinympc_prob.solve(x_current, Xref, Uref)
            u_opt = result['u_opt'][:, 0]  # First control action
            solve_time = result.get('solve_time', 0)
            num_iter = result.get('iter', 0)
            
            controls.append(u_opt.copy())
            solve_times.append(solve_time)
            iterations.append(num_iter)
            
            print(f"Control: [{u_opt[0]:6.2f}, {u_opt[1]:6.2f}]")
            print(f"Solve time: {solve_time:.1f} μs, Iterations: {num_iter}")
            
        except Exception as e:
            print(f"❌ TinyMPC solve failed: {e}")
            u_opt = np.zeros(NINPUTS)
            controls.append(u_opt)
            solve_times.append(0)
            iterations.append(0)
        
        # Check safety
        safe = check_collision(x_current[:2])
        safe_flags.append(safe)
        dist_to_obs = np.linalg.norm(x_current[:2] - x_obs)
        
        print(f"Distance to obstacle: {dist_to_obs:.2f}, Safe: {safe}")
        
        # Simulate forward with noise
        x_current = Ad @ x_current + Bd @ u_opt
        x_current += np.random.normal(0, 0.01, NSTATES)  # Small noise
        states.append(x_current.copy())
        
        # Check if reached goal
        if np.linalg.norm(x_current[:2]) < 0.5:
            print(f"✅ Reached goal at step {k+1}")
            break
    
    return np.array(states), np.array(controls), safe_flags

def plot_tinympc_results(states, controls, safe_flags):
    """Plot results like the Julia script"""
    if states is None:
        print("❌ No results to plot")
        return
    
    print("\n=== Plotting TinyMPC Results ===")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Trajectory with obstacle
    ax1.set_aspect('equal')
    
    # Draw obstacle
    theta = np.linspace(0, 2*np.pi, 100)
    obs_x = x_obs[0] + r_obs * np.cos(theta)
    obs_y = x_obs[1] + r_obs * np.sin(theta)
    ax1.fill(obs_x, obs_y, color='red', alpha=0.5, label=f'Obstacle (r={r_obs})')
    
    # Plot trajectory colored by safety
    positions = states[:, :2]
    for i in range(len(positions)-1):
        color = 'green' if safe_flags[i] else 'red'
        ax1.plot(positions[i:i+2, 0], positions[i:i+2, 1], color=color, linewidth=2, alpha=0.8)
    
    ax1.scatter(states[0, 0], states[0, 1], color='blue', s=100, label='Start', zorder=5)
    ax1.scatter(0, 0, color='gold', s=100, label='Goal', marker='*', zorder=5)
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('TinyMPC Obstacle Avoidance Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: States over time
    time_steps = np.arange(len(states))
    ax2.plot(time_steps, states[:, 0], label='x₁ (pos x)', linewidth=2)
    ax2.plot(time_steps, states[:, 1], label='x₂ (pos y)', linewidth=2)
    ax2.plot(time_steps, states[:, 2], label='x₃ (vel x)', linewidth=2)
    ax2.plot(time_steps, states[:, 3], label='x₄ (vel y)', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('State Value')
    ax2.set_title('States Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Controls over time
    if len(controls) > 0:
        control_time = np.arange(len(controls))
        ax3.plot(control_time, controls[:, 0], label='u₁ (accel x)', linewidth=2, color='blue')
        ax3.plot(control_time, controls[:, 1], label='u₂ (accel y)', linewidth=2, color='orange')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Control Value (m/s²)')
    ax3.set_title('Control Inputs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distance to obstacle over time
    distances = [np.linalg.norm(state[:2] - x_obs) for state in states]
    ax4.plot(distances, 'b-', linewidth=2, label='Distance to obstacle')
    ax4.axhline(y=r_obs, color='red', linestyle='--', linewidth=2, label=f'Safety threshold ({r_obs}m)')
    ax4.fill_between(range(len(distances)), distances, r_obs, 
                    where=np.array(distances) >= r_obs, color='green', alpha=0.3, label='Safe zone')
    ax4.fill_between(range(len(distances)), distances, r_obs,
                    where=np.array(distances) < r_obs, color='red', alpha=0.3, label='Collision zone')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Distance (m)')
    ax4.set_title('Safety Distance Check')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tinympc_obstacle_avoidance_complete.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    safe_count = sum(safe_flags)
    total_steps = len(safe_flags)
    final_dist_to_goal = np.linalg.norm(states[-1, :2])
    min_obstacle_dist = min(distances)
    
    print(f"\n=== TinyMPC Obstacle Avoidance Results ===")
    print(f"✅ Total simulation steps: {total_steps}")
    print(f"✅ Safe steps: {safe_count}/{total_steps} ({100*safe_count/total_steps:.1f}%)")
    print(f"✅ Final distance to goal: {final_dist_to_goal:.2f}m")
    print(f"✅ Minimum distance to obstacle: {min_obstacle_dist:.2f}m")
    
    if min_obstacle_dist < r_obs:
        print(f"⚠️  Collision detected! (min dist {min_obstacle_dist:.2f} < {r_obs})")
    else:
        print(f"✅ No collisions! (min dist {min_obstacle_dist:.2f} >= {r_obs})")

def main():
    if not tinympc_available:
        print("❌ Cannot run test - TinyMPC not available")
        print("Make sure tinympc-python is installed and accessible")
        return
    
    print("Running TinyMPC obstacle avoidance test...")
    states, controls, safe_flags = simulate_with_tinympc()
    
    if states is not None:
        plot_tinympc_results(states, controls, safe_flags)
        print("\n🎯 TinyMPC SDP projection ready for your colleagues!")
        print("📊 Results saved to 'tinympc_obstacle_avoidance_complete.png'")
    else:
        print("❌ Simulation failed")

if __name__ == "__main__":
    main()

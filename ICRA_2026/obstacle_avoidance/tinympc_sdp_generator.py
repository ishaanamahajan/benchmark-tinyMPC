#!/usr/bin/env python3
"""
TinyMPC SDP Workspace Generator for Obstacle Avoidance
Converts the Julia SDP problem (tinysdp_big.jl) to TinyMPC format with PSD projections
"""

import tinympc
import numpy as np
import os
from pathlib import Path

class TinyMPCSDPGenerator:
    def __init__(self):
        # Problem parameters from tinysdp_big.jl
        self.N = 31  # number of timesteps
        self.x_initial = np.array([-10.0, 0.1, 0.0, 0.0])
        self.x_obs = np.array([-5.0, 0.0])  # obstacle center
        self.r_obs = 2.0  # obstacle radius
        
        # System dimensions
        self.nx = 4  # physical state dimension
        self.nu = 2  # control dimension
        self.nxx = 16  # number of elements in xx' (4x4 matrix)
        self.nxu = 8   # number of elements in ux'
        self.nux = 8   # number of elements in xu'
        self.nuu = 4   # number of elements in uu'
        
        # Extended dimensions for SDP
        self.nx_ext = self.nx + self.nxx  # 4 + 16 = 20 (physical state + second moments)
        self.nu_ext = self.nu + self.nxu + self.nux + self.nuu  # 2 + 8 + 8 + 4 = 22
        
        # Weights from Julia
        self.q_xx = 0.1
        self.r_xx = 10.0
        self.R_xx = 500.0
        self.reg = 1e-6
        
        # System matrices from Julia (discrete-time double integrator)
        self.Ad = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1], 
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        self.Bd = np.array([[0.5, 0],
                           [0, 0.5],
                           [1, 0],
                           [0, 1]])
        
        print("🔧 TinyMPC SDP Generator Initialized")
        print(f"   Physical states: {self.nx}, Controls: {self.nu}")
        print(f"   Extended states: {self.nx_ext}, Extended controls: {self.nu_ext}")
        print(f"   Horizon: {self.N}, Obstacle: center {self.x_obs}, radius {self.r_obs}")
    
    def create_extended_dynamics(self):
        """Create extended system matrices for SDP formulation"""
        # Extended A matrix: [Ad, 0; 0, kron(Ad, Ad)]
        A_ext = np.zeros((self.nx_ext, self.nx_ext))
        A_ext[:self.nx, :self.nx] = self.Ad
        A_ext[self.nx:, self.nx:] = np.kron(self.Ad, self.Ad)
        
        # Extended B matrix: [Bd, 0; 0, kron(Bd,Ad), kron(Ad,Bd), kron(Bd,Bd)]
        B_ext = np.zeros((self.nx_ext, self.nu_ext))
        B_ext[:self.nx, :self.nu] = self.Bd
        
        # Second moment dynamics
        B_ext[self.nx:, self.nu:self.nu+self.nxu] = np.kron(self.Bd, self.Ad)
        B_ext[self.nx:, self.nu+self.nxu:self.nu+self.nxu+self.nux] = np.kron(self.Ad, self.Bd)
        B_ext[self.nx:, self.nu+self.nxu+self.nux:] = np.kron(self.Bd, self.Bd)
        
        return A_ext, B_ext
    
    def create_extended_costs(self):
        """Create extended cost matrices for SDP formulation"""
        # Q matrix: regularization + second moment cost
        Q_ext = self.reg * np.eye(self.nx_ext)
        Q_ext[self.nx:, self.nx:] += self.q_xx * np.eye(self.nxx)
        
        # R matrix: regularization + input costs
        R_ext = self.reg * np.eye(self.nu_ext)
        R_ext[self.nu+self.nxu+self.nux:, self.nu+self.nxu+self.nux:] += self.R_xx * np.eye(self.nuu)
        
        # Linear cost terms (from Julia formulation)
        q_ext = np.zeros(self.nx_ext)
        q_ext[self.nx:] = np.vectorize(lambda i, j: self.q_xx if i == j else 0)(
            *np.meshgrid(range(self.nx), range(self.nx))).flatten()
        
        r_ext = np.zeros(self.nu_ext)
        r_ext[self.nu+self.nxu+self.nux:] = np.vectorize(lambda i, j: self.r_xx if i == j else 0)(
            *np.meshgrid(range(self.nu), range(self.nu))).flatten()
        
        return Q_ext, R_ext, q_ext, r_ext
    
    def create_initial_condition(self):
        """Create initial condition: [x_initial; vec(x_initial * x_initial')]"""
        x0_ext = np.zeros(self.nx_ext)
        x0_ext[:self.nx] = self.x_initial
        x0_ext[self.nx:] = (self.x_initial[:, np.newaxis] @ self.x_initial[np.newaxis, :]).flatten()
        return x0_ext
    
    def create_constraints(self):
        """Create constraint bounds for obstacle avoidance and box constraints"""
        # State bounds (generous for now, obstacle avoidance handled by SDP constraint)
        x_min = np.full(self.nx_ext, -100.0)
        x_max = np.full(self.nx_ext, 100.0)
        
        # Control bounds
        u_min = np.full(self.nu_ext, -10.0)
        u_max = np.full(self.nu_ext, 10.0)
        
        # Tighten physical control bounds
        u_min[:self.nu] = -2.0  # Physical control limits
        u_max[:self.nu] = 2.0
        
        return x_min, x_max, u_min, u_max
    
    def generate_tinympc_workspace(self, output_dir="tinympc_sdp_generated"):
        """Generate TinyMPC workspace for SDP obstacle avoidance"""
        print("🔧 Generating TinyMPC SDP workspace...")
        
        # Create extended system
        A_ext, B_ext = self.create_extended_dynamics()
        Q_ext, R_ext, q_ext, r_ext = self.create_extended_costs()
        x0_ext = self.create_initial_condition()
        x_min, x_max, u_min, u_max = self.create_constraints()
        
        print(f"   Extended system: A {A_ext.shape}, B {B_ext.shape}")
        print(f"   Extended costs: Q {Q_ext.shape}, R {R_ext.shape}")
        
        # Setup TinyMPC problem
        tinympc_prob = tinympc.TinyMPC()
        tinympc_prob.setup(
            A_ext, B_ext, Q_ext, R_ext, self.N,
            rho=100.0,  # ADMM penalty parameter
            x_min=x_min, x_max=x_max,
            u_min=u_min, u_max=u_max,
            abs_pri_tol=1e-3, abs_dua_tol=1e-3,
            max_iter=1000, check_termination=10
        )
        
        # Generate code
        os.makedirs(output_dir, exist_ok=True)
        tinympc_prob.codegen(output_dir)
        
        # Copy to TinyMPC projects
        self.copy_to_mcu_projects(output_dir)
        
        # Generate problem-specific headers
        self.generate_problem_headers(output_dir)
        
        print(f"✅ TinyMPC SDP workspace generated in {output_dir}")
        return True
    
    def copy_to_mcu_projects(self, generated_dir):
        """Copy generated files to MCU project directories"""
        projects = [
            "tinympc/tinympc_teensy",
            "tinympc/tinympc_stm32"
        ]
        
        for project in projects:
            if os.path.exists(project):
                print(f"   📁 Copying to {project}")
                
                # Copy main workspace file
                os.system(f'cp {generated_dir}/src/tiny_data.cpp {project}/src/tiny_data_workspace.cpp')
                
                # Copy headers
                if os.path.exists(f'{generated_dir}/tinympc/tiny_data.hpp'):
                    os.system(f'cp {generated_dir}/tinympc/tiny_data.hpp {project}/src/tinympc/glob_opts.hpp')
    
    def generate_problem_headers(self, output_dir):
        """Generate problem-specific header files with obstacle parameters"""
        header_content = f'''#pragma once
// Auto-generated SDP Obstacle Avoidance Problem Parameters
// Generated from Julia tinysdp_big.jl

// Problem dimensions
#define NX_PHYSICAL {self.nx}      // Physical state dimension
#define NU_PHYSICAL {self.nu}      // Physical control dimension  
#define NX_EXTENDED {self.nx_ext}  // Extended state dimension (includes second moments)
#define NU_EXTENDED {self.nu_ext}  // Extended control dimension
#define NHORIZON {self.N}          // Time horizon

// Obstacle parameters
#define OBS_CENTER_X {self.x_obs[0]}f
#define OBS_CENTER_Y {self.x_obs[1]}f  
#define OBS_RADIUS {self.r_obs}f

// Initial condition
#define X0_POS_X {self.x_initial[0]}f
#define X0_POS_Y {self.x_initial[1]}f
#define X0_VEL_X {self.x_initial[2]}f
#define X0_VEL_Y {self.x_initial[3]}f

// Cost weights
#define Q_XX {self.q_xx}f
#define R_XX {self.r_xx}f
#define R_UU {self.R_xx}f

// Enable SDP projection
#define ENABLE_SDP_PROJECTION 1
#define SDP_MATRIX_SIZE 4  // Size of PSD constraint matrix [1 x'; x XX]

// Obstacle avoidance constraint function
// Returns constraint value: should be >= 0 for feasibility
// constraint = tr(XX[0:1, 0:1]) - 2*obs_center'*x[0:1] + obs_center'*obs_center - radius^2
inline float obstacle_constraint(const float* x, const float* XX_flat) {{
    // Extract position
    float px = x[0];
    float py = x[1];
    
    // Extract relevant second moments XX[0:1, 0:1] 
    // XX is stored as flattened 4x4 matrix in column-major order
    float XX_00 = XX_flat[0];   // XX(0,0)
    float XX_11 = XX_flat[5];   // XX(1,1) 
    
    // Compute constraint: tr(XX[0:1,0:1]) - 2*obs'*x + obs'*obs - r^2
    float trace_XX = XX_00 + XX_11;
    float obs_term = {self.x_obs[0]}f * {self.x_obs[0]}f + {self.x_obs[1]}f * {self.x_obs[1]}f;
    float pos_term = 2.0f * ({self.x_obs[0]}f * px + {self.x_obs[1]}f * py);
    
    return trace_XX - pos_term + obs_term - {self.r_obs}f * {self.r_obs}f;
}}
'''
        
        with open(f'{output_dir}/sdp_problem.hpp', 'w') as f:
            f.write(header_content)
        
        print(f"   📄 Generated sdp_problem.hpp with obstacle parameters")

def main():
    print("🚀 TinyMPC SDP Obstacle Avoidance Workspace Generator")
    print("=" * 60)
    
    # Create generator
    generator = TinyMPCSDPGenerator()
    
    # Generate workspace
    success = generator.generate_tinympc_workspace()
    
    if success:
        print("\n✅ SDP workspace generation complete!")
        print("📁 Files ready for:")
        print("   • TinyMPC Teensy: tinympc/tinympc_teensy/")
        print("   • TinyMPC STM32: tinympc/tinympc_stm32/")
        print("🎯 Next steps:")
        print("   1. Upload sketch to microcontroller")
        print("   2. Test SDP projection with obstacle avoidance")
        print("   3. Compare with Julia/Mosek solution")
    else:
        print("❌ SDP workspace generation failed!")

if __name__ == "__main__":
    main()



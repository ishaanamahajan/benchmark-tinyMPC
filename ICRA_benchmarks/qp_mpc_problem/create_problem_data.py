#!/usr/bin/env python3
"""
Create problem data for gen_code_osqp.py
Input dim = 16, State dim = 10, N = 10
"""

import numpy as np

# Problem parameters
nx = 10  # state dimension
nu = 16  # input dimension  
N = 10   # horizon (Nh)
Nsim = 100  # simulation steps

# Generate random system matrices
np.random.seed(42)  # For reproducibility

# System dynamics: x+ = Ax + Bu
A = np.random.randn(nx, nx) * 0.1
A = A + np.eye(nx)  # Make it stable
B = np.random.randn(nx, nu) * 0.5

# Cost matrices
Q = np.eye(nx)
R = 0.1 * np.eye(nu)
Qf = Q  # Terminal cost

# Bounds
xmin = -5.0 * np.ones((nx, 1))
xmax = 5.0 * np.ones((nx, 1))
umin = -3.0 * np.ones((nu, 1))
umax = 3.0 * np.ones((nu, 1))

# Reference trajectory (transpose to match expected format)
x_bar = np.zeros((nx, Nsim + N + 1))
for i in range(Nsim + N + 1):
    x_bar[:, i] = 0.1 * np.sin(0.1 * i) * np.ones(nx)

print(f"Problem dimensions: nx={nx}, nu={nu}, N={N}")
print(f"Reference trajectory shape: {x_bar.shape}")

# Save problem data in the expected format, matching the directory structure
import os
problem_dir = "random_problems/prob_nx_10_nu_16"
os.makedirs(problem_dir, exist_ok=True)

np.savez(f'{problem_dir}/rand_prob_osqp_params.npz',
         nx=nx,
         nu=nu, 
         Nh=N,
         Nsim=Nsim,
         A=A,
         B=B,
         Q=Q,
         R=R,
         Qf=Qf,
         x_bar=x_bar,
         xmin=xmin,
         xmax=xmax,
         umin=umin,
         umax=umax)

print("✓ Created rand_prob_osqp_params.npz with:")
print(f"  - nx={nx}, nu={nu}, Nh={N}, Nsim={Nsim}")
print(f"  - A shape: {A.shape}, B shape: {B.shape}")
print(f"  - Q shape: {Q.shape}, R shape: {R.shape}")
print(f"  - x_bar shape: {x_bar.shape}")
print(f"  - Bounds: xmin/xmax shape: {xmin.shape}, umin/umax shape: {umin.shape}")
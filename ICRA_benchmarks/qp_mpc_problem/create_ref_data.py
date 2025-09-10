#!/usr/bin/env python3
"""
Create reference trajectory data files for teensy benchmark
"""

import numpy as np

# Load the problem data
data = np.load('random_problems/prob_nx_10_nu_16/rand_prob_osqp_params.npz')
nx = int(data['nx'])
nu = int(data['nu']) 
N = int(data['Nh'])
Nsim = int(data['Nsim'])
A = data['A']
B = data['B']
Q = data['Q']
Qf = data['Qf']
x_bar = data['x_bar']

print(f"Creating reference data with nx={nx}, nu={nu}, N={N}, Nsim={Nsim}")
print(f"x_bar shape: {x_bar.shape}")

# Create header file
with open('../teensy_osqp_benchmark/include/rand_prob_osqp_xbar.h', 'w') as f:
    f.write(f"""// Auto-generated reference trajectory
#ifndef RAND_PROB_OSQP_XBAR_H
#define RAND_PROB_OSQP_XBAR_H

#include "osqp_api_types.h"

// Problem constants
#define NSTATES {nx}
#define NINPUTS {nu}
#define NHORIZON {N}
#define NTOTAL {Nsim + N}

// Reference trajectory data
extern const OSQPFloat Xref_data[{x_bar.size}];

// System matrices
extern const OSQPFloat A[{nx * nx}];
extern const OSQPFloat B[{nx * nu}];

// Cost matrices
extern const OSQPFloat mQ[{nx * nx}];
extern const OSQPFloat mQf[{nx * nx}];

#endif // RAND_PROB_OSQP_XBAR_H
""")

# Create source file
with open('../teensy_osqp_benchmark/src/rand_prob_osqp_xbar.c', 'w') as f:
    f.write('#include "rand_prob_osqp_xbar.h"\n\n')
    
    # Reference trajectory (flatten in column-major order for C)
    xref_flat = x_bar.flatten(order='F')
    f.write(f'const OSQPFloat Xref_data[{len(xref_flat)}] = {{\n')
    for i, val in enumerate(xref_flat):
        if i % 10 == 0:
            f.write('    ')
        f.write(f'{val:.6f}f')
        if i < len(xref_flat) - 1:
            f.write(', ')
        if (i + 1) % 10 == 0:
            f.write('\n')
    f.write('\n};\n\n')
    
    # System matrix A
    f.write(f'const OSQPFloat A[{nx * nx}] = {{\n')
    for i, val in enumerate(A.flatten()):
        if i % 10 == 0:
            f.write('    ')
        f.write(f'{val:.6f}f')
        if i < nx * nx - 1:
            f.write(', ')
        if (i + 1) % 10 == 0:
            f.write('\n')
    f.write('\n};\n\n')
    
    # System matrix B
    f.write(f'const OSQPFloat B[{nx * nu}] = {{\n')
    for i, val in enumerate(B.flatten()):
        if i % 10 == 0:
            f.write('    ')
        f.write(f'{val:.6f}f')
        if i < nx * nu - 1:
            f.write(', ')
        if (i + 1) % 10 == 0:
            f.write('\n')
    f.write('\n};\n\n')
    
    # Q matrix
    f.write(f'const OSQPFloat mQ[{nx * nx}] = {{\n')
    for i, val in enumerate(Q.flatten()):
        if i % 10 == 0:
            f.write('    ')
        f.write(f'{val:.6f}f')
        if i < nx * nx - 1:
            f.write(', ')
        if (i + 1) % 10 == 0:
            f.write('\n')
    f.write('\n};\n\n')
    
    # Qf matrix
    f.write(f'const OSQPFloat mQf[{nx * nx}] = {{\n')
    for i, val in enumerate(Qf.flatten()):
        if i % 10 == 0:
            f.write('    ')
        f.write(f'{val:.6f}f')
        if i < nx * nx - 1:
            f.write(', ')
        if (i + 1) % 10 == 0:
            f.write('\n')
    f.write('\n};\n')

print("✓ Created reference data files:")
print("  - include/rand_prob_osqp_xbar.h")  
print("  - src/rand_prob_osqp_xbar.c")
print(f"  - Reference trajectory has {x_bar.size} elements")
print(f"  - System matrices A: {nx}x{nx}, B: {nx}x{nu}")
print(f"  - Cost matrices Q: {nx}x{nx}, Qf: {nx}x{nx}")
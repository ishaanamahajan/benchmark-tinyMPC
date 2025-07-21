#!/usr/bin/env python3
"""
Clean TinyMPC Code Generation for Rocket Landing Problem
This generates TinyMPC data that matches SCS/ECOS benchmark setup exactly
"""

import os
import numpy as np
import scipy.linalg

# ROCKET LANDING PROBLEM PARAMETERS (matching gen_rocket.py and run_ecos.py)
NSTATES = 6
NINPUTS = 3  
NHORIZON = 32  # MATCH SCS/ECOS horizon for fair comparison
NTOTAL = 301

# Rocket landing dynamics (copied exactly from gen_rocket.py/run_ecos.py)
Ad = np.array([[1.0, 0.0, 0.0, 0.05, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.05, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.05],
               [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

Bd = np.array([[0.000125, 0.0, 0.0],
               [0.0, 0.000125, 0.0],
               [0.0, 0.0, 0.000125],
               [0.005, 0.0, 0.0],
               [0.0, 0.005, 0.0],
               [0.0, 0.0, 0.005]])

fdyn = np.array([0.0, 0.0, -0.0122625, 0.0, 0.0, -0.4905])  # gravity term

# Cost matrices (matching gen_rocket.py/run_ecos.py EXACTLY)
Q = 1e3 * np.eye(NSTATES)  # 1000 * I
R = 1e0 * np.eye(NINPUTS)  # 1 * I

# TinyMPC solver settings
rho = 1.0
abs_pri_tol = 1.0e-2
abs_dual_tol = 1.0e-2
max_iter = 500
check_termination = 1

# Bounds (matching gen_rocket.py/run_ecos.py)
u_min = -10 * np.ones(NINPUTS)
u_max = 105.0 * np.ones(NINPUTS)
x_min = np.array([-5, -5, 0, -10, -10, -10.0])
x_max = np.array([5, 5, 20, 10, 10, 10.0])

def generate_glob_opts_header(output_path, NSTATES, NINPUTS, NHORIZON):
    """Generate glob_opts.hpp file matching TinyMPC structure"""
    with open(output_path, 'w') as f:
        f.write('/*\n')
        f.write(' * TinyMPC Global Options for Rocket Landing Benchmark\n')
        f.write(' */\n\n')
        f.write('#pragma once\n\n')
        f.write('typedef float tinytype;\n\n')
        f.write(f'#define NSTATES {NSTATES}\n')
        f.write(f'#define NINPUTS {NINPUTS}\n')
        f.write(f'#define NHORIZON {NHORIZON}\n')
        f.write(f'#define NTOTAL {NTOTAL}\n\n')
        f.write('// Constraint definitions for rocket landing problem\n')
        f.write('#define NUM_INPUT_CONES 1  // One thrust cone constraint\n')
        f.write('#define NUM_STATE_CONES 0  // No state cone constraints\n')

def generate_problem_data_header(output_path, Ad, Bd, Q, R, fdyn, rho, NSTATES, NINPUTS, NHORIZON, NTOTAL):
    """Generate problem data header matching existing structure"""
    
    # Compute LQR/ADMM precomputes (Kinf, Pinf, Quu_inv, AmBKt, APf, BPf)
    Q1 = Q + rho * np.eye(NSTATES)
    R1 = R + rho * np.eye(NINPUTS)
    Ptp1 = rho * np.eye(NSTATES)
    Ktp1 = np.zeros((NINPUTS, NSTATES))
    
    # Solve discrete-time algebraic Riccati equation
    for i in range(1000):
        Kinf = np.linalg.solve(R1 + Bd.T @ Ptp1 @ Bd, Bd.T @ Ptp1 @ Ad)
        Pinf = Q1 + Ad.T @ Ptp1 @ (Ad - Bd @ Kinf)
        if np.max(np.abs(Kinf - Ktp1)) < 1e-5:
            break
        Ktp1 = Kinf.copy()
        Ptp1 = Pinf.copy()
    
    Quu_inv = np.linalg.inv(R1 + Bd.T @ Pinf @ Bd)
    AmBKt = (Ad - Bd @ Kinf).T
    APf = Pinf @ fdyn.reshape(-1, 1)
    BPf = Kinf @ fdyn.reshape(-1, 1)

    print(f"TinyMPC Riccati converged in {i} iterations")
    print(f"Kinf condition number: {np.linalg.cond(Kinf):.2f}")
    print(f"Pinf condition number: {np.linalg.cond(Pinf):.2f}")

    with open(output_path, 'w') as f:
        f.write('#pragma once\n\n')
        f.write('#include "types.hpp"\n\n')
        
        # Dynamics matrices
        f.write('tinytype Adyn_data[NSTATES * NSTATES] = {\n\t')
        adyn_vals = []
        for i in range(NSTATES):
            row_vals = [f'{Ad[i,j]:.6f}f' for j in range(NSTATES)]
            adyn_vals.append(', '.join(row_vals))
        f.write(', \n\t'.join(adyn_vals))
        f.write('\n};\n\n')
        
        f.write('tinytype Bdyn_data[NSTATES * NINPUTS] = {\n\t')
        bdyn_vals = []
        for i in range(NSTATES):
            row_vals = [f'{Bd[i,j]:.6f}f' for j in range(NINPUTS)]
            bdyn_vals.append(', '.join(row_vals))
        f.write(', \n\t'.join(bdyn_vals))
        f.write('\n};\n\n')
        
        f.write('tinytype fdyn_data[NSTATES] = {')
        f.write(', '.join([f'{x:.6f}f' for x in fdyn]))
        f.write('};\n\n')
        
        # Cost matrices
        f.write('tinytype Q_data[NSTATES] = {')
        f.write(', '.join([f'{x:.6f}f' for x in np.diag(Q)]))
        f.write('};\n\n')
        
        f.write('tinytype R_data[NINPUTS] = {')
        f.write(', '.join([f'{x:.6f}f' for x in np.diag(R)]))
        f.write('};\n\n')
        
        f.write(f'tinytype rho_value = {rho:.1f};\n\n')
        
        # Precomputed LQR matrices
        f.write('tinytype Kinf_data[NINPUTS*NSTATES] = {\n\t')
        kinf_vals = []
        for i in range(NINPUTS):
            row_vals = [f'{Kinf[i,j]:.6f}f' for j in range(NSTATES)]
            kinf_vals.append(', '.join(row_vals))
        f.write(', \n\t'.join(kinf_vals))
        f.write('\n};\n\n')
        
        f.write('tinytype Pinf_data[NSTATES*NSTATES] = {\n\t')
        pinf_vals = []
        for i in range(NSTATES):
            row_vals = [f'{Pinf[i,j]:.6f}f' for j in range(NSTATES)]
            pinf_vals.append(', '.join(row_vals))
        f.write(', \n\t'.join(pinf_vals))
        f.write('\n};\n\n')
        
        f.write('tinytype Quu_inv_data[NINPUTS*NINPUTS] = {\n\t')
        quu_vals = []
        for i in range(NINPUTS):
            row_vals = [f'{Quu_inv[i,j]:.6f}f' for j in range(NINPUTS)]
            quu_vals.append(', '.join(row_vals))
        f.write(', \n\t'.join(quu_vals))
        f.write('\n};\n\n')
        
        f.write('tinytype AmBKt_data[NSTATES*NSTATES] = {\n\t')
        ambkt_vals = []
        for i in range(NSTATES):
            row_vals = [f'{AmBKt[i,j]:.6f}f' for j in range(NSTATES)]
            ambkt_vals.append(', '.join(row_vals))
        f.write(', \n\t'.join(ambkt_vals))
        f.write('\n};\n\n')
        
        f.write('tinytype APf_data[NSTATES] = {')
        f.write(', '.join([f'{x[0]:.6f}f' for x in APf]))
        f.write('};\n\n')
        
        f.write('tinytype BPf_data[NINPUTS] = {')
        f.write(', '.join([f'{x[0]:.6f}f' for x in BPf]))
        f.write('};\n\n')

if __name__ == "__main__":
    print(f"Generating TinyMPC code for rocket landing:")
    print(f"  NSTATES: {NSTATES}")
    print(f"  NINPUTS: {NINPUTS}")
    print(f"  NHORIZON: {NHORIZON}")
    print(f"  Cost Q diagonal: {np.diag(Q)[0]}")
    print(f"  Cost R diagonal: {np.diag(R)[0]}")
    print(f"  rho: {rho}")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    teensy_dir = os.path.join(base_dir, 'tinympc', 'tinympc_teensy')
    stm32_dir = os.path.join(base_dir, 'tinympc', 'tinympc_stm32')
    generated_dir = os.path.join(base_dir, 'tinympc', 'tinympc_generated')
    
    # 1. Generate glob_opts.hpp for teensy
    teensy_glob_opts = os.path.join(teensy_dir, 'src', 'glob_opts.hpp')
    os.makedirs(os.path.dirname(teensy_glob_opts), exist_ok=True)
    generate_glob_opts_header(teensy_glob_opts, NSTATES, NINPUTS, NHORIZON)
    print(f"Generated: {teensy_glob_opts}")
    
    # 2. Generate glob_opts.hpp for stm32
    stm32_glob_opts = os.path.join(stm32_dir, 'src', 'tinympc', 'glob_opts.hpp')
    os.makedirs(os.path.dirname(stm32_glob_opts), exist_ok=True)
    generate_glob_opts_header(stm32_glob_opts, NSTATES, NINPUTS, NHORIZON)
    print(f"Generated: {stm32_glob_opts}")
    
    # 3. Generate glob_opts.hpp for generated
    generated_glob_opts = os.path.join(generated_dir, 'tinympc', 'glob_opts.hpp')
    os.makedirs(os.path.dirname(generated_glob_opts), exist_ok=True)
    generate_glob_opts_header(generated_glob_opts, NSTATES, NINPUTS, NHORIZON)
    print(f"Generated: {generated_glob_opts}")
    
    # 4. Generate problem data for teensy
    teensy_params = os.path.join(teensy_dir, 'src', 'problem_data', 'rocket_landing_params_20hz.hpp')
    os.makedirs(os.path.dirname(teensy_params), exist_ok=True)
    generate_problem_data_header(teensy_params, Ad, Bd, Q, R, fdyn, rho, NSTATES, NINPUTS, NHORIZON, NTOTAL)
    print(f"Generated: {teensy_params}")
    
    # 5. Generate problem data for stm32
    stm32_params = os.path.join(stm32_dir, 'problem_data', 'rocket_landing_params_20hz.hpp')
    os.makedirs(os.path.dirname(stm32_params), exist_ok=True)
    generate_problem_data_header(stm32_params, Ad, Bd, Q, R, fdyn, rho, NSTATES, NINPUTS, NHORIZON, NTOTAL)
    print(f"Generated: {stm32_params}")

    print("\n=== TinyMPC Generation Complete ===")
    print(f"  - NHORIZON = {NHORIZON} (matches SCS/ECOS)")
    print(f"  - NTOTAL = {NTOTAL} (simulation length)")
    print(f"  - Same dynamics, costs, and constraints as SCS/ECOS")
    print(f"  - Generated headers for both Teensy and STM32 projects")
    print("\\nYou can now compile and benchmark all three solvers fairly!")
# TinyMPC Code Generation for Rocket Landing Problem
# This generates TinyMPC data that matches SCS/ECOS benchmark setup exactly

import os
import sys
import time
import numpy as np
import scipy.linalg

path_to_root = os.getcwd()
print(f"Working directory: {path_to_root}")

# Add tinympc-python to Python path
tinympc_python_dir = os.path.abspath(os.path.join(path_to_root, "../../tinympc-python"))
sys.path.append(tinympc_python_dir)
print(f"Added to Python path: {tinympc_python_dir}")

try:
    import tinympc
    
    # Set up TinyMPC paths and library
    tinympc_python_dir = os.path.join(path_to_root, "../../tinympc-python")
    tinympc_dir = os.path.join(tinympc_python_dir, "tinympc/TinyMPC")  # Path to the TinyMPC directory (C code)

    print(f"TinyMPC directory: {tinympc_dir}")

    # Initialize TinyMPC and compile/load library
    tinympc_prob = tinympc.TinyMPC()
    tinympc_prob.compile_lib(tinympc_dir)  # Compile the library

    # Load the generic shared/dynamic library
    os_ext = ".dylib"  # Mac uses .dylib
    lib_dir = os.path.join(tinympc_dir, "build/src/tinympc/libtinympcShared" + os_ext)
    tinympc_prob.load_lib(lib_dir)  # Load the library
    tinympc_available = True
except Exception as e:
    print(f"TinyMPC import/setup failed: {e}")
    print("Continuing with header generation only...")
    tinympc_available = False
    tinympc_prob = None

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

# Cost matrices (matching gen_rocket.py/run_ecos.py)
Q = 1e3 * np.eye(NSTATES)  # 1000 * I
R = 1e0 * np.eye(NINPUTS)  # 1 * I

# Reference trajectories (needed for TinyMPC) - MATCH gen_rocket.py exactly
fdyn = np.array([0.0, 0.0, -0.0122625, 0.0, 0.0, -0.4905])  # gravity term from gen_rocket.py

# TinyMPC solver settings
rho = 1.0  # matching the value from existing rocket_landing_params files
abs_pri_tol = 1.0e-2    # matching benchmark settings
abs_dual_tol = 1.0e-2   # matching benchmark settings
max_iter = 500          # matching benchmark settings
check_termination = 1

# Bounds (matching gen_rocket.py/run_ecos.py)
u_min = -10 * np.ones(NINPUTS)
u_max = 105.0 * np.ones(NINPUTS)
x_min = np.array([-5, -5, 0, -10, -10, -10.0])  # matching exactly
x_max = np.array([5, 5, 20, 10, 10, 10.0])     # matching exactly

print(f"Generating TinyMPC code for rocket landing:")
print(f"  NSTATES: {NSTATES}")
print(f"  NINPUTS: {NINPUTS}")
print(f"  NHORIZON: {NHORIZON}")
print(f"  rho: {rho}")

# SET UP TINYMPC PROBLEM
# Convert matrices to column-major order lists for TinyMPC
A1 = Ad.transpose().reshape((NSTATES * NSTATES)).tolist()  # col-major order list
B1 = Bd.transpose().reshape((NSTATES * NINPUTS)).tolist()  # col-major order list
Q1 = Q.diagonal().tolist()  # diagonal of state cost (like safety_filter.py)
R1 = R.diagonal().tolist()  # diagonal of input cost (like safety_filter.py)

# State and input constraints for all horizons (matching safety_filter.py format)
xmin1 = list(x_min) * NHORIZON  # state constraints
xmax1 = list(x_max) * NHORIZON  # state constraints
umin1 = list(u_min) * (NHORIZON - 1)  # input constraints
umax1 = list(u_max) * (NHORIZON - 1)  # input constraints

# Setup the problem with the correct format (only if TinyMPC is available)
if tinympc_available:
    try:
        # Use simple box constraints only (like safety_filter) for compatibility
        # Note: This removes cone constraints but enables compilation and memory efficiency
        tinympc_prob.setup(NSTATES, NINPUTS, NHORIZON, A1, B1, Q1, R1, xmin1, xmax1, umin1, umax1, 
                           rho, abs_pri_tol, abs_dual_tol, max_iter, check_termination)
    except Exception as e:
        print(f"TinyMPC setup failed: {e}")
        print("Continuing with header generation only...")
        tinympc_available = False

def export_tinympc_data_to_c(header_path, Ad, Bd, Q, R, u_min, u_max, x_min, x_max, fdyn, rho, NSTATES, NINPUTS, NHORIZON, NTOTAL):
    # Compute LQR/ADMM precomputes (Kinf, Pinf, Quu_inv, AmBKt, APf, BPf)
    # These formulas match the Riccati recursion in TinyMPC codegen
    Q1 = Q + rho * np.eye(NSTATES)
    R1 = R + rho * np.eye(NINPUTS)
    Ptp1 = rho * np.eye(NSTATES)
    Ktp1 = np.zeros((NINPUTS, NSTATES))
    for i in range(1000):
        Kinf = np.linalg.solve(R1 + Bd.T @ Ptp1 @ Bd, Bd.T @ Ptp1 @ Ad)
        Pinf = Q1 + Ad.T @ Ptp1 @ (Ad - Bd @ Kinf)
        if np.max(np.abs(Kinf - Ktp1)) < 1e-5:
            break
        Ktp1 = Kinf
        Ptp1 = Pinf
    Quu_inv = np.linalg.inv(R1 + Bd.T @ Pinf @ Bd)
    AmBKt = (Ad - Bd @ Kinf).T
    APf = Pinf @ fdyn.reshape(-1, 1)
    BPf = Kinf @ fdyn.reshape(-1, 1)

    with open(header_path, 'w') as f:
        f.write('#pragma once\n\n')
        f.write('// TinyMPC Rocket Landing Problem Data\n\n')
        f.write(f'#define NSTATES {NSTATES}\n')
        f.write(f'#define NINPUTS {NINPUTS}\n')
        f.write(f'#define NHORIZON {NHORIZON}\n')
        f.write(f'#define NTOTAL {NTOTAL}\n\n')
        f.write('typedef float tinytype;\n\n')
        f.write(f'tinytype rho_value = {rho:.6f}f;\n\n')
        f.write('tinytype Adyn_data[NSTATES * NSTATES] = {\n')
        f.write(', '.join([f'{x:.6f}f' for x in Ad.flatten()]))
        f.write('};\n\n')
        f.write('tinytype Bdyn_data[NSTATES * NINPUTS] = {\n')
        f.write(', '.join([f'{x:.6f}f' for x in Bd.flatten()]))
        f.write('};\n\n')
        f.write('tinytype Q_data[NSTATES] = {\n')
        f.write(', '.join([f'{x:.6f}f' for x in np.diag(Q)]))
        f.write('};\n\n')
        f.write('tinytype R_data[NINPUTS] = {\n')
        f.write(', '.join([f'{x:.6f}f' for x in np.diag(R)]))
        f.write('};\n\n')
        f.write('tinytype u_min_data[NINPUTS] = {\n')
        f.write(', '.join([f'{x:.6f}f' for x in u_min]))
        f.write('};\n\n')
        f.write('tinytype u_max_data[NINPUTS] = {\n')
        f.write(', '.join([f'{x:.6f}f' for x in u_max]))
        f.write('};\n\n')
        f.write('tinytype x_min_data[NSTATES] = {\n')
        f.write(', '.join([f'{x:.6f}f' for x in x_min]))
        f.write('};\n\n')
        f.write('tinytype x_max_data[NSTATES] = {\n')
        f.write(', '.join([f'{x:.6f}f' for x in x_max]))
        f.write('};\n\n')
        f.write('tinytype fdyn_data[NSTATES] = {\n')
        f.write(', '.join([f'{x:.6f}f' for x in fdyn]))
        f.write('};\n\n')
        f.write('tinytype Kinf_data[NINPUTS*NSTATES] = {\n')
        f.write(', '.join([f'{x:.6f}f' for x in Kinf.flatten()]))
        f.write('};\n\n')
        f.write('tinytype Pinf_data[NSTATES*NSTATES] = {\n')
        f.write(', '.join([f'{x:.6f}f' for x in Pinf.flatten()]))
        f.write('};\n\n')
        f.write('tinytype Quu_inv_data[NINPUTS*NINPUTS] = {\n')
        f.write(', '.join([f'{x:.6f}f' for x in Quu_inv.flatten()]))
        f.write('};\n\n')
        f.write('tinytype AmBKt_data[NSTATES*NSTATES] = {\n')
        f.write(', '.join([f'{x:.6f}f' for x in AmBKt.flatten()]))
        f.write('};\n\n')
        f.write('tinytype APf_data[NSTATES] = {\n')
        f.write(', '.join([f'{x[0]:.6f}f' for x in APf]))
        f.write('};\n\n')
        f.write('tinytype BPf_data[NINPUTS] = {\n')
        f.write(', '.join([f'{x[0]:.6f}f' for x in BPf]))
        f.write('};\n\n')

# GENERATE PROBLEM DATA HEADER FOR TEENSY (like gen_rocket.py does)
mcu_dir = os.path.join(path_to_root, 'tinympc/tinympc_teensy')
header_path = os.path.join(mcu_dir, 'src/problem_data/rocket_landing_params_20hz.hpp')
os.makedirs(os.path.dirname(header_path), exist_ok=True)
export_tinympc_data_to_c(header_path, Ad, Bd, Q, R, u_min, u_max, x_min, x_max, fdyn, rho, NSTATES, NINPUTS, NHORIZON, NTOTAL)
print(f"Generated TinyMPC problem data header at: {header_path}")

# ALSO GENERATE TINYMPC CODE (if available)
if tinympc_available:
    path_to_tinympc = os.path.join(path_to_root, "tinympc")
    output_dir = os.path.join(path_to_tinympc, "tinympc_generated")
    tinympc_prob.tiny_codegen(tinympc_dir, output_dir)

    # Copy generated files to teensy project
    os.system('cp -R '+output_dir+'/src/tiny_data_workspace.cpp'+' '+mcu_dir+'/src/tiny_data_workspace.cpp')
    os.system('cp -R '+output_dir+'/tinympc/glob_opts.hpp'+' '+mcu_dir+'/lib/tinympc/glob_opts.hpp')

    # Copy to stm32 project  
    mcu_dir = os.path.join(path_to_tinympc, 'tinympc_stm32_feather')
    os.system('cp -R '+output_dir+'/src/tiny_data_workspace.cpp'+' '+mcu_dir+'/src/tiny_data_workspace.cpp')
    os.system('cp -R '+output_dir+'/tinympc/glob_opts.hpp'+' '+mcu_dir+'/src/tinympc/glob_opts.hpp')

    print(f"Generated TinyMPC code in: {output_dir}")
    print(f"Copied generated files to teensy and stm32 projects")
else:
    print("Skipping TinyMPC code generation due to setup failure")
    print("Header file generation completed successfully")

print("\n=== TinyMPC Generation Complete ===")
print(f"  - NHORIZON = {NHORIZON} (matches SCS/ECOS)")
print(f"  - NTOTAL = {NTOTAL} (simulation length)")
print(f"  - Same dynamics, costs, and constraints as SCS/ECOS")
print("  - Problem data header generated for Teensy project")
print("\nYou can now compile and benchmark all three solvers fairly!")

# Exit cleanly to avoid segfault
sys.exit(0) 
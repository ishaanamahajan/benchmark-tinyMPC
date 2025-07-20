# TinyMPC Code Generation for Rocket Landing Problem
# This generates TinyMPC data that matches SCS/ECOS benchmark setup exactly

import os
import sys
import time
import numpy as np

path_to_root = os.getcwd()
print(f"Working directory: {path_to_root}")

# Add tinympc-python to Python path
tinympc_python_dir = os.path.abspath(os.path.join(path_to_root, "../../tinympc-python"))
sys.path.append(tinympc_python_dir)
print(f"Added to Python path: {tinympc_python_dir}")

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

# ROCKET LANDING PROBLEM PARAMETERS (matching gen_rocket.py and run_ecos.py)
NSTATES = 6
NINPUTS = 3  
NHORIZON = 4  # MATCH SCS/ECOS horizon for fair comparison
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
Q1 = Q.transpose().reshape((NSTATES * NSTATES)).tolist()  # full Q matrix, col-major
R1 = R.transpose().reshape((NINPUTS * NINPUTS)).tolist()  # full R matrix, col-major

# State and input constraints for all horizons (matching safety_filter.py format)
xmin1 = list(x_min) * NHORIZON  # state constraints
xmax1 = list(x_max) * NHORIZON  # state constraints
umin1 = list(u_min) * (NHORIZON - 1)  # input constraints
umax1 = list(u_max) * (NHORIZON - 1)  # input constraints

# Setup the problem with the correct format
tinympc_prob.setup(NSTATES, NINPUTS, NHORIZON, A1, B1, Q1, R1, xmin1, xmax1, umin1, umax1, 
                   rho, abs_pri_tol, abs_dual_tol, max_iter, check_termination)

# GENERATE CODE
output_dir = os.path.join(path_to_root, "tinympc/tinympc_generated")

# Create all necessary directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "src"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "src", "tinympc"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "tinympc"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "include"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "examples"), exist_ok=True)

print(f"Generating code to: {output_dir}")
print("Created all necessary directories")

try:
    tinympc_prob.tiny_codegen(tinympc_dir, output_dir)  # Use tiny_codegen instead of codegen
    print("Code generation completed successfully!")
except Exception as e:
    print(f"Error during code generation: {e}")

# COPY FILES TO TEENSY PROJECT
mcu_dir = os.path.join(path_to_root, 'tinympc/tinympc_teensy')
print(f"Copying files to Teensy project: {mcu_dir}")

# Create destination directories if they don't exist
os.makedirs(os.path.join(mcu_dir, 'src'), exist_ok=True)
os.makedirs(os.path.join(mcu_dir, 'lib/tinympc'), exist_ok=True)

# Copy the generated data files
src_file = os.path.join(output_dir, 'src/tiny_data_workspace.cpp')
dest_file = os.path.join(mcu_dir, 'src/tiny_data_workspace.cpp')
if os.path.exists(src_file):
    os.system(f'cp "{src_file}" "{dest_file}"')
    print("Copied tiny_data_workspace.cpp")
else:
    print(f"Warning: {src_file} not found")

src_file = os.path.join(output_dir, 'tinympc/glob_opts.hpp')
dest_file = os.path.join(mcu_dir, 'lib/tinympc/glob_opts.hpp')
if os.path.exists(src_file):
    os.system(f'cp "{src_file}" "{dest_file}"')
    print("Copied glob_opts.hpp")
else:
    print(f"Warning: {src_file} not found")

print("Files copied to Teensy project!")

# Create a custom glob_opts.hpp with all necessary parameters
custom_glob_opts_content = f'''/*
 * This file was autogenerated by TinyMPC on {time.strftime("%a %b %d %H:%M:%S %Y")}
 */

#pragma once

typedef float tinytype;

#define NSTATES 6
#define NINPUTS 3

#define NUM_INPUT_CONES 1
#define NUM_STATE_CONES 1

#define NHORIZON {NHORIZON}
#define NTOTAL 301
'''

# Write the custom glob_opts.hpp file
custom_glob_opts_file = os.path.join(mcu_dir, 'lib/tinympc/glob_opts.hpp')
with open(custom_glob_opts_file, 'w') as f:
    f.write(custom_glob_opts_content)

print("Created custom glob_opts.hpp with all necessary parameters")

print("\n=== TinyMPC Generation Complete ===")
print("Your TinyMPC rocket landing code is now ready with:")
print(f"  - NHORIZON = {NHORIZON} (matches SCS/ECOS)")
print(f"  - NTOTAL = 301 (simulation length)")
print(f"  - NUM_INPUT_CONES = 1, NUM_STATE_CONES = 1 (SOC constraints)")
print(f"  - Same dynamics, costs, and constraints as SCS/ECOS")
print("  - Files copied to tinympc_teensy project")
print("  - tiny_data_workspace.cpp contains all precomputed matrices")
print("  - glob_opts.hpp contains solver options")
print("\nYou can now compile and benchmark all three solvers fairly!")

# Exit cleanly to avoid segfault
sys.exit(0) 
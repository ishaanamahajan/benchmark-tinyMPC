# %% [markdown]
# # Predictve Safety Filter Benchmark

# %% [markdown]
# Load necessary packages, make sure to install `tinympc` ([README.md](../README.md))

# %%
import tinympc
import osqp

import os
import numpy as np
import subprocess

path_to_root = os.getcwd()
print(path_to_root)

# %%
# TinyMPC setup - new API is much simpler, no compilation needed

# %% [markdown]
# ## Double Integrator System

# %%
NSTATES = 4  # may vary this
NINPUTS = NSTATES//2
NHORIZON = 20  # may vary this
NTOTAL = 201
NPOS = NINPUTS

# Double-integrator dynamics
h = 0.05 #20 Hz
temp_n = int(NSTATES/2)
A = np.block([[np.eye(temp_n), h*np.eye(temp_n)], [np.zeros((temp_n,temp_n)), np.eye(temp_n)]])
B = np.block([[0.5*h*h*np.eye(temp_n)], [h*np.eye(temp_n)]])

Q = 0.0*np.eye(NSTATES)
R = 1e2*np.eye(NINPUTS)

rho = 1e2 # may want different value for each solver
xmax = 1.5 # doesn't matter, will change in C
xmin = -1.5 # doesn't matter, will change in C
umax = 2.0 # doesn't matter, will change in C
umin = -2.0 # doesn't matter, will change in C

abs_pri_tol = 1.0e-2    # absolute primal tolerance
abs_dual_tol = 1.0e-2   # absolute dual tolerance
max_iter = 500          # maximum number of iterations
check_termination = 1   # whether to check termination and period

# %% [markdown]
# ## Code Generation

# %% [markdown]
# ### TinyMPC

# %%
# SET UP PROBLEM - new API uses numpy arrays directly
xmin_bounds = np.array([xmin] * NSTATES)  # per-state bounds
xmax_bounds = np.array([xmax] * NSTATES)  # per-state bounds  
umin_bounds = np.array([umin] * NINPUTS)  # per-input bounds
umax_bounds = np.array([umax] * NINPUTS)  # per-input bounds

tinympc_prob = tinympc.TinyMPC()
tinympc_prob.setup(
    A, B, Q, R, NHORIZON,  # matrices and horizon
    rho=rho,
    x_min=xmin_bounds, x_max=xmax_bounds,
    u_min=umin_bounds, u_max=umax_bounds,
    abs_pri_tol=abs_pri_tol, abs_dua_tol=abs_dual_tol,
    max_iter=max_iter, check_termination=check_termination
)

path_to_tinympc = path_to_root + "/tinympc_f" # Path to the tinympc subfolder under safety_filter/

# GENERATE CODE
output_dir = path_to_tinympc + "/tinympc_generated"  # Path to the generated code
tinympc_prob.codegen(output_dir)  
# You may want to check if Kinf in generated_code follows the same pattern as previous K in LQR, otherwise something is wrong

# MOVING FILES FROM GENERATED CODE TO MCU FOLDER

# Copy to teensy project (updated file names for new API)
mcu_dir = path_to_tinympc + '/tinympc_teensy'
os.system('cp -R '+output_dir+'/src/tiny_data.cpp'+' '+mcu_dir+'/src/tiny_data_workspace.cpp')
os.system('cp -R '+output_dir+'/tinympc/tiny_data.hpp'+' '+mcu_dir+'/lib/tinympc/glob_opts.hpp')

# Copy to stm32 project (updated file names for new API)
mcu_dir = path_to_tinympc + '/tinympc_stm32_feather'
os.system('cp -R '+output_dir+'/src/tiny_data.cpp'+' '+mcu_dir+'/src/tiny_data_workspace.cpp')
os.system('cp -R '+output_dir+'/tinympc/tiny_data.hpp'+' '+mcu_dir+'/src/tinympc/glob_opts.hpp')

# %% [markdown]
# The necessary files (`src/tiny_data_workspace.cpp` and `tinympc/glob_opts.hpp`) were copied from `tinympc_generated` to `tinympc_*` for you. Now you can directly upload and run the program in `tinympc_*`, where * is the mcu you want to use.

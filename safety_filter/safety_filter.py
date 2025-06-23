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
tinympc_python_dir = path_to_root + "/../tinympc-python"
tinympc_dir = tinympc_python_dir + "/tinympc/TinyMPC"  # Path to the TinyMPC directory (C code)

tinympc_generic = tinympc.TinyMPC()
tinympc_generic.compile_lib(tinympc_dir)  # Compile the library (or use the binary provided)

# Load the generic shared/dynamic library. **You may want to change the extension of the library based on your OS -- Linux: .so, Mac: .dylib, Windows: .dll**
os_ext = ".so"  # CHANGE THIS BASED ON YOUR OS
lib_dir = tinympc_dir + "/build/src/tinympc/libtinympcShared" + os_ext  # Path to the compiled library
tinympc_generic.load_lib(lib_dir)  # Load the library

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
# SET UP PROBLEM
A1 = A.transpose().reshape((NSTATES * NSTATES)).tolist() # col-major order list
B1 = B.transpose().reshape((NSTATES * NINPUTS)).tolist() # col-major order list
Q1 = Q.diagonal().tolist()  # diagonal of state cost -- DON'T NEED FOR SAFETY FILTER
R1 = R.diagonal().tolist()  # diagonal of input cost

xmin1 = [xmin] * NSTATES * NHORIZON         # state constraints 
xmax1 = [xmax] * NSTATES * NHORIZON         # state constraints
umin1 = [umin] * NINPUTS * (NHORIZON - 1)   # input constraints
umax1 = [umax] * NINPUTS * (NHORIZON - 1)   # input constraints

tinympc_prob = tinympc.TinyMPC()
tinympc_prob.load_lib(lib_dir)  # Load the library
tinympc_prob.setup(NSTATES, NINPUTS, NHORIZON, A1, B1, Q1, R1, xmin1, xmax1, umin1, umax1, rho, abs_pri_tol, abs_dual_tol, max_iter, check_termination)

path_to_tinympc = path_to_root + "/tinympc_f" # Path to the tinympc subfolder under safety_filter/

# GENERATE CODE
output_dir = path_to_tinympc + "/tinympc_generated"  # Path to the generated code
tinympc_prob.tiny_codegen(tinympc_dir, output_dir)  
# You may want to check if Kinf in generated_code follows the same pattern as previous K in LQR, otherwise something is wrong

# MOVING FILES FROM GENERATED CODE TO MCU FOLDER

# Copy to teensy project
mcu_dir = path_to_tinympc + '/tinympc_teensy'
os.system('cp -R '+output_dir+'/src/tiny_data_workspace.cpp'+' '+mcu_dir+'/src/tiny_data_workspace.cpp')
os.system('cp -R '+output_dir+'/tinympc/glob_opts.hpp'+' '+mcu_dir+'/lib/tinympc/glob_opts.hpp')

# Copy to stm32 project
mcu_dir = path_to_tinympc + '/tinympc_stm32_feather'
os.system('cp -R '+output_dir+'/src/tiny_data_workspace.cpp'+' '+mcu_dir+'/src/tiny_data_workspace.cpp')
os.system('cp -R '+output_dir+'/tinympc/glob_opts.hpp'+' '+mcu_dir+'/src/tinympc/glob_opts.hpp')

# %% [markdown]
# The necessary files (`src/tiny_data_workspace.cpp` and `tinympc/glob_opts.hpp`) were copied from `tinympc_generated` to `tinympc_*` for you. Now you can directly upload and run the program in `tinympc_*`, where * is the mcu you want to use.

# %% [markdown]
# ### OSQP

# %%
from scipy import sparse
from utils import osqp_export_data_to_c, replace_in_file

# SET UP PROBLEM
A2 = sparse.csc_matrix(A)
B2 = sparse.csc_matrix(B)

x0 = np.ones(NSTATES)*0.5  # doesn't matter, will change in C

Xref = np.zeros((NSTATES, NTOTAL)) 
for k in range(NTOTAL):
    Xref[0:NPOS,k] = np.sin(1*k)*2*np.ones(temp_n)
Uref = np.ones((NINPUTS, NTOTAL-1))*1  # doesn't matter, will change in C

Q2 = Q
R2 = R

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(NHORIZON), Q2),
                       sparse.kron(sparse.eye(NHORIZON-1), R2)], format='csc')
# - linear objective
q = np.hstack([np.zeros((NHORIZON)*NSTATES), np.hstack([-R2@Uref[:,i] for i in range(NHORIZON-1)])])
# - linear dynamics
Ax = sparse.kron(sparse.eye(NHORIZON),-sparse.eye(NSTATES)) + sparse.kron(sparse.eye(NHORIZON, k=-1), A2)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, NHORIZON-1)), sparse.eye(NHORIZON-1)]), B2)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros((NHORIZON-1)*NSTATES)])
ueq = leq

# - input and state constraints # doesn't matter, will change in C
xmin2 = np.ones(NSTATES)*xmin
xmax2 = np.ones(NSTATES)*xmax
umin2 = np.ones(NINPUTS)*umin
umax2 = np.ones(NINPUTS)*umax
Aineq = sparse.eye((NHORIZON)*NSTATES + (NHORIZON-1)*NINPUTS)
lineq = np.hstack([np.kron(np.ones(NHORIZON), xmin2), np.kron(np.ones(NHORIZON-1), umin2)])
uineq = np.hstack([np.kron(np.ones(NHORIZON), xmax2), np.kron(np.ones(NHORIZON-1), umax2)])
# - OSQP constraints
AA = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
osqp_prob = osqp.OSQP()

# Setup workspace and change alpha parameter
osqp_prob.setup(P, q, AA, l, u, alpha=1.0, scaling=0, check_termination=check_termination, eps_abs=abs_pri_tol, eps_rel=abs_pri_tol, eps_prim_inf=1e-4, eps_dual_inf=1e-4, max_iter=max_iter, polish=False, rho=rho, adaptive_rho=False, warm_start=True)

# res = osqp_prob.solve()
# x = res.x[0:NSTATES*NHORIZON]
# u = res.x[NSTATES*NHORIZON:]
# print(x)
# print(u)


path_to_osqp = path_to_root + "/osqp" # Path to the tinympc subfolder under safety_filter/

# GENERATE CODE
output_dir = path_to_osqp + "/osqp_generated"  # Path to the generated code

osqp_prob.codegen(
    output_dir,   # Output folder for auto-generated code
    prefix='osqp_data_',         # Prefix for filenames and C variables; useful if generating multiple problems
    force_rewrite=True,        # Force rewrite if output folder exists?
    parameters='vectors',      # What do we wish to update in the generated code?
                                # One of 'vectors' (allowing update of q/l/u through prob.update_data_vec)
                                # or 'matrices' (allowing update of P/A/q/l/u
                                # through prob.update_data_vec or prob.update_data_mat)
    use_float=True,
    printing_enable=False,     # Enable solver printing?
    profiling_enable=False,    # Enable solver profiling?
    interrupt_enable=False,    # Enable user interrupt (Ctrl-C)?
    include_codegen_src=True,  # Include headers/sources/Makefile in the output folder,
                                # creating a self-contained compilable folder?
    extension_name='pyosqp',   # Name of the generated python extension; generates a setup.py; Set None to skip
    compile=False,             # Compile the above python extension into an importable module
                                # (allowing "import pyosqp")?
)

# MOVING FILES FROM GENERATED CODE TO MCU FOLDER

mcu_dir = path_to_osqp + '/osqp_teensy'
os.system('cp -R '+output_dir+'/osqp_configure.h'+' '+mcu_dir+'/lib/osqp/inc/osqp_configure.h')
os.system('cp -R '+output_dir+'/osqp_data_workspace.c'+' '+mcu_dir+'/src/osqp_data_workspace.c')
osqp_export_data_to_c(mcu_dir+'/src',A, B, R, NSTATES, NINPUTS, NHORIZON, NTOTAL)

mcu_dir = path_to_osqp + '/osqp_stm32_feather'
os.system('cp -R '+output_dir+'/osqp_configure.h'+' '+mcu_dir+'/src/osqp/inc/osqp_configure.h')
os.system('cp -R '+output_dir+'/osqp_data_workspace.c'+' '+mcu_dir+'/osqp_data_workspace.c')
osqp_export_data_to_c(mcu_dir+'/src/osqp/inc/public',A, B, R, NSTATES, NINPUTS, NHORIZON, NTOTAL)

file_path = mcu_dir+"/osqp_data_workspace.c"
old_lines = [
    '#include "types.h"',
    '#include "algebra_impl.h"',
    '#include "qdldl_interface.h"'
]
new_lines = [
    '#include "src/osqp/inc/private/types.h"',
    '#include "src/osqp/inc/private/algebra_impl.h"',
    '#include "src/osqp/inc/private/qdldl_interface.h"'
]

replace_in_file(file_path, old_lines, new_lines)


# %% [markdown]
# The necessary files were copied from `osqp_generated` to `osqp_teensy` for you. Now you can directly upload and run the generated program in `osqp_teensy`.



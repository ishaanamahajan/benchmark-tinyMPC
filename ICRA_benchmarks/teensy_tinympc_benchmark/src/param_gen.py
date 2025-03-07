"""
File: prob_data_gen.py
Author: Anoushka, Khai
Date: 2023-09-15
Description: A Python script to generate random MPC problem for OSQP (Python) and TinyMPC (C++).
Modified to generate 1000 random controllable matrices.
"""

import numpy as np
import os

def export_xref_to_c(declare, data, nx):
    string = declare + "= {\n"
    for i in range(data.shape[0]):
        string = string + "  "
        for j in range(nx):
            if i == data.shape[0] and j == nx:
                this_str = str(data[i][j]) + ",\t"
            else:
                this_str = str(data[i][j]) + ",\t"
            # str = str * this_str * "f"
            string = string + this_str
        string = string + "\n"
    string = string + "};"
    return string

def export_mat_to_c(declare, data):
    string = declare + "= {\n"
    for i in range(data.shape[0]):
        string = string + "  "
        for j in range(data.shape[1]):
            if i == data.shape[0] and j == data.shape[1]:
                this_str = str(data[i][j]) + ",\t"
            else:
                 this_str = str(data[i][j]) + ",\t"
            string = string + this_str
        string = string + "\n"
    string = string + "};"
    return string

def tinympc_export_data_to_c(xbar, A, B, Q, Qf, R, umin, umax, xmin, xmax, nx, nu, Nh, Nsim):
    include_statement = '#include "types.hpp"\n\n'
    boilerplate = "#pragma once\n\n"

    xbar_string = export_xref_to_c("const PROGMEM tinytype Xref_data["+str(Nsim)+"*"+str(nx)+"] ", xbar, nx)

    A_data_string = export_mat_to_c("const PROGMEM tinytype Adyn_data["+str(nx)+"*"+str(nx)+"] ", A) + "\n\n"
    B_data_string = export_mat_to_c("const PROGMEM tinytype Bdyn_data["+str(nx)+"*"+str(nu)+"] ", B) + "\n\n"
   
    Q_diag = np.diagonal(Q)
    Q_data_string = "const PROGMEM tinytype Q_data["+str(nx)+"] = {"
    for i in range(nx):
        Q_data_string = Q_data_string + str(Q_diag[i])
        if i != nx-1:
            Q_data_string = Q_data_string + ','
    Q_data_string = Q_data_string+"};\n\n"
    
    Qf_diag = np.diagonal(Qf)
    Qf_data_string = "const PROGMEM tinytype Qf_data["+str(nx)+"] = {"
    for i in range(nx):
        Qf_data_string = Qf_data_string + str(Qf_diag[i])
        if i != nx-1:
            Qf_data_string = Qf_data_string + ','
    Qf_data_string = Qf_data_string+"};\n\n"
    
    R_diag = np.diagonal(R)
    R_data_string = "const PROGMEM tinytype R_data["+str(nu)+"] = {"
    for i in range(nu):
        R_data_string = R_data_string + str(R_diag[i])
        if i != nu-1:
            R_data_string = R_data_string + ','
    R_data_string = R_data_string+"};\n\n"

    umin_string = export_mat_to_c("const PROGMEM tinytype umin["+str(nu)+"] ", umin) + "\n\n"
    umax_string = export_mat_to_c("const PROGMEM tinytype umax["+str(nu)+"] ", umax) + "\n\n"
    xmin_string = export_mat_to_c("const PROGMEM tinytype xmin["+str(nx)+"] ", xmin) + "\n\n"
    xmax_string = export_mat_to_c("const PROGMEM tinytype xmax["+str(nx)+"] ", xmax) + "\n\n"

    f = open('random_problems/prob_nx_'+str(nx)+'/constants.hpp','w')
    f.write("#define NSTATES "+str(nx)+'\n\n')
    f.write("#define NINPUTS "+str(nu)+'\n\n')
    f.write("#define NHORIZON "+str(Nh)+'\n\n')
    f.write("#define NTOTAL "+str(Nsim)+'\n\n')
    f.write('#define NSTATE_CONSTRAINTS 1\n\n')

    f.close()

    f = open('random_problems/prob_nx_'+str(nx)+"/rand_prob_tinympc_xbar.hpp", "w")
    # f = open("rand_prob_tinympc_xbar.hpp", "w")
    f.write(include_statement)
    f.write(boilerplate)
    f.write(xbar_string)
    f.close()

    f = open('random_problems/prob_nx_'+str(nx)+"/rand_prob_tinympc_params.hpp", "w")
    # f = open("rand_prob_tinympc_params.hpp", "w")
    f.write(include_statement)
    f.write(boilerplate)
    f.write(A_data_string)
    f.write(B_data_string)
    f.write(Q_data_string)
    f.write(Qf_data_string)
    f.write(R_data_string)
    f.write(umin_string)
    f.write(umax_string)
    f.write(xmin_string)
    f.write(xmax_string)
        
    K = np.zeros((nu,nx))
    P = np.zeros((nx,nx))
    Kprev = np.zeros((nu,nx))
    Pprev = np.zeros((nx,nx))

    # Compute Kinf, Pinf
    riccati_iters = 0
    riccati_err = 1e-10
    Pprev = Qf
    while True:
        K = np.linalg.inv(R + B.T @ Pprev @ B) @ (B.T @ Pprev @ A)
        P = Q + A.T @ Pprev @ (A - B@K)
        if np.max(np.abs(K - Kprev)) < riccati_err:
            break
        Kprev = K
        Pprev = P
        riccati_iters += 1

    # Cache precomputed values
    rho = 85.0
    Q = Q + rho*np.eye(nx)
    Qf = P + rho*np.eye(nx)
    R = R + rho*np.eye(nu)

    # Compute Kinf, Pinf
    riccati_iters = 0
    riccati_err = 1e-10
    Pprev = Qf
    while True:
        K = np.linalg.inv(R + B.T @ Pprev @ B) @ (B.T @ Pprev @ A)
        P = Q + A.T @ Pprev @ (A - B@K)
        if np.max(np.abs(K - Kprev)) < riccati_err:
            break
        Kprev = K
        Pprev = P
        riccati_iters += 1

    Kinf = K
    Pinf = P
    Quu_inv = np.linalg.inv(R + B.T @ Pinf@B)
    AmBKt = (A - B@K).T
    coeff_d2p_list = Kinf.T@R - AmBKt@Pinf@B

    rho_string = "const PROGMEM tinytype rho_value = "+str(rho)+";\n\n"
    Kinf_string = export_mat_to_c("const PROGMEM tinytype Kinf_data["+str(nu)+"*"+str(nx)+"] ", Kinf) + "\n\n"
    Pinf_string = export_mat_to_c("const PROGMEM tinytype Pinf_data["+str(nx)+"*"+str(nx)+"] ", Pinf) + "\n\n"
    Quu_inv_string = export_mat_to_c("const PROGMEM tinytype Quu_inv_data["+str(nu)+"*"+str(nu)+"] ", Quu_inv) + "\n\n"
    AmBKt_string = export_mat_to_c("const PROGMEM tinytype AmBKt_data["+str(nx)+"*"+str(nx)+"] ", AmBKt) + "\n\n"
    coeff_d2p_list_string = export_mat_to_c("const PROGMEM tinytype coeff_d2p_data["+str(nx)+"*"+str(nu)+"] ", coeff_d2p_list) + "\n\n"

    f.write(rho_string)
    f.write(Kinf_string)
    f.write(Pinf_string)
    f.write(Quu_inv_string)
    f.write(AmBKt_string)
    f.write(coeff_d2p_list_string)
    f.close()

def osqp_export_data_to_python(xbar, A, B, Q, Qf, R, umin, umax, xmin, xmax, nx, nu, Nh, Nsim):
    np.savez('random_problems/prob_nx_'+str(nx)+'/rand_prob_osqp_params.npz',nx=nx, nu=nu, Nh=Nh, Nsim=Nsim, Q=Q, R=R, A=A, B=B, Qf=Qf, x_bar=xbar, umin=umin, umax=umax, xmin=xmin, xmax=xmax)
    # np.savez('rand_prob_osqp_params.npz',nx=nx, nu=nu, Nh=Nh, Nsim=Nsim, Q=Q, R=R, A=A, B=B, Qf=Qf, x_bar=xbar, umin=umin, umax=umax, xmin=xmin, xmax=xmax)
    SIZE_Q = (Nh + 1) * nx + Nh * nu
    SIZE_LU = (Nh + 1) * nx * 2 + Nh * nu

    include_statement = '#include "osqp_api_types.h"\n\n'
    boilerplate = "#pragma once\n\n"

    xbar_string = export_xref_to_c("const PROGMEM OSQPFloat Xref_data["+str(Nsim)+"*"+str(nx)+"] ", xbar, nx)+'\n\n'

    # f = open("rand_prob_osqp_xbar.h", "w")
    f = open('random_problems/prob_nx_'+str(nx)+"/rand_prob_osqp_xbar.h", "w")
    f.write(include_statement)
    f.write(boilerplate)
    f.write(xbar_string)

    f.write("#define NSTATES "+str(nx)+'\n\n')
    f.write("#define NINPUTS "+str(nu)+'\n\n')
    f.write("#define NHORIZON "+str(Nh)+'\n\n')
    f.write("#define NTOTAL "+str(Nsim)+'\n\n')
    f.write("#define SIZE_Q "+str(SIZE_Q)+'\n\n')
    f.write("#define SIZE_LU "+str(SIZE_LU)+'\n\n')

    Q_string = export_mat_to_c("const PROGMEM OSQPFloat mQ["+str(nx)+"*"+str(nx)+"] ", -Q)+'\n\n'
    Qf_string = export_mat_to_c("const PROGMEM OSQPFloat mQf["+str(nx)+"*"+str(nx)+"] ", -Qf)+'\n\n'
    A_string = export_mat_to_c("const PROGMEM OSQPFloat A["+str(nx)+"*"+str(nx)+"] ", A)+'\n\n'
    B_string = export_mat_to_c("const PROGMEM OSQPFloat B["+str(nx)+"*"+str(nu)+"] ", B)+'\n\n'

    f.write(Q_string)
    f.write(Qf_string)
    f.write(A_string)
    f.write(B_string)

    f.close()

def generate_data(nx, nu, Nh, Nsim):
    np.random.seed(123)
    # Generate Q: Q_{ii} = U(0, 1)
    Q_diag = np.random.uniform(0,10,nx)
    Q = np.diag(Q_diag)

    # Generate R: R_{ii} = 0.1
    R_diag = 0.1*np.ones(nu)
    R = np.diag(R_diag)

    # Generate Qf: (N-1)*Q
    Qf = ((Nh)-1)*Q

    # reference trajectory xk, uk --> uk ~ N(0, 1), xk+1 = f(xk, uk)
    u = np.random.normal(size=(nu, Nsim-1))

    # A and B are random matrices with eigenvalues inside of the unit circle

    # Generate a controllable system
    while True:
        # SVD for both A and B to ensure that the eigenvalues are inside of the unit circle
        A = np.random.uniform(low=-1, high=1, size=(nx, nx))
        U, S, Vh = np.linalg.svd(A) # S is a vector of the non-zero singular values
        E = np.zeros((U.shape[0], Vh.shape[0])) # E is the SVD matrix of singular values Σ 
        S = S / np.max(S) # Scale the singular values so that 
        np.fill_diagonal(E, S)
        A = U @ E @ Vh.T

        B = np.random.uniform(low=-1, high=1, size=(nx,nu))

        # Check if the system is controllable
        C = np.zeros((nx,nx*nu)) # controllability matrix
        Ak = np.zeros((nx,nx)) + np.eye(nx)
        for k in range(nx):
            C[:, (k)*nu:(k)*nu+nu] = Ak@B
            Ak = Ak @ A

        if np.linalg.matrix_rank(C) == A.shape[0]: # only true if system is controllable
            break
        else:
            print("Not controllable")

    umax = np.zeros((nu,1)) + 3
    umin = np.zeros((nu,1)) - 3
    xmin = np.zeros((nx,1)) - 10000 # might add state constraints later
    xmax = np.zeros((nx,1)) + 10000

    x0 = np.zeros(nx)
    xbar = np.zeros((Nsim, nx))
    xbar[0,:] = x0

    for k in range(Nsim-1):
        xbar[k+1,:] = A @ xbar[k,:] + B @ u[:,k]


    ### SAVE ALL RANDOM PROBLEM DATA TO A HPP FILE TO BE USED BY ALL SOLVERS ###

    tinympc_export_data_to_c(xbar, A, B, Q, Qf, R, umin, umax, xmin, xmax, nx, nu, Nh, Nsim)
    osqp_export_data_to_python(xbar, A, B, Q, Qf, R, umin, umax, xmin, xmax, nx, nu, Nh, Nsim)
        
def export_matrix_to_raw_format(A, B, system_id, output_dir):
    """Export A and B matrices in raw array format with PROGMEM directive."""
    
    with open(f"{output_dir}/system_{system_id}.h", "w") as f:
        f.write("#pragma once\n\n")
        f.write("typedef float tinytype;\n\n")
        
        # Export A matrix
        f.write(f"const PROGMEM tinytype Adyn_data[{A.shape[0]}*{A.shape[1]}] = {{\n")
        for i in range(A.shape[0]):
            f.write("  ")
            for j in range(A.shape[1]):
                f.write(f"{A[i, j]},\t")
            f.write("\n")
        f.write("};\n\n\n")
        
        # Export B matrix
        f.write(f"const PROGMEM tinytype Bdyn_data[{B.shape[0]}*{B.shape[1]}] = {{\n")
        for i in range(B.shape[0]):
            f.write("  ")
            for j in range(B.shape[1]):
                f.write(f"{B[i, j]},\t")
            f.write("\n")
        f.write("};\n\n")
        
        # Add placeholder for Q, Qf, R data
        nx = A.shape[0]
        nu = B.shape[1]
        
        # Generate random Q diagonal values
        Q_diag = np.random.uniform(0, 10, nx)
        f.write(f"const PROGMEM tinytype Q_data[{nx}] = {{")
        for i in range(nx):
            f.write(f"{Q_diag[i]}")
            if i < nx - 1:
                f.write(",")
        f.write("};\n\n")
        
        # Generate Qf as 9*Q
        Qf_diag = 9 * Q_diag
        f.write(f"const PROGMEM tinytype Qf_data[{nx}] = {{")
        for i in range(nx):
            f.write(f"{Qf_diag[i]}")
            if i < nx - 1:
                f.write(",")
        f.write("};\n\n")
        
        # Generate R diagonal values (all 0.1)
        f.write(f"const PROGMEM tinytype R_data[{nu}] = {{")
        for i in range(nu):
            f.write("0.1")
            if i < nu - 1:
                f.write(",")
        f.write("};\n")

def generate_1000_matrices(nx=12, nu=4):
    """Generate 1000 random controllable matrices using the existing logic."""
    output_dir = "/Users/ishaanmahajan/benchmark-tinyMPC/ICRA_benchmarks/teensy_tinympc_benchmark/include/problem_data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate systems
    for i in range(1, 1001):
        if i % 10 == 0:
            print(f"Generating system {i}/1000")
        
        # Use different random seed for each system
        np.random.seed(123 + i)
        
        # Generate a controllable system using the exact same logic from generate_data
        while True:
            # SVD for both A and B to ensure that the eigenvalues are inside of the unit circle
            A = np.random.uniform(low=-1, high=1, size=(nx, nx))
            U, S, Vh = np.linalg.svd(A) # S is a vector of the non-zero singular values
            E = np.zeros((U.shape[0], Vh.shape[0])) # E is the SVD matrix of singular values Σ 
            S = S / np.max(S) # Scale the singular values so that 
            np.fill_diagonal(E, S)
            A = U @ E @ Vh.T

            B = np.random.uniform(low=-1, high=1, size=(nx, nu))

            # Check if the system is controllable
            C = np.zeros((nx, nx*nu)) # controllability matrix
            Ak = np.zeros((nx, nx)) + np.eye(nx)
            for k in range(nx):
                C[:, (k)*nu:(k)*nu+nu] = Ak@B
                Ak = Ak @ A

            if np.linalg.matrix_rank(C) == A.shape[0]: # only true if system is controllable
                break
            else:
                print(f"System {i} not controllable, regenerating...")
        
        # Export to header files in raw format
        export_matrix_to_raw_format(A, B, i, output_dir)
    
    # Create an index file that includes all matrices
    with open(f"{output_dir}/all_systems.h", "w") as f:
        f.write("#ifndef ALL_SYSTEMS_H\n")
        f.write("#define ALL_SYSTEMS_H\n\n")
        
        for i in range(1, 1001):
            f.write(f"#include \"system_{i}.h\"\n")
        
        f.write("\n#endif // ALL_SYSTEMS_H\n")
    
    print(f"Successfully generated 1000 random controllable systems in {output_dir}")

if __name__ == '__main__':
    # Generate 1000 random controllable matrices (12 states, 4 inputs)
    generate_1000_matrices(12, 4)
    
    # Comment out the original code
    # nu = 4
    # Nh = 10
    # Nsim = 200
    # for nx in [10]:
    #     os.system('mkdir random_problems/prob_nx_'+str(nx))
    #     generate_data(nx, nu, Nh, Nsim)
    #     print('generated: random_problems/prob_nx_'+str(nx))
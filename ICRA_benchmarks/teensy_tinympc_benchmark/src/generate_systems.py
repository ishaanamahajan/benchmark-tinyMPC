"""
Script to generate 100 random 12×4 MPC problems with sensitivity matrices
"""

import numpy as np
import os

def compute_lqr_sensitivity(A, B, Q, R, rho_base=85.0, delta_rho=1e-4):
    """
    Compute LQR solution and sensitivity derivatives with respect to rho.
    """
    n = A.shape[0]  # Number of states
    m = B.shape[1]  # Number of inputs
    
    # Function to compute LQR solution for a given rho
    def solve_lqr(rho):
        R_rho = R + rho * np.eye(m)
        
        # Solve discrete algebraic Riccati equation
        P = Q.copy()
        for _ in range(100):  # Iterate to convergence
            K = np.linalg.solve(R_rho + B.T @ P @ B, B.T @ P @ A)
            P_new = Q + A.T @ P @ (A - B @ K)
            if np.allclose(P, P_new, rtol=1e-6, atol=1e-8):
                break
            P = P_new
        
        K = np.linalg.solve(R_rho + B.T @ P @ B, B.T @ P @ A)
        C1 = np.linalg.inv(R_rho + B.T @ P @ B)
        C2 = A - B @ K
        
        return K, P, C1, C2
    
    # Compute LQR solution at rho + delta_rho
    K_plus, P_plus, C1_plus, C2_plus = solve_lqr(rho_base + delta_rho)
    
    # Compute LQR solution at rho - delta_rho
    K_minus, P_minus, C1_minus, C2_minus = solve_lqr(rho_base - delta_rho)
    
    # Compute derivatives using central differences
    dK_drho = (K_plus - K_minus) / (2 * delta_rho)
    dP_drho = (P_plus - P_minus) / (2 * delta_rho)
    dC1_drho = (C1_plus - C1_minus) / (2 * delta_rho)
    dC2_drho = (C2_plus - C2_minus) / (2 * delta_rho)
    
    return dK_drho, dP_drho, dC1_drho, dC2_drho

def export_mat_to_c(declare, data):
    string = declare + "= {\n"
    for i in range(data.shape[0]):
        string = string + "  "
        for j in range(data.shape[1]):
            if j == data.shape[1]-1:
                this_str = str(data[i][j]) + ",\t"
            else:
                this_str = str(data[i][j]) + ",\t"
            string = string + this_str
        string = string + "\n"
    string = string + "};"
    return string

def generate_single_system(nx, nu, seed, output_dir, file_num):
    np.random.seed(seed)
    
    # Generate Q: Q_{ii} = U(0, 10)
    Q_diag = np.random.uniform(0, 10, nx)
    Q = np.diag(Q_diag)

    # Generate R: R_{ii} = 0.1
    R_diag = 0.1 * np.ones(nu)
    R = np.diag(R_diag)

    # Generate Qf: 9*Q
    Qf = 9 * Q

    # Generate A and B matrices with eigenvalues inside the unit circle
    while True:
        # Generate random A matrix
        A = np.random.uniform(low=-1, high=1, size=(nx, nx))
        U, S, Vh = np.linalg.svd(A)
        E = np.zeros((U.shape[0], Vh.shape[0]))
        S = S / np.max(S)  # Scale singular values to be inside unit circle
        np.fill_diagonal(E, S)
        A = U @ E @ Vh.T

        # Generate random B matrix
        B = np.random.uniform(low=-1, high=1, size=(nx, nu))

        # Check if the system is controllable
        C = np.zeros((nx, nx*nu))  # controllability matrix
        Ak = np.eye(nx)
        for k in range(nx):
            C[:, k*nu:(k+1)*nu] = Ak @ B
            Ak = Ak @ A

        if np.linalg.matrix_rank(C) == nx:  # System is controllable
            break
        else:
            print(f"System {file_num} not controllable, regenerating...")

    # Input and state constraints
    umax = np.ones((nu, 1)) * 3.0
    umin = np.ones((nu, 1)) * -3.0
    xmax = np.ones((nx, 1)) * 10000.0
    xmin = np.ones((nx, 1)) * -10000.0

    # Set rho value
    rho = 85.0

    # Compute LQR solution
    R_reg = R + rho * np.eye(nu)
    Q_reg = Q + rho * np.eye(nx)
    
    # Solve discrete algebraic Riccati equation
    P = Q_reg.copy()
    K = np.zeros((nu, nx))
    for _ in range(100):  # Iterate to convergence
        K = np.linalg.solve(R_reg + B.T @ P @ B, B.T @ P @ A)
        P_new = Q_reg + A.T @ P @ (A - B @ K)
        if np.allclose(P, P_new, rtol=1e-6, atol=1e-8):
            break
        P = P_new

    # Create output file
    filename = os.path.join(output_dir, f"problem_{file_num}.hpp")
    with open(filename, 'w') as f:
        # Write header
        f.write("#pragma once\n")
        f.write("#include <Arduino.h>\n\n")
        f.write("// Define tinytype to match the one in types.hpp\n")
        f.write("typedef float tinytype;\n")
        
        # Write A matrix
        A_string = export_mat_to_c(f"const PROGMEM tinytype Adyn_data[{nx}*{nx}] ", A)
        f.write(A_string + "\n\n")
        
        # Write B matrix
        B_string = export_mat_to_c(f"const PROGMEM tinytype Bdyn_data[{nx}*{nu}] ", B)
        f.write(B_string + "\n\n")
        
        # Write Q diagonal
        Q_string = "const PROGMEM tinytype Q_data[" + str(nx) + "] = {"
        for i in range(nx):
            Q_string += str(Q_diag[i])
            if i != nx-1:
                Q_string += ","
        Q_string += "};\n\n"
        f.write(Q_string)
        
        # Write Qf diagonal
        Qf_diag = np.diagonal(Qf)
        Qf_string = "const PROGMEM tinytype Qf_data[" + str(nx) + "] = {"
        for i in range(nx):
            Qf_string += str(Qf_diag[i])
            if i != nx-1:
                Qf_string += ","
        Qf_string += "};\n\n"
        f.write(Qf_string)
        
        # Write R diagonal
        R_string = "const PROGMEM tinytype R_data[" + str(nu) + "] = {"
        for i in range(nu):
            R_string += str(R_diag[i])
            if i != nu-1:
                R_string += ","
        R_string += "};\n\n"
        f.write(R_string)
        
        # Write umin
        umin_string = export_mat_to_c(f"const PROGMEM tinytype umin[{nu}] ", umin)
        f.write(umin_string + "\n\n")
        
        # Write umax
        umax_string = export_mat_to_c(f"const PROGMEM tinytype umax[{nu}] ", umax)
        f.write(umax_string + "\n\n")
        
        # Write xmin
        xmin_string = export_mat_to_c(f"const PROGMEM tinytype xmin[{nx}] ", xmin)
        f.write(xmin_string + "\n\n")
        
        # Write xmax
        xmax_string = export_mat_to_c(f"const PROGMEM tinytype xmax[{nx}] ", xmax)
        f.write(xmax_string + "\n\n")
        
        # Write rho
        f.write(f"const PROGMEM tinytype rho_value = {rho};\n\n")
        
        Kinf = K
        Pinf = P
        Quu_inv = np.linalg.inv(R_reg + B.T @ Pinf @ B)
        AmBKt = (A - B @ K).T
        coeff_d2p = Kinf.T @ R_reg - AmBKt @ Pinf @ B
        
        # Write Kinf
        Kinf_string = export_mat_to_c(f"const PROGMEM tinytype Kinf_data[{nu}*{nx}] ", Kinf)
        f.write(Kinf_string + "\n\n")
        
        # Write Pinf
        Pinf_string = export_mat_to_c(f"const PROGMEM tinytype Pinf_data[{nx}*{nx}] ", Pinf)
        f.write(Pinf_string + "\n\n")
        
        # Write Quu_inv
        Quu_inv_string = export_mat_to_c(f"const PROGMEM tinytype Quu_inv_data[{nu}*{nu}] ", Quu_inv)
        f.write(Quu_inv_string + "\n\n")
        
        # Write AmBKt
        AmBKt_string = export_mat_to_c(f"const PROGMEM tinytype AmBKt_data[{nx}*{nx}] ", AmBKt)
        f.write(AmBKt_string + "\n\n")
        
        # Write coeff_d2p
        coeff_d2p_string = export_mat_to_c(f"const PROGMEM tinytype coeff_d2p_data[{nx}*{nu}] ", coeff_d2p)
        f.write(coeff_d2p_string + "\n\n")
        
        # Compute and add sensitivity matrices
        dK_drho, dP_drho, dC1_drho, dC2_drho = compute_lqr_sensitivity(A, B, Q, R, rho)
        
        # Format sensitivity matrices
        f.write("// LQR Sensitivity Matrices\n")
        f.write(export_mat_to_c(f"const PROGMEM tinytype dKinf_drho_data[{nu}*{nx}] ", dK_drho) + "\n\n")
        f.write(export_mat_to_c(f"const PROGMEM tinytype dPinf_drho_data[{nx}*{nx}] ", dP_drho) + "\n\n")
        f.write(export_mat_to_c(f"const PROGMEM tinytype dC1_drho_data[{nu}*{nu}] ", dC1_drho) + "\n\n")
        f.write(export_mat_to_c(f"const PROGMEM tinytype dC2_drho_data[{nx}*{nx}] ", dC2_drho))
        
    print(f"Generated system {file_num} with sensitivity matrices at {filename}")

def main():
    nx = 12
    nu = 4
    output_dir = "/Users/ishaanmahajan/replicate/benchmark-tinyMPC/ICRA_benchmarks/teensy_tinympc_benchmark/include/problem_data"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 100 different systems
    for i in range(1, 101):
        generate_single_system(nx, nu, seed=i+123, output_dir=output_dir, file_num=i)
        print(f"Generated {i}/100 systems")

if __name__ == "__main__":
    main()
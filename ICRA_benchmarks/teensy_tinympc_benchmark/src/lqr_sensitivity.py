import numpy as np
from scipy import linalg
import os
import re

def compute_lqr_sensitivity(A, B, Q, R, rho_base=0.1, delta_rho=1e-4):
    """
    Compute LQR solution and sensitivity derivatives with respect to rho.
    
    Args:
        A: System dynamics matrix (n x n)
        B: Input matrix (n x m)
        Q: State cost matrix (n x n)
        R: Input cost matrix (m x m)
        rho_base: Base value of rho
        delta_rho: Step size for finite difference
    
    Returns:
        dK_drho: Sensitivity of K with respect to rho
        dP_drho: Sensitivity of P with respect to rho
        dC1_drho: Sensitivity of C1 with respect to rho
        dC2_drho: Sensitivity of C2 with respect to rho
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

def extract_vector_from_file(content, vector_name, size):
    """
    Extract vector data from file content.
    """
    pattern = rf"{vector_name}_data\[{size}\]\s*=\s*\{{([\s\S]*?)\}};"
    match = re.search(pattern, content)
    
    if not match:
        raise ValueError(f"Could not find vector {vector_name} in file")
    
    data_str = match.group(1)
    # Remove comments if any
    data_str = re.sub(r'//.*?\n', '\n', data_str)
    # Extract all floating point numbers
    numbers = re.findall(r'[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][-+]?\d+)?', data_str)
    numbers = [float(num) for num in numbers]
    
    return np.array(numbers)

def extract_matrix_from_file(content, matrix_name, rows, cols):
    """
    Extract matrix data from file content.
    """
    pattern = rf"{matrix_name}_data\[{rows}[\s\*]*{cols}\]\s*=\s*\{{([\s\S]*?)\}};"
    match = re.search(pattern, content)
    
    if not match:
        raise ValueError(f"Could not find matrix {matrix_name} in file")
    
    data_str = match.group(1)
    # Remove comments if any
    data_str = re.sub(r'//.*?\n', '\n', data_str)
    # Extract all floating point numbers
    numbers = re.findall(r'[-+]?(?:\d*\.\d+|\d+\.\d*|\d+)(?:[eE][-+]?\d+)?', data_str)
    numbers = [float(num) for num in numbers]
    
    # Reshape into matrix
    matrix = np.array(numbers).reshape(rows, cols)
    return matrix

def format_matrix_for_cpp(name, matrix):
    """
    Format a matrix for C++ output.
    """
    rows, cols = matrix.shape
    result = f"const PROGMEM tinytype {name}_data[{rows}*{cols}] = {{\n  "
    
    for i in range(rows):
        for j in range(cols):
            result += f"{matrix[i, j]},\t"
        if i < rows - 1:
            result += "\n  "
    
    result += "\n};"
    return result

def process_problem_file(file_path):
    """
    Process a problem data file to compute and append LQR sensitivity matrices.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Infer dimensions from the file
    adyn_match = re.search(r'Adyn_data\[(\d+)\*(\d+)\]', content)
    bdyn_match = re.search(r'Bdyn_data\[(\d+)\*(\d+)\]', content)
    
    if not adyn_match or not bdyn_match:
        raise ValueError("Could not determine system dimensions from file")
    
    nx = int(adyn_match.group(1))
    nu = int(bdyn_match.group(2))
    
    print(f"Detected system dimensions: {nx} states, {nu} inputs")
    
    # Extract matrices
    A = extract_matrix_from_file(content, "Adyn", nx, nx)
    B = extract_matrix_from_file(content, "Bdyn", nx, nu)
    
    # Extract Q and R (diagonal vectors)
    Q_diag = extract_vector_from_file(content, "Q", nx)
    R_diag = extract_vector_from_file(content, "R", nu)
    
    # Create diagonal matrices
    Q = np.diag(Q_diag)
    R = np.diag(R_diag)
    
    # # Extract rho value
    # rho_match = re.search(r'rho_value = ([\d\.e\-]+)', content)
    # if rho_match:
    #     rho_base = float(rho_match.group(1))
    # else:
    #     rho_base = 0.1  # Default value
    
    # print(f"Using rho value: {rho_base}")

    rho_base = 85.0
    
    # Compute LQR sensitivity
    dK_drho, dP_drho, dC1_drho, dC2_drho = compute_lqr_sensitivity(A, B, Q, R, rho_base)
    
    # Format sensitivity matrices
    sensitivity_matrices = "\n\n// LQR Sensitivity Matrices\n"
    sensitivity_matrices += format_matrix_for_cpp("dKinf_drho", dK_drho) + "\n\n"
    sensitivity_matrices += format_matrix_for_cpp("dPinf_drho", dP_drho) + "\n\n"
    sensitivity_matrices += format_matrix_for_cpp("dC1_drho", dC1_drho) + "\n\n"
    sensitivity_matrices += format_matrix_for_cpp("dC2_drho", dC2_drho)
    
    # Append to file
    with open(file_path, 'a') as f:
        f.write(sensitivity_matrices)
    
    print(f"Sensitivity matrices appended to {file_path}")

def main():
    # Path to the problem data file
    file_path = "/Users/ishaanmahajan/replicate/benchmark-tinyMPC/ICRA_benchmarks/teensy_tinympc_benchmark/include/problem_data/problem_10.hpp"
    
    # Process the file
    process_problem_file(file_path)
    
    print("LQR sensitivity computation completed successfully!")

if __name__ == "__main__":
    main()
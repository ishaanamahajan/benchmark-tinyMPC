import jax
import jax.numpy as jnp
from jax import grad, jacfwd

# Enable 64-bit precision for better numerical accuracy
jax.config.update("jax_enable_x64", True)

# Define system dimensions
NSTATES = 12
NINPUTS = 4
rho_value = 5.0

# Load A and B matrices
Adyn_data = [
    1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0245250, 0.0000000, 0.0500000, 0.0000000, 0.0000000, 0.0000000, 0.0002044, 0.0000000,
    0.0000000, 1.0000000, 0.0000000, -0.0245250, 0.0000000, 0.0000000, 0.0000000, 0.0500000, 0.0000000, -0.0002044, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0500000, 0.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.9810000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0122625, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, -0.9810000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, -0.0122625, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000,
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000
]

Bdyn_data = [
    -0.0007069, 0.0007773, 0.0007091, -0.0007795,
    0.0007034, 0.0007747, -0.0007042, -0.0007739,
    0.0052554, 0.0052554, 0.0052554, 0.0052554,
    -0.1720966, -0.1895213, 0.1722891, 0.1893288,
    -0.1729419, 0.1901740, 0.1734809, -0.1907131,
    0.0123423, -0.0045148, -0.0174024, 0.0095748,
    -0.0565520, 0.0621869, 0.0567283, -0.0623632,
    0.0562756, 0.0619735, -0.0563386, -0.0619105,
    0.2102143, 0.2102143, 0.2102143, 0.2102143,
    -13.7677303, -15.1617018, 13.7831318, 15.1463003,
    -13.8353509, 15.2139209, 13.8784751, -15.2570451,
    0.9873856, -0.3611820, -1.3921880, 0.7659845
]

Q_data = [100.0000000, 100.0000000, 100.0000000, 4.0000000, 4.0000000, 400.0000000, 4.0000000, 4.0000000, 4.0000000, 2.0408163, 2.0408163, 4.0000000]
R_data = [4.0000000, 4.0000000, 4.0000000, 4.0000000]

# Reshape matrices
A = jnp.array(Adyn_data).reshape(NSTATES, NSTATES)
B = jnp.array(Bdyn_data).reshape(NSTATES, NINPUTS)
Q = jnp.diag(jnp.array(Q_data))
R = jnp.diag(jnp.array(R_data))

def solve_dare(A, B, Q, R, rho):
    """Solve the discrete-time algebraic Riccati equation with regularization."""
    n = A.shape[0]
    R_rho = R + rho * jnp.eye(R.shape[0])
    
    # Initialize P with Q
    def riccati_iteration(P, _):
        K = jnp.linalg.solve(R_rho + B.T @ P @ B, B.T @ P @ A)
        P_new = Q + A.T @ P @ (A - B @ K)
        return P_new, None
    
    # Run iterations until convergence
    P_init = Q
    P_final, _ = jax.lax.scan(riccati_iteration, P_init, None, length=100)
    
    # Compute K, C1, C2
    K = jnp.linalg.solve(R_rho + B.T @ P_final @ B, B.T @ P_final @ A)
    C1 = jnp.linalg.inv(R_rho + B.T @ P_final @ B)
    C2 = A - B @ K
    
    return K, P_final, C1, C2

# Create functions that return each component
def get_K(rho):
    K, _, _, _ = solve_dare(A, B, Q, R, rho)
    return K

def get_P(rho):
    _, P, _, _ = solve_dare(A, B, Q, R, rho)
    return P

def get_C1(rho):
    _, _, C1, _ = solve_dare(A, B, Q, R, rho)
    return C1

def get_C2(rho):
    _, _, _, C2 = solve_dare(A, B, Q, R, rho)
    return C2

# Compute Jacobians (derivatives) with respect to rho
dK_drho = jacfwd(get_K)(rho_value)
dP_drho = jacfwd(get_P)(rho_value)
dC1_drho = jacfwd(get_C1)(rho_value)
dC2_drho = jacfwd(get_C2)(rho_value)

# Convert to numpy arrays for printing
import numpy as np
dK_drho_np = np.array(dK_drho)
dP_drho_np = np.array(dP_drho)
dC1_drho_np = np.array(dC1_drho)
dC2_drho_np = np.array(dC2_drho)

# Print in the requested format
def print_matrix(name, matrix):
    rows, cols = matrix.shape
    print(f"const float {name}[{rows}][{cols}] = {{")
    for i in range(rows):
        print("    {", end="")
        for j in range(cols):
            print(f"{matrix[i, j]:.7f}", end="")
            if j < cols - 1:
                print(", ", end="")
        print("}" + ("," if i < rows - 1 else ""))
    print("};")

print_matrix("dKinf_drho", dK_drho_np)
print_matrix("dPinf_drho", dP_drho_np)
print_matrix("dC1_drho", dC1_drho_np)
print_matrix("dC2_drho", dC2_drho_np)
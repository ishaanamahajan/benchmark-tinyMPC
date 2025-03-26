import autograd.numpy as anp
from autograd import jacobian

class LQRSensitivity:
    def __init__(self):
        # Define dimensions
        self.NSTATES = 12
        self.NINPUTS = 4
        
        # Initialize matrices
        self.initialize_matrices()
        
    def initialize_matrices(self):
        """Initialize system matrices"""
        # Initialize Adyn as 12x12 matrix
        self.Adyn = anp.array([
            [1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0245250, 0.0000000, 0.0500000, 0.0000000, 0.0000000, 0.0000000, 0.0002044, 0.0000000],
            [0.0000000, 1.0000000, 0.0000000, -0.0245250, 0.0000000, 0.0000000, 0.0000000, 0.0500000, 0.0000000, -0.0002044, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0500000, 0.0000000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000, 0.0000000],
            [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0250000],
            [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.9810000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0122625, 0.0000000],
            [0.0000000, 0.0000000, 0.0000000, -0.9810000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, -0.0122625, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000],
            [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000]
        ])
        
        # Initialize Bdyn as 12x4 matrix
        self.Bdyn = anp.array([
            [-0.0007069, 0.0007773, 0.0007091, -0.0007795],
            [0.0007034, 0.0007747, -0.0007042, -0.0007739],
            [0.0052554, 0.0052554, 0.0052554, 0.0052554],
            [-0.1720966, -0.1895213, 0.1722891, 0.1893288],
            [-0.1729419, 0.1901740, 0.1734809, -0.1907131],
            [0.0123423, -0.0045148, -0.0174024, 0.0095748],
            [-0.0565520, 0.0621869, 0.0567283, -0.0623632],
            [0.0562756, 0.0619735, -0.0563386, -0.0619105],
            [0.2102143, 0.2102143, 0.2102143, 0.2102143],
            [-13.7677303, -15.1617018, 13.7831318, 15.1463003],
            [-13.8353509, 15.2139209, 13.8784751, -15.2570451],
            [0.9873856, -0.3611820, -1.3921880, 0.7659845]
        ])
        
        # Initialize Q and R matrices
        self.Q = anp.diag([100.0, 100.0, 100.0, 4.0, 4.0, 400.0, 4.0, 4.0, 4.0, 2.0408163, 2.0408163, 4.0])
        self.R = anp.diag([4.0, 4.0, 4.0, 4.0])
        
    def lqr_direct(self, rho):
        """Compute LQR solution for given rho"""
        R_rho = self.R + rho * anp.eye(self.NINPUTS)
        
        # Compute P
        P = self.Q.copy()
        for _ in range(100):
            K = anp.linalg.solve(R_rho + self.Bdyn.T @ P @ self.Bdyn, 
                               self.Bdyn.T @ P @ self.Adyn)
            P_new = self.Q + self.Adyn.T @ P @ self.Adyn - self.Adyn.T @ P @ self.Bdyn @ K
            if anp.allclose(P, P_new):
                break
            P = P_new
        
        K = anp.linalg.solve(R_rho + self.Bdyn.T @ P @ self.Bdyn, 
                            self.Bdyn.T @ P @ self.Adyn)
        C1 = anp.linalg.inv(R_rho + self.Bdyn.T @ P @ self.Bdyn)
        C2 = self.Adyn - self.Bdyn @ K
        
        return anp.concatenate([K.flatten(), P.flatten(), C1.flatten(), C2.flatten()])
    
    def compute_derivatives(self, rho=5.0):
        """Compute and format LQR sensitivity matrices"""
        print("Computing LQR sensitivity matrices...")
        
        # Compute derivatives
        derivs = jacobian(self.lqr_direct)(rho)
        
        # Reshape derivatives
        k_size = self.NINPUTS * self.NSTATES
        p_size = self.NSTATES * self.NSTATES
        c1_size = self.NINPUTS * self.NINPUTS
        c2_size = self.NSTATES * self.NSTATES
        
        # Store derivatives
        self.dK = derivs[:k_size].reshape(self.NINPUTS, self.NSTATES)
        self.dP = derivs[k_size:k_size+p_size].reshape(self.NSTATES, self.NSTATES)
        self.dC1 = derivs[k_size+p_size:k_size+p_size+c1_size].reshape(self.NINPUTS, self.NINPUTS)
        self.dC2 = derivs[k_size+p_size+c1_size:].reshape(self.NSTATES, self.NSTATES)
        
    def print_matrices(self):
        """Print matrices in C++ format"""
        def print_matrix(name, matrix):
            print(f"const float {name}[{matrix.shape[0]}][{matrix.shape[1]}] = {{")
            for i in range(matrix.shape[0]):
                row = [f"{x:8.4f}" for x in matrix[i]]
                print("    {" + ", ".join(row) + "}" + ("," if i < matrix.shape[0]-1 else ""))
            print("};")
            print()

        print("// Pre-computed sensitivity matrices")
        print_matrix("dKinf_drho", self.dK)
        print_matrix("dPinf_drho", self.dP)
        print_matrix("dC1_drho", self.dC1)
        print_matrix("dC2_drho", self.dC2)

if __name__ == "__main__":
    # Create instance and compute sensitivities
    lqr = LQRSensitivity()
    lqr.compute_derivatives(rho=85.0)  # Using rho=85.0 as in RhoAdapter
    lqr.print_matrices()
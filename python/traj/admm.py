import numpy as np

class ADMM:
    def __init__(self, Q, q, A, l, u,
                 x=None, μ=None, λ=None, z=None, 
                 ρ=0.1, ρ_min=1e-6, ρ_max=1e6, 
                 σ=1e-6, tol=1e-3, max_iter=2000, verbose=False):
        # Convert QP to ADMM format
        # self.Q = qp.Q
        # self.q = qp.q
        # self.A = np.vstack([qp.A, qp.G])
        # self.l = np.concatenate([qp.b, -np.inf * np.ones(qp.n_ineq)])
        # self.u = np.concatenate([qp.b, qp.h])
        self.Q = np.array(Q)
        self.q = np.array(q)
        self.A = np.array(A)
        self.l = np.array(l)
        self.u = np.array(u)
        
        self.nx = self.Q.shape[0]
        # self.n_eq = self.A.shape[0]
        # self.n_ineq = self.l.shape[0]
        self.nc = self.A.shape[0]
        self.eq_inds = np.where(np.abs(u - l) <= 1e-4)[0]

        self.x = np.zeros(self.nx) if x is None else np.array(x)
        self.x_tilde = np.zeros(self.nx)
        self.μ = np.zeros(self.nx) if μ is None else np.array(μ)
        self.λ = np.zeros(self.nc) if λ is None else np.array(λ)
        self.z = np.zeros(self.nc) if z is None else np.array(z)


        # Initialize scaling vector with ones
        scaling_vector = np.ones(len(self.l), dtype=self.l.dtype)
        # Set scaling factors for equality constraints to 1000
        scaling_vector[self.eq_inds] = 1000
        self.ρ_scaling_matrix = np.diag(scaling_vector)

        self.ρ = ρ
        self.ρ_min = ρ_min
        self.ρ_max = ρ_max
        self.σ = σ
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def primal_residual(self):
        return self.A @ self.x - self.z

    def dual_residual(self):
        return self.Q @ self.x + self.q + self.A.T @ self.λ

    def solve(self):
        if self.verbose:
            print("ADMM solver")
            print("iter         J            ρ            |r_p|∞          |r_d|∞     ")
            print("------------------------------------------------------------------------------------")
        
        I = np.eye(self.Q.shape[0])
        rho = self.ρ * self.ρ_scaling_matrix 
        for i in range(1, self.max_iter + 1):
            # Update x_tilde
            # self.x_tilde = np.linalg.solve(self.Q + self.ρ * self.A.T @ self.A + self.σ * I, 
            #                                self.ρ * self.A.T @ self.z + self.σ * self.x - self.μ - self.A.T @ self.λ - self.q)
            self.x_tilde = np.linalg.solve(self.Q + self.A.T @ rho @ self.A + self.σ * I, 
                                            self.A.T @ rho @ self.z + self.σ * self.x - self.μ - self.A.T @ self.λ - self.q)

            # Update x
            self.x = self.x_tilde + self.μ / self.σ

            # Update z
            # self.z = np.maximum(self.l, np.minimum(self.u, self.A @ self.x_tilde + self.λ / (self.ρ)))
            self.z = np.maximum(self.l, np.minimum(self.u, self.A @ self.x_tilde + np.linalg.solve(rho, self.λ)))

            # Update λ
            # self.λ = self.λ + self.ρ * (self.A @ self.x - self.z)
            self.λ = self.λ + rho @ (self.A @ self.x - self.z)

            # Update μ
            self.μ = self.μ + self.σ * (self.x_tilde - self.x)

            # Check convergence every 25 iterations
            if i % 25 == 0:
                primal_residual_norm = np.linalg.norm(self.primal_residual(), np.inf)
                dual_residual_norm = np.linalg.norm(self.dual_residual(), np.inf)

                if self.verbose:
                    print(f"{i:3d}  {0:12.6e}  {self.ρ:12.6e}  {primal_residual_norm:12.6e}  {dual_residual_norm:12.6e}")
                
                # Update ρ
                self.ρ *= np.sqrt(primal_residual_norm / dual_residual_norm)
                self.ρ = max(self.ρ_min, min(self.ρ_max, self.ρ))
                rho = self.ρ * self.ρ_scaling_matrix
                # Check for convergence
                if primal_residual_norm < self.tol and dual_residual_norm < self.tol:
                    return

        print("Warning: ADMM did not converge")
    
    def update(self, q=None, l=None, u=None):
        if q is not None:
            self.q = np.array(q)
        if l is not None:
            self.l = np.array(l)
        if u is not None:
            self.u = np.array(u)
        
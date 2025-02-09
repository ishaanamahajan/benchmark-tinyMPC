import math
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy as sqrt
from autograd.numpy.linalg import norm
from autograd.numpy.linalg import inv
from autograd import jacobian
from autograd.test_util import check_grads
from scipy.linalg import block_diag

np.set_printoptions(precision=4, suppress=True)

import time


def generate_figure8_reference(t):
    """Generate figure-8 reference with smooth start"""
    # Figure 8 parameters
    A = 0.5 # amplitude
    w = 2*np.pi/7 # frequency
    
    # Smooth start factor (ramps up in first second)
    smooth_start = min(t/1.0, 1.0)
    
    x_ref = np.zeros(12)
    
    # Positions with smooth start
    x_ref[0] = A * np.sin(w*t) * smooth_start
    x_ref[2] = A * np.sin(2*w*t)/2 * smooth_start
    
    # Velocities (derivatives with smooth start)
    x_ref[6] = A * w * np.cos(w*t) * smooth_start 
    x_ref[8] = A * w * np.cos(2*w*t) * smooth_start
    
    # Zero attitude and angular velocity
    x_ref[3:6] = np.zeros(3)
    x_ref[9:12] = np.zeros(3)
    
    return x_ref

# Quaternion functions
def hat(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0.0]])

def L(q):
    s = q[0]
    v = q[1:4]
    up = np.hstack([s, -v])
    down = np.hstack([v.reshape(3,1), s*np.eye(3) + hat(v)])
    L = np.vstack([up,down])
    return L

T = np.diag([1.0, -1, -1, -1])
H = np.vstack([np.zeros((1,3)), np.eye(3)])

def qtoQ(q):
    return H.T @ T @ L(q) @ T @ L(q) @ H

def G(q):
    return L(q) @ H

def rptoq(phi):
    return (1./math.sqrt(1+phi.T @ phi)) * np.hstack([1, phi])

def qtorp(q):
    return q[1:4]/q[0]

def E(q):
    up = np.hstack([np.eye(3), np.zeros((3,3)), np.zeros((3,6))])
    mid = np.hstack([np.zeros((4,3)), G(q), np.zeros((4,6))])
    down = np.hstack([np.zeros((6,3)), np.zeros((6,3)), np.eye(6)])
    E = np.vstack([up, mid, down])
    return E

# Quadrotor parameters
mass = 0.035
J = np.array([[16.6e-6, 0.83e-6, 0.72e-6],
              [0.83e-6, 16.6e-6, 1.8e-6],
              [0.72e-6, 1.8e-6, 29.3e-6]])
g = 9.81
thrustToTorque = 0.0008
el = 0.046/1.414213562
scale = 65535
kt = 2.245365e-6*scale
km = kt*thrustToTorque

freq = 50.0


h = 1/freq

Nx1 = 13
Nx = 12
Nu = 4

def quad_dynamics(x, u):
    r = x[0:3]
    q = x[3:7]/norm(x[3:7])
    v = x[7:10]
    omg = x[10:13]
    Q = qtoQ(q)

    dr = v
    dq = 0.5*L(q)@H@omg
    dv = np.array([0, 0, -g]) + (1/mass)*Q@np.array([[0, 0, 0, 0], 
                                                     [0, 0, 0, 0], 
                                                     [kt, kt, kt, kt]])@u
    domg = inv(J)@(-hat(omg)@J@omg + 
                   np.array([[-el*kt, -el*kt, el*kt, el*kt], 
                            [-el*kt, el*kt, el*kt, -el*kt], 
                            [-km, km, -km, km]])@u)

    return np.hstack([dr, dq, dv, domg])

def quad_dynamics_rk4(x, u):
    f1 = quad_dynamics(x, u)
    f2 = quad_dynamics(x + 0.5*h*f1, u)
    f3 = quad_dynamics(x + 0.5*h*f2, u)
    f4 = quad_dynamics(x + h*f3, u)
    xn = x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
    xnormalized = xn[3:7]/norm(xn[3:7])
    return np.hstack([xn[0:3], xnormalized, xn[7:13]])


class RhoAdapter:
    def __init__(self):
        self.tolerance = 1.1 



        self.rho_base = 1.0   # Start lower
        self.tolerance = 1.1  # Slightly larger steps
        self.rho_min = 1.0
        self.rho_max = 100.0
    


        # For paper comparison/analysis
        self.rho_history = []
        self.residual_history = []

        self.start_time = time.time()  



    def predict_rho(self, pri_res, dual_res, iterations, current_rho, cache, x_prev, u_prev, v_prev, z_prev, g_prev, y_prev, current_time=None):
        try:

            print("\nInitial variable checks:")
            print(f"x_prev shape: {x_prev.shape}, norm: {np.linalg.norm(x_prev)}")
            print(f"u_prev shape: {u_prev.shape}, norm: {np.linalg.norm(u_prev)}")
            print(f"v_prev shape: {v_prev.shape}, norm: {np.linalg.norm(v_prev)}")
            print(f"z_prev shape: {z_prev.shape}, norm: {np.linalg.norm(z_prev)}")
            print(f"g_prev shape: {g_prev.shape}, norm: {np.linalg.norm(g_prev)}")
            print(f"y_prev shape: {y_prev.shape}, norm: {np.linalg.norm(y_prev)}")
            
            # Also check the passed-in residuals
            print("\nPassed-in residuals:")
            print(f"pri_res: {pri_res}")
            print(f"dual_res: {dual_res}")

            # Get dimensions
            N = x_prev.shape[1]  # Horizon length (25)
            nx = x_prev.shape[0]  # State dimension (12)
            nu = u_prev.shape[0]  # Input dimension (4)
            
            print(f"N={N}, nx={nx}, nu={nu}")
            
            # 1. Form decision variable x = [x_0; u_0; x_1; u_1; ...; x_N] 
            x_decision = []
            for i in range(N):
                x_decision.append(x_prev[:, i].reshape(-1, 1))  # state
                if i < N-1:  # Don't append input for last timestep
                    x_decision.append(u_prev[:, i].reshape(-1, 1))  # input
            x = np.vstack(x_decision)  # Paper's x variable
            print(f"x shape: {x.shape}")  # Should be (396, 1)
            
            # 2. Form constraint matrix A for dynamics
            A_base = cache['A']  # System A matrix
            B_base = cache['B']  # System B matrix
            
            # # Build A matrix that enforces x_{k+1} = Ax_k + Bu_k
            # A_blocks = []
            # for i in range(N-1):
            #     row_block = np.zeros((nx, (nx+nu)*(N-1) + nx))
            #     col_idx = i*(nx+nu)
            #     row_block[:, col_idx:col_idx+nx] = A_base
            #     row_block[:, col_idx+nx:col_idx+nx+nu] = B_base
            #     row_block[:, col_idx+nx+nu:col_idx+2*nx+nu] = -np.eye(nx)
            #     A_blocks.append(row_block)
            # A = np.vstack(A_blocks)  # Paper's A matrix
            # print(f"A shape: {A.shape}")  # Should be (288, 396)


            # Form constraint matrix A for both dynamics and inputs
            A_dynamics = []  # For x_{k+1} = Ax_k + Bu_k
            A_inputs = []    # For input bounds

            print("\nA matrix components:")
            print(f"A_base (system dynamics):\n{A_base}")
            print(f"B_base (input matrix):\n{B_base}")


            for i in range(N-1):
                # Dynamics block
                dyn_block = np.zeros((nx, (nx+nu)*(N-1) + nx))
                col_idx = i*(nx+nu)
                dyn_block[:, col_idx:col_idx+nx] = A_base
                dyn_block[:, col_idx+nx:col_idx+nx+nu] = B_base
                dyn_block[:, col_idx+nx+nu:col_idx+2*nx+nu] = -np.eye(nx)
                A_dynamics.append(dyn_block)
                
                # Input block (only once!)
                input_block = np.zeros((nu, (nx+nu)*(N-1) + nx))
                input_block[:, col_idx+nx:col_idx+nx+nu] = np.eye(nu)
                A_inputs.append(input_block)

                        

            # Before stacking
            print("\nBefore stacking:")
            print(f"Dynamics block shape: {A_dynamics[0].shape}")  # Should be (12, 396)
            print(f"Input block shape: {A_inputs[0].shape}")       # Should be (4, 396)

            A = np.vstack([
                np.vstack(A_inputs),    # Input constraints first
                np.vstack(A_dynamics)   # Then dynamics constraints
            ])

            print(f"Final A shape: {A.shape}")



            # # 3. Form constrained variable z
            # z_blocks = []
            # for i in range(N-1):
            #     z_blocks.append(v_prev[:, i].reshape(-1, 1))
            # z = np.vstack(z_blocks)
            # print(f"z shape: {z.shape}")  # Should be (288, 1)

            # print(f"y_prev shape: {y_prev.shape}")  # Debug y_prev shape

            # 3. Form constrained variable z
            z_inputs = []    # For input bounds
            z_dynamics = []  # For dynamics
            for i in range(N-1):
                z_inputs.append(z_prev[:, i].reshape(-1, 1))     # Input slack variables
                z_dynamics.append(v_prev[:, i].reshape(-1, 1))   # Dynamics slack variables

            z = np.vstack([
                np.vstack(z_inputs),    # nu*(N-1) rows
                np.vstack(z_dynamics)   # nx*(N-1) rows
            ])

            # 6. Form dual variable y 
            y_inputs = []    # For input bounds
            y_dynamics = []  # For dynamics
            for i in range(N-1):
                y_inputs.append(y_prev[:, i].reshape(-1, 1))     # Input duals
                y_dynamics.append(g_prev[:, i].reshape(-1, 1))   # Dynamics duals

            y = np.vstack([
                np.vstack(y_inputs),    # nu*(N-1) rows
                np.vstack(y_dynamics)   # nx*(N-1) rows
            ])



            
            # 4. Form cost matrix P
            Q = cache['Q']
            R = cache['R']
            P_blocks = []
            for i in range(N):
                if i < N-1:
                    P_block = np.block([
                        [Q, np.zeros((nx, nu))],
                        [np.zeros((nu, nx)), R]
                    ])
                else:
                    P_block = Q
                P_blocks.append(P_block)
            P = block_diag(*P_blocks)
            print(f"P shape: {P.shape}")  # Should be (396, 396)

            # 5. Form cost gradient q
            q_blocks = []
            for i in range(N):
                t = current_time if current_time is not None else 0
                x_ref = generate_figure8_reference(t)
                delta_x = x_prev[:, i] - x_ref[:12]
                q_x = Q @ delta_x.reshape(-1, 1)
                if i < N-1:
                    q_u = R @ u_prev[:, i].reshape(-1, 1)
                    q_blocks.extend([q_x, q_u])
                else:
                    q_blocks.append(q_x)
            q = np.vstack(q_blocks)
            print(f"q shape: {q.shape}")  # Should be (396, 1)

            print("\nInput value checks:")
            print(f"x norm: {np.linalg.norm(x)}")
            print(f"y norm: {np.linalg.norm(y)}")
            print(f"z norm: {np.linalg.norm(z)}")

            # Check first few values
            print("\nFirst few values:")
            print(f"x[:5]: {x[:5].flatten()}")
            print(f"y[:5]: {y[:5].flatten()}")
            print(f"z[:5]: {z[:5].flatten()}")

            # Check A matrix
            print("\nA matrix check:")
            print(f"A non-zeros: {np.count_nonzero(A)}")
            print(f"First non-zero in A: {A[A!=0][0] if np.any(A!=0) else 'all zeros'}")

            
        
      
            # # 6. Form dual variable y (only for dynamics constraints)
            # y_blocks = []
            # for i in range(N-1):
            #     y_blocks.append(v_prev[:, i].reshape(-1, 1))
            # y = np.vstack(y_blocks)  # Using only dynamics duals
            # print(f"y shape: {y.shape}")  # Should be (288, 1)

            # 7. Compute residuals
            # Primal residual
            Ax = A @ x
            r_prim = Ax - z
            pri_res = np.linalg.norm(r_prim, ord=np.inf)
            pri_norm = max(np.linalg.norm(Ax, ord=np.inf), 
                          np.linalg.norm(z, ord=np.inf))


  

            # Dual residual
            r_dual = P @ x + q + A.T @ y
            dual_res = np.linalg.norm(r_dual, ord=np.inf)
            
            # Normalization terms
            Px = P @ x
            ATy = A.T @ y
            dual_norm = max(
                np.linalg.norm(Px, ord=np.inf),
                np.linalg.norm(ATy, ord=np.inf),
                np.linalg.norm(q, ord=np.inf)
            )

            # Add debug prints
            print("\nResidual components:")
            print(f"Ax norm: {np.linalg.norm(Ax)}")
            print(f"z norm: {np.linalg.norm(z)}")
            print(f"P@x norm: {np.linalg.norm(P@x)}")
            print(f"q norm: {np.linalg.norm(q)}")
            print(f"A.T@y norm: {np.linalg.norm(A.T@y)}")

            # 8. Compute new rho
            normalized_pri = pri_res / (pri_norm + 1e-10)
            normalized_dual = dual_res / (dual_norm + 1e-10)

            ratio = np.sqrt(normalized_pri / (normalized_dual + 1e-10))
            print(f"ratio: {ratio}")

            print(f"normalized_pri: {normalized_pri}")
            print(f"normalized_dual: {normalized_dual}")


            
            


            rho_new = current_rho * np.sqrt(normalized_pri / (normalized_dual + 1e-10))

            
            rho_new = np.clip(rho_new, self.rho_min, self.rho_max)


            # ideal_rho = current_rho * np.sqrt(normalized_pri / (normalized_dual + 1e-10))

            # #Find closest rho in sequence
            # self.current_idx = np.argmin(np.abs(self.rhos - ideal_rho))
            # rho_new = self.rhos[self.current_idx]
            
            print(f"\nResiduals and normalizations:")
            print(f"||r_prim||_∞: {pri_res}, pri_norm: {pri_norm}")
            print(f"||r_dual||_∞: {dual_res}, dual_norm: {dual_norm}")
            print(f"normalized ||r_prim||: {normalized_pri}")
            print(f"normalized ||r_dual||: {normalized_dual}")
            print(f"ρ_new: {rho_new}")
            
            self.rho_history.append(rho_new)
            return rho_new

        except Exception as e:
            print(f"Error in predict_rho: {e}")
            import traceback
            traceback.print_exc()
            return current_rho
        
        
   

    
        


    

class TinyMPC:
    def __init__(self, input_data, Nsteps, mode = 0):
        self.cache = {}
        self.cache['rho'] = input_data['rho']  # Fixed rho
        self.cache['A'] = input_data['A']
        self.cache['B'] = input_data['B']
        self.cache['Q'] = input_data['Q']
        self.cache['R'] = input_data['R']



        A = input_data['A']  # 12x12 
        B = input_data['B']  # 12x4

        print(f"A: {A}")
        
        # # Create stacked system matrix for trajectory tracking
        # self.cache['A_stacked'] = np.block([
        #     [A, B],  # [12x12, 12x4]
        #     [np.zeros((Nu, Nx)), np.eye(Nu)]  # [4x12, 4x4]
        # ])  # Final size: (12+4)x(12+4) = 16x16


        nx = self.cache['A'].shape[0]  # State dimension
        nu = self.cache['B'].shape[1]  # Input dimension


        # # Create stacked system matrix for trajectory tracking
        # self.cache['A_stacked'] = np.block([
        #     [A, B],  # [12x12, 12x4]
        #     [-np.eye((Nu, Nx)), -np.eye(Nu)]  # [4x12, 4x4]
        # ])  # Final size: (12+4)x(12+4) = 16x16



    
        self.compute_cache_terms()
        self.compute_lqr_sensitivity()
        self.set_tols_iters()
        self.x_prev = np.zeros((self.cache['A'].shape[0],Nsteps))
        self.u_prev = np.zeros((self.cache['B'].shape[1],Nsteps))
        self.N = Nsteps
        self.rho_adapter = RhoAdapter()

    def compute_cache_terms(self):
        Q_rho = self.cache['Q']
        R_rho = self.cache['R']
        R_rho += self.cache['rho'] * np.eye(R_rho.shape[0])
        Q_rho += self.cache['rho'] * np.eye(Q_rho.shape[0])

        A = self.cache['A']
        B = self.cache['B']
        Kinf = np.zeros(B.T.shape)
        Pinf = np.copy(Q)
        
        for k in range(5000):
            Kinf_prev = np.copy(Kinf)
            Kinf = inv(R_rho + B.T @ Pinf @ B) @ B.T @ Pinf @ A
            Pinf = Q_rho + A.T @ Pinf @ (A - B @ Kinf)
            
            if np.linalg.norm(Kinf - Kinf_prev, 2) < 1e-10:
                break

        AmBKt = (A - B @ Kinf).T
        Quu_inv = np.linalg.inv(R_rho + B.T @ Pinf @ B)

        self.cache['Kinf'] = Kinf
        self.cache['Pinf'] = Pinf
        self.cache['C1'] = Quu_inv
        self.cache['C2'] = AmBKt

    def backward_pass_grad(self, d, p, q, r):
        for k in range(self.N-2, -1, -1):
            d[:, k] = np.dot(self.cache['C1'], np.dot(self.cache['B'].T, p[:, k + 1]) + r[:, k])
            p[:, k] = q[:, k] + np.dot(self.cache['C2'], p[:, k + 1]) - np.dot(self.cache['Kinf'].T, r[:, k])

    def forward_pass(self, x, u, d):
        for k in range(self.N - 1):
            u[:, k] = -np.dot(self.cache['Kinf'], x[:, k]) - d[:, k]
            x[:, k + 1] = np.dot(self.cache['A'], x[:, k]) + np.dot(self.cache['B'], u[:, k])

    def update_primal(self, x, u, d, p, q, r):
        self.backward_pass_grad(d, p, q, r)
        self.forward_pass(x, u, d)

    def update_slack(self, z, v, y, g, u, x, umax = None, umin = None, xmax = None, xmin = None):
        for k in range(self.N - 1):
            z[:, k] = u[:, k] + y[:, k]
            v[:, k] = x[:, k] + g[:, k]

            if (umin is not None) and (umax is not None):
                z[:, k] = np.clip(z[:, k], umin, umax)

            if (xmin is not None) and (xmax is not None):
                v[:, k] = np.clip(v[:, k], xmin, xmax)

        v[:, self.N-1] = x[:, self.N-1] + g[:, self.N-1]
        if (xmin is not None) and (xmax is not None):
            v[:, self.N-1] = np.clip(v[:, self.N-1], xmin, xmax)

    def update_dual(self, y, g, u, x, z, v):
        for k in range(self.N - 1):
            y[:, k] += u[:, k] - z[:, k]
            g[:, k] += x[:, k] - v[:, k]
        g[:, self.N-1] += x[:, self.N-1] - v[:, self.N-1]

    def update_linear_cost(self, r, q, p, z, v, y, g, u_ref, x_ref):
        for k in range(self.N - 1):
            r[:, k] = -self.cache['R'] @ u_ref[:, k]
            r[:, k] -= self.cache['rho'] * (z[:, k] - y[:, k])

            q[:, k] = -self.cache['Q'] @ x_ref[:, k]
            q[:, k] -= self.cache['rho'] * (v[:, k] - g[:, k])

        p[:,self.N-1] = -np.dot(self.cache['Pinf'], x_ref[:, self.N-1])
        p[:,self.N-1] -= self.cache['rho'] * (v[:, self.N-1] - g[:, self.N-1])

        self.cache['q'] = q.copy()

    def set_bounds(self, umax = None, umin = None, xmax = None, xmin = None):
        if (umin is not None) and (umax is not None):
            self.umin = umin
            self.umax = umax
        if (xmin is not None) and (xmax is not None):
            self.xmin = xmin
            self.xmax = xmax

    def set_tols_iters(self, max_iter = 500, abs_pri_tol = 1e-2, abs_dua_tol = 1e-2):
        self.max_iter = max_iter
        self.abs_pri_tol = abs_pri_tol
        self.abs_dua_tol = abs_dua_tol


    
    def compute_lqr_sensitivity(self):
        print("Computing LQR sensitivity")
        def lqr_direct(rho):
            R_rho = self.cache['R'] + rho * np.eye(self.cache['R'].shape[0])
            A, B = self.cache['A'], self.cache['B']
            Q = self.cache['Q']
            
            # Compute fresh P for this rho
            P = Q  # Start with Q
            for _ in range(10):  # Few iterations for fresh P
                K = np.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
                P = Q + A.T @ P @ (A - B @ K)
            
            # Rest using fresh P
            K = np.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
            C1 = np.linalg.inv(R_rho + B.T @ P @ B)
            C2 = A - B @ K
            
            return np.concatenate([K.flatten(), P.flatten(), C1.flatten(), C2.flatten()])
        
        # Get derivatives using autodiff on the direct equations
        m, n = self.cache['Kinf'].shape
        derivs = jacobian(lqr_direct)(self.cache['rho'])
        
        # Reshape into respective matrices
        k_size = m * n
        p_size = n * n
        c1_size = m * m
        c2_size = n * n
        
        self.cache['dKinf_drho'] = derivs[:k_size].reshape(m, n)
        self.cache['dPinf_drho'] = derivs[k_size:k_size+p_size].reshape(n, n)
        self.cache['dC1_drho'] = derivs[k_size+p_size:k_size+p_size+c1_size].reshape(m, m)
        self.cache['dC2_drho'] = derivs[k_size+p_size+c1_size:].reshape(n, n)


    # def compute_lqr_sensitivity(self):
    #     print("Computing LQR sensitivity using numerical differentiation")
        
    #     # Small perturbation for finite difference
    #     eps = 1e-4
    #     rho = self.cache['rho']
        
    #     # Helper function to compute LQR matrices for a given rho
    #     def compute_lqr(rho_val):
    #         R_rho = self.cache['R'] + rho_val * np.eye(self.cache['R'].shape[0])
    #         A, B = self.cache['A'], self.cache['B']
    #         Q = self.cache['Q']
            
    #         # Compute P
    #         P = np.copy(Q)
    #         for _ in range(10):
    #             K = np.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
    #             P = Q + A.T @ P @ (A - B @ K)
            
    #         # Compute final matrices
    #         K = np.linalg.inv(R_rho + B.T @ P @ B) @ B.T @ P @ A
    #         C1 = np.linalg.inv(R_rho + B.T @ P @ B)
    #         C2 = A - B @ K
            
    #         return K, P, C1, C2
        
    #     # Compute at rho + eps
    #     K_plus, P_plus, C1_plus, C2_plus = compute_lqr(rho + eps)
        
    #     # Compute at rho - eps
    #     K_minus, P_minus, C1_minus, C2_minus = compute_lqr(rho - eps)
        
    #     # Central difference approximation of derivatives
    #     self.cache['dKinf_drho'] = (K_plus - K_minus) / (2 * eps)
    #     self.cache['dPinf_drho'] = (P_plus - P_minus) / (2 * eps)
    #     self.cache['dC1_drho'] = (C1_plus - C1_minus) / (2 * eps)
    #     self.cache['dC2_drho'] = (C2_plus - C2_minus) / (2 * eps)
        
    #     # Print max derivatives for debugging
    #     print(f"Max derivative magnitudes:")
    #     print(f"dKinf_drho: {np.max(np.abs(self.cache['dKinf_drho']))}")
    #     print(f"dPinf_drho: {np.max(np.abs(self.cache['dPinf_drho']))}")
    #     print(f"dC1_drho: {np.max(np.abs(self.cache['dC1_drho']))}")
    #     print(f"dC2_drho: {np.max(np.abs(self.cache['dC2_drho']))}")
    
    
    
    def update_rho(self, new_rho):
       
        old_rho = self.cache['rho']
        delta_rho = new_rho - old_rho

        print(f"Delta rho: {delta_rho}")
        
        self.cache['rho'] = new_rho
        self.cache['Kinf'] +=  delta_rho * self.cache['dKinf_drho']
        self.cache['Pinf'] +=  delta_rho * self.cache['dPinf_drho']
        self.cache['C1'] +=  delta_rho * self.cache['dC1_drho']
        self.cache['C2'] +=  delta_rho * self.cache['dC2_drho']
        
            
    def solve_admm(self, x_init, u_init, x_ref=None, u_ref=None, current_time=None):
        status = 0
        x = np.copy(x_init)
        u = np.copy(u_init)
        v = np.zeros(x.shape)
        z = np.zeros(u.shape)
        v_prev = np.zeros(x.shape)
        z_prev = np.zeros(u.shape)
        g = np.zeros(x.shape)
        y = np.zeros(u.shape)
        q = np.zeros(x.shape)
        r = np.zeros(u.shape)
        p = np.zeros(x.shape)
        d = np.zeros(u.shape)

        if (x_ref is None):
            x_ref = np.zeros(x.shape)
        if (u_ref is None):
            u_ref = np.zeros(u.shape)


        iteration_budget = 15


        for k in range(iteration_budget):

            

            # Check before primal update
            print("Before primal update:")
            print(f"x contains NaN: {np.any(np.isnan(x))}")
            print(f"u contains NaN: {np.any(np.isnan(u))}")
            
            self.update_primal(x, u, d, p, q, r)

            # Check after primal update
            print("After primal update:")
            print(f"x contains NaN: {np.any(np.isnan(x))}")
            print(f"u contains NaN: {np.any(np.isnan(u))}")
            


            self.update_slack(z, v, y, g, u, x, self.umax, self.umin, self.xmax, self.xmin)

            # Check after primal update
            print("After primal update:")
            print(f"x contains NaN: {np.any(np.isnan(x))}")
            print(f"u contains NaN: {np.any(np.isnan(u))}")
        

            self.update_dual(y, g, u, x, z, v)
            self.update_linear_cost(r, q, p, z, v, y, g, u_ref, x_ref)

            pri_res_input = np.max(np.abs(u - z))
            pri_res_state = np.max(np.abs(x - v))


            print(f"Residuals:")
            print(f"pri_res_input: {pri_res_input}")
            print(f"pri_res_state: {pri_res_state}")
            print(f"Current rho: {self.cache['rho']}")

            dua_res_input = np.max(np.abs(self.cache['rho'] * (z_prev - z)))
            dua_res_state = np.max(np.abs(self.cache['rho'] * (v_prev - v)))


            pri_res = max(pri_res_input, pri_res_state)
            dual_res = max(dua_res_input, dua_res_state)


            new_rho = self.rho_adapter.predict_rho(
                        pri_res, 
                        dual_res, 
                        k, 
                        self.cache['rho'],
                        self.cache,
                        x,  # current x
                        u,
                        v,
                        z,  # current z
                        g,
                        y,   # current y
                        current_time=current_time
            )

            self.update_rho(new_rho)
                
                # With this code, stats are  - 
                # Final position error: 0.8228 m
                # Average position error: 0.7367 m
                # Average control effort: 0.5232
                # Total iterations: 36119

                # if abs(new_rho - self.cache['rho']) > 1e-6:
                #     # print(f"\nRho update at k={k}:")
                #     # print(f"Old rho: {self.cache['rho']}, New rho: {new_rho}")
                #     self.update_rho(new_rho)

                
                # With this code, stats are exactly the same as above
            # if k%10 == 0:
            #     self.update_rho(new_rho)

            z_prev = np.copy(z)
            v_prev = np.copy(v)

            if (pri_res < self.abs_pri_tol and dual_res < self.abs_dua_tol):
                status = 1
                break

        self.x_prev = x
        self.u_prev = u
        return x, u, status, k





def delta_x_quat(x_curr, t):
    """Compute error between current state and reference"""
    x_ref = generate_figure8_reference(t)
    
    # Current quaternion
    q = x_curr[3:7]
    
    # Reference quaternion (hover)
    q_ref = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Quaternion error
    phi = qtorp(L(q_ref).T @ q)
    
    # Full state error (12 dimensions)
    delta_x = np.hstack([
        x_curr[0:3] - x_ref[0:3],    # position error
        phi,                          # attitude error (3 components)
        x_curr[7:10] - x_ref[6:9],   # velocity error
        x_curr[10:13] - x_ref[9:12]  # angular velocity error
    ])
    return delta_x

def tinympc_controller(x_curr, t):
    """MPC controller with time-varying reference"""
    # Generate reference trajectory for horizon
    x_ref = np.zeros((Nx, N))
    u_ref = np.zeros((Nu, N-1))
    
    for i in range(N):
        x_ref[:,i] = generate_figure8_reference(t + i*h)
    u_ref[:] = uhover.reshape(-1,1)

    delta_x = delta_x_quat(x_curr, t)
    
    # Debug prints
    if t < 0.1:  # Print only at start
        print("\nInitial tracking error:")
        print(f"Position error: {np.linalg.norm(delta_x[0:3]):.4f}")
        print(f"Attitude error: {np.linalg.norm(delta_x[3:6]):.4f}")
        print(f"Velocity error: {np.linalg.norm(delta_x[6:9]):.4f}")
    
    x_init = np.copy(tinympc.x_prev)
    x_init[:,0] = delta_x
    u_init = np.copy(tinympc.u_prev)

    x_out, u_out, status, k = tinympc.solve_admm(x_init, u_init, x_ref, u_ref, current_time=t)
    
    return uhover + u_out[:,0], k


def visualize_trajectory(x_all, u_all):
    # Convert lists to numpy arrays
    x_all = np.array(x_all)
    u_all = np.array(u_all)
    nsteps = len(x_all)
    t = np.arange(nsteps) * h
    
    # Figure 8 parameters
    A = 1.0  # amplitude
    w = 2*np.pi/6  # frequency
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot 1: 2D Trajectory
    plt.subplot(131)
    plt.plot(x_all[:, 0], x_all[:, 2], 'b-', label='Actual')
    
    # # Plot reference figure-8 (matching Julia)
    # t_ref = np.linspace(0, 8.0, 100)  # Longer time range
    # x_ref = np.sin(2*t_ref)           # Julia's x reference
    # z_ref = np.cos(t_ref)/2           # Julia's z reference
    # plt.plot(x_ref, z_ref, 'r--', label='Reference')

    t_ref = np.linspace(0, 5.0, 100)
    x_ref = A * np.sin(w*t_ref)
    z_ref = A * np.sin(2*w*t_ref)/2
    plt.plot(x_ref, z_ref, 'r--', label='Reference')

    
    plt.title('Figure-8 Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    # # Plot 2: Position vs Time
    # plt.subplot(132)
    # plt.plot(t, x_all[:, 0], 'b-', label='x')
    # plt.plot(t, x_all[:, 2], 'r-', label='z')
    # plt.plot(t, np.sin(2*t), 'b--', label='x_ref')
    # plt.plot(t, np.cos(t)/2, 'r--', label='z_ref')
    # plt.title('Position vs Time')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Position [m]')
    # plt.grid(True)
    # plt.legend()

    plt.subplot(132)
    plt.plot(t, x_all[:, 0], 'b-', label='x')
    plt.plot(t, x_all[:, 2], 'r-', label='z')
    plt.plot(t, A*np.sin(w*t), 'b--', label='x_ref')
    plt.plot(t, A*np.sin(2*w*t)/2, 'r--', label='z_ref')
    plt.title('Position vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Control Inputs
    plt.subplot(133)
    plt.plot(t, u_all)
    plt.plot(t, [uhover[0]]*nsteps, 'k--', label='hover')
    plt.title('Control Inputs')
    plt.xlabel('Time [s]')
    plt.ylabel('Motor Commands')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate final reference position
    final_t = t[-1]
    final_ref_x = A * np.sin(w*final_t)
    final_ref_z = A * np.sin(2*w*final_t)/2
    

if __name__ == "__main__":
    # Clear the rho file at start of simulation
    #open('data/raw_rhos.txt', 'w').close()
    
    # Initialize system
    rg = np.array([0.0, 0, 0.0])
    qg = np.array([1.0, 0, 0, 0])
    vg = np.zeros(3)
    omgg = np.zeros(3)
    xg = np.hstack([rg, qg, vg, omgg])
    uhover = (mass*g/kt/4)*np.ones(4)

    A_jac = jacobian(quad_dynamics_rk4, 0)
    B_jac = jacobian(quad_dynamics_rk4, 1)
    
    Anp1 = A_jac(xg, uhover)
    Bnp1 = B_jac(xg, uhover)
    
    Anp = E(qg).T @ Anp1 @ E(qg)
    Bnp = E(qg).T @ Bnp1

    # Initial state
    x0 = np.copy(xg)
    x0[0:3] = np.array([0.0, 0.0, 0.0])  # Start at origin
    x0[3:7] = rptoq(np.array([0.0, 0.0, 0.0]))  # Zero attitude error

    # Modify MPC parameters for better tracking
    N = 25  # horizon length
    rho = 5.0 # initial rho (Julia starts at 5 and multiplies by 5)
    
    # Much tighter weights for better tracking
    max_dev_x = np.array([
        0.01, 0.01, 0.01,    # position (exactly as Julia)
        0.5, 0.5, 0.05,      # attitude (asymmetric as Julia)
        0.5, 0.5, 0.5,       # velocity
        0.7, 0.7, 0.5        # angular velocity
    ])
    max_dev_u = np.array([0.1, 0.1, 0.1, 0.1])  # control bounds
    
    # Construct cost matrices exactly as Julia
    Q = np.diag(1./(max_dev_x**2))  # No extra scaling
    R = np.diag(1./(max_dev_u**2))
    
    # Setup TinyMPC with modified parameters
    input_data = {
        'rho': rho,
        'A': Anp,
        'B': Bnp,
        'Q': Q,
        'R': R
    }
    
    tinympc = TinyMPC(input_data, N)
    
    # Wider control bounds for more authority
    # u_max = np.array([0.5, 0.5, 0.5, 0.5])  # exactly as Julia
    # u_min = -u_max
    # # x_max = [float('inf')] * Nx
    # # x_min = [-float('inf')] * Nx

    # x_max = [3.0] * Nx
    # x_min = [-3.0] * Nx

    u_max = [1.0-uhover[0]] * Nu
    u_min = [-1*uhover[0]] * Nu
    x_max = [2.] * Nx
    x_min = [-2.0] * Nx
    tinympc.set_bounds(u_max, u_min, x_max, x_min)


    tinympc.set_bounds(u_max, u_min, x_max, x_min)

    # Set nominal trajectory
    from scipy.spatial.transform import Rotation as spRot
    R0 = spRot.from_quat(qg)
    eulerg = R0.as_euler('zxy')
    xg_euler = np.hstack((eulerg,xg[4:]))
    x_nom_tinyMPC = np.tile(0*xg_euler,(N,1)).T
    u_nom_tinyMPC = np.tile(uhover,(N-1,1)).T

    # Run simulation
    def simulate_with_controller(x0, controller, NSIM=200):  # Longer simulation
        x_all = []
        u_all = []
        x_curr = np.copy(x0)
        iterations = []
        rho_vals = []
        
        for i in range(NSIM):
            t = i * h
            u_curr, k = controller(x_curr, t)
            #u_curr_clipped = np.clip(u_curr, 0, 1)
            x_curr = quad_dynamics_rk4(x_curr, u_curr)
            
            # Reshape and store
            x_curr = np.array(x_curr).reshape(-1)
            u_curr = np.array(u_curr).reshape(-1)
            
            x_all.append(x_curr)
            u_all.append(u_curr)
            iterations.append(k)
            rho_vals.append(tinympc.cache['rho'])
            
        return x_all, u_all, iterations, rho_vals

    # Run simulation with modified parameters
    x_all, u_all, iterations, rho_vals = simulate_with_controller(x0, tinympc_controller, NSIM=400)

    # Visualize trajectory
    visualize_trajectory(x_all, u_all)

    #np.savetxt('data/iterations_traj_adapt_OSQP.txt', iterations)

    np.savetxt('data/iterations/adapt_traj.txt', iterations)
    np.savetxt('data/rho_vals/adapt_traj.txt', rho_vals)

    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.plot(iterations, label='Iterations')
    plt.ylabel('Iterations')
    plt.title('ADMM Iterations per Time Step')
    print("Total iterations:", sum(iterations))
    plt.grid(True)
    plt.legend()

    # Plot rho values
    plt.subplot(212)
    #plt.scatter(range(len(rho_history)), rho_history, label='Rho')
    plt.plot(tinympc.rho_adapter.rho_history, label='Rho')
    #plt.step(range(len(rho_history)), rho_history, label='Rho')
    plt.xlabel('Time Step')
    plt.ylabel('Rho Value')
    plt.grid(True)
    plt.legend()


    plt.show()

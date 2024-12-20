import math
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy as sqrt
from autograd.numpy.linalg import norm
from autograd.numpy.linalg import inv
from autograd import jacobian
from autograd.test_util import check_grads
import torch
import time

from scipy.linalg import block_diag
   
from autograd import jacobian


#DEFINING INITIAL RHO
initial_rho = 85.0

np.set_printoptions(precision=4, suppress=True)

# Quaternion functions (same as fixed version)
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

# Quadrotor parameters (same as fixed version)
mass = 0.035
J = np.array([[1.66e-5, 0.83e-6, 0.72e-6], 
              [0.83e-6, 1.66e-5, 1.8e-6], 
              [0.72e-6, 1.8e-6, 2.93e-5]])
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

# Dynamics functions with wind disturbances
def quad_dynamics(x, u, wind = np.array([0.0, 0.0, 0.0])):
    r = x[0:3]
    q = x[3:7]/norm(x[3:7])
    v = x[7:10]
    omg = x[10:13]
    Q = qtoQ(q)

    dr = v
    dq = 0.5*L(q)@H@omg
    dv = np.array([0, 0, -g]) + (1/mass)*Q@np.array([[0, 0, 0, 0], 
                                                     [0, 0, 0, 0], 
                                                     [kt, kt, kt, kt]])@u + wind
    domg = inv(J)@(-hat(omg)@J@omg + 
                   np.array([[-el*kt, -el*kt, el*kt, el*kt], 
                            [-el*kt, el*kt, el*kt, -el*kt], 
                            [-km, km, -km, km]])@u)

    return np.hstack([dr, dq, dv, domg])

#

def quad_dynamics_rk4(x, u):
    f1 = quad_dynamics(x, u)
    f2 = quad_dynamics(x + 0.5*h*f1, u)
    f3 = quad_dynamics(x + 0.5*h*f2, u)
    f4 = quad_dynamics(x + h*f3, u)
    xn = x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
    xnormalized = xn[3:7]/norm(xn[3:7])
    return np.hstack([xn[0:3], xnormalized, xn[7:13]])


# global variable for rhos 




class RhoAdapter:
    def __init__(self, min_rho=70.0, max_rho=100.0):
        self.min_rho = min_rho
        self.max_rho = max_rho

        self.rho_base = 85.0  # Center of our rho range
        self.tolerance = 1.1  # Growth factor between rho values

        self.rho_min = 60.0
        self.rho_max = 100.0
        
        
        # Pre-compute sequence of rhos and associated matrices
        self.rhos = self.setup_rho_sequence()
        self.current_idx = len(self.rhos)//2  # Start in middle
        
        # For paper comparison/analysis
        self.rho_history = []
        self.residual_history = []



    def setup_rho_sequence(self):
        """Generate geometric sequence of rhos (like ReLU-QP)"""
        rhos = [self.rho_base]
        # Generate smaller rhos
        rho = self.rho_base
        while rho >= self.rho_min:
            rho = rho / self.tolerance
            rhos.append(rho)
        # Generate larger rhos
        rho = self.rho_base
        while rho <= self.rho_max:
            rho = rho * self.tolerance
            rhos.append(rho)
        return np.sort(rhos)


    def predict_rho(self, pri_res, dual_res, iterations, current_rho, cache, x_prev, u_prev, v_prev, z_prev, g_prev, y_prev, current_time=None):
        try:

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

            for i in range(N-1):
                # Dynamics block
                dyn_block = np.zeros((nx, (nx+nu)*(N-1) + nx))
                col_idx = i*(nx+nu)
                dyn_block[:, col_idx:col_idx+nx] = A_base
                dyn_block[:, col_idx+nx:col_idx+nx+nu] = B_base
                dyn_block[:, col_idx+nx+nu:col_idx+2*nx+nu] = -np.eye(nx)
                A_dynamics.append(dyn_block)
                
                # Input block
                input_block = np.zeros((nu, (nx+nu)*(N-1) + nx))
                input_block[:, col_idx+nx:col_idx+nx+nu] = np.eye(nu)
                A_inputs.append(input_block)

            A = np.vstack([
                np.vstack(A_inputs),    # Input constraints first
                np.vstack(A_dynamics)   # Then dynamics constraints
            ])

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
                # For hover, reference is just xg (equilibrium)
                delta_x = x_prev[:, i] - xg[:12]  # Use xg instead of figure8 reference
                q_x = Q @ delta_x.reshape(-1, 1)
                if i < N-1:
                    # For hover, reference input is uhover
                    delta_u = u_prev[:, i] - uhover  # Use uhover as reference
                    q_u = R @ delta_u.reshape(-1, 1)
                    q_blocks.extend([q_x, q_u])
                else:
                    q_blocks.append(q_x)
            q = np.vstack(q_blocks)

            
        
      
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

            # 8. Compute new rho
            normalized_pri = pri_res / (pri_norm + 1e-10)
            normalized_dual = dual_res / (dual_norm + 1e-10)
            rho_new = current_rho * np.sqrt(normalized_pri / (normalized_dual + 1e-10))
            rho_new = np.clip(rho_new, self.rho_min, self.rho_max)


            #ideal_rho = current_rho * np.sqrt(normalized_pri / (normalized_dual + 1e-10))

            # Find closest rho in sequence
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
        self.cache['rho'] = input_data['rho']
        self.cache['A'] = input_data['A']
        self.cache['B'] = input_data['B']
        self.cache['Q'] = input_data['Q']
        self.cache['R'] = input_data['R']

        A = input_data['A']  # 12x12 
        B = input_data['B']  # 12x4
        Q = input_data['Q']  # 12x12
        R = input_data['R']  # 4x4

        nx = self.cache['A'].shape[0]  # State dimension
        nu = self.cache['B'].shape[1]  # Input dimension


        # A should be AB -I 

        # q should be [q, r] 
        

        # # Create stacked system matrix
        # self.cache['A_stacked'] = np.block([
        #     [A, B],  # [12x12, 12x4]
        #     [np.zeros((Nu, Nx)), -np.eye(Nu)]  # [4x12, 4x4]
        # ])  # Final size: (12+4)x(12+4) = 16x16

        # self.cache['A_stacked'] = np.block([
        #     [self.cache['A'], self.cache['B'], -np.eye(nx)],  # [A B -I]
        #     [np.zeros((nu, nx)), np.eye(nu), -np.eye(nu)]     # [0 I -I]
        #  ])



        # self.cache['A_stacked'] = np.block([
        #     [self.cache['A'], self.cache['B'], -np.eye(nx)]  # [A B -I]
        # ])

        # Stack cost matrices [Q 0; 0 R]
    #     self.cache['q'] = np.block([
    #         [self.cache['Q']],  # Q for states
    #         [self.cache['R']]   # R for inputs
    # ])
    
        self.compute_cache_terms()
        self.compute_lqr_sensitivity()
        self.set_tols_iters()
        self.x_prev = np.zeros((self.cache['A'].shape[0],Nsteps))
        self.u_prev = np.zeros((self.cache['B'].shape[1],Nsteps))
        self.N = Nsteps
        self.rho_adapter = RhoAdapter()
        self.last_k = float('inf')
        self.cache['z'] = None
        self.cache['z_prev'] = None
        
    def compute_cache_terms(self):
        Q_rho = self.cache['Q']
        R_rho = self.cache['R']
        R_rho += self.cache['rho'] * np.eye(R_rho.shape[0])
        Q_rho += self.cache['rho'] * np.eye(Q_rho.shape[0])

        A = self.cache['A']
        B = self.cache['B']
        Kinf = np.zeros(B.T.shape)
        Pinf = np.copy(Q)  # Changed from Q to Q_rho
        
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

        #print A, B, Q and R 
        # print(f"A : {A}")
        # print(f"B : {B}")
        # print(f"Q : {Q}")
        # print(f"R : {R}")

        #self.compute_lqr_sensitivity()

        


    def backward_pass_grad(self, d, p, q, r):
        for k in range(self.N-2, -1, -1):
            d[:, k] = np.dot(self.cache['C1'], np.dot(self.cache['B'].T, p[:, k + 1]) + r[:, k])
            p[:, k] = q[:, k] + np.dot(self.cache['C2'], p[:, k + 1]) - np.dot(self.cache['Kinf'].T, r[:, k])

    def forward_pass(self, x, u, d):
        for k in range(self.N - 1):
            u[:, k] = -np.dot(self.cache['Kinf'], x[:, k]) - d[:, k]
            x[:, k + 1] = np.dot(self.cache['A'], x[:, k]) + np.dot(self.cache['B'], u[:, k])

    def update_primal(self, x, u, d, p, q, r):
        """Update primal variables with checks"""
        try:
            self.backward_pass_grad(d, p, q, r)
            # print("\nAfter backward pass:")
            # print(f"d contains NaN: {np.any(np.isnan(d))}")
            # print(f"p contains NaN: {np.any(np.isnan(p))}")
            
            self.forward_pass(x, u, d)
            # print("\nAfter forward pass:")
            # print(f"x contains NaN: {np.any(np.isnan(x))}")
            # print(f"u contains NaN: {np.any(np.isnan(u))}")
        except Exception as e:
            print(f"Exception in primal update: {str(e)}")

    def update_slack(self, z, v, y, g, u, x):
        """Update slack variables with checks"""
        try:
            # print("\nBefore slack update:")
            # print(f"u shape: {u.shape}, z shape: {z.shape}")
            # print(f"x shape: {x.shape}, v shape: {v.shape}")
            
            # Project onto constraint sets
            z[:] = np.clip(u + (1/self.cache['rho'])*y, 
                          self.umin.reshape(-1,1), 
                          self.umax.reshape(-1,1))
            v[:] = np.clip(x + (1/self.cache['rho'])*g,
                          self.xmin.reshape(-1,1),
                          self.xmax.reshape(-1,1))
                      
            # print("\nAfter slack update:")
            # print(f"z contains NaN: {np.any(np.isnan(z))}")
            # print(f"v contains NaN: {np.any(np.isnan(v))}")
        except Exception as e:
            print(f"Exception in slack update: {str(e)}")

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
            self.umin = np.array(umin)
            self.umax = np.array(umax)
        if (xmin is not None) and (xmax is not None):
            self.xmin = np.array(xmin)
            self.xmax = np.array(xmax)

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

        #print the values, not norms

        # print(f"dKinf_drho: {self.cache['dKinf_drho']}")
        # print(f"dPinf_drho: {self.cache['dPinf_drho']}")
        # print(f"dC1_drho: {self.cache['dC1_drho']}")
        # print(f"dC2_drho: {self.cache['dC2_drho']}")
        



   
    



    def update_rho(self, new_rho):
        start_time = time.time()
        old_rho = self.cache['rho']
        delta_rho = new_rho - old_rho

        print(f"Delta rho: {delta_rho}")
        
        self.cache['rho'] = new_rho
        self.cache['Kinf'] += delta_rho * self.cache['dKinf_drho']
        self.cache['Pinf'] += delta_rho * self.cache['dPinf_drho']
        self.cache['C1'] += delta_rho * self.cache['dC1_drho']
        self.cache['C2'] += delta_rho * self.cache['dC2_drho']

        

        #print(f"Time taken to update rho: {time.time() - start_time} seconds")

        # self.cache['rho'] = new_rho
        # self.compute_cache_terms()

    # Rest of the methods same as fixed version except solve_admm
    def solve_admm(self, x_init, u_init, x_ref = None, u_ref = None):
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

        x += np.random.normal(0, 1e-3, x.shape)
        u += np.random.normal(0, 1e-3, u.shape)



        # Add initial debug prints here
        print("\nFirst iteration debug:")
        print(f"Initial rho: {self.cache['rho']}")
        print(f"Kinf norm: {np.linalg.norm(self.cache['Kinf'])}")
        print(f"Pinf norm: {np.linalg.norm(self.cache['Pinf'])}")
        print(f"C1 norm: {np.linalg.norm(self.cache['C1'])}")
        print(f"C2 norm: {np.linalg.norm(self.cache['C2'])}")

        self.cache['z'] = z
        self.cache['z_prev'] = z_prev

        if (x_ref is None):
            x_ref = np.zeros(x.shape)
        if (u_ref is None):
            u_ref = np.zeros(u.shape)

        # Wind disturbance setup
        wind_magnitude = 0.005  # m/s
        wind_direction = np.array([1.0, 0.0, 0.0])  # Diagonal wind in xy-plane
        wind_direction = wind_direction / np.linalg.norm(wind_direction)
        wind_effect = wind_magnitude * wind_direction


        x[0:3, :] += wind_effect.reshape(-1, 1)  # Initial position offset

        for k in range(self.max_iter):

            

            # Store previous values
            print("\nBefore primal update:")
            print(f"x contains NaN: {np.any(np.isnan(x))}")
            print(f"u contains NaN: {np.any(np.isnan(u))}")
            print(f"z contains NaN: {np.any(np.isnan(z))}")
            print(f"v contains NaN: {np.any(np.isnan(v))}")
            
            # Primal update
            self.update_primal(x, u, d, p, q, r)
            
            print("\nAfter primal update:")
            print(f"x contains NaN: {np.any(np.isnan(x))}")
            print(f"u contains NaN: {np.any(np.isnan(u))}")
            
            # Slack update
            self.update_slack(z, v, y, g, u, x)
            
            print("\nAfter slack update:")
            print(f"z contains NaN: {np.any(np.isnan(z))}")
            print(f"v contains NaN: {np.any(np.isnan(v))}")
            
            # Calculate residuals with checks
            try:
                pri_res_input = np.max(np.abs(u - z))
                pri_res_state = np.max(np.abs(x - v))
                print(f"\nResidual calculation:")
                print(f"u-z max: {pri_res_input}")
                print(f"x-v max: {pri_res_state}")
            except Exception as e:
                print(f"Exception in residual calculation: {str(e)}")
            
            self.update_dual(y, g, u, x, z, v)
            self.update_linear_cost(r, q, p, z, v, y, g, u_ref, x_ref)

            pri_res_u = np.max(np.abs(u - z))
            pri_res_x = np.max(np.abs(x - v))
            pri_res = max(pri_res_u, pri_res_x)

            print(f"Pri Res ", pri_res)
            
            dual_res_u = np.max(np.abs(self.cache['rho'] * (z_prev - z)))
            dual_res_x = np.max(np.abs(self.cache['rho'] * (v_prev - v)))
            dual_res = max(dual_res_u, dual_res_x)

            #Add first iteration residuals debug
            
                # print("\nFirst iteration residuals:")
                # print(f"pri_res_u: {pri_res_u}")
                # print(f"pri_res_x: {pri_res_x}")
                # print(f"dual_res_u: {dual_res_u}")
                # print(f"dual_res_x: {dual_res_x}")
                # print(f"rho: {self.cache['rho']}")

            
            
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
                    y   # current y
            )

            self.update_rho(new_rho)

                
                # With this code, exactly the same as below
                # if abs(new_rho - self.cache['rho']) > 1e-6:
                #     print(f"\nRho update at k={k}:")
                #     print(f"Old rho: {self.cache['rho']}, New rho: {new_rho}")
                #     self.update_rho(new_rho)


            #     self.update_rho(new_rho)

            # new_rho = self.rho_adapter.predict_rho(
            #         pri_res, 
            #         dual_res, 
            #         k, 
            #         self.cache['rho'],
            #         self.cache,
            #         x,  # current x
            #         u,
            #         v,
            #         z,  # current z
            #         g,
            #         y   # current y
            #     )
                
            #     # With this code, exactly the same as below
            #     # if abs(new_rho - self.cache['rho']) > 1e-6:
            #     #     print(f"\nRho update at k={k}:")
            #     #     print(f"Old rho: {self.cache['rho']}, New rho: {new_rho}")
            #     #     self.update_rho(new_rho)



            
            #With this code - 
            # Trajectory Statistics:
            # Final position error: 0.0026 m
            # Final attitude error: 0.0061
            # Average control effort: 0.1167
            # Total iterations: 2235
            #self.update_rho(new_rho)

            z_prev = np.copy(z)
            v_prev = np.copy(v)

            if (pri_res < self.abs_pri_tol and dual_res < self.abs_dua_tol):
                status = 1
                break

            # # Check if we're actually converging
            # pri_res = max(pri_res_u, pri_res_x)
            # dual_res = max(dual_res_u, dual_res_x)
            
            # Add debug prints
            # if k < 10:  # Print first few iterations
            #     print(f"Iteration {k}:")
            #     print(f"Primal residual: {pri_res}")
            #     print(f"Dual residual: {dual_res}")
            #     print(f"Tolerance: {self.abs_pri_tol}, {self.abs_dua_tol}")

        self.x_prev = x
        self.u_prev = u
        return x, u, status, k


def vec(X):
    """Vectorize a matrix"""
    return X.flatten()

def reshape(x, shape):
    """Reshape vector to matrix"""
    return x.reshape(shape)

def vec(X):
    """Vectorize a matrix"""
    return X.reshape(-1, 1)

def reshape(x, shape):
    """Reshape vector to matrix"""
    return x.reshape(shape)

    


def delta_x_quat(x_curr):
    q = x_curr[3:7]
    phi = qtorp(L(qg).T @ q)
    delta_x = np.hstack([x_curr[0:3]-rg, phi, x_curr[7:10]-vg, x_curr[10:13]-omgg])
    return delta_x

def tinympc_controller(x_curr, x_nom, u_nom):
    delta_x = delta_x_quat(x_curr)
    #noise = np.random.normal(0, 0.01, (Nx,))*0
    
    # zero noise
    noise = np.zeros(Nx)

    delta_x_noise = (delta_x + noise).reshape(Nx).tolist()

    x_init = np.copy(tinympc.x_prev)
    x_init[:,0] = delta_x_noise
    u_init = np.copy(tinympc.u_prev)

    x_out, u_out, status, k = tinympc.solve_admm(x_init, u_init, x_nom, u_nom)
    print(f"Solved with status {status} and k {k}")

    return uhover+u_out[:,0], k

def visualize_trajectory(x_all, u_all):
    # Convert lists to numpy arrays for easier handling
    x_all = np.array(x_all)
    u_all = np.array(u_all)
    nsteps = len(x_all)
    steps = np.arange(nsteps)

    # Create subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Position plot
    ax1 = fig.add_subplot(311)
    ax1.plot(steps, x_all[:, 0], label="x", linewidth=2)
    ax1.plot(steps, x_all[:, 1], label="y", linewidth=2)
    ax1.plot(steps, x_all[:, 2], label="z", linewidth=2)
    ax1.plot(steps, [rg[0]]*nsteps, 'r--', label="x_goal")
    ax1.plot(steps, [rg[1]]*nsteps, 'g--', label="y_goal")
    ax1.plot(steps, [rg[2]]*nsteps, 'b--', label="z_goal")
    ax1.set_ylabel('Position [m]')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("Position Trajectories")

    # Attitude plot
    ax2 = fig.add_subplot(312)
    ax2.plot(steps, x_all[:, 3], label="q0", linewidth=2)
    ax2.plot(steps, x_all[:, 4], label="q1", linewidth=2)
    ax2.plot(steps, x_all[:, 5], label="q2", linewidth=2)
    ax2.plot(steps, x_all[:, 6], label="q3", linewidth=2)
    ax2.plot(steps, [qg[0]]*nsteps, 'r--', label="q0_goal")
    ax2.set_ylabel('Quaternion')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title("Attitude Trajectories")

    # Control inputs plot
    ax3 = fig.add_subplot(313)
    ax3.plot(steps, u_all[:, 0], label="u1", linewidth=2)
    ax3.plot(steps, u_all[:, 1], label="u2", linewidth=2)
    ax3.plot(steps, u_all[:, 2], label="u3", linewidth=2)
    ax3.plot(steps, u_all[:, 3], label="u4", linewidth=2)
    ax3.plot(steps, [uhover[0]]*nsteps, 'k--', label="hover_thrust")
    ax3.set_xlabel('Time steps')
    ax3.set_ylabel('Motor commands')
    ax3.legend()
    ax3.grid(True)
    ax3.set_title("Control Inputs")

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print("\nTrajectory Statistics:")
    print(f"Final position error: {np.linalg.norm(x_all[-1, :3] - rg):.4f} m")
    print(f"Final attitude error: {np.linalg.norm(x_all[-1, 3:7] - qg):.4f}")
    print(f"Average control effort: {np.mean(np.linalg.norm(u_all - uhover.reshape(1,-1), axis=1)):.4f}")

# Add after TinyMPC class, before if __name__ == "__main__":
def dlqr(A, B, Q, R, n_steps = 500):
    """Solve the discrete time lqr controller"""
    P = Q
    for i in range(n_steps):
        K = inv(R + B.T @ P @ B) @ B.T @ P @ A
        P = Q + A.T @ P @ (A - B @ K)
    return K, P
def test_fixed_vs_adaptive():
    # Setup same parameters as main
    rg = np.array([0.0, 0, 0.0])
    qg = np.array([1.0, 0, 0, 0])
    vg = np.zeros(3)
    omgg = np.zeros(3)
    xg = np.hstack([rg, qg, vg, omgg])
    uhover = (mass*g/kt/4)*np.ones(4)
    
    # Get system matrices
    A_jac = jacobian(quad_dynamics_rk4, 0)
    B_jac = jacobian(quad_dynamics_rk4, 1)
    Anp1 = A_jac(xg, uhover)
    Bnp1 = B_jac(xg, uhover)
    Anp = E(qg).T @ Anp1 @ E(qg)
    Bnp = E(qg).T @ Bnp1

    # Setup costs
    max_dev_x = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.05, 0.5, 0.5, 0.5, 0.7, 0.7, 0.2])
    max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])/6
    Q = np.diag(1./max_dev_x**2)
    R = np.diag(1./max_dev_u**2)
    K_lqr, P_lqr = dlqr(Anp, Bnp, Q, R)

    print("\n=== TESTING FIXED VS ADAPTIVE ===")
    
    # Test 1: Fixed rho=100
    input_data_fixed = {
        'rho': 100.0,
        'A': np.copy(Anp),
        'B': np.copy(Bnp),
        'Q': np.copy(P_lqr),
        'R': np.copy(R)
    }
    fixed_controller = TinyMPC(input_data_fixed, Nsteps=10)
    print("\nFIXED RHO TEST (rho=100)")
    print(f"Kinf norm: {np.linalg.norm(fixed_controller.cache['Kinf'])}")
    print(f"Pinf norm: {np.linalg.norm(fixed_controller.cache['Pinf'])}")
    print(f"C1 norm: {np.linalg.norm(fixed_controller.cache['C1'])}")
    print(f"C2 norm: {np.linalg.norm(fixed_controller.cache['C2'])}")

    # Test 2: Adaptive starting at rho=70
    input_data_adaptive = {
        'rho': 70.0,
        'A': np.copy(Anp),
        'B': np.copy(Bnp),
        'Q': np.copy(P_lqr),
        'R': np.copy(R)
    }
    adaptive_controller = TinyMPC(input_data_adaptive, Nsteps=10)
    
    # Force rho to 100 using adaptive update
    adaptive_controller.update_rho(100.0)
    
    print("\nADAPTIVE AFTER REACHING RHO=100")
    print(f"Kinf norm: {np.linalg.norm(adaptive_controller.cache['Kinf'])}")
    print(f"Pinf norm: {np.linalg.norm(adaptive_controller.cache['Pinf'])}")
    print(f"C1 norm: {np.linalg.norm(adaptive_controller.cache['C1'])}")
    print(f"C2 norm: {np.linalg.norm(adaptive_controller.cache['C2'])}")

# Main execution code (same as fixed version but with adaptive controller)
if __name__ == "__main__":


    #test_fixed_vs_adaptive()
    
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
    x0[0:3] += rg + np.array([0.2, 0.2, -0.2])
    x0[3:7] = rptoq(np.array([1.0, 0.0, 0.0]))

    # Setup MPC
    max_dev_x = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.05, 0.5, 0.5, 0.5, 0.7, 0.7, 0.2])
    max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])/6
    Q = np.diag(1./max_dev_x**2)
    R = np.diag(1./max_dev_u**2)

    def dlqr(A, B, Q, R, n_steps = 500):
        P = Q
        for i in range(n_steps):
            K = inv(R + B.T @ P @ B) @ B.T @ P @ A
            P = Q + A.T @ P @ (A - B @ K)
        return K, P

    K_lqr, P_lqr = dlqr(Anp, Bnp, Q, R)

    # Setup TinyMPC
    N = 10
    rho = 85.0
    input_data = {
        'rho': rho,
        'A': Anp,
        'B': Bnp,
        'Q': P_lqr,
        'R': R
    }

    tinympc = TinyMPC(input_data, N)

    u_max = [1.0-uhover[0]] * Nu
    umin = [-1*uhover[0]] * Nu
    x_max = [1000.] * Nx
    x_min = [-1000.0] * Nx
    tinympc.set_bounds(u_max, umin, x_max, x_min)

    # Set nominal trajectory
    from scipy.spatial.transform import Rotation as spRot
    R0 = spRot.from_quat(qg)
    eulerg = R0.as_euler('zxy')
    xg_euler = np.hstack((eulerg,xg[4:]))
    x_nom_tinyMPC = np.tile(0*xg_euler,(N,1)).T
    u_nom_tinyMPC = np.tile(uhover,(N-1,1)).T
    
    def simulate_with_controller(x0, x_nom, u_nom, controller, NSIM = 100):
        x_all = []
        u_all = []
        x_curr = np.copy(x0)
        iterations = []
        rho_vals = []
        
        for i in range(NSIM):
            u_curr, k = controller(x_curr, x_nom, u_nom)
            u_curr_clipped = np.clip(u_curr, 0, 1)
            #x_curr = quad_dynamics_rk4(x_curr, u_curr_clipped)

            x_curr = quad_dynamics_rk4(x_curr, u_curr)
            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()
            x_all.append(x_curr)
            u_all.append(u_curr)
            iterations.append(k)
            rho_vals.append(tinympc.cache['rho'])
        return x_all, u_all, iterations, rho_vals

    
   
    # Create single controller instance
    tinympc = TinyMPC(input_data, N)
    tinympc.set_bounds(u_max, umin, x_max, x_min)

    # Run simulation with online rho adaptation
    x_all, u_all, iterations, rho_vals = simulate_with_controller(x0, x_nom_tinyMPC, u_nom_tinyMPC, tinympc_controller)

    #np.savetxt('data/iterations_stacked_OSQP.txt', iterations)

    # Visualization (keep existing visualization code)
    visualize_trajectory(x_all, u_all)

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
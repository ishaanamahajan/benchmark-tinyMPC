# from sam and khai
import math
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy as sqrt
from autograd.numpy.linalg import norm
from autograd.numpy.linalg import inv
from autograd import jacobian
from autograd.test_util import check_grads
np.set_printoptions(precision=4, suppress=True)
import scipy.sparse as sp
import osqp
import matplotlib.pyplot as plt
from admm import ADMM
import time

def generate_figure8_trajectory(t):
    """Generate figure-8 reference with smooth start"""
    # Figure 8 parameters
    A = 0.5  # amplitude
    w = 2*np.pi/7  # frequency
    
    # Smooth start factor (ramps up in first second)
    smooth_start = min(t/1.0, 1.0)
    
    x_ref = np.zeros(12)
    
    # Positions with smooth start
    x_ref[0] = A * np.sin(w*t) * smooth_start
    x_ref[2] = A * np.sin(2*w*t)/2 * smooth_start
    
    # Velocities (derivatives with smooth start)
    x_ref[6] = A * w * np.cos(w*t) * smooth_start
    x_ref[8] = A * w * np.cos(2*w*t) * smooth_start
    
    return x_ref
#Quaternion stuff, check `Planning with Attitude` paper for more details
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
mass = 0.035  # mass
J = np.array([[1.66e-5, 0.83e-6, 0.72e-6], [0.83e-6, 1.66e-5, 1.8e-6], [0.72e-6, 1.8e-6, 2.93e-5]])  # inertia
g = 9.81  # gravity
thrustToTorque = 0.0008  # thrust to torque ratio
el = 0.046/1.414213562  # arm length
scale = 65535  # PWM scale
kt = 2.245365e-6*scale # thrust coefficient
km = kt*thrustToTorque # moment coefficient

freq = 50.0 # Control frequency
h = 1/freq #50 Hz

Nx1 = 13        # number of states (quaternion)
Nx = 12         # number of states (linearized)
Nu = 4          # number of controls (motor pwm signals, 0-1)

# Quadrotor dynamics
def quad_dynamics(x, u):
    r = x[0:3]  # position
    q = x[3:7]/norm(x[3:7])  # normalize quaternion
    v = x[7:10]  # linear velocity
    omg = x[10:13]  # angular velocity
    Q = qtoQ(q)  # quaternion to rotation matrix

    dr = v
    dq = 0.5*L(q)@H@omg
    dv = np.array([0, 0, -g]) + (1/mass)*Q@np.array([[0, 0, 0, 0], [0, 0, 0, 0], [kt, kt, kt, kt]])@u
    domg = inv(J)@(-hat(omg)@J@omg + np.array([[-el*kt, -el*kt, el*kt, el*kt], [-el*kt, el*kt, el*kt, -el*kt], [-km, km, -km, km]])@u)

    return np.hstack([dr, dq, dv, domg])

# RK4 integration
def quad_dynamics_rk4(x, u):
    f1 = quad_dynamics(x, u)
    f2 = quad_dynamics(x + 0.5*h*f1, u)
    f3 = quad_dynamics(x + 0.5*h*f2, u)
    f4 = quad_dynamics(x + h*f3, u)
    xn = x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
    xnormalized = xn[3:7]/norm(xn[3:7])  # normalize quaternion
    return np.hstack([xn[0:3], xnormalized, xn[7:13]])

def delta_x_quat(x_curr, t):
    """Compute error between current state and reference"""
    x_ref = generate_figure8_trajectory(t)
    
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

# Linearize the dynamics around xg, uhover
A_jac = jacobian(quad_dynamics_rk4, 0)  # jacobian wrt x
B_jac = jacobian(quad_dynamics_rk4, 1)  # jacobian wrt u

# Initial hovering state and control input
rg = np.array([0.0, 0, 1.0])  # Starting at height=1
qg = np.array([1.0, 0, 0, 0])
vg = np.zeros(3)
omgg = np.zeros(3)
xg = np.hstack([rg, qg, vg, omgg])
uhover = (mass*g/kt/4)*np.ones(4)  # hover thrust

# Check gradients
check_grads(quad_dynamics_rk4, modes=['rev'], order=2)(xg, uhover)

# Get linearized system matrices
Anp1 = A_jac(xg, uhover)
Bnp1 = B_jac(xg, uhover)

# Apply attitude transformation
Anp = E(qg).T @ Anp1 @ E(qg)
Bnp = E(qg).T @ Bnp1

# Initial state with some perturbation
x0 = np.copy(xg)
x0[0:3] = rg + np.array([0.2, 0.2, 0.0])  # perturbed position
x0[3:7] = rptoq(np.array([0.1, 0.1, 0.1]))  # perturbed attitude

# Cost matrices using Bryson's rule
max_dev_x = np.array([
    0.01, 0.01, 0.01,    # position (tighter)
    0.5, 0.5, 0.05,      # attitude
    0.5, 0.5, 0.5,       # velocity
    0.7, 0.7, 0.5        # angular velocity
])
max_dev_u = np.array([0.1, 0.1, 0.1, 0.1])  # tighter control bounds
Q = np.diag(1./max_dev_x**2)
R = np.diag(1./max_dev_u**2)


# MPC parameters
n = Nx  # number of states
m = Nu  # number of inputs
N = 25  # horizon length

# Control constraints
u_max = np.array([1.0 - uhover[0]] * m)
u_min = np.array([-uhover[0]] * m)

class MPCController:
    def __init__(self, Anp, Bnp, Q, R, N, u_min, u_max, x_ref, u_ref, solver="osqp"):
        self.Anp = Anp
        self.Bnp = Bnp
        self.Q = Q
        self.R = R
        self.N = N
        self.u_min = u_min
        self.u_max = u_max
        self.x_ref = x_ref
        self.u_ref = u_ref

        self.start_time = time.time()
        self.x_prev = np.zeros((Nx, N))
        self.u_prev = np.zeros((Nu, N-1))

        # Dimensions
        self.n = Anp.shape[0]
        self.m = Bnp.shape[1]
        self.num_x = self.n * (N + 1)
        self.num_u = self.m * N
        self.num_var = self.num_x + self.num_u

        # Cost matrices
        Q_bar = sp.block_diag([Q] * N + [Q])
        R_bar = sp.block_diag([R] * N)
        self.P = sp.block_diag([Q_bar, R_bar])
        self.q = np.zeros(self.num_var)

        # Build constraints
        A_eq = self._build_equality_constraints()
        A_u = sp.eye(self.num_u, self.num_var, k=self.num_x, format='csc')
        self.A = sp.vstack([A_eq, A_u], format='csc')

        # Constraint bounds
        self.b_eq = np.zeros(self.n * (N + 1))
        self.l = np.hstack([self.b_eq, np.tile(u_min, N)])
        self.u = np.hstack([self.b_eq, np.tile(u_max, N)])
        self.eq_indices = np.arange(self.n * (N + 1))

        # Setup solver
        self.solver = solver
        if solver == "osqp":
            self.prob = osqp.OSQP()
            self.prob.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, verbose=False)
        elif solver == "admm":
            self.prob = ADMM(Q=self.P.toarray(), q=self.q, 
                           A=self.A.toarray(), l=self.l, u=self.u, verbose=False)


    def _build_equality_constraints(self):
        A_eq_data = []
        A_eq_row = []
        A_eq_col = []
        
        def idx_x(k): return self.n * k
        def idx_u(k): return self.num_x + self.m * k
        
        for k in range(self.N):
            xk_start = idx_x(k)
            xk1_start = idx_x(k + 1)
            uk_start = idx_u(k)
            
            for i in range(self.n):
                A_eq_data.append(1.0)
                A_eq_row.append((k + 1) * self.n + i)
                A_eq_col.append(xk1_start + i)
                
                for j in range(self.n):
                    val = -self.Anp[i, j]
                    if val != 0:
                        A_eq_data.append(val)
                        A_eq_row.append((k + 1) * self.n + i)
                        A_eq_col.append(xk_start + j)
                
                for j in range(self.m):
                    val = -self.Bnp[i, j]
                    if val != 0:
                        A_eq_data.append(val)
                        A_eq_row.append((k + 1) * self.n + i)
                        A_eq_col.append(uk_start + j)
        
        for i in range(self.n):
            A_eq_data.append(1.0)
            A_eq_row.append(i)
            A_eq_col.append(i)
            
        return sp.csc_matrix((A_eq_data, (A_eq_row, A_eq_col)), 
                           shape=(self.n * (self.N + 1), self.num_var))

    def control(self, x_curr):
        """Modified control method to handle trajectory tracking"""
        t = time.time() - self.start_time
        
        # Generate reference trajectory for horizon
        x_ref = np.zeros((Nx, self.N))
        u_ref = np.zeros((Nu, self.N-1))
        
        # Reference quaternion (hover)
        q_ref = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Fill reference trajectory
        for i in range(self.N):
            ref = generate_figure8_trajectory(t + i*h)
            x_ref[0:3,i] = ref[0:3]      # position
            x_ref[3:6,i] = np.zeros(3)   # attitude (hover)
            x_ref[6:9,i] = ref[3:6]      # velocity
            x_ref[9:12,i] = ref[6:9]     # angular velocity
        
        u_ref[:] = self.u_ref.reshape(-1,1)
        
        # Compute state error
        delta_x = delta_x_quat(x_curr, t)
        
        # Initialize optimization variables
        x_init = np.copy(self.x_prev)
        x_init[:,0] = delta_x
        u_init = np.copy(self.u_prev)
        
        # Update reference trajectory
        self.x_ref = x_ref
        
        # Update constraints for current state
        self.b_eq[:self.n] = delta_x  # Use delta_x directly
        self.l[self.eq_indices] = self.b_eq
        self.u[self.eq_indices] = self.b_eq
        
        # Solve optimization problem
        if self.solver == "osqp":
            self.prob.update(l=self.l, u=self.u)
            res = self.prob.solve()
            if res.info.status_val != osqp.constant('OSQP_SOLVED'):
                return self.u_ref
            z_opt = res.x
        elif self.solver == "admm":
            self.prob.update(l=self.l, u=self.u)
            self.prob.solve()
            z_opt = self.prob.x

        # Extract and return control
        u_opt_dev = z_opt[self.num_x:self.num_x + self.m]
        return self.u_ref + u_opt_dev

# Simulation and visualization
def simulate_with_controller(x0, controller, NSIM=200):
    x_all = []
    u_all = []
    ref_all = []
    x_curr = np.copy(x0)
    controller.start_time = time.time()
    
    for i in range(NSIM):
        t = i/freq
        ref = generate_figure8_trajectory(t)
        ref_all.append(ref)
        
        u_curr = controller.control(x_curr)
        u_curr_clipped = np.clip(u_curr, 0, 1)
        x_curr = quad_dynamics_rk4(x_curr, u_curr_clipped)
        
        x_all.append(x_curr)
        u_all.append(u_curr_clipped)
    
    return np.array(x_all), np.array(u_all), np.array(ref_all)

# Initialize controller
#mpc = MPCController(Anp, Bnp, Q, R, N, u_min, u_max, xg, uhover, solver="admm")

mpc = MPCController(
    Anp, Bnp, Q, R, N, u_min, u_max,
    x_ref=xg,   # Reference state
    u_ref=uhover,  # Reference input
    solver="admm"
)

# Run simulation
import time
x_all, u_all, ref_all = simulate_with_controller(x0, mpc)

# Visualization
plt.figure(figsize=(15, 10))

# 3D trajectory
ax = plt.subplot(221, projection='3d')
ax.plot3D(x_all[:, 0], x_all[:, 1], x_all[:, 2], 'b-', label='Actual')
ax.plot3D(ref_all[:, 0], ref_all[:, 1], ref_all[:, 2], 'r--', label='Reference')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Position tracking
ax = plt.subplot(222)
t = np.arange(len(x_all))/freq
ax.plot(t, x_all[:, 0], 'b-', label='x')
ax.plot(t, x_all[:, 1], 'g-', label='y')
ax.plot(t, x_all[:, 2], 'r-', label='z')
ax.plot(t, ref_all[:, 0], 'b--')
ax.plot(t, ref_all[:, 1], 'g--')
ax.plot(t, ref_all[:, 2], 'r--')
ax.legend()
ax.set_xlabel('Time [s]')
ax.set_ylabel('Position [m]')

# Attitude
ax = plt.subplot(223)
ax.plot(t, x_all[:, 3:7])
ax.legend(['q0', 'q1', 'q2', 'q3'])
ax.set_xlabel('Time [s]')
ax.set_ylabel('Quaternion')

# Control inputs
ax = plt.subplot(224)
ax.plot(t, u_all)
ax.legend(['u1', 'u2', 'u3', 'u4'])
ax.set_xlabel('Time [s]')
ax.set_ylabel('Motor PWM')

plt.tight_layout()
plt.show()
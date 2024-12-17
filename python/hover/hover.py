# from sam and khai
import math
# %matplotlib inline
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
#from reluqp import ReLU_QP

# Note: autograd does not work with np.block

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
# thrustToTorque = 0.005964552
thrustToTorque = 0.0008  # thrust to torque ratio
el = 0.046/1.414213562  # arm length
scale = 65535  # PWM scale
kt = 2.245365e-6*scale # thrust coefficient, u is PWM in range [0...1], 0 is no thrust, 1 is max thrust
km = kt*thrustToTorque # moment coefficient

freq = 50.0 # >>>>>>>> CONTROL FREQUENCY <<<<<<<<<<
h = 1/freq #50 Hz

Nx1 = 13        # number of states (quaternion)
Nx = 12         # number of states (linearized): x, y, z, Rodriguez 3-parameters (p, q, r), vx, vy, vz, wx, wy, wz
Nu = 4          # number of controls (motor pwm signals, 0-1)

# Quadrotor dynamics -- single rigid body dynamics
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

# RK4 integration with zero-order hold on u
def quad_dynamics_rk4(x, u):
    f1 = quad_dynamics(x, u)
    f2 = quad_dynamics(x + 0.5*h*f1, u)
    f3 = quad_dynamics(x + 0.5*h*f2, u)
    f4 = quad_dynamics(x + h*f3, u)
    xn = x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
    xnormalized = xn[3:7]/norm(xn[3:7])  # normalize quaternion
    return np.hstack([xn[0:3], xnormalized, xn[7:13]])
# function to visualize the trajectory
def visualize_trajectory(x_all, u_all):
    # Set up the figure and axis for plotting
    fig, ax = plt.subplots(3, 1)

    # Plot the trajectory
    x_all = np.array(x_all)
    nsteps = len(x_all)
    steps = np.arange(nsteps)
    ax[0].plot(steps, x_all[:, 0], label="x", linewidth=2)
    ax[0].plot(steps, x_all[:, 1], label="y", linewidth=2)
    ax[0].plot(steps, x_all[:, 2], label="z", linewidth=2)
    ax[0].legend()
    ax[0].title.set_text("Position")

    ax[1].plot(steps, x_all[:, 3], label="q0", linewidth=2)
    ax[1].plot(steps, x_all[:, 4], label="q1", linewidth=2)
    ax[1].plot(steps, x_all[:, 5], label="q2", linewidth=2)
    ax[1].plot(steps, x_all[:, 6], label="q3", linewidth=2)
    ax[1].legend()
    ax[1].title.set_text("Attitude")

    u_all = np.array(u_all)
    nsteps = len(u_all)
    steps = np.arange(nsteps)
    ax[2].plot(steps, u_all[:, 0], label="u1", linewidth=2)
    ax[2].plot(steps, u_all[:, 1], label="u2", linewidth=2)
    ax[2].plot(steps, u_all[:, 2], label="u3", linewidth=2)
    ax[2].plot(steps, u_all[:, 3], label="u4", linewidth=2)
    # ax[2].legend()
    ax[2].title.set_text("Controls")
    plt.show()

# function to simulate with a controller
def simulate_with_controller(x0, x_nom, u_nom, controller, NSIM = 100):
    x_all = []
    u_all = []
    x_curr = np.copy(x0)
    # simulate the dynamics with the LQR controller
    for i in range(NSIM):
        u_curr = controller(x_curr, x_nom, u_nom)
        u_curr_clipped = np.clip(u_curr, 0, 1)
        x_curr = quad_dynamics_rk4(x_curr, u_curr_clipped)
        x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
        u_curr = u_curr.reshape(u_curr.shape[0]).tolist()
        x_all.append(x_curr)
        u_all.append(u_curr)
    return x_all, u_all

# Linearize the dynamics around xg, uhover
A_jac = jacobian(quad_dynamics_rk4, 0)  # jacobian wrt x
B_jac = jacobian(quad_dynamics_rk4, 1)  # jacobian wrt u

# Hovering state and control input
rg = np.array([0.0, 0, 0.0])
qg = np.array([1.0, 0, 0, 0])
vg = np.zeros(3)
omgg = np.zeros(3)
xg = np.hstack([rg, qg, vg, omgg])
uhover = (mass*g/kt/4)*np.ones(4)  # ~each motor thrust to compensate for gravity
print("Hovering Initial State and Control")
print(xg, uhover)

check_grads(quad_dynamics_rk4, modes=['rev'], order=2)(xg, uhover)

Anp1 = A_jac(xg, uhover)  # jacobian of the dynamics wrt x at xg, uhover
Bnp1 = B_jac(xg, uhover)  # jacobian of the dynamics wrt u at xg, uhover

# `Planning with Attitude` trick, attitude Jacobians
#  https://rexlab.ri.cmu.edu/papers/planning_with_attitude.pdf
Anp = E(qg).T @ Anp1 @ E(qg)
Bnp = E(qg).T @ Bnp1
# print("A = \n", Anp)
# print("B = \n", Bnp)

# x0[0:3] += rg + 3*np.random.randn(3)/3  # disturbed initial posRition
x0 = np.copy(xg)
x0[0:3] += rg + np.array([0.2, 0.2, -0.2])  # disturbed initial position
x0[3:7] = rptoq(np.array([1.0, 0.0, 0.0]))  # disturbed initial attitude
print("Perturbed Intitial State")
print(x0)

# Choose Q and R matrices based on Bryson's rule
max_dev_x = np.array([0.1, 0.1, 0.1,  0.5, 0.5, 0.05,  0.5, 0.5, 0.5,  0.7, 0.7, 0.2])
max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])/6
Q = np.diag(1./max_dev_x**2)
R = np.diag(1./max_dev_u**2)



# Number of states and inputs
n = Nx  # number of states == Nx
m = Nu   # number of inputs == Nu
N = 30  # horizon length

# Control input constraints
u_max = np.array([1.0 - uhover[0]] * m)
u_min = np.array([-uhover[0]] * m)

print("u_max = ", u_max)
print("u_min = ", u_min)


# Function to compute delta_x with quaternion handling
def delta_x_quat(x_curr, x_ref):
    q_curr = x_curr[3:7]
    q_ref = x_ref[3:7]
    # Compute the quaternion difference
    q_diff = L(q_ref).T @ q_curr
    # Convert quaternion difference to Rodrigues parameters
    phi = qtorp(q_diff)
    delta_x = np.hstack([
        x_curr[0:3] - x_ref[0:3],  # Position difference
        phi,                        # Attitude difference in Rodrigues parameters
        x_curr[7:10] - x_ref[7:10], # Linear velocity difference
        x_curr[10:13] - x_ref[10:13]  # Angular velocity difference
    ])
    return delta_x

# Updated MPCController class
class MPCController:
    def __init__(self, Anp, Bnp, Q, R, N, u_min, u_max, x_ref, u_ref, solver="osqp"):
        self.Anp = Anp
        self.Bnp = Bnp
        self.Q = Q
        self.R = R
        self.N = N
        self.u_min = u_min
        self.u_max = u_max
        self.x_ref = x_ref  # Reference state (goal state)
        self.u_ref = u_ref  # Reference input (hover input)

        # Number of states and inputs
        self.n = Anp.shape[0]
        self.m = Bnp.shape[1]

        # Total number of variables
        self.num_x = self.n * (N + 1)
        self.num_u = self.m * N
        self.num_var = self.num_x + self.num_u

        # Objective function matrices
        Q_bar = sp.block_diag([Q] * N + [Q])  # Include terminal cost
        R_bar = sp.block_diag([R] * N)
        self.P = sp.block_diag([Q_bar, R_bar])

        # Precompute q (linear term in the objective)
        # We will update q at each time step based on the reference trajectory
        self.q = np.zeros(self.num_var)

        # Equality constraints (dynamics)
        A_eq_data = []
        A_eq_row = []
        A_eq_col = []

        # Functions to get indices in z
        def idx_x(k):
            return self.n * k

        def idx_u(k):
            return self.num_x + self.m * k

        # Build the equality constraints
        for k in range(N):
            # Indices for variables
            xk_start = idx_x(k)
            xk1_start = idx_x(k + 1)
            uk_start = idx_u(k)

            # x_{k+1} - A x_k - B u_k = 0

            # x_{k+1} variables
            for i in range(self.n):
                A_eq_data.append(1.0)
                A_eq_row.append((k + 1) * self.n + i)
                A_eq_col.append(xk1_start + i)

            # x_k variables
            for i in range(self.n):
                for j in range(self.n):
                    val = -Anp[i, j]
                    if val != 0:
                        A_eq_data.append(val)
                        A_eq_row.append((k + 1) * self.n + i)
                        A_eq_col.append(xk_start + j)

            # u_k variables
            for i in range(self.n):
                for j in range(self.m):
                    val = -Bnp[i, j]
                    if val != 0:
                        A_eq_data.append(val)
                        A_eq_row.append((k + 1) * self.n + i)
                        A_eq_col.append(uk_start + j)

        # Initial state constraint x0 (we will update this at each time step)
        for i in range(self.n):
            A_eq_data.append(1.0)
            A_eq_row.append(i)
            A_eq_col.append(i)
        # Build the sparse matrix for equality constraints
        A_eq = sp.csc_matrix((A_eq_data, (A_eq_row, A_eq_col)), shape=(self.n * (N + 1), self.num_var))

        # Inequality constraints (input constraints)
        A_u = sp.eye(self.num_u, self.num_var, k=self.num_x, format='csc')

        # Combine constraints
        A = sp.vstack([A_eq, A_u], format='csc')

        # Build initial l and u
        self.b_eq = np.zeros(self.n * (N + 1))
        self.l = np.hstack([self.b_eq, np.tile(u_min, N)])
        self.u = np.hstack([self.b_eq, np.tile(u_max, N)])

        # Save the indices where b_eq is located in l and u (since we will update it)
        self.eq_indices = np.arange(self.n * (N + 1))

        self.solver = solver
        if self.solver == "osqp":
            # Set up the OSQP problem
            self.prob = osqp.OSQP()
            self.prob.setup(P=self.P, q=self.q, A=A, l=self.l, u=self.u, verbose=False)
        elif self.solver == "admm":
            # Convert matrices to numpy arrays
            self.P = self.P.toarray()
            self.A = A.toarray()
            self.prob = ADMM(Q=self.P, q=self.q, A=self.A, l=self.l, u=self.u, verbose=False)
            
        elif self.solver == "reluqp":
            self.P = self.P.toarray()
            self.A = A.toarray()
            self.prob = ReLU_QP()
            self.prob.setup(H=self.P, g=self.q, A=self.A, l=self.l, u=self.u, verbose=False)
        else:
            raise ValueError("Unknown solver")

    def control(self, x_curr):
        # Update the initial state constraint
        # Convert x_curr to the linearized state vector
        phi0 = qtorp(L(self.x_ref[3:7]).T @ x_curr[3:7])  # Difference in Rodrigues parameters
        x_init = np.hstack([
            x_curr[0:3] - self.x_ref[0:3],
            phi0,
            x_curr[7:10] - self.x_ref[7:10],
            x_curr[10:13] - self.x_ref[10:13]
        ])  # Linearized state deviation vector

        # Update b_eq[:n] = x_init
        self.b_eq[:self.n] = x_init

        # Update q based on the reference trajectory
        # Since our reference trajectory is constant (hovering), the linear term remains zero

        # Update l and u
        self.l[self.eq_indices] = self.b_eq
        self.u[self.eq_indices] = self.b_eq
        # Update the problem data
        self.prob.update(l=self.l, u=self.u)
        # Solve the QP
        if self.solver == "osqp":            
            res = self.prob.solve()

            # Check if the problem was solved successfully
            if res.info.status_val != osqp.constant('OSQP_SOLVED'):
                print("OSQP did not solve the problem!")
                return self.u_ref  # Return reference input if not solved
            # Extract the optimal control input deviation
            z_opt = res.x
        elif self.solver == "admm":
            self.prob.solve()
            z_opt = self.prob.x
        elif self.solver == "reluqp":
            res = self.prob.solve()
            z_opt = res.x
            pass

        
        u_opt_dev = z_opt[self.num_x:self.num_x + self.m]

        # Compute the actual control input
        u_opt = self.u_ref + u_opt_dev
        return u_opt

# Instantiate the MPC controller
mpc = MPCController(
    Anp, Bnp, Q, R, N, u_min, u_max,
    x_ref=xg,   # Reference state (hovering state)
    u_ref=uhover,  # Reference input (hovering input)
    solver="admm" # this is super slow
    # solver="osqp"
    #solver="reluqp"
)

# Modified simulate_with_controller function
def simulate_with_controller(x0, controller, NSIM=100):
    x_all = []
    u_all = []
    x_curr = np.copy(x0)
    # simulate the dynamics with the controller
    for i in range(NSIM):
        u_curr = controller(x_curr)
        u_curr_clipped = np.clip(u_curr, 0, 1)
        x_curr = quad_dynamics_rk4(x_curr, u_curr_clipped)
        x_all.append(x_curr)
        u_all.append(u_curr_clipped)
    return x_all, u_all

# Simulate the system
x0_sim = np.copy(x0)
NSIM = 100  # Number of simulation steps

x_all, u_all = simulate_with_controller(x0_sim, mpc.control, NSIM=NSIM)

# Visualization function
def visualize_trajectory(x_all, u_all):
    # Set up the figure and axis for plotting
    fig, ax = plt.subplots(3, 1, figsize=(12, 10))

    # Plot the trajectory
    x_all = np.array(x_all)
    nsteps = len(x_all)
    steps = np.arange(nsteps)
    ax[0].plot(steps, x_all[:, 0], label="x", linewidth=2)
    ax[0].plot(steps, x_all[:, 1], label="y", linewidth=2)
    ax[0].plot(steps, x_all[:, 2], label="z", linewidth=2)
    ax[0].legend()
    ax[0].set_title("Position")

    ax[1].plot(steps, x_all[:, 3], label="q0", linewidth=2)
    ax[1].plot(steps, x_all[:, 4], label="q1", linewidth=2)
    ax[1].plot(steps, x_all[:, 5], label="q2", linewidth=2)
    ax[1].plot(steps, x_all[:, 6], label="q3", linewidth=2)
    ax[1].legend()
    ax[1].set_title("Attitude (Quaternion)")

    u_all = np.array(u_all)
    nsteps = len(u_all)
    steps = np.arange(nsteps)
    ax[2].plot(steps, u_all[:, 0], label="u1", linewidth=2)
    ax[2].plot(steps, u_all[:, 1], label="u2", linewidth=2)
    ax[2].plot(steps, u_all[:, 2], label="u3", linewidth=2)
    ax[2].plot(steps, u_all[:, 3], label="u4", linewidth=2)
    ax[2].legend()
    ax[2].set_title("Controls")
    plt.tight_layout()
    plt.show()

# Visualize the trajectory
visualize_trajectory(x_all, u_all)
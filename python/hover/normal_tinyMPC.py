
# tinyMPC_hover.py
import math
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy as sqrt
from autograd.numpy.linalg import norm
from autograd.numpy.linalg import inv
from autograd import jacobian
from autograd.test_util import check_grads
np.set_printoptions(precision=4, suppress=True)

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

class TinyMPC:
    def __init__(self, input_data, Nsteps, mode = 0):
        self.cache = {}
        self.cache['rho'] = input_data['rho']  # Fixed rho
        self.cache['A'] = input_data['A']
        self.cache['B'] = input_data['B']
        self.cache['Q'] = input_data['Q']
        self.cache['R'] = input_data['R']
        self.compute_cache_terms()
        self.set_tols_iters()
        self.x_prev = np.zeros((self.cache['A'].shape[0],Nsteps))
        self.u_prev = np.zeros((self.cache['B'].shape[1],Nsteps))
        self.N = Nsteps

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

        if (x_ref is None):
            x_ref = np.zeros(x.shape)
        if (u_ref is None):
            u_ref = np.zeros(u.shape)

        for k in range(self.max_iter):
            self.update_primal(x, u, d, p, q, r)
            self.update_slack(z, v, y, g, u, x, self.umax, self.umin, self.xmax, self.xmin)
            self.update_dual(y, g, u, x, z, v)
            self.update_linear_cost(r, q, p, z, v, y, g, u_ref, x_ref)

            pri_res_input = np.max(np.abs(u - z))
            pri_res_state = np.max(np.abs(x - v))
            dua_res_input = np.max(np.abs(self.cache['rho'] * (z_prev - z)))
            dua_res_state = np.max(np.abs(self.cache['rho'] * (v_prev - v)))

            z_prev = np.copy(z)
            v_prev = np.copy(v)

            if (pri_res_input < self.abs_pri_tol and dua_res_input < self.abs_dua_tol and
                pri_res_state < self.abs_pri_tol and dua_res_state < self.abs_dua_tol):
                status = 1
                break

        self.x_prev = x
        self.u_prev = u
        return x, u, status, k

    


def delta_x_quat(x_curr):
    q = x_curr[3:7]
    phi = qtorp(L(qg).T @ q)
    delta_x = np.hstack([x_curr[0:3]-rg, phi, x_curr[7:10]-vg, x_curr[10:13]-omgg])
    return delta_x

def tinympc_controller(x_curr, x_nom, u_nom):
    delta_x = delta_x_quat(x_curr)
    noise = np.random.normal(0, 0.01, (Nx,))*0
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

if __name__ == "__main__":
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
    rho = 80.0
    input_data = {
        'rho': rho,
        'A': Anp,
        'B': Bnp,
        'Q': P_lqr,
        'R': R
    }




    tinympc = TinyMPC(input_data, N)

    
    
    
    u_max = [1.0-uhover[0]] * Nu
    u_min = [-1*uhover[0]] * Nu
    x_max = [1000.] * Nx
    x_min = [-1000.0] * Nx
    tinympc.set_bounds(u_max, u_min, x_max, x_min)

    # Set nominal trajectory
    from scipy.spatial.transform import Rotation as spRot
    R0 = spRot.from_quat(qg)
    eulerg = R0.as_euler('zxy')
    xg_euler = np.hstack((eulerg,xg[4:]))
    x_nom_tinyMPC = np.tile(0*xg_euler,(N,1)).T
    u_nom_tinyMPC = np.tile(uhover,(N-1,1)).T

    # Run simulation
    def simulate_with_controller(x0, x_nom, u_nom, controller, NSIM = 100):
        x_all = []
        u_all = []
        x_curr = np.copy(x0)
        iterations = []
        
        for i in range(NSIM):
            u_curr, k = controller(x_curr, x_nom, u_nom)
            #u_curr_clipped = np.clip(u_curr, 0, 1)
            x_curr = quad_dynamics_rk4(x_curr, u_curr)
            x_curr = x_curr.reshape(x_curr.shape[0]).tolist()
            u_curr = u_curr.reshape(u_curr.shape[0]).tolist()
            x_all.append(x_curr)
            u_all.append(u_curr)
            iterations.append(k)
        return x_all, u_all, iterations

    x_all, u_all, iterations = simulate_with_controller(x0, x_nom_tinyMPC, u_nom_tinyMPC, tinympc_controller)

    np.savetxt('data/iterations/normal_hover.txt', iterations)
    

    # Visualize trajectory
    visualize_trajectory(x_all, u_all)

    

    # Plot iterations
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, label='Fixed rho')
    plt.xlabel('Time Step')
    print("Total iterations:", sum(iterations))
    plt.ylabel('Iterations')
    plt.title('ADMM Iterations per Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()


    # rho_start = 100
    # rho_end = 70
    # rho_step = -5
    # rho_values = np.arange(rho_start, rho_end-1, rho_step)
    # iterations_per_rho = []
    
    # # Run simulation for each rho value
    # for rho in rho_values:
    #     print(f"\nTesting rho = {rho}")
        
    #     # Reset TinyMPC with new rho
    #     input_data = {
    #         'rho': rho,
    #         'A': Anp,
    #         'B': Bnp,
    #         'Q': P_lqr,
    #         'R': R
    #     }
    #     tinympc = TinyMPC(input_data, N)

    #     u_max = [1.0-uhover[0]] * Nu
    #     u_min = [-1*uhover[0]] * Nu
    #     x_max = [1000.] * Nx
    #     x_min = [-1000.0] * Nx
    #     tinympc.set_bounds(u_max, u_min, x_max, x_min)

    #     # Initial state
    #     x0 = np.copy(xg)
    #     x0[0:3] += rg + np.array([0.2, 0.2, -0.2])
    #     x0[3:7] = rptoq(np.array([1.0, 0.0, 0.0]))

       
        
    #     # Run simulation
    #     x_all, u_all, iterations = simulate_with_controller(x0, x_nom_tinyMPC, u_nom_tinyMPC, tinympc_controller)
    #     total_iterations = sum(iterations)
    #     iterations_per_rho.append(total_iterations)
    #     print(f"Total iterations: {total_iterations}")
    
    # # Plot results
    # plt.figure(figsize=(10, 5))
    # plt.plot(rho_values, iterations_per_rho, 'b.-')
    # plt.xlabel('Rho Value')
    # plt.ylabel('Total Iterations')
    # plt.title('Effect of Rho on ADMM Iterations')
    # plt.grid(True)
    # plt.show()









def generate_mpc_data():
    # Match your dimensions
    nx = 12  # states
    nu = 4   # inputs
    Nh = 10  # horizon
    Nsim = 200
    
    # Generate a stable random system (similar to your original generator)
    while True:
        A = np.random.uniform(low=-1, high=1, size=(nx, nx))
        U, S, Vh = np.linalg.svd(A)
        S = S / np.max(S)  # Scale eigenvalues to be inside unit circle
        A = U @ np.diag(S) @ Vh
        
        B = np.random.uniform(low=-1, high=1, size=(nx, nu))
        
        # Check controllability
        C = np.zeros((nx, nx*nu))
        Ak = np.eye(nx)
        for k in range(nx):
            C[:, k*nu:(k+1)*nu] = Ak @ B
            Ak = Ak @ A
            
        if np.linalg.matrix_rank(C) == nx:
            break
    
    # Cost matrices (matching TinyMPC hover)
    max_dev_x = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.05, 0.5, 0.5, 0.5, 0.7, 0.7, 0.2])
    max_dev_u = np.array([0.5, 0.5, 0.5, 0.5])/6
    Q = np.diag(1./max_dev_x**2)
    R = np.diag(1./max_dev_u**2)

    # Compute LQR cost (matching TinyMPC)
    def dlqr(A, B, Q, R, n_steps=500):
        P = Q
        for i in range(n_steps):
            K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
            P = Q + A.T @ P @ (A - B @ K)
        return K, P

    K_lqr, P_lqr = dlqr(A, B, Q, R)
    
    # Quadrotor parameters (matching TinyMPC)
    mass = 0.035  # kg
    g = 9.81      # m/s^2
    scale = 65535
    kt = 2.245365e-6 * scale
    hover_thrust = (mass * g / kt / 4)
    
    # Constraints (matching TinyMPC)
    umax = (1.0 - hover_thrust) * np.ones((nu, 1))
    umin = (-hover_thrust) * np.ones((nu, 1))
    xmax = 1000 * np.ones((nx, 1))
    xmin = -1000 * np.ones((nx, 1))
    
    # Cache computation (matching TinyMPC exactly)
    rho = 80.0  # Same value as TinyMPC
    Q_rho = P_lqr + rho * np.eye(nx)  # Use P_lqr instead of Q
    R_rho = R + rho * np.eye(nu)
    
    # Compute Kinf, Pinf (matching TinyMPC implementation)
    Kinf = np.zeros(B.T.shape)
    Pinf = np.copy(Q_rho)
    
    for k in range(5000):
        Kinf_prev = np.copy(Kinf)
        Kinf = np.linalg.inv(R_rho + B.T @ Pinf @ B) @ B.T @ Pinf @ A
        Pinf = Q_rho + A.T @ Pinf @ (A - B @ Kinf)
        
        if np.linalg.norm(Kinf - Kinf_prev, 2) < 1e-10:
            break
    
    AmBKt = (A - B @ Kinf).T
    Quu_inv = np.linalg.inv(R_rho + B.T @ Pinf @ B)
    
    # Export everything
    export_data_to_cpp(xbar, A, B, P_lqr, Pinf, R, Kinf, Quu_inv, AmBKt, 
                      umin, umax, xmin, xmax, rho, nx, nu)
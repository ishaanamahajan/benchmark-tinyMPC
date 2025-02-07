import numpy as np
from scipy.linalg import block_diag

def construct_A_stacked(A, B, N=5, nx=12, nu=4):
    total_rows = (nu * (N-1)) + (nx * (N-1))  # 64 rows
    total_cols = (nx * N) + (nu * (N-1))      # 76 cols
    
    A_stacked = np.zeros((total_rows, total_cols))
    
    # Input constraints (first nu*(N-1) rows)
    for i in range(N-1):
        row_start = i * nu
        col_start = nx + i * (nx + nu)  # Skip first nx cols, then skip block size
        A_stacked[row_start:row_start+nu, col_start:col_start+nu] = np.eye(nu)
    
    # Dynamics constraints (last nx*(N-1) rows)
    for i in range(N-1):
        row_start = nu*(N-1) + i*nx  # Start after input constraints
        col_start = i*(nx + nu)
        
        # Add A, B, -I matrices
        A_stacked[row_start:row_start+nx, col_start:col_start+nx] = A
        A_stacked[row_start:row_start+nx, col_start+nx:col_start+nx+nu] = B
        A_stacked[row_start:row_start+nx, col_start+nx+nu:col_start+2*nx+nu] = -np.eye(nx)
    
    return A_stacked

def construct_P(Q, R, N=5, nx=12, nu=4):
    # Build block diagonal P matrix
    P_blocks = []
    for i in range(N):
        if i < N-1:
            P_blocks.append(block_diag(Q, R))
        else:
            P_blocks.append(Q)
    return block_diag(*P_blocks)

def construct_q(Q, R, x_prev, u_prev, xg, uhover, N=5):
    q_blocks = []
    for i in range(N):
        # For hover, reference is just xg
        delta_x = x_prev[:, i] - xg[:12]
        q_x = Q @ delta_x.reshape(-1, 1)
        if i < N-1:
            # For hover, reference input is uhover
            delta_u = u_prev[:, i] - uhover
            q_u = R @ delta_u.reshape(-1, 1)
            q_blocks.extend([q_x, q_u])
        else:
            q_blocks.append(q_x)
    return np.vstack(q_blocks)

def print_matrix_cpp(matrix, name):
    """Generate C++ code for a matrix"""
    if matrix.ndim == 2:
        print(f"const float {name}[{matrix.shape[0]}][{matrix.shape[1]}] = {{0}};  // Initialize to zero")
        print("{")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if abs(matrix[i,j]) > 1e-10:  # Only print non-zero elements
                    print(f"    {name}[{i}][{j}] = {matrix[i,j]}f;")
        print("}")
    else:  # Vector
        print(f"const float {name}[{matrix.shape[0]}] = {{")
        for i in range(matrix.shape[0]):
            if abs(matrix[i]) > 1e-10:  # Only print non-zero elements
                print(f"    {matrix[i]}f,")
        print("};")

def main():
    # System dimensions
    nx, nu = 12, 4
    N = 5
    
    # Constants from your output
    xg = np.array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    uhover = np.array([0.58333335, 0.58333335, 0.58333335, 0.58333335])
    
    # Load Adyn matrix (12x12)
    Adyn = np.array([
        [-0.013073761504966662, 0.22461100944713758, 0.2544408073658377, 0.1323348019582432, -0.05119190975443125, -0.2853635217317808, 0.005852846280465838, 0.07011807025248573, -0.2968322415804105, -0.013342503116046556, 0., 0.],
        [-0.06137248955268807, 0.2556656045343489, -0.03063256502887507, 0.11552939765513578, -0.058969127243344646, -0.07512087891577696, -0.2702501391833329, 0.127746986825234, 0.2908888109041632, -0.020948991811369395, 0., 0.],
        [-0.0037584741172171717, -0.16552794408938437, -0.08875096809930084, 0.042372984402228635, 0.21786326884394774, -0.22975411646799596, -0.07774672994090405, -0.11375421443400495, -0.058180667232261285, -0.4098828152231077, 0., 0.],
        [0.12242331773993388, 0.12471677936291188, -0.13911303499908662, -0.26582147700536507, -0.06658710699507481, 0.1492285342092377, 0.2972460153275449, 0.0391296849734597, -0.2510614748851446, -0.005822319071535572, 0., 0.],
        [-0.143324536463163, -0.20137617437023497, -0.003491287843558525, -0.40319248933888935, 0.14221711686704866, 0.014805096771462429, -0.2455110123152518, 0.20304991085071805, -0.07610204893227299, 0.116184055528124, 0., 0.],
        [0.3210840769778985, 0.03371740843854953, -0.07351063257254649, 0.12192570383078735, -0.09990897508376877, 0.16135562872560028, -0.09661345679939942, -0.3049664444342707, -0.2697298445157618, -0.08068945894468639, 0., 0.],
        [-0.2453297827460556, 0.0781710615539388, 0.21751900328258827, 0.16456053729640305, 0.039794460418853844, -0.341101291599811, 0.06377132349933874, -0.4229502226809967, 0.1311149585415748, -0.2636762989898564, 0., 0.],
        [0.1423211580260777, -0.23414683773964046, -0.0841343426124434, -0.3162995888947139, 0.05837127123614148, -0.0007534977634424823, 0.2791294857680328, 0.04530726363395446, 0.16481169971975138, 0.12548148413744514, 0., 0.],
        [0.1957264057858685, 0.027686871374062134, 0.043621258418221606, -0.08965321413278575, 0.3543881498570847, -0.10454970854675709, -0.1424335459450487, -0.21015125400397391, -0.17906097779100996, -0.13920921968532168, 0., 0.],
        [0.17549826346412045, 0.035767072993639695, 0.13962148612526337, -0.10341080386062167, 0.34497227607248887, -0.07652298009340157, -0.1685188116906241, -0.08983973705154376, 0.24099638700900639, -0.012089801465588858, 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ])
    
    # Load Bdyn matrix (12x4)
    Bdyn = np.array([
        [0.5847337443977123, 0.9156896162787989, -0.6632152796182489, -0.7287620922223179],
        [-0.06009044137962216, 0.7220150513443002, 0.5341385478526204, 0.1163562164376648],
        [-0.6864334687845264, -0.2174748617885558, 0.4386915533094997, -0.25317463011527597],
        [-0.921761299446676, 0.16776792214640257, 0.4402691593858987, 0.42954298328491336],
        [-0.6702683229305435, -0.3280168211921326, -0.18165541540616048, -0.1590371745457484],
        [-0.7716830209906447, -0.22893693859729036, 0.013264373087658488, 0.42112134423274306],
        [0.13889537768820004, -0.19013887920220185, 0.8551947255090679, 0.19616713326347623],
        [0.949581767410826, 0.7347527127002798, 0.34725220325022876, 0.7986269908494112],
        [0.9824793562366523, -0.5582454660717111, 0.38211303980760825, 0.8375585557182579],
        [-0.9651996799670686, 0.5989779930651686, -0.8211941002179848, 0.8331085948904269],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]
    ])
    
    # Load Q matrix (12x12)
    Q_diag = np.array([6.964691855978616, 2.8613933495037944, 2.268514535642031, 5.513147690828912,
                       7.1946896978556305, 4.23106460124461, 9.807641983846155, 6.848297385848633,
                       4.809319014843609, 3.9211751819415053, 0., 0.])
    Q = np.diag(Q_diag)
    
    # Load R matrix (4x4)
    R = np.diag([0.1, 0.1, 0.1, 0.1])
    
    # Sample state and input trajectories for hover
    x_prev = np.zeros((nx, N))  # Starting from zero state
    u_prev = np.zeros((nu, N-1))  # Starting from zero input
    
    # Generate MPC matrices
    A_stacked = construct_A_stacked(Adyn, Bdyn, N)
    P = construct_P(Q, R, N)
    q = construct_q(Q, R, x_prev, u_prev, xg, uhover, N)
    
    # Print C++ code
    print("// Generated MPC matrices")
    print_matrix_cpp(A_stacked, "A_stacked")
    print_matrix_cpp(P, "P")
    print_matrix_cpp(q, "q")

if __name__ == "__main__":
    main()
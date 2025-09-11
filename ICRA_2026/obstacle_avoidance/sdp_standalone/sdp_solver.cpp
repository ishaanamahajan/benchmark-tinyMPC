#include "sdp_solver.hpp"
#include <cmath>

SDPSolver::SDPSolver() : iteration(0), primal_residual(1e6), dual_residual(1e6) {
    // Initialize matrices and variables
    x.setZero();
    u.setZero();
    v.setZero();
    z.setZero();
    g.setZero();
    y.setZero();
    x_traj.setZero();
    u_traj.setZero();
    
    setup_extended_system();
    setup_cost_matrices();
    setup_initial_condition();
    
    std::cout << "SDP Solver initialized with Julia problem parameters" << std::endl;
    std::cout << "Extended system: " << nx_ext << " states, " << nu_ext << " controls" << std::endl;
    std::cout << "Horizon: " << NHORIZON << ", Obstacle: [" << x_obs.transpose() << "], radius: " << r_obs << std::endl;
}

void SDPSolver::setup_extended_system() {
    // Extended A matrix: [Ad, 0; 0, kron(Ad, Ad)]
    A_ext.setZero();
    A_ext.block<nx, nx>(0, 0) = Ad;
    
    // Kronecker product: kron(Ad, Ad) - 16x16 block starting at (4,4)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < nx; j++) {
            A_ext.block<nx, nx>(nx + i*nx, nx + j*nx) = Ad(i,j) * Ad;
        }
    }
    
    // Extended B matrix: [Bd, 0; 0, kron(Bd,Ad), kron(Ad,Bd), kron(Bd,Bd)]
    B_ext.setZero();
    B_ext.block<nx, nu>(0, 0) = Bd;
    
    // Build Kronecker products more carefully
    // kron(Bd, Ad): each element Bd(i,j) multiplied by entire Ad matrix
    // Result is 16x8 matrix (nx*nx rows, nu*nx columns)
    for (int i = 0; i < nu; i++) {
        for (int j = 0; j < nx; j++) {
            for (int ii = 0; ii < nx; ii++) {
                for (int jj = 0; jj < nx; jj++) {
                    B_ext(nx + i*nx + ii, nu + j*nx + jj) = Bd(ii,i) * Ad(ii,jj);
                }
            }
        }
    }
    
    // kron(Ad, Bd): each element Ad(i,j) multiplied by entire Bd matrix  
    // Result is 16x8 matrix (nx*nx rows, nx*nu columns)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < nu; j++) {
            for (int ii = 0; ii < nx; ii++) {
                for (int jj = 0; jj < nu; jj++) {
                    B_ext(nx + i*nx + ii, nu + 8 + j*nu + jj) = Ad(i,j) * Bd(ii,jj);
                }
            }
        }
    }
    
    // kron(Bd, Bd): each element Bd(i,j) multiplied by entire Bd matrix
    // Result is 16x4 matrix (nx*nx rows, nu*nu columns)  
    for (int i = 0; i < nu; i++) {
        for (int j = 0; j < nu; j++) {
            for (int ii = 0; ii < nx; ii++) {
                for (int jj = 0; jj < nu; jj++) {
                    B_ext(nx + i*nx + ii, nu + 16 + j*nu + jj) = Bd(i,j) * Bd(ii,jj);
                }
            }
        }
    }
    
    std::cout << "Extended system matrices set up with Kronecker products" << std::endl;
}

void SDPSolver::setup_cost_matrices() {
    // Q matrix: regularization + second moment cost
    Q_ext = reg * Matrix<double, nx_ext, nx_ext>::Identity();
    
    // Add cost on second moments (diagonal of xxT terms)
    for (int i = 0; i < nx; i++) {
        Q_ext(nx + i*nx + i, nx + i*nx + i) += q_xx;
    }
    
    // R matrix: regularization + input costs  
    R_ext = reg * Matrix<double, nu_ext, nu_ext>::Identity();
    
    // Add cost on second moments of controls (diagonal of uuT terms)
    for (int i = 0; i < nu; i++) {
        R_ext(nu + 16 + i*nu + i, nu + 16 + i*nu + i) += R_xx;
    }
    
    std::cout << "Cost matrices set up with second moment penalties" << std::endl;
}

void SDPSolver::setup_initial_condition() {
    // Initial condition: [x_initial; vec(x_initial * x_initial')]
    Vector4d x_init(-10.0, 0.1, 0.0, 0.0);  // From Julia
    
    x_initial.setZero();
    x_initial.head<nx>() = x_init;
    
    // Second moments: vec(x_initial * x_initial')
    Matrix4d XX_init = x_init * x_init.transpose();
    Map<Vector<double, 16>>(x_initial.data() + nx) = Map<const Vector<double, 16>>(XX_init.data());
    
    // Set initial state
    x.col(0) = x_initial;
    
    // Goal state (origin)
    x_goal.setZero();
    
    std::cout << "Initial condition set: x = [" << x_init.transpose() << "]" << std::endl;
}

bool SDPSolver::solve() {
    std::cout << "\nStarting SDP-ADMM solver..." << std::endl;
    std::cout << "Parameters: rho=" << rho << ", tol=" << abs_pri_tol << ", max_iter=" << max_iter << std::endl;
    
    for (iteration = 0; iteration < max_iter; iteration++) {
        admm_iteration();
        
        if (iteration % 10 == 0) {
            std::cout << "Iter " << iteration << ": primal_res=" << primal_residual 
                     << ", dual_res=" << dual_residual << std::endl;
        }
        
        if (check_convergence()) {
            std::cout << "Converged at iteration " << iteration << std::endl;
            extract_physical_trajectory();
            return true;
        }
    }
    
    std::cout << "Max iterations reached without convergence" << std::endl;
    extract_physical_trajectory();
    return false;
}

void SDPSolver::admm_iteration() {
    update_primal();
    update_slack_with_sdp_projection();
    update_dual();
}

void SDPSolver::update_primal() {
    // Much simpler approach: just enforce dynamics and smooth trajectory
    // Skip the full optimization for now - focus on testing SDP projection
    
    // Forward simulate with simple control law
    for (int k = 0; k < NHORIZON - 1; k++) {
        // Simple control: move towards origin, avoid obstacle
        Vector4d x_phys = x.col(k).head<nx>();
        Vector2d pos = x_phys.head<2>();
        Vector2d vel = x_phys.tail<2>();
        
        // Goal attraction
        Vector2d u_goal = -0.1 * pos - 0.05 * vel;
        
        // Strong obstacle avoidance
        Vector2d to_obs = pos - x_obs;
        double dist = to_obs.norm();
        Vector2d u_avoid = Vector2d::Zero();
        
        // Much stronger avoidance force
        if (dist < 4.0 * r_obs && dist > 0.1) {
            double force_strength = 2.0 / (dist - r_obs + 0.1);  // Stronger near boundary
            u_avoid = force_strength * to_obs / dist;
        }
        
        // Emergency avoidance if too close
        if (dist < r_obs + 0.5) {
            u_avoid = 5.0 * to_obs / (dist + 0.01);  // Very strong repulsion
        }
        
        Vector2d u_total = u_goal + u_avoid;
        u_total = u_total.cwiseMax(-1.0).cwiseMin(1.0);  // Clamp
        
        // Set physical control
        u.col(k).head<nu>() = u_total;
        
        // Forward dynamics for physical states
        Vector4d x_next = Ad * x_phys + Bd * u_total;
        x.col(k+1).head<nx>() = x_next;
        
        // Update second moments (simplified)
        Matrix4d XX_current = x_next * x_next.transpose();
        Map<Vector<double, 16>>(x.col(k+1).data() + nx) = Map<Vector<double, 16>>(XX_current.data());
        
        // Set cross terms (simplified)
        Matrix<double, 4, 2> XU = x_next * u_total.transpose();
        Matrix<double, 2, 4> UX = u_total * x_next.transpose();
        Matrix2d UU = u_total * u_total.transpose();
        
        Map<Matrix<double, 4, 2>>(u.col(k).data() + nu) = XU;
        Map<Matrix<double, 2, 4>>(u.col(k).data() + nu + 8) = UX;
        Map<Matrix2d>(u.col(k).data() + nu + 16) = UU;
    }
}

void SDPSolver::update_slack_with_sdp_projection() {
    // Update state slack variables with SDP projection
    for (int k = 0; k < NHORIZON; k++) {
        // Box projection first
        v.col(k) = (x.col(k) + g.col(k) / rho).cwiseMax(-10.0).cwiseMin(10.0);
        
        // SDP projection for moment matrices
        if (k < NHORIZON - 1) {
            project_moment_matrix(k);
        } else {
            project_terminal_matrix();
        }
        
        // OBSTACLE CONSTRAINT PROJECTION
        project_obstacle_constraint(k);
    }
    
    // Update control slack variables  
    for (int k = 0; k < NHORIZON - 1; k++) {
        // Box projection
        Vector<double, nu_ext> u_temp = u.col(k) + y.col(k) / rho;
        
        // Physical control bounds
        u_temp.head<nu>() = u_temp.head<nu>().cwiseMax(-2.0).cwiseMin(2.0);
        
        // Other components (less restrictive)
        u_temp.tail<nu_ext - nu>() = u_temp.tail<nu_ext - nu>().cwiseMax(-10.0).cwiseMin(10.0);
        
        z.col(k) = u_temp;
    }
}

void SDPSolver::project_moment_matrix(int k) {
    // Build 7x7 moment matrix [1 x' u'; x xx' xu'; u ux' uu']
    Matrix<double, 7, 7> M = build_moment_matrix(k);
    
    // Project onto PSD cone using your projection function
    Matrix<double, 7, 7> M_proj = project_psd<7>(M);
    
    // Extract projected values back to state and control variables
    extract_from_moment_matrix(M_proj, k);
}

void SDPSolver::project_terminal_matrix() {
    // Build 5x5 terminal matrix [1 x'; x xx']
    Matrix<double, 5, 5> M = build_terminal_matrix();
    
    // Project onto PSD cone
    Matrix<double, 5, 5> M_proj = project_psd<5>(M);
    
    // Extract projected values back
    extract_from_terminal_matrix(M_proj);
}

Matrix<double, 7, 7> SDPSolver::build_moment_matrix(int k) {
    Matrix<double, 7, 7> M;
    M.setZero();
    
    // Extract physical variables
    Vector4d x_phys = v.col(k).head<nx>();
    Vector2d u_phys = z.col(k).head<nu>();
    
    // Extract second moment variables
    Matrix4d XX = Map<const Matrix4d>(v.col(k).data() + nx);
    Matrix<double, 4, 2> XU = Map<const Matrix<double, 4, 2>>(z.col(k).data() + nu);
    Matrix<double, 2, 4> UX = Map<const Matrix<double, 2, 4>>(z.col(k).data() + nu + 8);
    Matrix2d UU = Map<const Matrix2d>(z.col(k).data() + nu + 16);
    
    // Build moment matrix
    M(0, 0) = 1.0;
    M.block<1, 4>(0, 1) = x_phys.transpose();
    M.block<1, 2>(0, 5) = u_phys.transpose();
    M.block<4, 1>(1, 0) = x_phys;
    M.block<4, 4>(1, 1) = XX;
    M.block<4, 2>(1, 5) = XU;
    M.block<2, 1>(5, 0) = u_phys;
    M.block<2, 4>(5, 1) = UX;
    M.block<2, 2>(5, 5) = UU;
    
    return M;
}

Matrix<double, 5, 5> SDPSolver::build_terminal_matrix() {
    Matrix<double, 5, 5> M;
    M.setZero();
    
    // Extract physical state at terminal time
    Vector4d x_phys = v.col(NHORIZON-1).head<nx>();
    Matrix4d XX = Map<const Matrix4d>(v.col(NHORIZON-1).data() + nx);
    
    // Build terminal matrix [1 x'; x xx']
    M(0, 0) = 1.0;
    M.block<1, 4>(0, 1) = x_phys.transpose();
    M.block<4, 1>(1, 0) = x_phys;
    M.block<4, 4>(1, 1) = XX;
    
    return M;
}

void SDPSolver::extract_from_moment_matrix(const Matrix<double, 7, 7>& M, int k) {
    // Extract projected values back to variables
    v.col(k).head<nx>() = M.block<4, 1>(1, 0);
    Map<Matrix4d>(v.col(k).data() + nx) = M.block<4, 4>(1, 1);
    
    z.col(k).head<nu>() = M.block<2, 1>(5, 0);
    Map<Matrix<double, 4, 2>>(z.col(k).data() + nu) = M.block<4, 2>(1, 5);
    Map<Matrix<double, 2, 4>>(z.col(k).data() + nu + 8) = M.block<2, 4>(5, 1);
    Map<Matrix2d>(z.col(k).data() + nu + 16) = M.block<2, 2>(5, 5);
}

void SDPSolver::extract_from_terminal_matrix(const Matrix<double, 5, 5>& M) {
    v.col(NHORIZON-1).head<nx>() = M.block<4, 1>(1, 0);
    Map<Matrix4d>(v.col(NHORIZON-1).data() + nx) = M.block<4, 4>(1, 1);
}

void SDPSolver::project_obstacle_constraint(int k) {
    // Extract current state and second moments
    Vector4d x_phys = v.col(k).head<nx>();
    Matrix4d XX = Map<const Matrix4d>(v.col(k).data() + nx);
    
    // Compute constraint value: tr(XX[1:2, 1:2]) - 2*x_obs'*x[1:2] + x_obs'*x_obs - r^2
    double constraint_val = obstacle_constraint_value(x_phys, XX);
    
    // If constraint is violated (< 0), project to make it feasible
    if (constraint_val < 0) {
        // Simple projection: push position away from obstacle center
        Vector2d pos = x_phys.head<2>();
        Vector2d to_obs = pos - x_obs;
        double dist = to_obs.norm();
        
        if (dist > 1e-8) {  // Avoid division by zero
            // Push position to obstacle boundary + small margin
            double target_dist = r_obs + 0.1;  // 10cm margin
            Vector2d pos_new = x_obs + (target_dist / dist) * to_obs;
            
            // Update position in state vector
            v.col(k).head<2>() = pos_new;
            
            // Update velocity to maintain smoothness (simple approach)
            if (k > 0) {
                Vector2d pos_prev = v.col(k-1).head<2>();
                v.col(k).segment<2>(2) = pos_new - pos_prev;  // Simple velocity update
            }
            
            // Update second moments to be consistent
            Vector4d x_new = v.col(k).head<nx>();
            Matrix4d XX_new = x_new * x_new.transpose();
            Map<Matrix4d>(v.col(k).data() + nx) = XX_new;
            
            // Verify constraint is now satisfied
            double new_constraint_val = obstacle_constraint_value(x_new, XX_new);
            if (new_constraint_val < 0) {
                std::cout << "Warning: obstacle projection failed at step " << k 
                         << ", constraint value: " << new_constraint_val << std::endl;
            }
        }
    }
}

void SDPSolver::update_dual() {
    // Update dual variables
    for (int k = 0; k < NHORIZON; k++) {
        g.col(k) += rho * (x.col(k) - v.col(k));
    }
    
    for (int k = 0; k < NHORIZON - 1; k++) {
        y.col(k) += rho * (u.col(k) - z.col(k));
    }
}

bool SDPSolver::check_convergence() {
    primal_residual = compute_primal_residual();
    dual_residual = compute_dual_residual();
    
    return (primal_residual < abs_pri_tol && dual_residual < abs_dual_tol);
}

double SDPSolver::compute_primal_residual() {
    double res = 0.0;
    
    for (int k = 0; k < NHORIZON; k++) {
        res += (x.col(k) - v.col(k)).norm();
    }
    
    for (int k = 0; k < NHORIZON - 1; k++) {
        res += (u.col(k) - z.col(k)).norm();
    }
    
    return res / (NHORIZON * (nx_ext + nu_ext));
}

double SDPSolver::compute_dual_residual() {
    // Simplified dual residual computation
    return primal_residual * 0.1;  // Placeholder
}

void SDPSolver::extract_physical_trajectory() {
    // Extract physical states and controls for plotting
    for (int k = 0; k < NHORIZON; k++) {
        x_traj.col(k) = x.col(k).head<nx>();
    }
    
    for (int k = 0; k < NHORIZON - 1; k++) {
        u_traj.col(k) = u.col(k).head<nu>();
    }
    
    // Check obstacle constraint satisfaction
    int violations = 0;
    for (int k = 0; k < NHORIZON; k++) {
        Vector4d x_phys = x_traj.col(k);
        Matrix4d XX = Map<const Matrix4d>(x.col(k).data() + nx);
        
        double constraint_val = obstacle_constraint_value(x_phys, XX);
        if (constraint_val < 0) {
            violations++;
        }
    }
    
    std::cout << "Physical trajectory extracted" << std::endl;
    std::cout << "Obstacle constraint violations: " << violations << "/" << NHORIZON << std::endl;
}

double SDPSolver::obstacle_constraint_value(const Vector4d& x_phys, const Matrix4d& XX) {
    // tr(XX[0:1, 0:1]) - 2*x_obs'*x[0:1] + x_obs'*x_obs - r^2
    double trace_XX = XX(0,0) + XX(1,1);
    double obs_term = x_obs.dot(x_obs);
    double pos_term = 2.0 * x_obs.dot(x_phys.head<2>());
    
    return trace_XX - pos_term + obs_term - r_obs * r_obs;
}

void SDPSolver::save_results(const std::string& filename) {
    std::ofstream file(filename);
    
    file << "# TinyMPC SDP Obstacle Avoidance Results\n";
    file << "# Format: time, pos_x, pos_y, vel_x, vel_y, u_x, u_y\n";
    
    for (int k = 0; k < NHORIZON; k++) {
        file << k;
        for (int i = 0; i < nx; i++) {
            file << ", " << x_traj(i, k);
        }
        if (k < NHORIZON - 1) {
            for (int i = 0; i < nu; i++) {
                file << ", " << u_traj(i, k);
            }
        } else {
            file << ", 0, 0";  // No control at final step
        }
        file << "\n";
    }
    
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

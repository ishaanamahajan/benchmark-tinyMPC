#include <iostream>
#include <Eigen/Eigenvalues>

#include "admm.hpp"
#include "rho_benchmark.hpp"    

#define DEBUG_MODULE "TINYALG"

extern "C" {

/**
    * Update linear terms from Riccati backward pass
    */
void backward_pass_grad(TinySolver *solver)
{
    for (int i = solver->work->N - 2; i >= 0; i--)
    {
        (solver->work->d.col(i)).noalias() = solver->cache->Quu_inv * (solver->work->Bdyn.transpose() * solver->work->p.col(i + 1) + solver->work->r.col(i));
        (solver->work->p.col(i)).noalias() = solver->work->q.col(i) + solver->cache->AmBKt.lazyProduct(solver->work->p.col(i + 1)) - (solver->cache->Kinf.transpose()).lazyProduct(solver->work->r.col(i)); 
    }
}

/**
    * Use LQR feedback policy to roll out trajectory
    */
void forward_pass(TinySolver *solver)
{
    for (int i = 0; i < solver->work->N - 1; i++)
    {
        (solver->work->u.col(i)).noalias() = -solver->cache->Kinf.lazyProduct(solver->work->x.col(i)) - solver->work->d.col(i);
        // solver->work->u.col(i) << .001, .02, .3, 4;
        // DEBUG_PRINT("u(0): %f\n", solver->work->u.col(0)(0));
        // multAdyn(solver->Ax->cache.Adyn, solver->work->x.col(i));
        (solver->work->x.col(i + 1)).noalias() = solver->work->Adyn.lazyProduct(solver->work->x.col(i)) + solver->work->Bdyn.lazyProduct(solver->work->u.col(i));
    }
}

/**
    * Project slack (auxiliary) variables into their feasible domain, defined by
    * projection functions related to each constraint
    * TODO: pass in meta information with each constraint assigning it to a
    * projection function
    */
void update_slack(TinySolver *solver)
{
    solver->work->znew = solver->work->u + solver->work->y;
    solver->work->vnew = solver->work->x + solver->work->g;

    // Box constraints on input
    if (solver->settings->en_input_bound)
    {
        solver->work->znew = solver->work->u_max.cwiseMin(solver->work->u_min.cwiseMax(solver->work->znew));
    }

    // Box constraints on state
    if (solver->settings->en_state_bound)
    {
        solver->work->vnew = solver->work->x_max.cwiseMin(solver->work->x_min.cwiseMax(solver->work->vnew));
    }

    // SDP CONSTRAINTS - PROJECT MOMENT MATRICES
    project_sdp_constraints(solver);
}

/**
    * Update next iteration of dual variables by performing the augmented
    * lagrangian multiplier update
    */
void update_dual(TinySolver *solver)
{
    solver->work->y = solver->work->y + solver->work->u - solver->work->znew;
    solver->work->g = solver->work->g + solver->work->x - solver->work->vnew;
}

/**
    * Update linear control cost terms in the Riccati feedback using the changing
    * slack and dual variables from ADMM
    */
void update_linear_cost(TinySolver *solver)
{
    solver->work->r = -(solver->work->Uref.array().colwise() * solver->work->R.array()); // Uref = 0 so commented out for speed up. Need to uncomment if using Uref
    (solver->work->r).noalias() -= solver->cache->rho * (solver->work->znew - solver->work->y);
    solver->work->q = -(solver->work->Xref.array().colwise() * solver->work->Q.array());
    (solver->work->q).noalias() -= solver->cache->rho * (solver->work->vnew - solver->work->g);
    solver->work->p.col(solver->work->N - 1) = -(solver->work->Xref.col(solver->work->N - 1).transpose().lazyProduct(solver->cache->Pinf));
    (solver->work->p.col(solver->work->N - 1)).noalias() -= solver->cache->rho * (solver->work->vnew.col(solver->work->N - 1) - solver->work->g.col(solver->work->N - 1));
}

/**
    * Check for termination condition by evaluating whether the largest absolute
    * primal and dual residuals for states and inputs are below threhold.
    */
bool termination_condition(TinySolver *solver)
{
    if (solver->work->iter % solver->settings->check_termination == 0)
    {
        solver->work->primal_residual_state = (solver->work->x - solver->work->vnew).cwiseAbs().maxCoeff();
        solver->work->dual_residual_state = ((solver->work->v - solver->work->vnew).cwiseAbs().maxCoeff()) * solver->cache->rho;
        solver->work->primal_residual_input = (solver->work->u - solver->work->znew).cwiseAbs().maxCoeff();
        solver->work->dual_residual_input = ((solver->work->z - solver->work->znew).cwiseAbs().maxCoeff()) * solver->cache->rho;

        if (solver->work->primal_residual_state < solver->settings->abs_pri_tol &&
            solver->work->primal_residual_input < solver->settings->abs_pri_tol &&
            solver->work->dual_residual_state < solver->settings->abs_dua_tol &&
            solver->work->dual_residual_input < solver->settings->abs_dua_tol)
        {
            return true;                 
        }
    }
    return false;
}

int solve(TinySolver *solver)
{
    // Initialize variables
    solver->solution->solved = 0;
    solver->solution->iter = 0;
    solver->work->status = 11; // TINY_UNSOLVED
    solver->work->iter = 0;

    // Setup for adaptive rho
    RhoAdapter adapter;
    adapter.rho_min = solver->settings->adaptive_rho_min;
    adapter.rho_max = solver->settings->adaptive_rho_max;
    adapter.clip = solver->settings->adaptive_rho_enable_clipping;
    
    RhoBenchmarkResult rho_result;

    // Store previous values for residuals
    tinyMatrix v_prev = solver->work->vnew;
    tinyMatrix z_prev = solver->work->znew;

    for (int i = 0; i < solver->settings->max_iter; i++)
    {
        // Solve linear system with Riccati and roll out to get new trajectory
        forward_pass(solver);

        // Project slack variables into feasible domain
        update_slack(solver);

        // Compute next iteration of dual variables
        update_dual(solver);

        // Update linear control cost terms using reference trajectory, duals, and slack variables
        update_linear_cost(solver);

        solver->work->iter += 1;

        

        if (solver->settings->adaptive_rho) {

            // Calculate residuals for adaptive rho
            tinytype pri_res_input = (solver->work->u - solver->work->znew).cwiseAbs().maxCoeff();
            tinytype pri_res_state = (solver->work->x - solver->work->vnew).cwiseAbs().maxCoeff();
            tinytype dua_res_input = solver->cache->rho * (solver->work->znew - z_prev).cwiseAbs().maxCoeff();
            tinytype dua_res_state = solver->cache->rho * (solver->work->vnew - v_prev).cwiseAbs().maxCoeff();

            // Update rho every 5 iterations
            if (i> 0 && i % 5 == 0) {
                benchmark_rho_adaptation(
                    &adapter,
                    solver->work->x,
                    solver->work->u,
                    solver->work->vnew,
                    solver->work->znew,
                    solver->work->g,
                    solver->work->y,
                    solver->cache,
                    solver->work,
                    solver->work->N,
                    &rho_result
                );
                
                // Update matrices using Taylor expansion
                update_matrices_with_derivatives(solver->cache, rho_result.final_rho);
            }
        }
            
        // Store previous values for next iteration
        z_prev = solver->work->znew;
        v_prev = solver->work->vnew;

        // Check for whether cost is minimized by calculating residuals
        if (termination_condition(solver)) {
            solver->work->status = 1; // TINY_SOLVED

            // Save solution
            solver->solution->iter = solver->work->iter;
            solver->solution->solved = 1;
            solver->solution->x = solver->work->vnew;
            solver->solution->u = solver->work->znew;

            std::cout << "Solver converged in " << solver->work->iter << " iterations" << std::endl;

            return 0;
        }

        // Save previous slack variables
        solver->work->v = solver->work->vnew;
        solver->work->z = solver->work->znew;

        backward_pass_grad(solver);
    }
    
    solver->solution->iter = solver->work->iter;
    solver->solution->solved = 0;
    solver->solution->x = solver->work->vnew;
    solver->solution->u = solver->work->znew;
    return 1;
}

/**
 * Project SDP constraints for obstacle avoidance
 * Uses your custom project_psd<M>() function integrated into TinyMPC ADMM
 */
void project_sdp_constraints(TinySolver *solver) {
    // SDP problem dimensions (from Julia formulation)
    const int nx_phys = 4;  // Physical state dimension
    const int nu_phys = 2;  // Physical control dimension
    
    // Obstacle parameters (from Julia)
    const tinytype x_obs_x = -5.0;
    const tinytype x_obs_y = 0.0;
    const tinytype r_obs = 2.0;
    
    // For each time step, project the moment matrices
    for (int k = 0; k < solver->work->N; k++) {
        
        if (k < solver->work->N - 1) {
            // PROJECT 7x7 MOMENT MATRIX [1 x' u'; x XX XU; u UX UU]
            
            // Extract physical variables
            Vector4d x_phys = solver->work->vnew.col(k).head<nx_phys>();
            Vector2d u_phys = solver->work->znew.col(k).head<nu_phys>();
            
            // Build moment matrix components (simplified for now)
            Matrix4d XX = x_phys * x_phys.transpose();
            Matrix<double, 4, 2> XU = x_phys * u_phys.transpose();
            Matrix<double, 2, 4> UX = u_phys * x_phys.transpose();
            Matrix2d UU = u_phys * u_phys.transpose();
            
            // Build 7x7 moment matrix
            Matrix<tinytype, 7, 7> M;
            M.setZero();
            M(0, 0) = 1.0;
            M.block<1, 4>(0, 1) = x_phys.transpose();
            M.block<1, 2>(0, 5) = u_phys.transpose();
            M.block<4, 1>(1, 0) = x_phys;
            M.block<4, 4>(1, 1) = XX;
            M.block<4, 2>(1, 5) = XU;
            M.block<2, 1>(5, 0) = u_phys;
            M.block<2, 4>(5, 1) = UX;
            M.block<2, 2>(5, 5) = UU;
            
            // PROJECT USING YOUR FUNCTION!
            Matrix<tinytype, 7, 7> M_proj = project_psd<7>(M, 1e-8);
            
            // Extract projected values back to TinyMPC variables
            solver->work->vnew.col(k).head<nx_phys>() = M_proj.block<4, 1>(1, 0);
            solver->work->znew.col(k).head<nu_phys>() = M_proj.block<2, 1>(5, 0);
            
            // TODO: Update extended state components (second moments) if using full 20-state formulation
            
        } else {
            // PROJECT 5x5 TERMINAL MATRIX [1 x'; x XX]
            
            Vector4d x_phys = solver->work->vnew.col(k).head<nx_phys>();
            Matrix4d XX = x_phys * x_phys.transpose();
            
            // Build 5x5 terminal matrix
            Matrix<tinytype, 5, 5> M;
            M.setZero();
            M(0, 0) = 1.0;
            M.block<1, 4>(0, 1) = x_phys.transpose();
            M.block<4, 1>(1, 0) = x_phys;
            M.block<4, 4>(1, 1) = XX;
            
            // PROJECT USING YOUR FUNCTION!
            Matrix<tinytype, 5, 5> M_proj = project_psd<5>(M, 1e-8);
            
            // Extract projected values back
            solver->work->vnew.col(k).head<nx_phys>() = M_proj.block<4, 1>(1, 0);
        }
        
        // OBSTACLE CONSTRAINT PROJECTION
        Vector4d x_phys = solver->work->vnew.col(k).head<nx_phys>();
        Matrix4d XX = x_phys * x_phys.transpose();
        
        // Compute constraint: tr(XX[1:2,1:2]) - 2*x_obs'*x[1:2] + x_obs'*x_obs - r^2
        tinytype trace_XX = XX(0,0) + XX(1,1);
        tinytype obs_term = x_obs_x * x_obs_x + x_obs_y * x_obs_y;
        tinytype pos_term = 2.0 * (x_obs_x * x_phys(0) + x_obs_y * x_phys(1));
        tinytype constraint_val = trace_XX - pos_term + obs_term - r_obs * r_obs;
        
        // If violated, project position away from obstacle
        if (constraint_val < 0) {
            Vector2d pos = x_phys.head<2>();
            Vector2d x_obs_vec(x_obs_x, x_obs_y);
            Vector2d to_obs = pos - x_obs_vec;
            tinytype dist = to_obs.norm();
            
            std::cout << "OBSTACLE VIOLATION at step " << k << ": pos=[" << pos.transpose() 
                     << "], dist=" << dist << ", constraint=" << constraint_val << std::endl;
            
            if (dist > 1e-6) {
                // Push to safe distance
                tinytype target_dist = r_obs + 0.3;  // Larger margin
                Vector2d pos_new = x_obs_vec + (target_dist / dist) * to_obs;
                solver->work->vnew.col(k).head<2>() = pos_new;
                
                std::cout << "   PROJECTED to: [" << pos_new.transpose() << "]" << std::endl;
            }
        }
    }
}

} /* extern "C" */

/**
 * PSD projection function - your exact implementation
 * Projects matrix onto positive semidefinite cone using eigendecomposition
 */
template<int M>
EIGEN_STRONG_INLINE Matrix<tinytype, M, M>
project_psd(const Matrix<tinytype, M, M>& S_in, tinytype eps)
{
    using MatM = Matrix<tinytype, M, M>;
    
    // 1) Make symmetric to avoid numerical asymmetry
    MatM S = tinytype(0.5) * (S_in + S_in.transpose());

    // 2) Eigendecomposition (self-adjoint is fastest & most stable)
    Eigen::SelfAdjointEigenSolver<MatM> es;
    es.compute(S, Eigen::ComputeEigenvectors);
    
    // 3) Clamp eigenvalues to be nonnegative (or eps floor)
    Matrix<tinytype, M, 1> d = es.eigenvalues();
    for (int i = 0; i < M; ++i) {
        d(i) = d(i) < eps ? eps : d(i);
    }

    // 4) Reconstruct: V * diag(d) * V^T
    const MatM& V = es.eigenvectors();
    return V * d.asDiagonal() * V.transpose();
}
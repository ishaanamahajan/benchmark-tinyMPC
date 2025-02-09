#include <iostream>

#include "admm.hpp"

#include "rho_benchmark.hpp"

#include "problem_data/rand_prob_tinympc_params.hpp"

#define DEBUG_MODULE "TINYALG"

extern "C" {

// #include "debug.h"

static uint32_t startTimestamp;

void solve_lqr(struct tiny_problem *problem, const struct tiny_params *params) {
    problem->u.col(0) = -params->cache.Kinf[problem->cache_level] * (problem->x.col(0) - params->Xref.col(0));
}


void solve_admm(struct tiny_problem *problem, struct tiny_params *params) {
    // Initialize dual variables to zero only on first solve
    if (problem->solve_count == 0) {
        problem->y.setZero();
        problem->g.setZero();
    }
    problem->solve_count++;
    
    //Serial.println("\n=== Starting solve_admm ===");
    
    // Print key parameters
    // Serial.print("Tolerances at solve - pri_tol: ");
    // Serial.print(params->abs_pri_tol, 8);
    // Serial.print(", dua_tol: ");
    // Serial.println(params->abs_dua_tol, 8);
    
    // // Print norms of key matrices
    // Serial.print("State norm: "); Serial.println(problem->x.norm());
    // Serial.print("Input norm: "); Serial.println(problem->u.norm());
    // Serial.print("Q norm: "); Serial.println(params->Q[0].norm());
    // Serial.print("Xref norm: "); Serial.println(params->Xref.norm());
    
    startTimestamp = micros();
    
    problem->status = 0;
    problem->iter = 0;
    problem->rho_time = 0;

    // Initial updates
    forward_pass(problem, params);
    update_slack(problem, params);
    update_dual(problem, params);
    update_linear_cost(problem, params);

    // Keep track of previous values for residuals
    tiny_MatrixNxNh v_prev = problem->v;
    tiny_MatrixNuNhm1 z_prev = problem->z;

    uint32_t admm_start = micros();
    
    // Debug print initial values
    // Serial.println("\nInitial values:");
    // Serial.print("max_iter: "); Serial.println(params->max_iter);
    // Serial.print("abs_pri_tol: "); Serial.println(params->abs_pri_tol);
    // Serial.print("abs_dua_tol: "); Serial.println(params->abs_dua_tol);
    
    // Calculate initial residuals
    problem->primal_residual_input = (problem->u - problem->znew).cwiseAbs().maxCoeff();
    problem->primal_residual_state = (problem->x - problem->vnew).cwiseAbs().maxCoeff();
    problem->dual_residual_input = (params->cache.rho[problem->cache_level] * (z_prev - problem->znew)).cwiseAbs().maxCoeff();
    problem->dual_residual_state = (params->cache.rho[problem->cache_level] * (v_prev - problem->vnew)).cwiseAbs().maxCoeff();

    // Serial.println("Initial residuals:");
    // Serial.print("pri_in: "); Serial.println(problem->primal_residual_input);
    // Serial.print("pri_st: "); Serial.println(problem->primal_residual_state);
    // Serial.print("dua_in: "); Serial.println(problem->dual_residual_input);
    // Serial.print("dua_st: "); Serial.println(problem->dual_residual_state);
    
    for (int i = 0; i < params->max_iter; i++) {
        update_primal(problem, params);
        update_slack(problem, params);
        update_dual(problem, params);
        update_linear_cost(problem, params);

        // Calculate and store residuals
        problem->primal_residual_input = (problem->u - problem->znew).cwiseAbs().maxCoeff();
        problem->primal_residual_state = (problem->x - problem->vnew).cwiseAbs().maxCoeff();
        problem->dual_residual_input = (params->cache.rho[problem->cache_level] * (z_prev - problem->znew)).cwiseAbs().maxCoeff();
        problem->dual_residual_state = (params->cache.rho[problem->cache_level] * (v_prev - problem->vnew)).cwiseAbs().maxCoeff();

        // Print every 50 iterations
        if (problem->iter % 50 == 0) {
            Serial.print("Iter "); Serial.print(problem->iter);
            Serial.print(" pri_in: "); Serial.print(problem->primal_residual_input, 6);
            Serial.print(" pri_st: "); Serial.print(problem->primal_residual_state, 6);
            Serial.print(" dua_in: "); Serial.print(problem->dual_residual_input, 6);
            Serial.print(" dua_st: "); Serial.println(problem->dual_residual_state, 6);
        }

        // Check convergence with more detailed output
        if (problem->primal_residual_input < params->abs_pri_tol && 
            problem->primal_residual_state < params->abs_pri_tol &&
            problem->dual_residual_input < params->abs_dua_tol && 
            problem->dual_residual_state < params->abs_dua_tol) {
            
            //Serial.println("Converged!");
            problem->status = 1;  // Converged
            break;
        }

        // Update previous values
        v_prev = problem->vnew;
        z_prev = problem->znew;
        
        problem->v = problem->vnew;
        problem->z = problem->znew;
        
        problem->iter += 1;
    }
    
    // Only adapt rho after ADMM convergence
    if (params->rho_adapter.analytical_method) {
        uint32_t rho_start = micros();
        float pri_res = std::max(problem->primal_residual_state, problem->primal_residual_input);
        float dual_res = std::max(problem->dual_residual_state, problem->dual_residual_input);

        RhoBenchmarkResult result;
        benchmark_rho_adaptation(
            problem->x.data(),
            problem->u.data(),
            problem->v.data(),
            pri_res,
            dual_res,
            &result,
            &params->rho_adapter
        );
        
        params->cache.rho[problem->cache_level] = result.final_rho;
        problem->rho_time = micros() - rho_start;
    }
    
    problem->admm_time = micros() - admm_start;
    problem->solve_time = micros() - startTimestamp;
}

/**
 * Do backward Riccati pass then forward roll out
*/
void update_primal(struct tiny_problem *problem, const struct tiny_params *params) {
    backward_pass_grad(problem, params);
    forward_pass(problem, params);
}

/**
 * Update linear terms from Riccati backward pass
*/
void backward_pass_grad(struct tiny_problem *problem, const struct tiny_params *params) {
    for (int i=NHORIZON-2; i>=0; i--) {
        // d[:, k] = C1 @ (B.T @ p[:, k+1] + r[:, k])
        auto temp = params->cache.Bdyn[problem->cache_level].transpose() * problem->p.col(i+1) + 
                   problem->r.col(i);
        problem->d.col(i).noalias() = params->cache.Quu_inv[problem->cache_level] * temp;
        
        // p[:, k] = q[:, k] + C2 @ p[:, k+1] - Kinf.T @ r[:, k]
        problem->p.col(i).noalias() = problem->q.col(i) + 
                                     params->cache.AmBKt[problem->cache_level] * problem->p.col(i+1) - 
                                     params->cache.Kinf[problem->cache_level].transpose() * problem->r.col(i);
    }
}

/**
 * Use LQR feedback policy to roll out trajectory
*/
void forward_pass(struct tiny_problem *problem, const struct tiny_params *params) {
    // Serial.println("\n=== Forward Pass ===");
    
    // // Debug K matrix first
    // Serial.println("K matrix (first row):");
    for(int j = 0; j < NSTATES; j++) {
        // Serial.print(params->cache.Kinf[problem->cache_level](0,j), 6);
        // Serial.print(" ");
    }
   
    
    for (int k = 0; k < NHORIZON-1; k++) {
        // 1. Print state and reference
        // Serial.print("k="); Serial.print(k);
        // Serial.print(" x:"); Serial.print(problem->x.col(k)(0), 6);
        // Serial.print(" xref:"); Serial.print(params->Xref.col(k)(0), 6);
        
        // 2. Compute and print control components
        auto state_error = problem->x.col(k) - params->Xref.col(k);
        auto feedback = -params->cache.Kinf[problem->cache_level] * state_error;
        auto feedforward = -problem->d.col(k);
        
        // Serial.print(" fb:"); Serial.print(feedback(0), 6);
        // Serial.print(" ff:"); Serial.print(feedforward(0), 6);
        
        // 3. Update control and state
        problem->u.col(k) = feedback + feedforward;
        problem->x.col(k+1) = params->cache.Adyn[problem->cache_level] * problem->x.col(k) + 
                             params->cache.Bdyn[problem->cache_level] * problem->u.col(k);
        
        // Serial.print(" u:"); Serial.print(problem->u.col(k)(0), 6);
        // Serial.print(" next_x:"); Serial.println(problem->x.col(k+1)(0), 6);
    }
}

/**
 * Project slack (auxiliary) variables into their feasible domain, defined by
 * projection functions related to each constraint
 * TODO: pass in meta information with each constraint assigning it to a
 * projection function
*/
void update_slack(struct tiny_problem *problem, const struct tiny_params *params) {
    //Serial.println("\n=== Slack Update ===");
    
    for (int k = 0; k < NHORIZON-1; k++) {
        // Input bounds - use vectors directly
        auto temp_z = problem->u.col(k) + problem->y.col(k);
        for (int i = 0; i < NINPUTS; i++) {
            problem->znew(i,k) = std::min(std::max(temp_z(i), params->u_min(i)), params->u_max(i));
        }
        
        // State bounds - use vectors directly
        auto temp_v = problem->x.col(k) + problem->g.col(k);
        for (int i = 0; i < NSTATES; i++) {
            problem->vnew(i,k) = std::min(std::max(temp_v(i), params->x_min(i)), params->x_max(i));
        }
        
        // Debug output for first iteration
        if (k == 0) {
            // Serial.println("Before update (k=0):");
            // Serial.print("u: "); Serial.print(problem->u.col(0).norm());
            // Serial.print(" y: "); Serial.print(problem->y.col(0).norm());
            // Serial.print(" x: "); Serial.print(problem->x.col(0).norm());
            // Serial.print(" g: "); Serial.println(problem->g.col(0).norm());
            
            // Serial.println("After update (k=0):");
            // Serial.print("temp_z: "); Serial.print(temp_z.norm());
            // Serial.print(" znew: "); Serial.print(problem->znew.col(0).norm());
            // Serial.print(" temp_v: "); Serial.print(temp_v.norm());
            // Serial.print(" vnew: "); Serial.println(problem->vnew.col(0).norm());
        }
    }
    
    // Final state constraint
    auto temp_v = problem->x.col(NHORIZON-1) + problem->g.col(NHORIZON-1);
    for (int i = 0; i < NSTATES; i++) {
        problem->vnew(i,NHORIZON-1) = std::min(std::max(temp_v(i), params->x_min(i)), params->x_max(i));
    }
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
*/
void update_dual(struct tiny_problem *problem, const struct tiny_params *params) {
    // Store previous values for residual computation
    tiny_MatrixNxNh v_prev = problem->v;
    tiny_MatrixNuNhm1 z_prev = problem->z;
    
    // Regular dual updates
    for (int k = 0; k < NHORIZON-1; k++) {
        problem->y.col(k) += problem->u.col(k) - problem->znew.col(k);
        problem->g.col(k) += problem->x.col(k) - problem->vnew.col(k);
    }
    problem->g.col(NHORIZON-1) += problem->x.col(NHORIZON-1) - problem->vnew.col(NHORIZON-1);
    
    // Update residuals
    problem->primal_residual_input = (problem->u - problem->znew).cwiseAbs().maxCoeff();
    problem->primal_residual_state = (problem->x - problem->vnew).cwiseAbs().maxCoeff();
    problem->dual_residual_input = (params->cache.rho[problem->cache_level] * (z_prev - problem->znew)).cwiseAbs().maxCoeff();
    problem->dual_residual_state = (params->cache.rho[problem->cache_level] * (v_prev - problem->vnew)).cwiseAbs().maxCoeff();
}

/**
 * Update linear control cost terms in the Riccati feedback using the changing
 * slack and dual variables from ADMM
*/
void update_linear_cost(struct tiny_problem *problem, const struct tiny_params *params) {
    // Debug dimensions first
    // Serial.println("\nDimensions check:");
    // Serial.print("r: "); Serial.print(problem->r.rows()); Serial.print("x"); Serial.println(problem->r.cols());
    // Serial.print("q: "); Serial.print(problem->q.rows()); Serial.print("x"); Serial.println(problem->q.cols());
    // Serial.print("p: "); Serial.print(problem->p.rows()); Serial.print("x"); Serial.println(problem->p.cols());

    for (int k = 0; k < NHORIZON-1; k++) {
        // Input cost - EXACTLY like Python: r[:, k] = -self.cache['R'] @ u_ref[:, k]
        problem->r.col(k) = -(params->R[problem->cache_level].asDiagonal() * params->Uref.col(k));
        problem->r.col(k) -= params->cache.rho[problem->cache_level] * 
                            (problem->znew.col(k) - problem->y.col(k));
            
        // State cost - EXACTLY like Python: q[:, k] = -self.cache['Q'] @ x_ref[:, k]
        problem->q.col(k) = -(params->Q[problem->cache_level].asDiagonal() * params->Xref.col(k));
        problem->q.col(k) -= params->cache.rho[problem->cache_level] * 
                            (problem->vnew.col(k) - problem->g.col(k));
    }

    // Terminal cost - EXACTLY like Python: p[:,self.N-1] = -np.dot(self.cache['Pinf'], x_ref[:, self.N-1])
    problem->p.col(NHORIZON-1) = -(params->cache.Pinf[problem->cache_level] * 
                                  params->Xref.col(NHORIZON-1));
    problem->p.col(NHORIZON-1) -= params->cache.rho[problem->cache_level] * 
                                 (problem->vnew.col(NHORIZON-1) - problem->g.col(NHORIZON-1));
}

void compute_cache_terms(struct tiny_params *params) {
    Serial.println("\n=== Computing Cache Terms ===");
    
    // First, let's print the pre-computed K (from PROGMEM)
    Serial.println("Pre-computed K (first row):");
    for(int j = 0; j < NSTATES; j++) {
        float k_val = pgm_read_float(&Kinf_data[j]);
        Serial.print(k_val, 6); Serial.print(" ");
    }
    Serial.println();
    
    // Create augmented cost matrices
    tiny_MatrixNxNx Q_rho;
    tiny_MatrixNuNu R_rho;
    Q_rho.setZero();
    R_rho.setZero();
    
    // Load Q and R from PROGMEM
    for(int i = 0; i < NSTATES; i++) {
        Q_rho(i,i) = pgm_read_float(&Q_data[i]) + params->cache.rho[0];
    }
    for(int i = 0; i < NINPUTS; i++) {
        R_rho(i,i) = pgm_read_float(&R_data[i]) + params->cache.rho[0];
    }
    
    // DLQR iteration
    tiny_MatrixNxNx P;
    // Initialize P with pre-computed Pinf
    for(int i = 0; i < NSTATES; i++) {
        for(int j = 0; j < NSTATES; j++) {
            P(i,j) = pgm_read_float(&Pinf_data[i*NSTATES + j]);
        }
    }
    
    tiny_MatrixNuNx BtP;
    tiny_MatrixNuNu temp;
    
    // Single iteration to compute K and other cache terms
    BtP = params->cache.Bdyn[0].transpose() * P;
    temp = R_rho + BtP * params->cache.Bdyn[0];
    
    // Add small regularization
    for(int j = 0; j < NINPUTS; j++) {
        temp(j,j) += 1e-8f;
    }
    
    // Compute and store all cache terms
    params->cache.Quu_inv[0] = temp.lu().solve(tiny_MatrixNuNu::Identity());
    params->cache.Kinf[0] = params->cache.Quu_inv[0] * BtP * params->cache.Adyn[0];
    params->cache.AmBKt[0] = params->cache.Adyn[0] - params->cache.Bdyn[0] * params->cache.Kinf[0];
    params->cache.Pinf[0] = P;  // Store pre-computed P
    
    // Print computed K for comparison
    Serial.println("Computed K (first row):");
    for(int j = 0; j < NSTATES; j++) {
        Serial.print(params->cache.Kinf[0](0,j), 6); Serial.print(" ");
    }
    Serial.println();
    
    // Print difference norm
    float diff_norm = 0;
    for(int i = 0; i < NINPUTS; i++) {
        for(int j = 0; j < NSTATES; j++) {
            float k_precomputed = pgm_read_float(&Kinf_data[i*NSTATES + j]);
            diff_norm += pow(k_precomputed - params->cache.Kinf[0](i,j), 2);
        }
    }
    diff_norm = sqrt(diff_norm);
    Serial.print("K matrix difference norm: "); Serial.println(diff_norm, 6);
    
    Serial.println("=== Cache Terms Complete ===\n");
}




} /* extern "C" */
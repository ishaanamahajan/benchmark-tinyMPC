#include <iostream>

#include "admm.hpp"

#include "rho_benchmark.hpp"

#include "problem_data/rand_prob_tinympc_params.hpp"


#include "types.hpp"

#define DEBUG_MODULE "TINYALG"

extern "C" {


// #include "debug.h"

static uint32_t startTimestamp;

void solve_lqr(struct tiny_problem *problem, const struct tiny_params *params) {
    // Use single Kinf matrix instead of cache.Kinf[cache_level]
    problem->u.col(0) = -params->Kinf * (problem->x.col(0) - params->Xref.col(0));
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

    // Store previous values for residuals using Eigen matrices
    tiny_MatrixNxNh v_prev = problem->vnew;
    tiny_MatrixNuNhm1 z_prev = problem->znew;
    
    uint32_t admm_start = micros();
    
    // Debug print initial values
    // Serial.println("\nInitial values:");
    // Serial.print("max_iter: "); Serial.println(params->max_iter);
    // Serial.print("abs_pri_tol: "); Serial.println(params->abs_pri_tol);
    // Serial.print("abs_dua_tol: "); Serial.println(params->abs_dua_tol);
    
    // Calculate initial residuals - use single rho value
    float pri_res_input = (problem->u - problem->znew).lpNorm<Eigen::Infinity>();
    float pri_res_state = (problem->x - problem->vnew).lpNorm<Eigen::Infinity>();
    float dua_res_input = params->rho * (problem->znew - z_prev).lpNorm<Eigen::Infinity>();
    float dua_res_state = params->rho * (problem->vnew - v_prev).lpNorm<Eigen::Infinity>();

    // Serial.println("Initial residuals:");
    // Serial.print("pri_in: "); Serial.println(problem->primal_residual_input);
    // Serial.print("pri_st: "); Serial.println(problem->primal_residual_state);
    // Serial.print("dua_in: "); Serial.println(problem->dual_residual_input);
    // Serial.print("dua_st: "); Serial.println(problem->dual_residual_state);
    
    for (int iter = 0; iter < params->max_iter; iter++) {
        // Update steps in same order as Python
        update_primal(problem, params);
        update_slack(problem, params);
        update_dual(problem, params);
        update_linear_cost(problem, params);

        // Calculate residuals (matching Python exactly)
        float pri_res_input = (problem->u - problem->znew).lpNorm<Eigen::Infinity>();
        float pri_res_state = (problem->x - problem->vnew).lpNorm<Eigen::Infinity>();
        float dua_res_input = params->rho * (problem->znew - z_prev).lpNorm<Eigen::Infinity>();
        float dua_res_state = params->rho * (problem->vnew - v_prev).lpNorm<Eigen::Infinity>();

        // Store current values for next iteration
        z_prev = problem->znew;
        v_prev = problem->vnew;

        // Debug print every 50 iterations
        if (iter % 50 == 0) {
            Serial.print("Iter "); Serial.print(iter);
            Serial.print(" pri_in: "); Serial.print(pri_res_input, 6);
            Serial.print(" pri_st: "); Serial.print(pri_res_state, 6);
            Serial.print(" dua_in: "); Serial.print(dua_res_input, 6);
            Serial.print(" dua_st: "); Serial.println(dua_res_state, 6);
        }

        // Check convergence like Python lines 263-267
        if (pri_res_input < params->abs_pri_tol && 
            pri_res_state < params->abs_pri_tol &&
            dua_res_input < params->abs_dua_tol && 
            dua_res_state < params->abs_dua_tol) {
            
            //Serial.println("Converged!");
            problem->status = 1;  // Converged
            break;
        }

        problem->iter += 1;
    }
    
    // // Update rho if adapter exists (like Python line 278-279)
    // if (params->rho_adapter.analytical_method) {
    //     uint32_t rho_start = micros();
    //     float pri_res = std::max(pri_res_state, pri_res_input);
    //     float dual_res = std::max(dua_res_state, dua_res_input);

    //     RhoBenchmarkResult result;
    //     benchmark_rho_adaptation(
    //         problem->x.data(),
    //         problem->u.data(),
    //         problem->v.data(),
    //         pri_res,
    //         dual_res,
    //         &result,
    //         &params->rho_adapter
    //     );
        
    //     params->cache.rho[problem->cache_level] = result.final_rho;
    //     problem->rho_time = micros() - rho_start;
    // }
    
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
        // Use single matrices instead of cache arrays
        auto temp = params->B.transpose() * problem->p.col(i+1) + problem->r.col(i);
        problem->d.col(i).noalias() = params->C1 * temp;
        
        problem->p.col(i).noalias() = problem->q.col(i) + 
                                     params->C2 * problem->p.col(i+1) - 
                                     params->Kinf.transpose() * problem->r.col(i);
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
        // Use single Kinf matrix
        auto feedback = -params->Kinf * state_error;
        auto feedforward = -problem->d.col(k);
        
        // Serial.print(" fb:"); Serial.print(feedback(0), 6);
        // Serial.print(" ff:"); Serial.print(feedforward(0), 6);
        
        // 3. Update control and state
        problem->u.col(k) = feedback + feedforward;
        // Use single A and B matrices
        problem->x.col(k+1) = params->A * problem->x.col(k) + 
                             params->B * problem->u.col(k);
        
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
    // Like Python lines 134-137
    for (int k = 0; k < NHORIZON-1; k++) {
        // Clip input vectors
        problem->znew.col(k) = (problem->u.col(k) + problem->y.col(k))
            .cwiseMax(params->u_min.col(k))
            .cwiseMin(params->u_max.col(k));

        // Clip state vectors
        problem->vnew.col(k) = (problem->x.col(k) + problem->g.col(k))
            .cwiseMax(params->x_min.col(k))
            .cwiseMin(params->x_max.col(k));
    }

    // Terminal state
    problem->vnew.col(NHORIZON-1) = (problem->x.col(NHORIZON-1) + problem->g.col(NHORIZON-1))
        .cwiseMax(params->x_min.col(NHORIZON-1))
        .cwiseMin(params->x_max.col(NHORIZON-1));
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
*/
void update_dual(struct tiny_problem *problem, const struct tiny_params *params) {
    // Match Python's update_dual
    for (int k = 0; k < NHORIZON-1; k++) {
        problem->y.col(k) += problem->u.col(k) - problem->znew.col(k);
        problem->g.col(k) += problem->x.col(k) - problem->vnew.col(k);
    }
    // Don't forget terminal state update
    problem->g.col(NHORIZON-1) += problem->x.col(NHORIZON-1) - problem->vnew.col(NHORIZON-1);
}

/**
 * Update linear control cost terms in the Riccati feedback using the changing
 * slack and dual variables from ADMM
*/
void update_linear_cost(struct tiny_problem *problem, const struct tiny_params *params) {
    for (int k = 0; k < NHORIZON-1; k++) {
        // Use single R, Q matrices and single rho value
        problem->r.col(k) = -(params->R * params->Uref.col(k));
        problem->r.col(k) -= params->rho * (problem->znew.col(k) - problem->y.col(k));
        
        problem->q.col(k) = -(params->Q * params->Xref.col(k));
        problem->q.col(k) -= params->rho * (problem->vnew.col(k) - problem->g.col(k));
    }
    
    // Terminal cost using single Pinf matrix
    problem->p.col(NHORIZON-1) = -(params->Pinf * params->Xref.col(NHORIZON-1));
    problem->p.col(NHORIZON-1) -= params->rho * (problem->vnew.col(NHORIZON-1) - 
                                                problem->g.col(NHORIZON-1));
}

} /* extern "C" */
#include <iostream>

#include "admm.hpp"

#include "rho_benchmark.hpp"

#include "problem_data/rand_prob_tinympc_params.hpp"

#include <Eigen/Cholesky>


#include "types.hpp"

#define DEBUG_MODULE "TINYALG"

extern "C" {


// #include "debug.h"

static uint32_t startTimestamp;

void solve_lqr(struct tiny_problem *problem, const struct tiny_params *params) {
    problem->u.col(0) = -params->Kinf * (problem->x.col(0) - params->Xref.col(0));
}


void solve_admm(struct tiny_problem *problem, struct tiny_params *params) {
    // Measure initialization
    uint32_t init_start = micros();
    
    // IMPORTANT: Reset ALL solver state variables
    problem->y.setZero();
    problem->g.setZero();
    problem->v.setZero();
    problem->z.setZero();
    problem->vnew.setZero();
    problem->znew.setZero();
    // problem->p.setZero();
    // problem->q.setZero();
    // problem->r.setZero();
    // problem->d.setZero();
    
    // Reset convergence variables
    problem->status = 0;
    problem->iter = 0;
    
    // Reset timing variables
    problem->fixed_timings.init_time = 0;
    problem->fixed_timings.admm_time = 0;
    problem->fixed_timings.total_time = 0;
    
    // Reset cache terms in params
    //params->compute_cache_terms();
    
    startTimestamp = micros();
    
    problem->fixed_timings.rho_time = 0;

    // Initial updates
    forward_pass(problem, params);
    update_slack(problem, params);
    update_dual(problem, params);
    update_linear_cost(problem, params);

    // Store previous values for residuals
    tiny_MatrixNxNh v_prev = problem->vnew;
    tiny_MatrixNuNhm1 z_prev = problem->znew;
    
    uint32_t admm_start = micros();
    
    // Calculate initial residuals
    float pri_res_input = (problem->u - problem->znew).lpNorm<Eigen::Infinity>();
    float pri_res_state = (problem->x - problem->vnew).lpNorm<Eigen::Infinity>();
    float dua_res_input = params->rho * (problem->znew - z_prev).lpNorm<Eigen::Infinity>();
    float dua_res_state = params->rho * (problem->vnew - v_prev).lpNorm<Eigen::Infinity>();
    
    for (int iter = 0; iter < params->max_iter; iter++) {
        update_primal(problem, params);
        update_slack(problem, params);
        update_dual(problem, params);
        update_linear_cost(problem, params);

        pri_res_input = (problem->u - problem->znew).lpNorm<Eigen::Infinity>();
        pri_res_state = (problem->x - problem->vnew).lpNorm<Eigen::Infinity>();
        dua_res_input = params->rho * (problem->znew - z_prev).lpNorm<Eigen::Infinity>();
        dua_res_state = params->rho * (problem->vnew - v_prev).lpNorm<Eigen::Infinity>();

        z_prev = problem->znew;
        v_prev = problem->vnew;

        // Store residuals
        problem->primal_residual_input = pri_res_input;
        problem->primal_residual_state = pri_res_state;
        problem->dual_residual_input = dua_res_input;
        problem->dual_residual_state = dua_res_state;

        if (pri_res_input <= params->abs_pri_tol && 
            pri_res_state <= params->abs_pri_tol &&
            dua_res_input <= params->abs_dua_tol && 
            dua_res_state <= params->abs_dua_tol) {
            problem->status = 1;
            break;
        }
        
        problem->iter++;
    }
    
    problem->fixed_timings.admm_time = micros() - admm_start;
    problem->fixed_timings.total_time = problem->fixed_timings.init_time + 
                                      problem->fixed_timings.admm_time;
}

void solve_admm_adaptive(struct tiny_problem *problem, struct tiny_params *params, RhoAdapter *adapter) {
    // Measure initialization
    uint32_t init_start = micros();
    problem->y.setZero();
    problem->g.setZero();
    problem->v.setZero();
    problem->z.setZero();
    problem->vnew.setZero();
    problem->znew.setZero();
    // problem->p.setZero();
    // problem->q.setZero();
    // problem->r.setZero();
    // problem->d.setZero();
    
    // Reset convergence variables
    problem->status = 0;
    problem->iter = 0;
    problem->solve_count++;
    problem->adaptive_timings.init_time = micros() - init_start;
    
    problem->adaptive_timings.rho_time = 0;  // Reset rho time
    
    startTimestamp = micros();
    
    problem->status = 0;
    problem->iter = 0;

    // Initial updates
    forward_pass(problem, params);
    update_slack(problem, params);
    update_dual(problem, params);
    update_linear_cost(problem, params);

    tiny_MatrixNxNh v_prev = problem->vnew;
    tiny_MatrixNuNhm1 z_prev = problem->znew;
    
    uint32_t admm_start = micros();
    
    // Rho adaptation variables
    RhoBenchmarkResult rho_result;
    
    for (int iter = 0; iter < params->max_iter; iter++) {
        update_primal(problem, params);
        update_slack(problem, params);
        update_dual(problem, params);
        update_linear_cost(problem, params);

        float pri_res_input = (problem->u - problem->znew).lpNorm<Eigen::Infinity>();
        float pri_res_state = (problem->x - problem->vnew).lpNorm<Eigen::Infinity>();
        float dua_res_input = params->rho * (problem->znew - z_prev).lpNorm<Eigen::Infinity>();
        float dua_res_state = params->rho * (problem->vnew - v_prev).lpNorm<Eigen::Infinity>();

        // Update rho every 10 iterations
        if (iter > 0 && iter % 10 == 0) {
            uint32_t rho_update_start = micros();
            
            // Call benchmark_rho_adaptation with current state
            benchmark_rho_adaptation(
                problem->x.col(0).data(),
                problem->u.col(0).data(),
                problem->vnew.col(0).data(),
                max(pri_res_input, pri_res_state),
                max(dua_res_input, dua_res_state),
                &rho_result,
                adapter,
                params->rho
            );
            
            // // Update rho and cache terms using Taylor expansion
            // if (abs(rho_result.final_rho - params->rho) > adapter->tolerance) {
            //     float old_rho = params->rho;
            //     params->rho = rho_result.final_rho;
            //     update_cache_taylor(params->rho, old_rho);
            // }
            
            problem->adaptive_timings.rho_time += micros() - rho_update_start;
        }

        z_prev = problem->znew;
        v_prev = problem->vnew;

        // Store residuals
        problem->primal_residual_input = pri_res_input;
        problem->primal_residual_state = pri_res_state;
        problem->dual_residual_input = dua_res_input;
        problem->dual_residual_state = dua_res_state;
        
        // Check convergence
        if (pri_res_input <= params->abs_pri_tol && 
            pri_res_state <= params->abs_pri_tol &&
            dua_res_input <= params->abs_dua_tol && 
            dua_res_state <= params->abs_dua_tol) {
            problem->status = 1;
            break;
        }
        
        problem->iter++;
    }
    
    problem->adaptive_timings.admm_time = micros() - admm_start;
    problem->adaptive_timings.total_time = problem->adaptive_timings.init_time + 
                                         problem->adaptive_timings.admm_time + 
                                         problem->adaptive_timings.rho_time;
}

void update_primal(struct tiny_problem *problem, const struct tiny_params *params) {
    backward_pass_grad(problem, params);
    forward_pass(problem, params);
}

void backward_pass_grad(struct tiny_problem *problem, const struct tiny_params *params) {
    for (int i = NHORIZON-2; i >= 0; i--) {
        auto temp = params->B.transpose() * problem->p.col(i+1) + problem->r.col(i);
        problem->d.col(i).noalias() = params->C1 * temp;
        
        problem->p.col(i).noalias() = problem->q.col(i) + 
                                     params->C2 * problem->p.col(i+1) - 
                                     params->Kinf.transpose() * problem->r.col(i);
    }
}

void forward_pass(struct tiny_problem *problem, const struct tiny_params *params) {
    for (int k = 0; k < NHORIZON-1; k++) {
        auto state_error = problem->x.col(k) - params->Xref.col(k);
        auto feedback = -params->Kinf * state_error;
        auto feedforward = -problem->d.col(k);
        
        problem->u.col(k) = feedback + feedforward;
        problem->x.col(k+1) = params->A * problem->x.col(k) + 
                             params->B * problem->u.col(k);
    }
}

void update_slack(struct tiny_problem *problem, const struct tiny_params *params) {
    for (int k = 0; k < NHORIZON-1; k++) {
        problem->znew.col(k) = (problem->u.col(k) + problem->y.col(k))
            .cwiseMax(params->u_min.col(k))
            .cwiseMin(params->u_max.col(k));

        problem->vnew.col(k) = (problem->x.col(k) + problem->g.col(k))
            .cwiseMax(params->x_min.col(k))
            .cwiseMin(params->x_max.col(k));
    }

    problem->vnew.col(NHORIZON-1) = (problem->x.col(NHORIZON-1) + problem->g.col(NHORIZON-1))
        .cwiseMax(params->x_min.col(NHORIZON-1))
        .cwiseMin(params->x_max.col(NHORIZON-1));
}

void update_dual(struct tiny_problem *problem, const struct tiny_params *params) {
    for (int k = 0; k < NHORIZON-1; k++) {
        problem->y.col(k) += problem->u.col(k) - problem->znew.col(k);
        problem->g.col(k) += problem->x.col(k) - problem->vnew.col(k);
    }
    problem->g.col(NHORIZON-1) += problem->x.col(NHORIZON-1) - problem->vnew.col(NHORIZON-1);
}

void update_linear_cost(struct tiny_problem *problem, const struct tiny_params *params) {
    for (int k = 0; k < NHORIZON-1; k++) {
        problem->r.col(k) = -(params->R * params->Uref.col(k));
        problem->r.col(k) -= params->rho * (problem->znew.col(k) - problem->y.col(k));
        
        problem->q.col(k) = -(params->Q * params->Xref.col(k));
        problem->q.col(k) -= params->rho * (problem->vnew.col(k) - problem->g.col(k));
    }
    
    problem->p.col(NHORIZON-1) = -(params->Pinf * params->Xref.col(NHORIZON-1));
    problem->p.col(NHORIZON-1) -= params->rho * (problem->vnew.col(NHORIZON-1) - 
                                                problem->g.col(NHORIZON-1));
}

} /* extern "C" */
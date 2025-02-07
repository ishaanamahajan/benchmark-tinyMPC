#include <iostream>

#include "admm.hpp"

#include "rho_benchmark.hpp"

#define DEBUG_MODULE "TINYALG"

extern "C" {

// #include "debug.h"

static uint32_t startTimestamp;

void solve_lqr(struct tiny_problem *problem, const struct tiny_params *params) {
    problem->u.col(0) = -params->cache.Kinf[problem->cache_level] * (problem->x.col(0) - params->Xref.col(0));
}


void solve_admm(struct tiny_problem *problem, struct tiny_params *params) {
    Serial.println("\n=== Starting solve_admm ===");
    
    // Print key parameters
    Serial.print("Tolerances at solve - pri_tol: ");
    Serial.print(params->abs_pri_tol, 8);
    Serial.print(", dua_tol: ");
    Serial.println(params->abs_dua_tol, 8);
    
    // Print norms of key matrices
    Serial.print("State norm: "); Serial.println(problem->x.norm());
    Serial.print("Input norm: "); Serial.println(problem->u.norm());
    Serial.print("Q norm: "); Serial.println(params->Q[0].norm());
    Serial.print("Xref norm: "); Serial.println(params->Xref.norm());
    
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
    Serial.println("\nInitial values:");
    Serial.print("max_iter: "); Serial.println(params->max_iter);
    Serial.print("abs_pri_tol: "); Serial.println(params->abs_pri_tol);
    Serial.print("abs_dua_tol: "); Serial.println(params->abs_dua_tol);
    
    // Calculate initial residuals
    problem->primal_residual_input = (problem->u - problem->znew).cwiseAbs().maxCoeff();
    problem->primal_residual_state = (problem->x - problem->vnew).cwiseAbs().maxCoeff();
    problem->dual_residual_input = (params->cache.rho[problem->cache_level] * (z_prev - problem->znew)).cwiseAbs().maxCoeff();
    problem->dual_residual_state = (params->cache.rho[problem->cache_level] * (v_prev - problem->vnew)).cwiseAbs().maxCoeff();

    Serial.println("Initial residuals:");
    Serial.print("pri_in: "); Serial.println(problem->primal_residual_input);
    Serial.print("pri_st: "); Serial.println(problem->primal_residual_state);
    Serial.print("dua_in: "); Serial.println(problem->dual_residual_input);
    Serial.print("dua_st: "); Serial.println(problem->dual_residual_state);
    
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

        if (i % 100 == 0) {  // Print every 100 iterations
            Serial.print("Iter "); Serial.print(i);
            Serial.print(" Residuals - pri_in: "); Serial.print(problem->primal_residual_input);
            Serial.print(" pri_st: "); Serial.print(problem->primal_residual_state);
            Serial.print(" dua_in: "); Serial.print(problem->dual_residual_input);
            Serial.print(" dua_st: "); Serial.println(problem->dual_residual_state);
        }

        // Update previous values
        v_prev = problem->vnew;
        z_prev = problem->znew;
        
        problem->v = problem->vnew;
        problem->z = problem->znew;
        
        problem->iter += 1;

        // Check convergence using stored residuals
        if ((problem->primal_residual_input < params->abs_pri_tol) && 
            (problem->primal_residual_state < params->abs_pri_tol) &&
            (problem->dual_residual_input < params->abs_dua_tol) && 
            (problem->dual_residual_state < params->abs_dua_tol))
        {
            Serial.println("Converged!");
            problem->status = 1;
            break;
        }
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
        // Match Python's computation exactly
        (problem->d.col(i)).noalias() = params->cache.Quu_inv[problem->cache_level] * 
            (params->cache.Bdyn[problem->cache_level].transpose() * problem->p.col(i+1) + problem->r.col(i));
        
        (problem->p.col(i)).noalias() = problem->q.col(i) + 
            params->cache.AmBKt[problem->cache_level] * problem->p.col(i+1) - 
            params->cache.Kinf[problem->cache_level].transpose() * problem->r.col(i);
    }
}

/**
 * Use LQR feedback policy to roll out trajectory
*/
void forward_pass(struct tiny_problem *problem, const struct tiny_params *params) {
    for (int i=0; i<NHORIZON-1; i++) {
        (problem->u.col(i)).noalias() = -params->cache.Kinf[problem->cache_level].lazyProduct(problem->x.col(i)) - problem->d.col(i);
        // problem->u.col(i) << .001, .02, .3, 4;
        // DEBUG_PRINT("u(0): %f\n", problem->u.col(0)(0));
        // multAdyn(problem->Ax, params->cache.Adyn[problem->cache_level], problem->x.col(i));
        (problem->x.col(i+1)).noalias() = params->cache.Adyn[problem->cache_level].lazyProduct(problem->x.col(i)) + params->cache.Bdyn[problem->cache_level].lazyProduct(problem->u.col(i));
        // (problem->x.col(i+1)).noalias() = params->cache.Adyn.lazyProduct(problem->x.col(i)) + params->cache.Bdyn.lazyProduct(problem->u.col(i));
    }
}

/**
 * Project slack (auxiliary) variables into their feasible domain, defined by
 * projection functions related to each constraint
 * TODO: pass in meta information with each constraint assigning it to a
 * projection function
*/
void update_slack(struct tiny_problem *problem, const struct tiny_params *params) {
    // Box constraints on input
    // Get current time

    problem->znew = params->u_max.cwiseMin(params->u_min.cwiseMax(problem->u + problem->y));
    problem->vnew = params->x_max.cwiseMin(params->x_min.cwiseMax(problem->x + problem->g));

    // Half space constraints on state
    // TODO: support multiple half plane constraints per knot point
    //      currently this only works for one constraint per knot point
    // TODO: can potentially take advantage of the fact that A_constraints[3:end] is zero and just do
    //      v.col(i) = x.col(i) - dist*A_constraints[i] since we have to copy x[3:end] into v anyway
    //      downside is it's not clear this is happening externally and so values of A_constraints
    //      not set to zero (other than the first three) can cause the algorithm to fail
    // TODO: the only state values changing here are the first three (x, y, z) so it doesn't make sense
    //      to do operations on the remaining 9 when projecting (or doing anything related to the dual
    //      or auxiliary variables). v and g could be of size (3) and everything would work the same.
    //      The only reason this doesn't break is because in the update_linear_cost function subtracts
    //      g from v and so the last nine entries are always zero.
    // problem->xg = problem->x + problem->g;
    // problem->dists = (params->A_constraints.transpose().cwiseProduct(problem->xg)).colwise().sum();
    // problem->dists -= params->x_max;
    // // startTimestamp = usecTimestamp();
    // problem->cache_level = 0;
    // for (int i=0; i<NHORIZON; i++) {
    //     problem->dist = (params->A_constraints[i].head(3)).lazyProduct(problem->xg.col(i).head(3)); // Distances can be computed in one step outside the for loop
    //     problem->dist -= params->x_max[i](0);
    //     problem->xyz_news.col(i) = problem->xg.col(i).head(3) - problem->dist*params->A_constraints[i].head(3).transpose();
    //     // DEBUG_PRINT("dist: %f\n", dist);
    //     if (problem->dist <= 0) {
    //         problem->vnew.col(i) = problem->xg.col(i);
    //     }
    //     else {
    //         problem->cache_level = 1; // Constraint violated, use second cache level
    //         problem->xyz_new = problem->xg.col(i).head(3) - problem->dist*params->A_constraints[i].head(3).transpose();
    //         problem->vnew.col(i) << problem->xyz_new, problem->xg.col(i).tail(NSTATES-3);
    //     }
    // }
    // problem->vnew = problem->xg;
    // DEBUG_PRINT("s: %d\n", usecTimestamp() - startTimestamp);
}

/**
 * Update next iteration of dual variables by performing the augmented
 * lagrangian multiplier update
*/
void update_dual(struct tiny_problem *problem, const struct tiny_params *params) {
    problem->y = problem->y + problem->u - problem->znew;
    problem->g = problem->g + problem->x - problem->vnew;
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




} /* extern "C" */
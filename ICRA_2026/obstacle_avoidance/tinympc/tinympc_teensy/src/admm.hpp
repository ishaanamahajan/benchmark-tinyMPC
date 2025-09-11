#pragma once

#include "types.hpp"

#ifdef __cplusplus
extern "C"
{
#endif

    int tiny_solve(TinySolver *solver);
    
    // Helper functions
    Matrix<tinytype, 3, 1> project_soc(Matrix<tinytype, 3, 1> s, float mu);
    bool termination_condition(TinySolver *solver);
    void reset_dual(TinySolver *solver);
    void reset_problem(TinySolver *solver);

    // Core ADMM functions
    void backward_pass_grad(TinySolver *solver);
    void forward_pass(TinySolver *solver);
    void update_slack(TinySolver *solver);
    void update_dual(TinySolver *solver);
    void update_linear_cost(TinySolver *solver);
    
    // SDP constraint functions
    void project_sdp_constraints(TinySolver *solver);
    
    #ifdef ENABLE_SDP_PROJECTION
    void project_moment_matrix_at_step(TinySolver *solver, int k);
    void project_terminal_matrix_at_step(TinySolver *solver, int k);
    void project_obstacle_constraint_at_step(TinySolver *solver, int k);
    #endif

#ifdef __cplusplus
}

// SDP projection function (C++ template, outside extern "C")
template<int M>
Eigen::Matrix<tinytype, M, M> project_psd(const Eigen::Matrix<tinytype, M, M>& S_in, tinytype eps = tinytype(0));

#endif
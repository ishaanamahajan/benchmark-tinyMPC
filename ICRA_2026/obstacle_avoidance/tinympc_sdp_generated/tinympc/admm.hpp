#pragma once

#include "types.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int solve(TinySolver *solver);

void update_primal(TinySolver *solver);
void backward_pass_grad(TinySolver *solver);
void forward_pass(TinySolver *solver);
void update_slack(TinySolver *solver);
void update_dual(TinySolver *solver);
void update_linear_cost(TinySolver *solver);
bool termination_condition(TinySolver *solver);

// SDP constraint projection function
void project_sdp_constraints(TinySolver *solver);

#ifdef __cplusplus
}

// SDP projection function template (C++ only, outside extern "C")
template<int M>
Matrix<tinytype, M, M> project_psd(const Matrix<tinytype, M, M>& S_in, tinytype eps);

#endif
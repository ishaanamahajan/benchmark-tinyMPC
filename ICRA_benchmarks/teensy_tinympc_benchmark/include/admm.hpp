#pragma once

#include "types.hpp"

// Move template functions before extern "C"
template<typename VectorType>
void clip_vector(VectorType& result, const VectorType& input, 
                const VectorType& lower, const VectorType& upper) {
    result = input.cwiseMax(lower).cwiseMin(upper);
}

template<typename MatrixType>
float max_abs_diff(const MatrixType& A, const MatrixType& B) {
    return (A - B).template lpNorm<Eigen::Infinity>();
}

#ifdef __cplusplus
extern "C" {
#endif

// Main solver functions
void solve_lqr(struct tiny_problem *problem, const struct tiny_params *params);
void solve_admm(struct tiny_problem *problem, struct tiny_params *params);

// ADMM helper functions
void update_primal(struct tiny_problem *problem, const struct tiny_params *params);
void backward_pass_grad(struct tiny_problem *problem, const struct tiny_params *params);
void forward_pass(struct tiny_problem *problem, const struct tiny_params *params);
void update_slack(struct tiny_problem *problem, const struct tiny_params *params);
void update_dual(struct tiny_problem *problem, const struct tiny_params *params);
void update_linear_cost(struct tiny_problem *problem, const struct tiny_params *params);

#ifdef __cplusplus
}
#endif
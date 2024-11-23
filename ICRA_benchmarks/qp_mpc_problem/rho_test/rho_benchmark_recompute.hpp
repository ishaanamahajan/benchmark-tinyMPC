#pragma once
#include <stdint.h>

// Same result structure
struct RhoBenchmarkResult {
    float initial_rho;
    float final_rho;
    float pri_res;
    float dual_res;
    uint32_t time_us;
};

// New function for full recomputation
void benchmark_rho_recompute(
    const float* x_prev,
    const float* u_prev,
    const float* z_prev,
    float pri_res,
    float dual_res,
    RhoBenchmarkResult* result
);
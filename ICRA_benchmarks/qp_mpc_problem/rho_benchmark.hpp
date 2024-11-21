#pragma once
#include <stdint.h>

// Simple structure to hold benchmark results
struct RhoBenchmarkResult {
    float initial_rho;     // Starting rho value (85.0)
    float final_rho;       // What rho adapted to
    float pri_res;         // Primal residual that triggered adaptation
    float dual_res;        // Dual residual that triggered adaptation
    uint32_t time_us;      // Time taken for adaptation computation
};

// Main benchmark function
void benchmark_rho_adaptation(
    float pri_res,         // Current primal residual
    float dual_res,        // Current dual residual
    RhoBenchmarkResult* result
);
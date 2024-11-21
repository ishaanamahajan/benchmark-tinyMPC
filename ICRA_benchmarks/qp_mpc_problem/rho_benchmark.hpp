#pragma once
#include <stdint.h>

// Problem dimensions
#define BENCH_NX 12  // State dimension
#define BENCH_NU 4   // Input dimension

// Structure to hold benchmark results
struct RhoBenchmarkResult {
    float initial_rho;     // Starting rho value (85.0)
    float final_rho;       // What rho adapted to
    float pri_res;         // Primal residual that triggered adaptation
    float dual_res;        // Dual residual that triggered adaptation
    uint32_t time_us;      // Time taken for adaptation computation
};

// Main benchmark functions
void benchmark_rho_adaptation(float pri_res, float dual_res, RhoBenchmarkResult* result);
void update_cache_taylor(float new_rho, float old_rho);
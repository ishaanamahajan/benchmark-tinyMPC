#include <iostream>
#include "rho_benchmark.hpp"

int main() {
    // Test data
    float x_prev[BENCH_NX] = {0};
    float u_prev[BENCH_NU] = {0};
    float z_prev[BENCH_NX] = {0};
    
    RhoBenchmarkResult result;

    // Just try to compile both functions
    benchmark_rho_adaptation(x_prev, u_prev, z_prev, 1.0, 0.5, &result);
    benchmark_rho_recompute(x_prev, u_prev, z_prev, 1.0, 0.5, &result);
    
    return 0;
}
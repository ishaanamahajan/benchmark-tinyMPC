#include "rho_benchmark.hpp"
#include <Arduino.h>

void benchmark_rho_adaptation(
    float pri_res,
    float dual_res,
    RhoBenchmarkResult* result
) {
    // Initial setup
    result->initial_rho = 85.0f;
    result->pri_res = pri_res;
    result->dual_res = dual_res;
    
    // Start timing
    uint32_t start = micros();
    
    // Perform rho adaptation (same as Python)
    float ratio = pri_res / dual_res;
    ratio = min(max(ratio, 0.001f), 1.0f);
    float new_rho = result->initial_rho * sqrt(ratio);
    new_rho = min(max(new_rho, 70.0f), 100.0f);
    
    // End timing
    result->time_us = micros() - start;
    result->final_rho = new_rho;
}
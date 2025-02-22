#include "rho_benchmark_stm32.h"

// Keep timing functions but use HAL
static uint32_t get_micros(void) {
    return HAL_GetTick() * 1000;  // Convert ms to us
}

void benchmark_rho_adaptation_stm32(
    const float* x_prev,
    const float* u_prev,
    const float* v_prev,
    float pri_res,
    float dual_res,
    RhoBenchmarkResult* result,
    RhoAdapter* adapter
) {
    uint32_t start = get_micros();
    
    // Same computation logic as original but using STM32 timing
    float normalized_pri = pri_res / (pri_norm + 1e-10f);
    float normalized_dual = dual_res / (dual_norm + 1e-10f);
    float ratio = normalized_pri / (normalized_dual + 1e-10f);

    float new_rho = adapter->rho_base * sqrt(ratio);
    
    if (adapter->clip) {
        new_rho = min(max(new_rho, adapter->rho_min), adapter->rho_max);
    }
    
    result->time_us = get_micros() - start;
    result->initial_rho = adapter->rho_base;
    result->final_rho = new_rho;
    result->pri_res = pri_res;
    result->dual_res = dual_res;
}

void run_benchmarks_stm32(void) {
    printf("\r\nStarting STM32 MPC Benchmark Test\r\n");
    printf("Method,Trial,SolveTime,ADMMTime,RhoTime,Iterations,FinalRho\r\n");
    
    // Initialize problem & params similar to original
    RhoAdapter adapter = {
        .rho_base = 85.0f,
        .rho_min = 40.0f,
        .rho_max = 100.0f,
        .tolerance = 1.1f,
        .clip = true
    };
    
    // Run your benchmark tests but with printf output
    const int NUM_TRIALS = 1000;
    for(int i = 0; i < NUM_TRIALS; i++) {
        // Your benchmark logic here but using printf instead of Serial
        printf("Fixed,%d,%f,%f,%f,%d,%f\n", 
               i, solve_time, admm_time, rho_time, iters, rho);
        
        HAL_Delay(500);  // Similar delay as original
    }
    
    printf("Benchmark Complete!\r\n");
}
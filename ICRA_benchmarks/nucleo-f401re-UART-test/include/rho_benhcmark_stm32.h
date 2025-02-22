#pragma once
#include "main.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

// Keep same dimensions
#define BENCH_NX 12
#define BENCH_NU 4
#define BENCH_N 10

// Same structs but without Arduino dependency
struct RhoAdapter {
    float rho_base;
    float rho_min;
    float rho_max;
    float tolerance;
    bool clip;
};

struct RhoBenchmarkResult {
    uint32_t time_us;
    float initial_rho;
    float final_rho;
    float pri_res;
    float dual_res;
    float pri_norm;
    float dual_norm;
};

struct SolverStats {
    float avg_solve_time;
    float avg_admm_time;
    float avg_rho_time;
    float avg_iters;
    float std_solve_time;
    float std_iters;
    float solve_times[1000];
    float admm_times[1000];
    float rho_times[1000];
    float iterations[1000];
};

// Function declarations
void benchmark_rho_adaptation_stm32(
    const float* x_prev,
    const float* u_prev,
    const float* v_prev,
    float pri_res,
    float dual_res,
    RhoBenchmarkResult* result,
    RhoAdapter* adapter
);

void run_benchmarks_stm32(void);
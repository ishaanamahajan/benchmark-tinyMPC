// rho_benchmark.hpp
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <Arduino.h>

// Dimensions
#define BENCH_NX 12
#define BENCH_NU 4

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
};

// Cache matrices declarations
extern float Kinf[BENCH_NU][BENCH_NX];
extern float Pinf[BENCH_NX][BENCH_NX];
extern float C1[BENCH_NU][BENCH_NU];
extern float C2[BENCH_NX][BENCH_NX];

// External dependencies
extern const float A_stacked[BENCH_NX + BENCH_NU][BENCH_NX + BENCH_NU];
extern const float q[BENCH_NX + BENCH_NU];
extern const float P[BENCH_NX + BENCH_NU][BENCH_NX + BENCH_NU];

// Function declarations
void initialize_benchmark_cache();
void update_cache_taylor(float new_rho, float old_rho);
void benchmark_rho_adaptation(
    const float* x_prev,
    const float* u_prev,
    const float* z_prev,
    float pri_res,
    float dual_res,
    RhoBenchmarkResult* result,
    RhoAdapter* adapter
);

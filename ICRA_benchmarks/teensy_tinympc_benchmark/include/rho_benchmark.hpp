// rho_benchmark.hpp
#pragma once
#include <cstdint>  // for uint32_t
#include <cstring>  // for memcpy
#include <cmath>    // for sqrt, abs
#include <algorithm> // for min, max
#include <Arduino.h>

// Dimensions
#define BENCH_NX 12
#define BENCH_NU 4

// Result structure
struct RhoBenchmarkResult {
    uint32_t time_us;
    float initial_rho;
    float final_rho;
};

// Cache matrices declarations (defined as static in cpp)
extern float Kinf[BENCH_NU][BENCH_NX];
extern float Pinf[BENCH_NX][BENCH_NX];
extern float C1[BENCH_NU][BENCH_NU];
extern float C2[BENCH_NX][BENCH_NX];

// Pre-computed sensitivity matrices declarations
extern const float dKinf_drho[BENCH_NU][BENCH_NX];
extern const float dPinf_drho[BENCH_NX][BENCH_NX];
extern const float dC1_drho[BENCH_NU][BENCH_NU];
extern const float dC2_drho[BENCH_NX][BENCH_NX];

// External dependencies needed by benchmark
extern float x_prev[BENCH_NX];
extern float u_prev[BENCH_NU];
extern float z_prev[BENCH_NX];
extern const float A_stacked[BENCH_NX + BENCH_NU][BENCH_NX + BENCH_NU];
extern const float q[BENCH_NX + BENCH_NU];

// Function declarations
void update_cache_taylor(float new_rho, float old_rho);
void initialize_benchmark_cache();  // This needs to be implemented
void benchmark_rho_adaptation(float pri_res, float dual_res, RhoBenchmarkResult* result);\





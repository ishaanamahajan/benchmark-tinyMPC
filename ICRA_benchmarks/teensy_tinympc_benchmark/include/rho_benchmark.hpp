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
#define BENCH_N 10  // Horizon length, adjust as needed

enum RhoMethod {
    OPTIMAL,    // Sqrt ratio method
    SIMPLE,     // Simple heuristic
};

struct RhoAdapter {
    float rho_base;
    float rho_min;
    float rho_max;
    float tolerance;
    bool clip;
    RhoMethod method;  // Added method selection
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

// Cache matrices declarations
extern float Kinf[BENCH_NU][BENCH_NX];
extern float Pinf[BENCH_NX][BENCH_NX];
extern float C1[BENCH_NU][BENCH_NU];
extern float C2[BENCH_NX][BENCH_NX];

// Derivative matrices for analytical method
extern const float dKinf_drho[BENCH_NU][BENCH_NX];
extern const float dPinf_drho[BENCH_NX][BENCH_NX];
extern const float dC1_drho[BENCH_NU][BENCH_NU];
extern const float dC2_drho[BENCH_NX][BENCH_NX];

// System matrices
extern const float A[BENCH_NX][BENCH_NX];
extern const float B[BENCH_NX][BENCH_NU];
extern const float Q[BENCH_NX][BENCH_NX];
extern const float R[BENCH_NU][BENCH_NU];

// Reference states and inputs
extern const float xg[BENCH_NX];
extern const float uhover[BENCH_NU];

// Function declarations
void initialize_benchmark_cache();

void format_matrices(
    const float* x_prev,
    const float* u_prev,
    const float* v_prev,
    const float* z_prev,
    const float* g_prev,
    const float* y_prev,
    float* A_out,
    float* z_out,
    float* y_out,
    float* P_out,
    float* q_out,
    int N
);

void compute_residuals(
    const float* x,
    const float* A,
    const float* z,
    const float* y,
    const float* P,
    const float* q,
    float& pri_res,
    float& dual_res,
    float& pri_norm,
    float& dual_norm
);

float predict_rho(
    float pri_res,
    float dual_res,
    float pri_norm,
    float dual_norm,
    float current_rho,
    const RhoAdapter* adapter
);

void update_matrices(
    float new_rho,
    float old_rho
);

void benchmark_rho_adaptation(
    const float* x_prev,
    const float* u_prev,
    const float* v_prev,
    float pri_res,
    float dual_res,
    RhoBenchmarkResult* result,
    RhoAdapter* adapter,
    float rho
);

void update_cache_taylor(float new_rho, float old_rho);
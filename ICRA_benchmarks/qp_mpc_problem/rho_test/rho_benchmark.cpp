#include "rho_benchmark.hpp"
#include <Arduino.h>

// Cache matrices that get updated
float Kinf[BENCH_NU][BENCH_NX];
float Pinf[BENCH_NX][BENCH_NX];
float C1[BENCH_NU][BENCH_NU];
float C2[BENCH_NX][BENCH_NX];

void initialize_benchmark_cache() {
    memcpy(Kinf, KINF_INIT, sizeof(Kinf));
    memcpy(Pinf, PINF_INIT, sizeof(Pinf));
    memcpy(C1, C1_INIT, sizeof(C1));
    memcpy(C2, C2_INIT, sizeof(C2));
}




void update_cache_taylor(float new_rho, float old_rho) {
    float delta_rho = new_rho - old_rho;
    
    // Update K using Taylor expansion
    for(int i = 0; i < BENCH_NU; i++) {
        for(int j = 0; j < BENCH_NX; j++) {
            Kinf[i][j] += delta_rho * dKinf_drho[i][j];
        }
    }
    
    // Update P using Taylor expansion
    for(int i = 0; i < BENCH_NX; i++) {
        for(int j = 0; j < BENCH_NX; j++) {
            Pinf[i][j] += delta_rho * dPinf_drho[i][j];
        }
    }
    
    // Update C1 using Taylor expansion
    for(int i = 0; i < BENCH_NU; i++) {
        for(int j = 0; j < BENCH_NU; j++) {
            C1[i][j] += delta_rho * dC1_drho[i][j];
        }
    }
    
    // Update C2 using Taylor expansion
    for(int i = 0; i < BENCH_NX; i++) {
        for(int j = 0; j < BENCH_NX; j++) {
            C2[i][j] += delta_rho * dC2_drho[i][j];
        }
    }
}

// Add new helper function
float compute_max_norm(const float* vec, int size) {
    float max_val = 0.0f;
    for(int i = 0; i < size; i++) {
        max_val = max(max_val, abs(vec[i]));
    }
    return max_val;
}

// Add residual computation
void compute_residuals(
    const float* x,
    const float* A,
    const float* z,
    const float* y,
    const float* P,
    const float* q,
    int size,
    float* pri_res,
    float* dual_res,
    float* pri_norm,
    float* dual_norm
) {
    // Primal residual: r = Ax - z
    float* Ax = new float[size];
    matrix_multiply(A, x, Ax, size, size, 1);
    
    *pri_res = 0.0f;
    for(int i = 0; i < size; i++) {
        *pri_res = max(*pri_res, abs(Ax[i] - z[i]));
    }
    
    // Normalization terms
    *pri_norm = max(compute_max_norm(Ax, size), 
                   compute_max_norm(z, size));

    // Dual residual: r = Px + q + A'y
    float* Px = new float[size];
    float* ATy = new float[size];
    matrix_multiply(P, x, Px, size, size, 1);
    matrix_multiply_transpose(A, y, ATy, size, size, 1);
    
    *dual_res = 0.0f;
    for(int i = 0; i < size; i++) {
        *dual_res = max(*dual_res, abs(Px[i] + q[i] + ATy[i]));
    }
    
    // Normalization terms
    *dual_norm = max(max(compute_max_norm(Px, size),
                        compute_max_norm(ATy, size)),
                    compute_max_norm(q, size));
    
    delete[] Ax;
    delete[] Px;
    delete[] ATy;
}

// Update the main benchmark function
void benchmark_rho_adaptation(
    const float* x_prev,
    const float* u_prev,
    const float* z_prev,
    float pri_res,
    float dual_res,
    RhoBenchmarkResult* result,
    RhoAdapter* adapter
) {
    initialize_benchmark_cache();
    
    uint32_t start = micros();
    
    // Get current state
    float x_k[BENCH_NX];
    float u_k[BENCH_NU];
    float z_k[BENCH_NX];
    float y_k[BENCH_NX + BENCH_NU];
    
    memcpy(x_k, x_prev, BENCH_NX * sizeof(float));
    memcpy(u_k, u_prev, BENCH_NU * sizeof(float));
    memcpy(z_k, z_prev, BENCH_NX * sizeof(float));
    
    // Compute residuals and norms
    float pri_norm, dual_norm;
    compute_residuals(x_k, A_stacked, z_k, y_k, P, q, 
                     BENCH_NX + BENCH_NU,
                     &pri_res, &dual_res, &pri_norm, &dual_norm);
    
    // Adaptive rho logic
    float normalized_pri = pri_res / (pri_norm + 1e-10f);
    float normalized_dual = dual_res / (dual_norm + 1e-10f);
    float ratio = normalized_pri / (normalized_dual + 1e-10f);
    
    float new_rho = adapter->rho_base * sqrtf(ratio);
    
    if (adapter->clip) {
        new_rho = min(max(new_rho, adapter->rho_min), adapter->rho_max);
    }
    
    // Update cache using Taylor expansion
    update_cache_taylor(new_rho, adapter->rho_base);
    
    // Store results
    result->initial_rho = adapter->rho_base;
    result->final_rho = new_rho;
    result->pri_res = pri_res;
    result->dual_res = dual_res;
    result->time_us = micros() - start;
}
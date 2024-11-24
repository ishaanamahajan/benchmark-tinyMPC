#include "rho_benchmark.hpp"
//#include <Arduino.h>

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



void benchmark_rho_adaptation(float pri_res, float dual_res, RhoBenchmarkResult* result) {
    initialize_benchmark_cache();

    result->initial_rho = 85.0f;
    
    uint32_t start = micros();
    
    // Get state and input from previous iteration
    // Assuming x_prev, u_prev, z_prev, y_k are stored somewhere
    float x_k[BENCH_NX];  // state (12x1)
    float u_k[BENCH_NU];  // input (4x1)
    float z_k[BENCH_NX];  // slack (12x1)
    float y_k[BENCH_NX + BENCH_NU];  // [x;u] (16x1)

    memcpy(x_k, x_prev, BENCH_NX * sizeof(float));
    memcpy(u_k, u_prev, BENCH_NU * sizeof(float));
    memcpy(z_k, z_prev, BENCH_NX * sizeof(float));
    
    // Build y_k = [x_k; u_k]
    memcpy(y_k, x_k, BENCH_NX * sizeof(float));
    memcpy(y_k + BENCH_NX, u_k, BENCH_NU * sizeof(float));
    
    // Compute primal scaling
    float Ax_norm = 0.0f;
    // Compute A_stacked @ x_bar
    for(int i = 0; i < BENCH_NX + BENCH_NU; i++) {
        float sum = 0.0f;
        for(int j = 0; j < BENCH_NX + BENCH_NU; j++) {
            sum += A_stacked[i][j] * (j < BENCH_NX ? x_k[j] : u_k[j-BENCH_NX]);
        }
        Ax_norm = std::max(Ax_norm, std::abs(sum));
    }
    
    // Compute |z_k|_∞
    float z_norm = 0.0f;
    for(int i = 0; i < BENCH_NX; i++) {
        z_norm = std::max(z_norm, std::abs(z_k[i]));
    }
    
    float prim_scaling = pri_res / std::max(Ax_norm, z_norm);
    
    // Compute dual scaling
    float Px_norm = 0.0f;
    // Compute |Pinf @ x_k|_∞
    for(int i = 0; i < BENCH_NX; i++) {
        float sum = 0.0f;
        for(int j = 0; j < BENCH_NX; j++) {
            sum += Pinf[i][j] * x_k[j];
        }
        Px_norm = std::max(Px_norm, std::abs(sum));
    }
    
    float ATy_norm = 0.0f;
    // Compute |A_stacked.T @ y_k|_∞
    for(int i = 0; i < BENCH_NX + BENCH_NU; i++) {
        float sum = 0.0f;
        for(int j = 0; j < BENCH_NX + BENCH_NU; j++) {
            sum += A_stacked[j][i] * y_k[j];
        }
        ATy_norm = std::max(ATy_norm, std::abs(sum));
    }
    
    float q_norm = 0.0f;
    // Compute |q|_∞
    for(int i = 0; i < BENCH_NX + BENCH_NU; i++) {
        q_norm = std::max(q_norm, std::abs(q[i]));
    }
    
    float dual_scaling = dual_res / std::max(std::max(Px_norm, ATy_norm), q_norm);
    
    // Update rho
    float ratio = prim_scaling / dual_scaling;
    ratio = std::min(std::max(ratio, 0.001f), 1.0f);
    float new_rho = result->initial_rho * std::sqrt(ratio);
    new_rho = std::min(std::max(new_rho, 70.0f), 100.0f);
    
    // Update cache using Taylor expansion
    update_cache_taylor(new_rho, result->initial_rho);
    
    result->time_us = micros() - start;
    result->final_rho = new_rho;
}
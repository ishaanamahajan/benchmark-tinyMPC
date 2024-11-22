#include "rho_benchmark_recompute.hpp"
#include <Arduino.h>

// Cache matrices that get recomputed
static float Kinf[BENCH_NU][BENCH_NX];
static float Pinf[BENCH_NX][BENCH_NX];
static float C1[BENCH_NU][BENCH_NU];
static float C2[BENCH_NX][BENCH_NX];

// System matrices (from your generated problem)
const float A[BENCH_NX][BENCH_NX] = { /* Your A matrix */ };
const float B[BENCH_NX][BENCH_NU] = { /* Your B matrix */ };
const float Q[BENCH_NX][BENCH_NX] = { /* Your Q matrix */ };
const float R[BENCH_NU][BENCH_NU] = { /* Your R matrix */ };
const float A_stacked[BENCH_NX + BENCH_NU][BENCH_NX + BENCH_NU] = { /* Your A_stacked */ };
const float q[BENCH_NX + BENCH_NU] = { /* Your q vector */ };

// Helper function to recompute cache matrices
void recompute_cache(float rho) {
    // 1. Compute Q_rho and R_rho
    float Q_rho[BENCH_NX][BENCH_NX];
    float R_rho[BENCH_NU][BENCH_NU];
    
    // Q_rho = Q + rho * I
    for(int i = 0; i < BENCH_NX; i++) {
        for(int j = 0; j < BENCH_NX; j++) {
            Q_rho[i][j] = Q[i][j] + (i == j ? rho : 0.0f);
        }
    }
    
    // R_rho = R + rho * I
    for(int i = 0; i < BENCH_NU; i++) {
        for(int j = 0; j < BENCH_NU; j++) {
            R_rho[i][j] = R[i][j] + (i == j ? rho : 0.0f);
        }
    }
    
    // 2. Initialize Kinf and Pinf
    float Kinf_prev[BENCH_NU][BENCH_NX] = {0};  // For convergence check
    
    // Pinf starts as Q_rho
    for(int i = 0; i < BENCH_NX; i++) {
        for(int j = 0; j < BENCH_NX; j++) {
            Pinf[i][j] = Q_rho[i][j];
        }
    }
    
    // 3. Iterate until convergence (max 5000 iterations)
    for(int k = 0; k < 5000; k++) {
        // Store previous Kinf for convergence check
        for(int i = 0; i < BENCH_NU; i++) {
            for(int j = 0; j < BENCH_NX; j++) {
                Kinf_prev[i][j] = Kinf[i][j];
            }
        }
        
        // Compute BTPinfB (BENCH_NU x BENCH_NU)
        float BTPinfB[BENCH_NU][BENCH_NU] = {0};
        for(int i = 0; i < BENCH_NU; i++) {
            for(int j = 0; j < BENCH_NU; j++) {
                float sum = 0;
                for(int k = 0; k < BENCH_NX; k++) {
                    for(int l = 0; l < BENCH_NX; l++) {
                        sum += B[l][i] * Pinf[l][k] * B[k][j];
                    }
                }
                BTPinfB[i][j] = sum;
            }
        }
        
        // Compute BTPinfA (BENCH_NU x BENCH_NX)
        float BTPinfA[BENCH_NU][BENCH_NX] = {0};
        for(int i = 0; i < BENCH_NU; i++) {
            for(int j = 0; j < BENCH_NX; j++) {
                float sum = 0;
                for(int k = 0; k < BENCH_NX; k++) {
                    for(int l = 0; l < BENCH_NX; l++) {
                        sum += B[l][i] * Pinf[l][k] * A[k][j];
                    }
                }
                BTPinfA[i][j] = sum;
            }
        }
        
        // Compute Kinf = inv(R_rho + BTPinfB) * BTPinfA
        float temp[BENCH_NU][BENCH_NU];
        for(int i = 0; i < BENCH_NU; i++) {
            for(int j = 0; j < BENCH_NU; j++) {
                temp[i][j] = R_rho[i][j] + BTPinfB[i][j];
            }
        }
        // Need matrix inverse here
        float temp_inv[BENCH_NU][BENCH_NU];
        matrix_inverse(temp, temp_inv, BENCH_NU);
        
        // Multiply temp_inv * BTPinfA to get Kinf
        matrix_multiply(temp_inv, BTPinfA, Kinf, BENCH_NU, BENCH_NU, BENCH_NX);
        
        // Update Pinf = Q_rho + AT * Pinf * (A - B*Kinf)
        float BK[BENCH_NX][BENCH_NX] = {0};
        matrix_multiply(B, Kinf, BK, BENCH_NX, BENCH_NU, BENCH_NX);
        
        float AmBK[BENCH_NX][BENCH_NX];
        matrix_subtract(A, BK, AmBK, BENCH_NX, BENCH_NX);
        
        float new_Pinf[BENCH_NX][BENCH_NX];
        matrix_multiply(Pinf, AmBK, new_Pinf, BENCH_NX, BENCH_NX, BENCH_NX);
        matrix_multiply_transpose_left(A, new_Pinf, Pinf, BENCH_NX, BENCH_NX, BENCH_NX);
        matrix_add(Pinf, Q_rho, Pinf, BENCH_NX, BENCH_NX);
        
        // Check convergence
        float diff_norm = matrix_norm(Kinf, Kinf_prev, BENCH_NU, BENCH_NX);
        if(diff_norm < 1e-10) break;
    }
    
    // 4. Compute C1 = inv(R_rho + BTPinfB)
    float BTPinfB[BENCH_NU][BENCH_NU] = {0};
    compute_BTPinfB(B, Pinf, BTPinfB);
    matrix_add(R_rho, BTPinfB, temp, BENCH_NU, BENCH_NU);
    matrix_inverse(temp, C1, BENCH_NU);
    
    // 5. Compute C2 = (A - B*Kinf)T
    float BK[BENCH_NX][BENCH_NX] = {0};
    matrix_multiply(B, Kinf, BK, BENCH_NX, BENCH_NU, BENCH_NX);
    matrix_subtract(A, BK, C2, BENCH_NX, BENCH_NX);
    matrix_transpose(C2, C2, BENCH_NX, BENCH_NX);
}

void benchmark_rho_recompute(
    const float* x_prev,
    const float* u_prev,
    const float* z_prev,
    float pri_res,
    float dual_res,
    RhoBenchmarkResult* result
) {
    result->initial_rho = 85.0f;
    
    uint32_t start = micros();
    
    // Copy previous values
    float x_k[BENCH_NX];
    float u_k[BENCH_NU];
    float z_k[BENCH_NX];
    float y_k[BENCH_NX + BENCH_NU];
    
    memcpy(x_k, x_prev, BENCH_NX * sizeof(float));
    memcpy(u_k, u_prev, BENCH_NU * sizeof(float));
    memcpy(z_k, z_prev, BENCH_NX * sizeof(float));
    memcpy(y_k, x_k, BENCH_NX * sizeof(float));
    memcpy(y_k + BENCH_NX, u_k, BENCH_NU * sizeof(float));
    
    / Compute primal scaling
    float Ax_norm = 0.0f;
    // Compute A_stacked @ x_bar
    for(int i = 0; i < BENCH_NX + BENCH_NU; i++) {
        float sum = 0.0f;
        for(int j = 0; j < BENCH_NX + BENCH_NU; j++) {
            sum += A_stacked[i][j] * (j < BENCH_NX ? x_k[j] : u_k[j-BENCH_NX]);
        }
        Ax_norm = max(Ax_norm, abs(sum));
    }
    
    // Compute |z_k|_∞
    float z_norm = 0.0f;
    for(int i = 0; i < BENCH_NX; i++) {
        z_norm = max(z_norm, abs(z_k[i]));
    }
    
    float prim_scaling = pri_res / max(Ax_norm, z_norm);
    
    // Compute dual scaling
    float Px_norm = 0.0f;
    // Compute |Pinf @ x_k|_∞
    for(int i = 0; i < BENCH_NX; i++) {
        float sum = 0.0f;
        for(int j = 0; j < BENCH_NX; j++) {
            sum += Pinf[i][j] * x_k[j];
        }
        Px_norm = max(Px_norm, abs(sum));
    }
    
    float ATy_norm = 0.0f;
    // Compute |A_stacked.T @ y_k|_∞
    for(int i = 0; i < BENCH_NX + BENCH_NU; i++) {
        float sum = 0.0f;
        for(int j = 0; j < BENCH_NX + BENCH_NU; j++) {
            sum += A_stacked[j][i] * y_k[j];
        }
        ATy_norm = max(ATy_norm, abs(sum));
    }
    
    float q_norm = 0.0f;
    // Compute |q|_∞
    for(int i = 0; i < BENCH_NX + BENCH_NU; i++) {
        q_norm = max(q_norm, abs(q[i]));
    }
    
    float dual_scaling = dual_res / max(max(Px_norm, ATy_norm), q_norm);
    
    // Update rho
    float ratio = prim_scaling / dual_scaling;
    ratio = min(max(ratio, 0.001f), 1.0f);
    float new_rho = result->initial_rho * sqrt(ratio);
    new_rho = min(max(new_rho, 70.0f), 100.0f);
    
    // Recompute cache with new rho
    recompute_cache(new_rho);
    
    result->time_us = micros() - start;
    result->final_rho = new_rho;
}
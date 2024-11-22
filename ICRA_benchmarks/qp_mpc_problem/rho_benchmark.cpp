#include "rho_benchmark.hpp"
#include <Arduino.h>

// Cache matrices that get updated
static float Kinf[BENCH_NU][BENCH_NX];
static float Pinf[BENCH_NX][BENCH_NX];
static float C1[BENCH_NU][BENCH_NU];
static float C2[BENCH_NX][BENCH_NX];

// Pre-computed sensitivity matrices at rho = 85.0
const float dKinf_drho[BENCH_NU][BENCH_NX] = {
    { 0.0001, -0.0000, -0.0016,  0.0002,  0.0005,  0.0033,  0.0001, -0.0000, -0.0009,  0.0000,  0.0001,  0.0010},
    {-0.0001,  0.0000, -0.0016, -0.0001, -0.0004, -0.0033, -0.0001,  0.0000, -0.0009, -0.0000, -0.0001, -0.0010},
    { 0.0000, -0.0000, -0.0016,  0.0001,  0.0004,  0.0033,  0.0000, -0.0000, -0.0009,  0.0000,  0.0001,  0.0010},
    {-0.0001,  0.0000, -0.0016, -0.0002, -0.0004, -0.0033, -0.0000,  0.0000, -0.0009, -0.0000, -0.0001, -0.0010}
};

const float dPinf_drho[BENCH_NX][BENCH_NX] = {
    { 0.0636, -0.0079, -0.0000,  0.0408,  0.2425,  0.4183,  0.0425, -0.0059, -0.0000,  0.0056,  0.0294,  0.1505},
    {-0.0079,  0.0589,  0.0000, -0.1954, -0.0409, -0.1666, -0.0059,  0.0378,  0.0000, -0.0217, -0.0056, -0.0600},
    { 0.0000,  0.0000,  9.0348,  0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  6.1357,  0.0000, -0.0000,  0.0000},
    { 0.0408, -0.1954, -0.0000,  0.7039,  0.3142,  1.8467,  0.0357, -0.1284, -0.0000,  0.0834,  0.0506,  0.7094},
    { 0.2425, -0.0409,  0.0000,  0.3142,  1.2380,  4.6235,  0.1788, -0.0358, -0.0000,  0.0507,  0.1764,  1.7752},
    { 0.4183, -0.1666,  0.0000,  1.8467,  4.6235, 34.2096,  0.4407, -0.1758,  0.0000,  0.3224,  0.8063, 12.9370},
    { 0.0425, -0.0059, -0.0000,  0.0357,  0.1788,  0.4407,  0.0293, -0.0046, -0.0000,  0.0053,  0.0231,  0.1643},
    {-0.0059,  0.0378,  0.0000, -0.1284, -0.0358, -0.1758, -0.0046,  0.0244,  0.0000, -0.0145, -0.0053, -0.0656},
    {-0.0000,  0.0000,  6.1357, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  4.2496, -0.0000,  0.0000,  0.0000},
    { 0.0056, -0.0217,  0.0000,  0.0834,  0.0507,  0.3224,  0.0053, -0.0145, -0.0000,  0.0109,  0.0086,  0.1258},
    { 0.0294, -0.0056,  0.0000,  0.0506,  0.1764,  0.8063,  0.0231, -0.0053,  0.0000,  0.0086,  0.0274,  0.3145},
    { 0.1505, -0.0600,  0.0000,  0.7094,  1.7752, 12.9370,  0.1643, -0.0656,  0.0000,  0.1258,  0.3145,  5.0369}
};

// dC1_drho is all zeros
const float dC1_drho[BENCH_NU][BENCH_NU] = {
    {-0.0, -0.0, -0.0, -0.0},
    {-0.0, -0.0, -0.0, -0.0},
    {-0.0, -0.0, -0.0, -0.0},
    {-0.0, -0.0, -0.0, -0.0}
};

const float dC2_drho[BENCH_NX][BENCH_NX] = {
    { 0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000},
    {-0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000},
    { 0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000},
    { 0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000},
    { 0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000},
    { 0.0000, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000},
    { 0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000},
    {-0.0000,  0.0000,  0.0000, -0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000,  0.0000},
    { 0.0000,  0.0000,  0.0005, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0003,  0.0000,  0.0000, -0.0000},
    { 0.0000, -0.0002,  0.0000,  0.0008,  0.0001, -0.0001,  0.0000, -0.0002,  0.0000,  0.0001,  0.0000, -0.0000},
    { 0.0002, -0.0000,  0.0000,  0.0001,  0.0007, -0.0002,  0.0002, -0.0000, -0.0000,  0.0000,  0.0001, -0.0001},
    { 0.0000, -0.0000, -0.0000,  0.0000,  0.0001,  0.0011,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000,  0.0003}
};

static bool cache_initialized = false;

void initialize_benchmark_cache() {
    if (!cache_initialized) {
        // Use recompute_cache to populate initial values at rho = 85.0
        recompute_cache(85.0f);
        cache_initialized = true;
    }
}



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
    
    // Update cache using Taylor expansion
    update_cache_taylor(new_rho, result->initial_rho);
    
    result->time_us = micros() - start;
    result->final_rho = new_rho;
}
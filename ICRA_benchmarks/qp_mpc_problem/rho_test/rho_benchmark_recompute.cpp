#include "rho_benchmark.hpp"
#include <Arduino.h>

// Cache matrices that get updated
static float Kinf[BENCH_NU][BENCH_NX];
static float Pinf[BENCH_NX][BENCH_NX];
static float C1[BENCH_NU][BENCH_NU];
static float C2[BENCH_NX][BENCH_NX];

void initialize_benchmark_cache() {
    memcpy(Kinf, KINF_INIT, sizeof(Kinf));
    memcpy(Pinf, PINF_INIT, sizeof(Pinf));
    memcpy(C1, C1_INIT, sizeof(C1));
    memcpy(C2, C2_INIT, sizeof(C2));
}


// Helper: Matrix multiplication C = A * B
void matrix_multiply(const float* A, const float* B, float* C, 
                    int A_rows, int A_cols, int B_cols) {
    for(int i = 0; i < A_rows; i++) {
        for(int j = 0; j < B_cols; j++) {
            C[i * B_cols + j] = 0;
            for(int k = 0; k < A_cols; k++) {
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}

// Helper: Matrix transpose
void matrix_transpose(const float* A, float* At, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            At[j * rows + i] = A[i * cols + j];
        }
    }
}

// Helper: Matrix addition C = A + B
void matrix_add(const float* A, const float* B, float* C, int rows, int cols) {
    for(int i = 0; i < rows * cols; i++) {
        C[i] = A[i] + B[i];
    }
}

// Helper: Matrix subtraction C = A - B
void matrix_subtract(const float* A, const float* B, float* C, int rows, int cols) {
    for(int i = 0; i < rows * cols; i++) {
        C[i] = A[i] - B[i];
    }
}

// Helper: Matrix inverse using Gaussian elimination
void matrix_inverse(const float* A, float* Ainv, int n) {
    // Create augmented matrix [A|I]
    float aug[n][2*n];
    float temp;
    
    // Initialize augmented matrix
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            aug[i][j] = A[i * n + j];
            aug[i][j+n] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Gaussian elimination
    for(int i = 0; i < n; i++) {
        // Find pivot
        temp = aug[i][i];
        for(int j = 0; j < 2*n; j++) {
            aug[i][j] /= temp;
        }
        
        // Eliminate column
        for(int j = 0; j < n; j++) {
            if(i != j) {
                temp = aug[j][i];
                for(int k = 0; k < 2*n; k++) {
                    aug[j][k] -= aug[i][k] * temp;
                }
            }
        }
    }
    
    // Extract inverse
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            Ainv[i * n + j] = aug[i][j+n];
        }
    }
}

void recompute_cache(float rho) {
    // Temporary matrices
    float Q_rho[BENCH_NX][BENCH_NX];
    float R_rho[BENCH_NU][BENCH_NU];
    float BT[BENCH_NU][BENCH_NX];
    float AT[BENCH_NX][BENCH_NX];
    float temp1[BENCH_NU][BENCH_NX];
    float temp2[BENCH_NU][BENCH_NU];
    float temp3[BENCH_NX][BENCH_NX];
    
    // 1. Add rho*I to Q and R
    memcpy(Q_rho, Q, sizeof(Q_rho));
    memcpy(R_rho, R, sizeof(R_rho));
    for(int i = 0; i < BENCH_NX; i++) Q_rho[i][i] += rho;
    for(int i = 0; i < BENCH_NU; i++) R_rho[i][i] += rho;
    
    // 2. Compute matrix transposes
    matrix_transpose((float*)B, (float*)BT, BENCH_NX, BENCH_NU);
    matrix_transpose((float*)A, (float*)AT, BENCH_NX, BENCH_NX);
    
    // 3. Initialize Pinf = Q_rho
    memcpy(Pinf, Q_rho, sizeof(Pinf));
    
    // 4. Iterative computation
    float Kinf_prev[BENCH_NU][BENCH_NX];
    float diff = 1.0f;
    int iter = 0;
    const int max_iter = 5000;
    const float tolerance = 1e-10f;
    
    while(diff > tolerance && iter < max_iter) {
        memcpy(Kinf_prev, Kinf, sizeof(Kinf));
        
        // Compute Kinf = inv(R_rho + B'PB)B'PA
        matrix_multiply((float*)BT, (float*)Pinf, (float*)temp1, BENCH_NU, BENCH_NX, BENCH_NX);
        matrix_multiply((float*)temp1, (float*)B, (float*)temp2, BENCH_NU, BENCH_NX, BENCH_NU);
        matrix_add((float*)R_rho, (float*)temp2, (float*)temp2, BENCH_NU, BENCH_NU);
        matrix_multiply((float*)temp1, (float*)A, (float*)temp1, BENCH_NU, BENCH_NX, BENCH_NX);
        matrix_inverse((float*)temp2, (float*)temp2, BENCH_NU);
        matrix_multiply((float*)temp2, (float*)temp1, (float*)Kinf, BENCH_NU, BENCH_NU, BENCH_NX);
        
        // Compute Pinf = Q_rho + A'P(A - BK)
        matrix_multiply((float*)B, (float*)Kinf, (float*)temp3, BENCH_NX, BENCH_NU, BENCH_NX);
        matrix_subtract((float*)A, (float*)temp3, (float*)temp3, BENCH_NX, BENCH_NX);
        matrix_multiply((float*)AT, (float*)Pinf, (float*)temp1, BENCH_NX, BENCH_NX, BENCH_NX);
        matrix_multiply((float*)temp1, (float*)temp3, (float*)temp3, BENCH_NX, BENCH_NX, BENCH_NX);
        matrix_add((float*)Q_rho, (float*)temp3, (float*)Pinf, BENCH_NX, BENCH_NX);
        
        // Compute difference
        diff = 0;
        for(int i = 0; i < BENCH_NU * BENCH_NX; i++) {
            float d = ((float*)Kinf)[i] - ((float*)Kinf_prev)[i];
            diff += d*d;
        }
        diff = sqrt(diff);
        
        iter++;
    }
    
    // 5. Compute C1 = inv(R_rho + B'PB)
    matrix_multiply((float*)BT, (float*)Pinf, (float*)temp1, BENCH_NU, BENCH_NX, BENCH_NX);
    matrix_multiply((float*)temp1, (float*)B, (float*)temp2, BENCH_NU, BENCH_NX, BENCH_NU);
    matrix_add((float*)R_rho, (float*)temp2, (float*)temp2, BENCH_NU, BENCH_NU);
    matrix_inverse((float*)temp2, (float*)C1, BENCH_NU);
    
    // 6. Compute C2 = (A - BK)'
    matrix_multiply((float*)B, (float*)Kinf, (float*)temp3, BENCH_NX, BENCH_NU, BENCH_NX);
    matrix_subtract((float*)A, (float*)temp3, (float*)temp3, BENCH_NX, BENCH_NX);
    matrix_transpose((float*)temp3, (float*)C2, BENCH_NX, BENCH_NX);
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
    
    // Update cache using Full Recomputation
    recompute_cache(new_rho);
    
    result->time_us = micros() - start;
    result->final_rho = new_rho;
}
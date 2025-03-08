#include "rho_benchmark.hpp"
#include "mpc_matrices.hpp"
#include <Arduino.h>
#include "types.hpp"

// These need to be non-const to match header
float Kinf[BENCH_NU][BENCH_NX];
float Pinf[BENCH_NX][BENCH_NX];
float C1[BENCH_NU][BENCH_NU];
float C2[BENCH_NX][BENCH_NX];

// These derivatives will now be computed analytically
float dKinf_drho[BENCH_NU][BENCH_NX];
float dPinf_drho[BENCH_NX][BENCH_NX];
float dC1_drho[BENCH_NU][BENCH_NU];
float dC2_drho[BENCH_NX][BENCH_NX];

// Helper function for computing max norm directly
float compute_max_norm(const float* vec, int size) {
    float max_val = 0.0f;
    for(int i = 0; i < size; i++) {
        max_val = max(max_val, abs(vec[i]));
    }
    return max_val;
}

// New function to initialize benchmark matrices from tiny_params
void initialize_benchmark_matrices(struct tiny_params *params) {
    // Allocate arrays for derivatives
    float dKinf_drho_local[BENCH_NU * BENCH_NX];
    float dPinf_drho_local[BENCH_NX * BENCH_NX];
    float dC1_drho_local[BENCH_NU * BENCH_NU];
    float dC2_drho_local[BENCH_NX * BENCH_NX];
    
    // Compute sensitivity matrices analytically
    params->compute_sensitivity_matrices(
        dKinf_drho_local, 
        dPinf_drho_local, 
        dC1_drho_local, 
        dC2_drho_local
    );
    
    // Copy matrices to global arrays
    for (int i = 0; i < BENCH_NU; i++) {
        for (int j = 0; j < BENCH_NX; j++) {
            Kinf[i][j] = params->Kinf(i, j);
            dKinf_drho[i][j] = dKinf_drho_local[i * BENCH_NX + j];
        }
    }
    
    for (int i = 0; i < BENCH_NX; i++) {
        for (int j = 0; j < BENCH_NX; j++) {
            Pinf[i][j] = params->Pinf(i, j);
            dPinf_drho[i][j] = dPinf_drho_local[i * BENCH_NX + j];
        }
    }
    
    for (int i = 0; i < BENCH_NU; i++) {
        for (int j = 0; j < BENCH_NU; j++) {
            C1[i][j] = params->C1(i, j);
            dC1_drho[i][j] = dC1_drho_local[i * BENCH_NU + j];
        }
    }
    
    for (int i = 0; i < BENCH_NX; i++) {
        for (int j = 0; j < BENCH_NX; j++) {
            C2[i][j] = params->C2(i, j);
            dC2_drho[i][j] = dC2_drho_local[i * BENCH_NX + j];
        }
    }
}

void update_cache_taylor(float new_rho, float old_rho) {
    float delta_rho = new_rho - old_rho;
    
    // Update matrices using Taylor expansion
    for(int i = 0; i < BENCH_NU; i++) {
        for(int j = 0; j < BENCH_NX; j++) {
            Kinf[i][j] += delta_rho * dKinf_drho[i][j];
        }
    }
    
    for(int i = 0; i < BENCH_NX; i++) {
        for(int j = 0; j < BENCH_NX; j++) {
            Pinf[i][j] += delta_rho * dPinf_drho[i][j];
        }
    }
    
    for(int i = 0; i < BENCH_NU; i++) {
        for(int j = 0; j < BENCH_NU; j++) {
            C1[i][j] += delta_rho * dC1_drho[i][j];
        }
    }
    
    for(int i = 0; i < BENCH_NX; i++) {
        for(int j = 0; j < BENCH_NX; j++) {
            C2[i][j] += delta_rho * dC2_drho[i][j];
        }
    }
}

// This function computes the Ax product directly without building A_STACKED
void compute_Ax_direct(const float* x_k, const float* u_k, float* result, int nx, int nu) {
    // First nx elements: State dynamics
    for(int i = 0; i < nx; i++) {
        result[i] = 0.0f;
        for(int j = 0; j < nx; j++) {
            result[i] += C2[i][j] * x_k[j];
        }
        for(int j = 0; j < nu; j++) {
            result[i] -= Kinf[j][i] * u_k[j];  // Negative because Kinf applies to control
        }
    }
    
    // Next nu elements: Control inputs
    for(int i = 0; i < nu; i++) {
        result[nx + i] = u_k[i];  // Identity matrix for control inputs
    }
}

// This function computes Px directly without building P_DATA
void compute_Px_direct(const float* x_k, const float* u_k, float* result, int nx, int nu) {
    // First nx elements: State cost
    for(int i = 0; i < nx; i++) {
        result[i] = 0.0f;
        for(int j = 0; j < nx; j++) {
            result[i] += Pinf[i][j] * x_k[j];
        }
    }
    
    // Next nu elements: Control cost
    for(int i = 0; i < nu; i++) {
        result[nx + i] = 0.0f;
        for(int j = 0; j < nu; j++) {
            result[nx + i] += C1[i][j] * u_k[j];
        }
    }
}

// This function computes A^T y directly without building A_STACKED
void compute_ATy_direct(const float* x_k, const float* u_k, float* result, int nx, int nu) {
    // Transpose of the dynamics part
    for(int i = 0; i < nx; i++) {
        result[i] = 0.0f;
        for(int j = 0; j < nx; j++) {
            result[i] += C2[j][i] * x_k[j];  // Transpose by swapping indices
        }
    }
    
    // Transpose of the input part
    for(int i = 0; i < nu; i++) {
        result[nx + i] = 0.0f;
        for(int j = 0; j < nx; j++) {
            result[nx + i] -= Kinf[i][j] * x_k[j];  // Transpose
        }
        result[nx + i] += u_k[i];  // Identity part
    }
}

// Direct computation of cost vector q without storing it
float compute_q_norm_direct(const float* x_k, const float* u_k, const float* x_ref, const float* u_ref, int nx, int nu) {
    float max_q = 0.0f;
    
    // State cost elements
    float temp_x[nx];
    for(int i = 0; i < nx; i++) {
        temp_x[i] = x_k[i] - x_ref[i];
    }
    
    for(int i = 0; i < nx; i++) {
        float q_i = 0.0f;
        for(int j = 0; j < nx; j++) {
            q_i += Pinf[i][j] * temp_x[j];
        }
        max_q = max(max_q, abs(q_i));
    }
    
    // Control cost elements
    float temp_u[nu];
    for(int i = 0; i < nu; i++) {
        temp_u[i] = u_k[i] - u_ref[i];
    }
    
    for(int i = 0; i < nu; i++) {
        float q_i = 0.0f;
        for(int j = 0; j < nu; j++) {
            q_i += C1[i][j] * temp_u[j];
        }
        max_q = max(max_q, abs(q_i));
    }
    
    return max_q;
}

void benchmark_rho_adaptation(
    const float* x_prev,
    const float* u_prev,
    const float* v_prev,
    float pri_res,
    float dual_res,
    RhoBenchmarkResult* result,
    RhoAdapter* adapter,
    float current_rho
) {
    // Start timing
    uint32_t start = micros();
    
    // Copy input state
    float x_k[BENCH_NX];
    float u_k[BENCH_NU];
    float z_k[BENCH_NX + BENCH_NU]; 
    
    // Pre-allocate buffers on stack
    float temp_Ax[BENCH_NX + BENCH_NU]; 
    float temp_Px[BENCH_NX + BENCH_NU];
    float temp_ATy[BENCH_NX + BENCH_NU];
    
    // Copy input data
    memcpy(x_k, x_prev, BENCH_NX * sizeof(float));
    memcpy(u_k, u_prev, BENCH_NU * sizeof(float));
    
    // Fill z_k with slack variables
    memcpy(z_k, v_prev, BENCH_NX * sizeof(float));
    for(int i = 0; i < BENCH_NU; i++) {
        z_k[BENCH_NX + i] = u_k[i];
    }
    
    // Directly compute matrices
    compute_Ax_direct(x_k, u_k, temp_Ax, BENCH_NX, BENCH_NU);
    float Ax_norm = compute_max_norm(temp_Ax, BENCH_NX + BENCH_NU);
    
    // Compute slack norm
    float z_norm = compute_max_norm(z_k, BENCH_NX + BENCH_NU);
    
    // Compute P*x directly
    compute_Px_direct(x_k, u_k, temp_Px, BENCH_NX, BENCH_NU);
    float Px_norm = compute_max_norm(temp_Px, BENCH_NX + BENCH_NU);
    
    // Compute A^T*y directly
    compute_ATy_direct(x_k, u_k, temp_ATy, BENCH_NX, BENCH_NU);
    float ATy_norm = compute_max_norm(temp_ATy, BENCH_NX + BENCH_NU);
    
    // Use zero reference for simplicity - this is just for normalization
    float zero_ref[BENCH_NX] = {0};
    float zero_u_ref[BENCH_NU] = {0};
    float q_norm = compute_q_norm_direct(x_k, u_k, zero_ref, zero_u_ref, BENCH_NX, BENCH_NU);
    
    // Compute scalings with epsilon
    const float eps = 1e-10f;
    float pri_norm = max(Ax_norm, z_norm);
    float dual_norm = max(max(Px_norm, ATy_norm), q_norm);

    float normalized_pri = pri_res / (pri_norm + eps);
    float normalized_dual = dual_res / (dual_norm + eps);
    float ratio = normalized_pri / (normalized_dual + eps);

    // Update rho using same formula as Python
    float new_rho = current_rho * sqrt(ratio);

    // Apply clipping if enabled
    if (adapter->clip) {
        new_rho = min(max(new_rho, adapter->rho_min), adapter->rho_max);
    }
    
    // Store results
    result->time_us = micros() - start;
    result->initial_rho = current_rho;
    result->final_rho = new_rho;
    result->pri_res = pri_res;
    result->dual_res = dual_res;
    result->pri_norm = pri_norm;
    result->dual_norm = dual_norm;
}
#include "rho_benchmark.hpp"
#include "mpc_matrices.hpp"
#include <Arduino.h>

// These need to be non-const to match header
float Kinf[BENCH_NU][BENCH_NX];
float Pinf[BENCH_NX][BENCH_NX];
float C1[BENCH_NU][BENCH_NU];
float C2[BENCH_NX][BENCH_NX];

// These initialization arrays can stay const
float PINF_INIT[12][12] = {
    {74092.187704f, -73.167319f, 0.000000f, 198.053342f, 78082.549386f, 1717.656003f, 26348.957464f, -44.979013f, 0.000000f, 7.184539f, 399.820082f, 213.830097f},
    {-73.167319f, 73964.909547f, -0.000000f, -77730.404405f, -197.999696f, -686.738633f, -44.969404f, 26269.155532f, -0.000000f, -387.712050f, -7.186499f, -85.452259f},
    {0.000000f, -0.000000f, 44963.622514f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.000000f, 6235.729084f, 0.000000f, 0.000000f, 0.000000f},
    {198.053342f, -77730.404405f, 0.000000f, 295376.035129f, 910.998615f, 4979.087349f, 149.190105f, -54213.403856f, 0.000000f, 1507.510149f, 49.618732f, 687.275807f},
    {78082.549386f, -197.999696f, 0.000000f, 910.998615f, 297086.229647f, 12449.458845f, 54486.369572f, -149.178137f, 0.000000f, 49.618853f, 1600.826569f, 1718.550470f},
    {1717.656003f, -686.738633f, 0.000000f, 4979.087349f, 12449.458845f, 249846.316019f, 1575.992749f, -630.228271f, 0.000000f, 348.342573f, 870.953406f, 14433.220501f},
    {26348.957464f, -44.969404f, 0.000000f, 149.190105f, 54486.369572f, 1575.992749f, 15386.757218f, -30.020107f, 0.000000f, 6.160320f, 283.309542f, 196.304757f},
    {-44.979013f, 26269.155532f, -0.000000f, -54213.403856f, -149.178137f, -630.228271f, -30.020107f, 15332.380946f, -0.000000f, -272.368631f, -6.160959f, -78.483363f},
    {0.000000f, -0.000000f, 6235.729084f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.000000f, 3312.994426f, 0.000000f, 0.000000f, 0.000000f},
    {7.184539f, -387.712050f, -0.000000f, 1507.510149f, 49.618853f, 348.342573f, 6.160320f, -272.368631f, 0.000000f, 185.041879f, 6.126061f, 92.499883f},
    {399.820082f, -7.186499f, 0.000000f, 49.618732f, 1600.826569f, 870.953406f, 283.309542f, -6.160959f, 0.000000f, 6.126061f, 197.178200f, 231.262934f},
    {213.830097f, -85.452259f, 0.000000f, 687.275807f, 1718.550470f, 14433.220501f, 196.304757f, -78.483363f, 0.000000f, 92.499883f, 231.262934f, 3894.405009f},
};
float C1_INIT[4][4] = {
    {0.001228f, 0.000008f, 0.001186f, 0.000009f},
    {0.000008f, 0.001223f, 0.000011f, 0.001189f},
    {0.001186f, 0.000011f, 0.001222f, 0.000013f},
    {0.000009f, 0.001189f, 0.000013f, 0.001220f},
};
float C2_INIT[12][12] = {
    {0.999988f, -0.000000f, 0.000000f, 0.000021f, -0.019007f, 0.001094f, -0.002486f, -0.000003f, 0.000000f, 0.004264f, -3.801469f, 0.218865f},
    {-0.000000f, 0.999988f, -0.000000f, 0.019009f, -0.000021f, -0.000437f, -0.000003f, -0.002486f, -0.000000f, 3.801852f, -0.004220f, -0.087345f},
    {-0.000000f, -0.000000f, 0.995404f, 0.000000f, -0.000000f, 0.000000f, -0.000000f, -0.000000f, -0.459639f, 0.000000f, -0.000000f, 0.000000f},
    {0.000000f, -0.003873f, 0.000000f, 0.922624f, 0.000081f, 0.001660f, 0.000011f, -0.382279f, 0.000000f, -15.475193f, 0.016263f, 0.331965f},
    {0.003873f, -0.000000f, 0.000000f, 0.000082f, 0.922605f, 0.004161f, 0.382277f, -0.000011f, 0.000000f, 0.016384f, -15.478918f, 0.832192f},
    {0.000000f, -0.000000f, 0.000000f, 0.000245f, 0.000615f, 0.995117f, 0.000080f, -0.000032f, 0.000000f, 0.049073f, 0.122914f, -0.976627f},
    {0.019991f, -0.000000f, 0.000000f, 0.000015f, -0.013626f, 0.000769f, 0.998218f, -0.000002f, 0.000000f, 0.003037f, -2.725219f, 0.153866f},
    {-0.000000f, 0.019991f, -0.000000f, 0.013625f, -0.000015f, -0.000307f, -0.000002f, 0.998218f, -0.000000f, 2.724958f, -0.003010f, -0.061389f},
    {0.000000f, -0.000000f, 0.017587f, 0.000000f, 0.000000f, 0.000000f, -0.000000f, -0.000000f, 0.758742f, -0.000000f, -0.000000f, 0.000000f},
    {0.000000f, -0.000010f, 0.000000f, 0.004679f, 0.000007f, 0.000099f, 0.000001f, -0.001266f, 0.000000f, -0.064134f, 0.001439f, 0.019701f},
    {0.000010f, -0.000000f, 0.000000f, 0.000007f, 0.004682f, 0.000247f, 0.001266f, -0.000001f, 0.000000f, 0.001444f, -0.063588f, 0.049403f},
    {0.000000f, -0.000000f, 0.000000f, 0.000029f, 0.000073f, 0.008722f, 0.000010f, -0.000004f, 0.000000f, 0.005877f, 0.014697f, 0.744465f},
};
const float KINF_INIT[4][12] = {
    {-0.216731f, 0.190402f, 1.366579f, -0.848711f, -1.066524f, -3.009834f, -0.164837f, 0.140262f, 0.717299f, -0.068030f, -0.097495f, -0.792505f},
    {0.201444f, 0.138892f, 1.366579f, -0.491749f, 1.003969f, 3.010949f, 0.153864f, 0.095762f, 0.717299f, -0.024156f, 0.093195f, 0.792117f},
    {0.126087f, -0.154187f, 1.366579f, 0.554307f, 0.330238f, -3.012637f, 0.080966f, -0.106738f, 0.717299f, 0.028459f, -0.001447f, -0.791015f},
    {-0.110799f, -0.175107f, 1.366579f, 0.786153f, -0.267683f, 3.011522f, -0.069994f, -0.129287f, 0.717299f, 0.063727f, 0.005747f, 0.791402f},
};

// These derivative arrays can stay const
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

// Helper functions
float compute_max_norm(const float* vec, int size) {
    float max_val = 0.0f;
    for(int i = 0; i < size; i++) {
        max_val = max(max_val, abs(vec[i]));
    }
    return max_val;
}

void matrix_multiply(const float* A, const float* x, float* result, int rows, int cols, int n) {
    for(int i = 0; i < rows; i++) {
        result[i] = 0;
        for(int j = 0; j < cols; j++) {
            result[i] += A[i * cols + j] * x[j];
        }
    }
}

void matrix_multiply_transpose(const float* A, const float* x, float* result, int rows, int cols, int n) {
    for(int i = 0; i < cols; i++) {
        result[i] = 0;
        for(int j = 0; j < rows; j++) {
            result[i] += A[j * cols + i] * x[j];
        }
    }
}

void initialize_benchmark_cache() {
    memcpy(Kinf, KINF_INIT, sizeof(Kinf));
    memcpy(Pinf, PINF_INIT, sizeof(Pinf));
    memcpy(C1, C1_INIT, sizeof(C1));
    memcpy(C2, C2_INIT, sizeof(C2));
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

void benchmark_rho_adaptation(
    const float* x_prev,
    const float* u_prev,
    const float* v_prev,
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
    memcpy(z_k, v_prev, BENCH_NX * sizeof(float));
    
    // Build y_k = [x_k; u_k]
    memcpy(y_k, x_k, BENCH_NX * sizeof(float));
    memcpy(y_k + BENCH_NX, u_k, BENCH_NU * sizeof(float));
    
    // Compute norms for scaling
    float Ax_norm = 0.0f;
    float z_norm = compute_max_norm(z_k, BENCH_NX);
    float Px_norm = 0.0f;
    float ATy_norm = 0.0f;
    float q_norm = compute_max_norm(Q_DATA, BENCH_NX + BENCH_NU);
    
    // Compute A_stacked @ x_k
    float* temp = new float[BENCH_NX + BENCH_NU];
    matrix_multiply((float*)A_STACKED_DATA, y_k, temp, BENCH_NX + BENCH_NU, BENCH_NX + BENCH_NU, 1);
    Ax_norm = compute_max_norm(temp, BENCH_NX + BENCH_NU);
    
    // Compute P @ x_k
    matrix_multiply((float*)P_DATA, y_k, temp, BENCH_NX + BENCH_NU, BENCH_NX + BENCH_NU, 1);
    Px_norm = compute_max_norm(temp, BENCH_NX + BENCH_NU);
    
    // Compute A_stacked.T @ y_k
    matrix_multiply_transpose((float*)A_STACKED_DATA, y_k, temp, BENCH_NX + BENCH_NU, BENCH_NX + BENCH_NU, 1);
    ATy_norm = compute_max_norm(temp, BENCH_NX + BENCH_NU);
    
    delete[] temp;
    
    // Compute scalings with epsilon
    const float eps = 1e-10f;
    float pri_norm = max(Ax_norm, z_norm);
    float dual_norm = max(max(Px_norm, ATy_norm), q_norm);

    float new_rho;
    if (adapter->method == SIMPLE) {
        // Simple heuristic based on ratio
        float ratio = pri_res / (dual_res + 1e-8f);
        
        if (ratio > 3.0f) {  // Primal residual much larger
            new_rho = adapter->rho_base * 1.1f;
        } else if (ratio < 0.33f) {  // Dual residual much larger
            new_rho = adapter->rho_base * 0.9f;
        } else {
            new_rho = adapter->rho_base;
        }
    } else {  // OPTIMAL method
        float normalized_pri = pri_res / (pri_norm + eps);
        float normalized_dual = dual_res / (dual_norm + eps);
        float ratio = normalized_pri / (normalized_dual + eps);
        new_rho = adapter->rho_base * sqrt(ratio);
    }

    // Apply clipping if enabled
    if (adapter->clip) {
        new_rho = min(max(new_rho, adapter->rho_min), adapter->rho_max);
    }
    
    // Update cache using Taylor expansion
    update_cache_taylor(new_rho, adapter->rho_base);
    
    // Store results
    result->time_us = micros() - start;
    result->initial_rho = adapter->rho_base;
    result->final_rho = new_rho;
    result->pri_res = pri_res;
    result->dual_res = dual_res;
    result->pri_norm = pri_norm;
    result->dual_norm = dual_norm;
}
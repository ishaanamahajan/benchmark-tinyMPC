#pragma once

#include <Eigen.h>
#include <Eigen/LU>
#include "constants.hpp"
#include "Arduino.h"
#include "rho_benchmark.hpp"

using Eigen::Matrix;

// Move typedefs outside extern "C"
typedef float tinytype;

typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;
typedef Matrix<tinytype, NINPUTS, 1> tiny_VectorNu;
typedef Matrix<tinytype, NSTATE_CONSTRAINTS, 1> tiny_VectorNc;
typedef Matrix<tinytype, NSTATES, NSTATES> tiny_MatrixNxNx;
typedef Matrix<tinytype, NSTATES, NINPUTS> tiny_MatrixNxNu;
typedef Matrix<tinytype, NINPUTS, NSTATES> tiny_MatrixNuNx;
typedef Matrix<tinytype, NINPUTS, NINPUTS> tiny_MatrixNuNu;
typedef Matrix<tinytype, NSTATE_CONSTRAINTS, NSTATES> tiny_MatrixNcNx;
typedef Matrix<tinytype, NSTATES, NHORIZON, Eigen::ColMajor> tiny_MatrixNxNh;
typedef Matrix<tinytype, NINPUTS, NHORIZON-1, Eigen::ColMajor> tiny_MatrixNuNhm1;

struct tiny_params {
    static constexpr float DEFAULT_PRI_TOL = 1e-2f;  // Match Python's default
    static constexpr float DEFAULT_DUA_TOL = 1e-2f;
    static constexpr int DEFAULT_MAX_ITER = 500;
    
    // System matrices (loaded from rand_prob_tinympc_params.hpp)
    tiny_MatrixNxNx A;            
    tiny_MatrixNxNu B;            
    
    // Cost matrices (single matrices, not arrays)
    tiny_MatrixNxNx Q;            
    tiny_MatrixNuNu R;            
    
    // Reference trajectories
    tiny_MatrixNxNh Xref;         
    tiny_MatrixNuNhm1 Uref;       
    
    // Bounds (loaded from rand_prob_tinympc_params.hpp)
    tiny_VectorNu u_min;
    tiny_VectorNu u_max;
    tiny_VectorNx x_min;
    tiny_VectorNx x_max;
    
    // Cache terms (loaded from rand_prob_tinympc_params.hpp)
    float rho;                    // Single fixed rho value
    tiny_MatrixNuNx Kinf;         // Single feedback gain
    tiny_MatrixNxNx Pinf;         // Single terminal cost
    tiny_MatrixNuNu C1;           // Single Quu_inv
    tiny_MatrixNxNx C2;           // Single AmBKt
    
    // Solver parameters
    float abs_pri_tol;
    float abs_dua_tol;
    int max_iter;

    tiny_params() : 
        rho(rho_value),           // Use the value from rand_prob_tinympc_params.hpp
        abs_pri_tol(DEFAULT_PRI_TOL),
        abs_dua_tol(DEFAULT_DUA_TOL),
        max_iter(DEFAULT_MAX_ITER)
    {
        Serial.println("=== tiny_params constructor running! ===");
        
        // Load matrices from the header file
        load_matrices_from_header();
        
        // Initialize reference trajectories
        Xref.setZero();
        Uref.setZero();
    }

private:
    void load_matrices_from_header() {
        // Load system matrices
        memcpy(A.data(), Adyn_data, sizeof(Adyn_data));
        memcpy(B.data(), Bdyn_data, sizeof(Bdyn_data));
        
        // Load cost matrices as diagonal matrices
        Q = tiny_MatrixNxNx::Zero();
        R = tiny_MatrixNuNu::Zero();
        for(int i = 0; i < NSTATES; i++) Q(i,i) = Q_data[i];
        for(int i = 0; i < NINPUTS; i++) R(i,i) = R_data[i];
        
        // Load cache terms
        memcpy(Kinf.data(), Kinf_data, sizeof(Kinf_data));
        memcpy(Pinf.data(), Pinf_data, sizeof(Pinf_data));
        memcpy(C1.data(), Quu_inv_data, sizeof(Quu_inv_data));
        memcpy(C2.data(), AmBKt_data, sizeof(AmBKt_data));
        
        // Load bounds
        for(int i = 0; i < NINPUTS; i++) {
            u_min(i) = umin[i];
            u_max(i) = umax[i];
        }
        for(int i = 0; i < NSTATES; i++) {
            x_min(i) = xmin[i];
            x_max(i) = xmax[i];
        }
    }
};

struct tiny_problem {
    // Problem state and inputs
    tiny_MatrixNxNh x;            // State trajectory
    tiny_MatrixNuNhm1 u;          // Input trajectory
    tiny_MatrixNuNhm1 d;          // Backward pass gradient terms
    
    // ADMM variables
    tiny_MatrixNxNh v;            // State slack variables
    tiny_MatrixNuNhm1 z;          // Input slack variables
    tiny_MatrixNxNh vnew;         // New state slack variables
    tiny_MatrixNuNhm1 znew;       // New input slack variables
    tiny_MatrixNxNh g;            // State dual variables
    tiny_MatrixNuNhm1 y;          // Input dual variables
    
    // Cost terms
    tiny_MatrixNxNh q;            // State cost terms
    tiny_MatrixNuNhm1 r;          // Input cost terms
    tiny_MatrixNxNh p;            // Terminal cost terms
    
    // Residuals
    float primal_residual_state;  // Primal residual for states
    float primal_residual_input;  // Primal residual for inputs
    float dual_residual_state;    // Dual residual for states
    float dual_residual_input;    // Dual residual for inputs
    
    // Solver status
    int status;                   // Solver status (0=not converged, 1=converged)
    int iter;                     // Current iteration count
    int max_iter;                 // Maximum iterations
    float abs_tol;               // Absolute tolerance
    
    // Timing
    uint32_t solve_time;          // Total solve time
    uint32_t admm_time;           // ADMM iteration time
    uint32_t rho_time;           // Rho adaptation time
    int solve_count = 0;  // Add this line

    tiny_problem() :
        primal_residual_state(0),
        primal_residual_input(0),
        dual_residual_state(0),
        dual_residual_input(0),
        status(0),
        iter(0),
        max_iter(500),
        abs_tol(1e-3f),
        solve_time(0),
        admm_time(0),
        rho_time(0)
    {
        Serial.println("=== tiny_problem constructor running! ===");
        
        // Set non-zero initial state
        x.setZero();
        x.col(0) << 1.0f, 2.0f, 3.0f, 4.0f;  // Adjust size based on NSTATES
        
        // Initialize other matrices
        u.setRandom();  // Random initial inputs
        d.setZero();
        v.setZero();
        z.setZero();
        vnew.setZero();
        znew.setZero();
        g.setZero();
        y.setZero();
        q.setZero();
        r.setZero();
        p.setZero();
    }
};

#ifdef __cplusplus
extern "C" {
#endif

void multAdyn(tiny_VectorNx &Ax, const tiny_MatrixNxNx &A, const tiny_VectorNx &x);

#ifdef __cplusplus
}
#endif

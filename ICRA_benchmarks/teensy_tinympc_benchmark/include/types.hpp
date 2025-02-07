#pragma once

#include <Eigen.h>
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

// Define structs outside extern "C"
struct tiny_cache {
    tiny_cache() {
        rho[0] = 85.0f;
        rho[1] = 85.0f;
        
        // Zero initialize all matrices
        for (int i = 0; i < 2; i++) {
            Adyn[i].setZero();
            Bdyn[i].setZero();
            Kinf[i].setZero();
            Pinf[i].setZero();
            Quu_inv[i].setZero();
            AmBKt[i].setZero();
            coeff_d2p[i].setZero();
        }
    }

    tiny_MatrixNxNx Adyn[2];
    tiny_MatrixNxNu Bdyn[2];
    tinytype rho[2];
    tiny_MatrixNuNx Kinf[2];
    tiny_MatrixNxNx Pinf[2];
    tiny_MatrixNuNu Quu_inv[2];
    tiny_MatrixNxNx AmBKt[2];
    tiny_MatrixNxNu coeff_d2p[2];
};

struct tiny_params {
    static constexpr float DEFAULT_PRI_TOL = 1e-3f;
    static constexpr float DEFAULT_DUA_TOL = 1e-3f;
    static constexpr int DEFAULT_MAX_ITER = 500;
    
    tiny_params() : 
        abs_pri_tol(DEFAULT_PRI_TOL),
        abs_dua_tol(DEFAULT_DUA_TOL),
        max_iter(DEFAULT_MAX_ITER)
    {
        Serial.println("=== tiny_params constructor running! ===");
        
        // Zero initialize all matrices
        u_min.setZero();
        u_max.setZero();
        x_min.setZero();
        x_max.setZero();
        
        // Set non-zero references and costs
        Xref.setRandom();  // Random reference trajectory
        Uref.setRandom();  // Random control inputs

        for (int i = 0; i < 2; i++) {
            Q[i].setConstant(1.0f);  // Unit state cost
            Qf[i].setConstant(1.0f); // Unit terminal cost
            R[i].setConstant(0.1f);  // Small input cost
        }

        // Print key values after initialization
        Serial.print("Constructor values - pri_tol: ");
        Serial.print(abs_pri_tol, 8);
        Serial.print(", dua_tol: ");
        Serial.println(abs_dua_tol, 8);
        Serial.print("Xref norm: "); Serial.println(Xref.norm());
        Serial.print("Q[0] norm: "); Serial.println(Q[0].norm());

        for (int i = 0; i < NHORIZON; i++) {
            A_constraints[i].setZero();
        }
    }

    // Member variables
    tiny_VectorNx Q[2];           // State cost for each cache level
    tiny_VectorNx Qf[2];          // Terminal state cost for each cache level
    tiny_VectorNu R[2];           // Input cost for each cache level
    tiny_MatrixNuNhm1 u_min;      // Input lower bounds
    tiny_MatrixNuNhm1 u_max;      // Input upper bounds
    tiny_MatrixNxNh x_min;        // State lower bounds
    tiny_MatrixNxNh x_max;        // State upper bounds
    tiny_MatrixNcNx A_constraints[NHORIZON];  // State constraint matrices
    RhoAdapter rho_adapter;       // Rho adaptation parameters
    tiny_MatrixNxNh Xref;         // Reference state trajectory
    tiny_MatrixNuNhm1 Uref;       // Reference input trajectory
    tiny_cache cache;             // Cached matrices
    float abs_pri_tol;            // Absolute primal tolerance
    float abs_dua_tol;            // Absolute dual tolerance
    int max_iter;                 // Maximum iterations
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
    int cache_level;             // Current cache level (0 or 1)
    
    // Timing
    uint32_t solve_time;          // Total solve time
    uint32_t admm_time;           // ADMM iteration time
    uint32_t rho_time;           // Rho adaptation time

    tiny_problem() :
        primal_residual_state(0),
        primal_residual_input(0),
        dual_residual_state(0),
        dual_residual_input(0),
        status(0),
        iter(0),
        max_iter(500),
        abs_tol(1e-3f),
        cache_level(0),
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

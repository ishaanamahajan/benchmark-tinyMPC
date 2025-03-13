#pragma once

#include <Eigen.h>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include "constants.hpp"
#include "Arduino.h"
#include "rho_benchmark.hpp"
//#include "problem_data/rand_prob_tinympc_params.hpp"  // Contains all the matrix data
#include "problem_data/problem_61.hpp"
using Eigen::Matrix;

// Move typedefs outside extern "C"
typedef float tinytype;

typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;
typedef Matrix<tinytype, NINPUTS, 1> tiny_VectorNu;
typedef Matrix<tinytype, NSTATE_CONSTRAINTS, 1> tiny_VectorNc;
typedef Matrix<tinytype, NSTATES, NSTATES> tiny_MatrixNxN;
typedef Matrix<tinytype, NSTATES, NINPUTS> tiny_MatrixNxB;
typedef Matrix<tinytype, NINPUTS, NSTATES> tiny_MatrixNuN;
typedef Matrix<tinytype, NINPUTS, NINPUTS> tiny_MatrixNuNu;
typedef Matrix<tinytype, NSTATE_CONSTRAINTS, NSTATES> tiny_MatrixNcNx;
typedef Matrix<tinytype, NSTATES, NHORIZON, Eigen::ColMajor> tiny_MatrixNxNh;
typedef Matrix<tinytype, NINPUTS, NHORIZON-1, Eigen::ColMajor> tiny_MatrixNuNhm1;

// Add reference trajectory types
typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixXref;
typedef Matrix<tinytype, NINPUTS, NHORIZON-1> tiny_MatrixUref;

struct tiny_params {
    static constexpr float DEFAULT_PRI_TOL = 1e-2f;  // Match Python's default
    static constexpr float DEFAULT_DUA_TOL = 1e-2f;
    static constexpr int DEFAULT_MAX_ITER = 500;
    
    // Raw matrices with correct dimensions
    tiny_MatrixNxN A;    // NSTATES x NSTATES
    tiny_MatrixNxB B;    // NSTATES x NINPUTS
    tiny_MatrixNxN Q;    // NSTATES x NSTATES
    tiny_MatrixNuNu R;   // NINPUTS x NINPUTS
    
    // Reference trajectories
    tiny_MatrixXref Xref;
    tiny_MatrixUref Uref;
    
    // Input/state bounds
    tiny_MatrixXref x_min;
    tiny_MatrixXref x_max;
    tiny_MatrixUref u_min;
    tiny_MatrixUref u_max;
    
    // Computed cache terms (will be calculated)
    tiny_MatrixNuN Kinf; // NINPUTS x NSTATES
    tiny_MatrixNxN Pinf; // NSTATES x NSTATES
    tiny_MatrixNuNu C1;  // NINPUTS x NINPUTS
    tiny_MatrixNxN C2;   // NSTATES x NSTATES
    
    // Other parameters
    tinytype rho;
    tinytype abs_pri_tol;
    tinytype abs_dua_tol;
    int max_iter;

    // Constructor to initialize from raw data
    tiny_params() : 
        rho(85.0),
        abs_pri_tol(DEFAULT_PRI_TOL),
        abs_dua_tol(DEFAULT_DUA_TOL),
        max_iter(DEFAULT_MAX_ITER)
    {
        // Initialize matrices to zero first
        A.setZero();
        B.setZero();
        Q.setZero();
        R.setZero();

        // Load matrices with correct dimensions
        load_matrix_from_progmem(A, Adyn_data);
        load_matrix_from_progmem(B, Bdyn_data);
        load_diagonal_matrix_from_progmem(Q, Q_data);
        load_diagonal_matrix_from_progmem(R, R_data);
        
        // Initialize reference trajectories to zero
        Xref.setZero();
        Uref.setZero();
        
        // Initialize bounds
        x_min.setConstant(-10000.0);  // From rand_prob_tinympc_params.hpp
        x_max.setConstant(10000.0);
        u_min.setConstant(-3.0);
        u_max.setConstant(3.0);
        
        // Compute cache terms
        compute_cache_terms();
    }

    void compute_cache_terms() {

        // load_matrix_from_progmem(Kinf, Kinf_data);
        // load_matrix_from_progmem(Pinf, Pinf_data);
        // load_matrix_from_progmem(C1, Quu_inv_data);  // C1 is Quu_inv
        // load_matrix_from_progmem(C2, AmBKt_data); 


        // Add regularization
        tiny_MatrixNuNu R_rho = R + rho * tiny_MatrixNuNu::Identity();
        tiny_MatrixNxN Q_rho = Q + rho * tiny_MatrixNxN::Identity();

        // DLQR computation
        Kinf = tiny_MatrixNuN::Zero();
        Pinf = Q;

        for (int k = 0; k < 5000; k++) {
            tiny_MatrixNuN Kinf_prev = Kinf;
            
            tiny_MatrixNuNu BtPB = B.transpose() * Pinf * B;
            BtPB += 1e-8 * tiny_MatrixNuNu::Identity();
            
            tiny_MatrixNuN BtPA = B.transpose() * Pinf * A;
            
            Kinf = BtPB.ldlt().solve(BtPA);
            
            Pinf = Q_rho + A.transpose() * Pinf * (A - B * Kinf);

            if ((Kinf - Kinf_prev).norm() < 1e-10)
                break;
        }

        // Compute C1 and C2
        tiny_MatrixNuNu temp = R_rho + B.transpose() * Pinf * B;
        C1 = temp.ldlt().solve(tiny_MatrixNuNu::Identity());
        C2 = (A - B * Kinf).transpose();
    }

    void load_matrix_from_progmem(tiny_MatrixNxN &matrix, const tinytype *data) {
        for (int i = 0; i < NSTATES; i++) {
            for (int j = 0; j < NSTATES; j++) {
                matrix(i,j) = pgm_read_float(&data[i * NSTATES + j]);
            }
        }
    }

    void load_matrix_from_progmem(tiny_MatrixNxB &matrix, const tinytype *data) {
        for (int i = 0; i < NSTATES; i++) {
            for (int j = 0; j < NINPUTS; j++) {
                matrix(i,j) = pgm_read_float(&data[i * NINPUTS + j]);
            }
        }
    }

    void load_diagonal_matrix_from_progmem(tiny_MatrixNxN &matrix, const tinytype *data) {
        matrix.setZero();
        for (int i = 0; i < NSTATES; i++) {
            matrix(i,i) = pgm_read_float(&data[i]);
        }
    }

    void load_diagonal_matrix_from_progmem(tiny_MatrixNuNu &matrix, const tinytype *data) {
        matrix.setZero();
        for (int i = 0; i < NINPUTS; i++) {
            matrix(i,i) = pgm_read_float(&data[i]);
        }
    }

    void load_matrix_from_progmem(tiny_MatrixNuN &matrix, const tinytype *data) {
        for (int i = 0; i < NINPUTS; i++) {
            for (int j = 0; j < NSTATES; j++) {
                matrix(i,j) = pgm_read_float(&data[i * NSTATES + j]);
            }
        }
    }

    void load_matrix_from_progmem(tiny_MatrixNuNu &matrix, const tinytype *data) {
        for (int i = 0; i < NINPUTS; i++) {
            for (int j = 0; j < NINPUTS; j++) {
                matrix(i,j) = pgm_read_float(&data[i * NINPUTS + j]);
            }
        }
    }
};

struct SolverTimings {
    uint32_t init_time;
    uint32_t admm_time;
    uint32_t rho_time;
    uint32_t total_time;
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
    uint32_t init_time;  // Add this field
    int solve_count = 0;  // Add this line

    // Separate timing structs for each solver
    SolverTimings fixed_timings;
    SolverTimings adaptive_timings;

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
        rho_time(0),
        init_time(0)
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

void multAdyn(tiny_VectorNx &Ax, const tiny_MatrixNxN &A, const tiny_VectorNx &x);

#ifdef __cplusplus
}
#endif
// Rocket Landing Benchmark - No Serial Printing During Loop
// This version stores data in arrays and prints everything at the end
// to avoid serial buffer issues and get complete dataset

#include <iostream>
#include "src/admm.hpp"
#include "problem_data/rocket_landing_params_20hz.hpp"
#include "Arduino.h"

#define NRUNS (NTOTAL - NHORIZON - 1)

Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
Eigen::IOFormat SaveData(4, 0, ", ", "\n");

extern "C"
{
    // Global variables for TinyMPC objects
    TinyBounds bounds;
    TinySocs socs;
    TinyWorkspace work;
    TinyCache cache;
    TinySettings settings;
    
    // Data storage arrays - adjust size if needed
    float tracking_errors[300];
    int iterations[300];
    unsigned long solve_times[300];
    float controls_x[300], controls_y[300], controls_z[300];
    int completed_steps = 0;
    
    void setup()
    {
        Serial.begin(9600);
        delay(5000);
        
        Serial.println("Serial initialized");
        Serial.println("Start TinyMPC Rocket Landing - Batch Mode");
        Serial.println("==========================================");
        Serial.print("Expected steps: "); Serial.println(NRUNS);
        Serial.print("Horizon length: "); Serial.println(NHORIZON);
        Serial.println("Running benchmark...");
        
        // Initialize TinyMPC objects
        work.bounds = &bounds;
        work.socs = &socs;
        TinySolver solver{&settings, &cache, &work};
        
        /* Map data from problem_data (array in row-major order) */
        //////// Cache
        cache.rho = rho_value;
        cache.Kinf = Eigen::Map<Matrix<tinytype, NINPUTS, NSTATES, Eigen::RowMajor>>(Kinf_data);
        cache.Pinf = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Pinf_data);
        cache.Quu_inv = Eigen::Map<Matrix<tinytype, NINPUTS, NINPUTS, Eigen::RowMajor>>(Quu_inv_data);
        cache.AmBKt = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(AmBKt_data);
        cache.APf = Eigen::Map<Matrix<tinytype, NSTATES, 1>>(APf_data);
        cache.BPf = Eigen::Map<Matrix<tinytype, NINPUTS, 1>>(BPf_data);
        
        //////// Workspace (dynamics and LQR cost matrices)
        work.Adyn = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Adyn_data);
        work.Bdyn = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(Bdyn_data);
        work.fdyn = Eigen::Map<Matrix<tinytype, NSTATES, 1>>(fdyn_data);
        work.Q = Eigen::Map<tiny_VectorNx>(Q_data);
        work.R = Eigen::Map<tiny_VectorNu>(R_data);
        
        //////// Box constraints
        tiny_VectorNu u_min_one_time_step(-10.0, -10.0, -10.0);
        tiny_VectorNu u_max_one_time_step(105.0, 105.0, 105.0);
        work.bounds->u_min = u_min_one_time_step.replicate(1, NHORIZON-1);
        work.bounds->u_max = u_max_one_time_step.replicate(1, NHORIZON-1);
        tiny_VectorNx x_min_one_time_step(-5.0, -5.0, -0.5, -10.0, -10.0, -20.0);
        tiny_VectorNx x_max_one_time_step(5.0, 5.0, 100.0, 10.0, 10.0, 20.0);
        work.bounds->x_min = x_min_one_time_step.replicate(1, NHORIZON);
        work.bounds->x_max = x_max_one_time_step.replicate(1, NHORIZON);
        
        //////// Second order cone constraints
        work.socs->cu[0] = 0.25; // coefficients for input cones (mu)
        work.socs->cx[0] = 0.6; // coefficients for state cones (mu)
        work.socs->Acu[0] = 0; // start indices for input cones
        work.socs->Acx[0] = 0; // start indices for state cones
        work.socs->qcu[0] = 3; // dimensions for input cones
        work.socs->qcx[0] = 3; // dimensions for state cones
        
        //////// Settings
        settings.abs_pri_tol = 0.01;
        settings.abs_dua_tol = 0.01;
        settings.max_iter = 100;
        settings.check_termination = 1;
        settings.en_state_bound = 0;
        settings.en_input_bound = 1;
        settings.en_state_soc = 0;
        settings.en_input_soc = 1;
        
        //////// Initialize other workspace values automatically
        reset_problem(&solver);
        
        tiny_VectorNx x0, x1; // current and next simulation states
        tiny_VectorNx xinit, xg; // initial and goal states
        
        // Initial state
        xinit << 4, 2, 20, -3, 2, -4.5;
        xg << 0, 0, 0, 0, 0, 0.0;
        x0 = xinit*1.1;
        
        // Uref stays constant
        for (int i=0; i<NHORIZON-1; i++) {
            work.Uref.col(i)(2) = 10;
        }
        for (int i=0; i<NHORIZON; i++) {
            work.Xref.col(i) = xinit + (xg - xinit)*tinytype(i)/(NTOTAL-1);
        }
        work.p.col(NHORIZON-1) = -cache.Pinf*work.Xref.col(NHORIZON-1);
        
        tinytype tracking_error = 0;
        unsigned long total_start = micros();
        
        // MAIN BENCHMARK LOOP - NO SERIAL PRINTING
        for (int k = 0; k < NRUNS; ++k)
        {
            // Calculate tracking error
            tracking_error = (x0 - work.Xref.col(1)).norm();
            tracking_errors[k] = tracking_error;
            
            // 1. Update measurement
            work.x.col(0) = x0;
            
            // 2. Update reference
            for (int i=0; i<NHORIZON; i++) {
                work.Xref.col(i) = xinit + (xg - xinit)*tinytype(i+k)/(NTOTAL-1);
            }
            
            // 3. Solve MPC problem
            unsigned long start_solve = micros();
            tiny_solve(&solver);
            unsigned long end_solve = micros();
            
            // Store results
            iterations[k] = work.iter;
            solve_times[k] = end_solve - start_solve;
            controls_x[k] = work.u.col(0)(0);
            controls_y[k] = work.u.col(0)(1);
            controls_z[k] = work.u.col(0)(2);
            
            // 4. Simulate forward
            x1 = work.Adyn * x0 + work.Bdyn * work.u.col(0) + work.fdyn;
            x0 = x1;
            
            completed_steps = k + 1;
        }
        
        unsigned long total_end = micros();
        unsigned long total_time = total_end - total_start;
        
        // Print essential data for plotting - one line per step
        Serial.println("DATA_START");
        for (int i = 0; i < completed_steps; i++) {
            // Format: step,solve_time,iterations
            Serial.print(i); Serial.print(",");
            Serial.print(solve_times[i]); Serial.print(",");
            Serial.println(iterations[i]);
            
            // Add small delay every 25 steps to prevent buffer overflow
            if (i % 25 == 0 && i > 0) {
                delay(5);
            }
        }
        Serial.println("DATA_END");
        
        // Summary stats
        float avg_solve_time = 0;
        int avg_iterations = 0;
        for (int i = 0; i < completed_steps; i++) {
            avg_solve_time += solve_times[i];
            avg_iterations += iterations[i];
        }
        avg_solve_time /= completed_steps;
        avg_iterations /= completed_steps;
        
        Serial.print("STEPS:"); Serial.println(completed_steps);
        Serial.print("AVG_SOLVE:"); Serial.println(avg_solve_time, 1);
        Serial.print("AVG_ITER:"); Serial.println(avg_iterations);
    }
} /* extern "C" */

void loop()
{
    // Do nothing - all work done in setup()
}
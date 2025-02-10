#include <Arduino.h>
#undef F
#include "admm.hpp"
#include "problem_data/rand_prob_tinympc_params.hpp"
#include "types.hpp"

void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("Starting MPC Benchmark Test");
    Serial.println("Trial,SolveTime,ADMMTime,Iterations");
    
    // Initialize problem & params
    Serial.println("\n=== Creating initial objects ===");
    tiny_problem problem;
    tiny_params params;
    
    // Set ADMM tolerances to match Python
    params.abs_pri_tol = 1e-2f;
    params.abs_dua_tol = 1e-2f;
    params.max_iter = 500;
    
    // Load matrices from header file
    Serial.println("=== Loading Matrices ===");
    
    // Debug print after loading
    Serial.println("=== Matrix Loading Complete ===");
    Serial.print("A norm: "); Serial.println(params.A.norm());
    Serial.print("B norm: "); Serial.println(params.B.norm());
    Serial.print("Q norm: "); Serial.println(params.Q.norm());
    Serial.print("R norm: "); Serial.println(params.R.norm());
    Serial.print("Kinf norm: "); Serial.println(params.Kinf.norm());
    Serial.print("Pinf norm: "); Serial.println(params.Pinf.norm());
    Serial.print("rho: "); Serial.println(params.rho);
    
    // Print bounds
    Serial.println("Bounds:");
    Serial.print("u_min: "); Serial.println(params.u_min(0));
    Serial.print("u_max: "); Serial.println(params.u_max(0));
    Serial.print("x_min: "); Serial.println(params.x_min(0));
    Serial.print("x_max: "); Serial.println(params.x_max(0));
    
    const int NUM_TRIALS = 5;
    
    // Run trials with fixed rho only
    for(int i = 0; i < NUM_TRIALS; i++) {
        Serial.println("\n=== Starting Trial " + String(i) + " ===");
        
        // Reset problem variables
        problem.status = 0;
        problem.iter = 0;
        problem.solve_time = 0;
        problem.admm_time = 0;
        
        // Set non-zero initial state and inputs
        problem.x.setZero();
        problem.x.col(0) << 1.0f, 2.0f, 3.0f, 4.0f;  // Set initial state
        problem.u.setRandom();  // Random initial inputs
        
        // Set non-zero references
        params.Xref.setRandom();
        params.Uref.setRandom();
        
        // Debug print before solve
        Serial.print("Before solve - pri_tol: ");
        Serial.print(params.abs_pri_tol, 8);
        Serial.print(", dua_tol: ");
        Serial.println(params.abs_dua_tol, 8);
        
        solve_admm(&problem, &params);
        
        Serial.print(i);
        Serial.print(",");
        Serial.print(problem.solve_time);
        Serial.print(",");
        Serial.print(problem.admm_time);
        Serial.print(",");
        Serial.println(problem.iter);
        
        Serial.flush();
        delay(500);
    }
    
    Serial.println("Benchmark Complete!");
}

void loop() {
    // Empty
}  
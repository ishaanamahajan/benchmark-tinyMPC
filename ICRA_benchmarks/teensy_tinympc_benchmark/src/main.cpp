#include <Arduino.h>
#undef F
#include "admm.hpp"
#include "problem_data/rand_prob_tinympc_params.hpp"
#include "types.hpp"
#include "rho_benchmark.hpp"

void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("Starting MPC Benchmark Test");
    Serial.println("Method,Trial,SolveTime,ADMMTime,RhoTime,Iterations,FinalRho");
    
    // Initialize problem & params
    tiny_problem problem;
    tiny_params params;
    
    // Set ADMM tolerances
    params.abs_pri_tol = 1e-2f;
    params.abs_dua_tol = 1e-2f;
    params.max_iter = 500;
    
    // Initialize RhoAdapter
    RhoAdapter adapter;
    adapter.rho_base = 85.0f;
    adapter.rho_min = 70.0f;
    adapter.rho_max = 100.0f;
    adapter.tolerance = 1.1f;
    adapter.clip = true;
    
    const int NUM_TRIALS = 5;
    
    // First run trials with fixed rho
    for(int i = 0; i < NUM_TRIALS; i++) {
        Serial.println("\n=== Starting Fixed Rho Trial " + String(i) + " ===");
        
        // Reset problem
        problem.status = 0;
        problem.iter = 0;
        problem.solve_time = 0;
        problem.admm_time = 0;
        problem.rho_time = 0;
        
        // Set test conditions
        problem.x.setZero();
        problem.x.col(0) << 1.0f, 2.0f, 3.0f, 4.0f;
        problem.u.setRandom();
        params.Xref.setRandom();
        params.Uref.setRandom();
        
        solve_admm(&problem, &params);
        
        Serial.print("Fixed,");
        Serial.print(i);
        Serial.print(",");
        Serial.print(problem.solve_time);
        Serial.print(",");
        Serial.print(problem.admm_time);
        Serial.print(",");
        Serial.print(problem.rho_time);
        Serial.print(",");
        Serial.print(problem.iter);
        Serial.print(",");
        Serial.println(params.rho);
        
        delay(500);
    }
    
    // Then run trials with adaptive rho
    for(int i = 0; i < NUM_TRIALS; i++) {
        Serial.println("\n=== Starting Adaptive Rho Trial " + String(i) + " ===");
        
        // Reset problem
        problem.status = 0;
        problem.iter = 0;
        problem.solve_time = 0;
        problem.admm_time = 0;
        problem.rho_time = 0;
        
        // Use same test conditions as fixed version
        problem.x.setZero();
        problem.x.col(0) << 1.0f, 2.0f, 3.0f, 4.0f;
        problem.u.setRandom();
        params.Xref.setRandom();
        params.Uref.setRandom();
        
        // Reset rho to base value
        params.rho = adapter.rho_base;
        params.compute_cache_terms();
        
        solve_admm_adaptive(&problem, &params, &adapter);
        
        Serial.print("Adaptive,");
        Serial.print(i);
        Serial.print(",");
        Serial.print(problem.solve_time);
        Serial.print(",");
        Serial.print(problem.admm_time);
        Serial.print(",");
        Serial.print(problem.rho_time);
        Serial.print(",");
        Serial.print(problem.iter);
        Serial.print(",");
        Serial.println(params.rho);
        
        delay(500);
    }
    
    Serial.println("Benchmark Complete!");
}

void loop() {
    // Empty
}  
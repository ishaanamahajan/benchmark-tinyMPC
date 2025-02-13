#include <Arduino.h>
#undef F
#include "admm.hpp"
#include "problem_data/rand_prob_tinympc_params.hpp"
#include "types.hpp"
#include "rho_benchmark.hpp"

// Add struct for collecting stats
struct SolverStats {
    float avg_solve_time;
    float avg_admm_time;
    float avg_rho_time;
    float avg_iters;
    float std_solve_time;
    float std_iters;
    
    // Arrays to store raw data
    float solve_times[1000];
    float admm_times[1000];
    float rho_times[1000];
    float iterations[1000];
};

// Function to compute statistics
void compute_stats(SolverStats* stats, int num_trials) {
    // Compute averages
    float sum_solve = 0, sum_admm = 0, sum_rho = 0, sum_iters = 0;
    for(int i = 0; i < num_trials; i++) {
        sum_solve += stats->solve_times[i];
        sum_admm += stats->admm_times[i];
        sum_rho += stats->rho_times[i];
        sum_iters += stats->iterations[i];
    }
    
    stats->avg_solve_time = sum_solve / num_trials;
    stats->avg_admm_time = sum_admm / num_trials;
    stats->avg_rho_time = sum_rho / num_trials;
    stats->avg_iters = sum_iters / num_trials;
    
    // Compute standard deviations
    float sum_sq_solve = 0, sum_sq_iters = 0;
    for(int i = 0; i < num_trials; i++) {
        sum_sq_solve += pow(stats->solve_times[i] - stats->avg_solve_time, 2);
        sum_sq_iters += pow(stats->iterations[i] - stats->avg_iters, 2);
    }
    
    stats->std_solve_time = sqrt(sum_sq_solve / num_trials);
    stats->std_iters = sqrt(sum_sq_iters / num_trials);
}

void print_stats(const char* method, SolverStats* stats) {
    Serial.println("\n=== " + String(method) + " Statistics ===");
    Serial.println("Average solve time: " + String(stats->avg_solve_time) + " µs");
    Serial.println("Average ADMM time: " + String(stats->avg_admm_time) + " µs");
    Serial.println("Average rho time: " + String(stats->avg_rho_time) + " µs");
    Serial.println("Average iterations: " + String(stats->avg_iters));
    Serial.println("Std Dev solve time: " + String(stats->std_solve_time) + " µs");
    Serial.println("Std Dev iterations: " + String(stats->std_iters));
}

// Add hover constants
const float MASS = 0.035f;
const float G = 9.81f;
const float KT = 2.245365e-6f * 65535.0f;

void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("Starting MPC Benchmark Test");
    Serial.println("Method,Trial,SolveTime,InitTime,ADMMTime,RhoTime,Iterations,FinalRho");
    
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
    
    const int NUM_TRIALS = 1000;
    SolverStats fixed_stats = {0};
    SolverStats adaptive_stats = {0};
    
    // First do hover test
    Serial.println("\n=== Starting Hover Tests ===");
    
    // Setup hover conditions
    float hover_thrust = (MASS * G) / (4.0f * KT);
    
    problem.x.setZero();
    problem.x.col(0) << 0.2f, 0.2f, -0.2f,  // position offset
                        1.0f, 0.0f, 0.0f,    // roll offset
                        0.0f, 0.0f, 0.0f,    // zero velocity
                        0.0f, 0.0f, 0.0f;    // zero angular rates
    
    params.Xref.setZero();  // target hover state
    params.Uref.setZero();
    // Set hover thrust for all timesteps
    for(int k = 0; k < NHORIZON-1; k++) {
        params.Uref.col(k) << hover_thrust, hover_thrust, hover_thrust, hover_thrust;
    }
    
    // Test fixed rho for hover
    Serial.println("\n=== Hover with Fixed Rho ===");
    // Reset everything
    problem.status = 0;
    problem.iter = 0;
    problem.solve_count = 0;
    problem.y.setZero();
    problem.g.setZero();
    problem.v.setZero();
    problem.z.setZero();
    problem.vnew.setZero();
    problem.znew.setZero();
    solve_admm(&problem, &params);
    Serial.print("Fixed Hover,");
    Serial.print("-1,");
    Serial.print(problem.fixed_timings.total_time);
    Serial.print(",");
    Serial.print(problem.fixed_timings.init_time);
    Serial.print(",");
    Serial.print(problem.fixed_timings.admm_time);
    Serial.print(",0,");  // No rho time for fixed
    Serial.print(problem.iter);
    Serial.print(",");
    Serial.println(params.rho);
    
    // Test adaptive rho for hover
    Serial.println("\n=== Hover with Adaptive Rho ===");
    // Reset everything again
    problem.status = 0;
    problem.iter = 0;
    problem.solve_count = 0;
    problem.y.setZero();
    problem.g.setZero();
    problem.v.setZero();
    problem.z.setZero();
    problem.vnew.setZero();
    problem.znew.setZero();
    params.rho = adapter.rho_base;
    params.compute_cache_terms();
    solve_admm_adaptive(&problem, &params, &adapter);
    Serial.print("Adaptive Hover,");
    Serial.print("-1,");
    Serial.print(problem.adaptive_timings.total_time);
    Serial.print(",");
    Serial.print(problem.adaptive_timings.init_time);
    Serial.print(",");
    Serial.print(problem.adaptive_timings.admm_time);
    Serial.print(",");
    Serial.print(problem.adaptive_timings.rho_time);
    Serial.print(",");
    Serial.print(problem.iter);
    Serial.print(",");
    Serial.println(params.rho);
    
    delay(1000);
    
    // Then run trials with fixed rho
    for(int i = 0; i < NUM_TRIALS; i++) {
        // Maybe add a progress indicator every 100 trials
        if(i % 100 == 0) {
            Serial.print("Trial ");
            Serial.print(i);
            Serial.println(" of 1000");
        }
        
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
        Serial.print(problem.init_time);
        Serial.print(",");
        Serial.print(problem.admm_time);
        Serial.print(",");
        Serial.print(problem.rho_time);
        Serial.print(",");
        Serial.print(problem.iter);
        Serial.print(",");
        Serial.println(params.rho);
        
        // Store stats
        fixed_stats.solve_times[i] = problem.solve_time;
        fixed_stats.admm_times[i] = problem.admm_time;
        fixed_stats.rho_times[i] = problem.rho_time;
        fixed_stats.iterations[i] = problem.iter;
        
        delay(10);  // Maybe reduce delay between trials
    }
    
    // Then run trials with adaptive rho
    for(int i = 0; i < NUM_TRIALS; i++) {
        // Maybe add a progress indicator every 100 trials
        if(i % 100 == 0) {
            Serial.print("Trial ");
            Serial.print(i);
            Serial.println(" of 1000");
        }
        
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
        Serial.print(problem.init_time);
        Serial.print(",");
        Serial.print(problem.admm_time);
        Serial.print(",");
        Serial.print(problem.rho_time);
        Serial.print(",");
        Serial.print(problem.iter);
        Serial.print(",");
        Serial.println(params.rho);
        
        // Store stats
        adaptive_stats.solve_times[i] = problem.solve_time;
        adaptive_stats.admm_times[i] = problem.admm_time;
        adaptive_stats.rho_times[i] = problem.rho_time;
        adaptive_stats.iterations[i] = problem.iter;
        
        delay(10);  // Maybe reduce delay between trials
    }
    
    // Compute and print statistics
    compute_stats(&fixed_stats, NUM_TRIALS);
    compute_stats(&adaptive_stats, NUM_TRIALS);
    
    print_stats("Fixed Rho", &fixed_stats);
    print_stats("Adaptive Rho", &adaptive_stats);
    
    Serial.println("Benchmark Complete!");
}

void loop() {
    // Empty
}  
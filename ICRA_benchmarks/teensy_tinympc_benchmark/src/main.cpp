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
    delay(3000);
    
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
    
    const int NUM_TRIALS = 1000;
    SolverStats fixed_stats = {0};
    SolverStats optimal_stats = {0};
    SolverStats simple_stats = {0};
    
    // First run fixed rho trials
    for(int i = 0; i < NUM_TRIALS; i++) {
        problem.status = 0;
        problem.iter = 0;
        problem.fixed_timings.total_time = 0;
        problem.fixed_timings.admm_time = 0;
        problem.fixed_timings.rho_time = 0;
        
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
        Serial.print(problem.fixed_timings.total_time);
        Serial.print(",");
        Serial.print(problem.fixed_timings.admm_time);
        Serial.print(",");
        Serial.print(problem.fixed_timings.rho_time);
        Serial.print(",");
        Serial.print(problem.iter);
        Serial.print(",");
        Serial.println(params.rho);
        
        fixed_stats.solve_times[i] = problem.fixed_timings.total_time;
        fixed_stats.admm_times[i] = problem.fixed_timings.admm_time;
        fixed_stats.rho_times[i] = problem.fixed_timings.rho_time;
        fixed_stats.iterations[i] = problem.iter;
        
        delay(100);
    }
    
    // Run optimal method trials
    adapter.method = OPTIMAL;
    for(int i = 0; i < NUM_TRIALS; i++) {
        problem.status = 0;
        problem.iter = 0;
        problem.adaptive_timings.total_time = 0;
        problem.adaptive_timings.admm_time = 0;
        problem.adaptive_timings.rho_time = 0;
        
        problem.x.setZero();
        problem.x.col(0) << 1.0f, 2.0f, 3.0f, 4.0f;
        problem.u.setRandom();
        params.Xref.setRandom();
        params.Uref.setRandom();
        
        params.rho = adapter.rho_base;
        params.compute_cache_terms();
        
        solve_admm_adaptive(&problem, &params, &adapter);
        
        Serial.print("Optimal,");
        Serial.print(i);
        Serial.print(",");
        Serial.print(problem.adaptive_timings.total_time);
        Serial.print(",");
        Serial.print(problem.adaptive_timings.admm_time);
        Serial.print(",");
        Serial.print(problem.adaptive_timings.rho_time);
        Serial.print(",");
        Serial.print(problem.iter);
        Serial.print(",");
        Serial.println(params.rho);
        
        optimal_stats.solve_times[i] = problem.adaptive_timings.total_time;
        optimal_stats.admm_times[i] = problem.adaptive_timings.admm_time;
        optimal_stats.rho_times[i] = problem.adaptive_timings.rho_time;
        optimal_stats.iterations[i] = problem.iter;
        
        delay(100);
    }
    
    // Run simple method trials
    adapter.method = SIMPLE;
    for(int i = 0; i < NUM_TRIALS; i++) {
        problem.status = 0;
        problem.iter = 0;
        problem.adaptive_timings.total_time = 0;
        problem.adaptive_timings.admm_time = 0;
        problem.adaptive_timings.rho_time = 0;
        
        problem.x.setZero();
        problem.x.col(0) << 1.0f, 2.0f, 3.0f, 4.0f;
        problem.u.setRandom();
        params.Xref.setRandom();
        params.Uref.setRandom();
        
        params.rho = adapter.rho_base;
        params.compute_cache_terms();
        
        solve_admm_adaptive(&problem, &params, &adapter);
        
        Serial.print("Simple,");
        Serial.print(i);
        Serial.print(",");
        Serial.print(problem.adaptive_timings.total_time);
        Serial.print(",");
        Serial.print(problem.adaptive_timings.admm_time);
        Serial.print(",");
        Serial.print(problem.adaptive_timings.rho_time);
        Serial.print(",");
        Serial.print(problem.iter);
        Serial.print(",");
        Serial.println(params.rho);
        
        simple_stats.solve_times[i] = problem.adaptive_timings.total_time;
        simple_stats.admm_times[i] = problem.adaptive_timings.admm_time;
        simple_stats.rho_times[i] = problem.adaptive_timings.rho_time;
        simple_stats.iterations[i] = problem.iter;
        
        delay(100);
    }
    
    // Compute and print statistics
    compute_stats(&fixed_stats, NUM_TRIALS);
    compute_stats(&optimal_stats, NUM_TRIALS);
    compute_stats(&simple_stats, NUM_TRIALS);
    
    print_stats("Fixed Rho", &fixed_stats);
    print_stats("Optimal Rho", &optimal_stats);
    print_stats("Simple Rho", &simple_stats);
    
    Serial.println("DONE");
}

void loop() {
    // Empty
}  
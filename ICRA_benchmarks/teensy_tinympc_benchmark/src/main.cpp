#include <Arduino.h>
#undef F
#include "admm.hpp"
#include "problem_data/rand_prob_tinympc_params.hpp"
#include "types.hpp"
#include "rho_benchmark.hpp"

const int NUM_TRIALS = 100;  // Number of trials for each rho value

void setup() {
    Serial.begin(115200);
    delay(3000);
    
    // Initialize problem & params
    tiny_problem problem;
    tiny_params params;

    RhoAdapter adapter;
    adapter.rho_base = 100.0f;
    adapter.rho_min = 1.0f;
    adapter.rho_max = 100.0f;
    adapter.tolerance = 1.1f;
    adapter.clip = true;
    
    // Set ADMM parameters
    params.abs_pri_tol = 1e-2f;
    params.abs_dua_tol = 1e-2f;
    params.max_iter = 500;
    params.rho = 100.0f;  // This will be replaced by sed in bash script
    
    // Print CSV header
    Serial.println("rho,trial,iterations,solve_time_us,success");
    
    // Run hover test NUM_TRIALS times
    for(int i = 0; i < NUM_TRIALS; i++) {
        // Reset problem state completely
        problem.status = 0;
        problem.iter = 0;
        problem.solve_count = 0;
        
        // Reset matrices
        problem.x.setZero();
        problem.u.setZero();
        problem.y.setZero();
        problem.g.setZero();
        problem.v.setZero();
        problem.z.setZero();
        problem.vnew.setZero();
        problem.znew.setZero();
        problem.p.setZero();
        problem.q.setZero();
        problem.r.setZero();
        problem.d.setZero();
        
        // Set hover test initial condition
        problem.x.col(0) << 0.2f, 0.2f, -0.2f,  // position offset
                           1.0f, 0.0f, 0.0f,    // roll offset
                           0.0f, 0.0f, 0.0f,    // zero velocity
                           0.0f, 0.0f, 0.0f;    // zero angular rates
        
        // Solve and time
        uint32_t solve_start = micros();
        solve_admm(&problem, &params);
        //solve_admm_adaptive(&problem, &params, &adapter);
        uint32_t solve_time = micros() - solve_start;
        
        // Print results
        Serial.print(params.rho);
        Serial.print(",");
        Serial.print(i);
        Serial.print(",");
        Serial.print(problem.iter);
        Serial.print(",");
        Serial.print(solve_time);
        Serial.print(",");
        Serial.println(problem.iter < 500 ? 1 : 0);  // success flag
    }
    
    Serial.println("END");  // Marker for data collection
}

void loop() {
    // Empty
}


// #include <Arduino.h>
// #undef F
// #include "admm.hpp"
// #include "problem_data/rand_prob_tinympc_params.hpp"
// #include "types.hpp"
// #include "rho_benchmark.hpp"

// // Add these constants at the top with other constants
// const float MASS = 0.035f;
// const float G = 9.81f;
// const float KT = 2.245365e-6f * 65535.0f;

// // Wind generation constants
// const float WIND_MEAN_X = 0.5f;
// const float WIND_MEAN_Y = 0.0f;
// const float WIND_MEAN_Z = 0.3f;
// const float WIND_FREQ = 0.5f;
// const float WIND_AMP = 0.5f;

// // New wind-related structures
// struct WindState {
//     float vx;
//     float vy;
//     float vz;
//     float t;
// };

// struct Trajectory {
//     Matrix<float, NSTATES, NHORIZON> Xref;
//     Matrix<float, NINPUTS, NHORIZON-1> Uref;
// };

// // Wind generation function
// WindState generate_wind(float t) {
//     WindState wind;
//     wind.vx = WIND_MEAN_X + WIND_AMP * sin(WIND_FREQ * t);
//     wind.vy = WIND_MEAN_Y + WIND_AMP * cos(WIND_FREQ * t);
//     wind.vz = WIND_MEAN_Z + 0.2f * WIND_AMP * sin(2.0f * WIND_FREQ * t);
//     wind.t = t;
//     return wind;
// }

// // Wind trajectory generation
// Trajectory generate_wind_trajectory(float start_time, float duration) {
//     Trajectory traj;
//     const float dt = duration / (NHORIZON - 1);
    
//     traj.Xref.setZero();
//     traj.Uref.setZero();
    
//     float hover_thrust = (MASS * G) / (4.0f * KT);
    
//     for(int k = 0; k < NHORIZON; k++) {
//         float t = start_time + k * dt;
//         WindState wind = generate_wind(t);
        
//         // Position with wind effects
//         traj.Xref(0,k) = 0.5f * wind.vx * dt * dt;
//         traj.Xref(1,k) = 0.5f * wind.vy * dt * dt;
//         traj.Xref(2,k) = 0.5f * wind.vz * dt * dt;
        
//         if(k < NHORIZON-1) {
//             float wind_comp_x = wind.vx * 0.1f;
//             float wind_comp_y = wind.vy * 0.1f;
//             float wind_comp_z = wind.vz * 0.1f;
            
//             traj.Uref(0,k) = hover_thrust + wind_comp_x;
//             traj.Uref(1,k) = hover_thrust + wind_comp_y;
//             traj.Uref(2,k) = hover_thrust + wind_comp_z;
//             traj.Uref(3,k) = hover_thrust;
//         }
//     }
//     return traj;
// }

// // Add wind toggle
// const bool USE_WIND = true; 

// // Add struct for collecting stats
// struct SolverStats {
//     float avg_solve_time;
//     float avg_admm_time;
//     float avg_rho_time;
//     float avg_iters;
//     float std_solve_time;
//     float std_iters;
    
//     // Arrays to store raw data
//     float solve_times[1000];
//     float admm_times[1000];
//     float rho_times[1000];
//     float iterations[1000];
// };

// // Add struct for problem cases
// struct ProblemCase {
//     int trial_num;
//     float fixed_time;
//     float fixed_iters;
//     float adaptive_time;
//     float adaptive_iters;
//     Matrix<float, NSTATES, NHORIZON> Xref;
//     Matrix<float, NINPUTS, NHORIZON-1> Uref;
// };

// // Add these variables
// const int MAX_PROBLEM_CASES = 50;
// ProblemCase problem_cases[MAX_PROBLEM_CASES];
// int num_problem_cases = 0;

// // At the top with other globals
// int max_iter_count = 0;  // Counter for cases hitting max iterations

// // Function to compute statistics
// void compute_stats(SolverStats* stats, int num_trials) {
//     // Compute averages
//     float sum_solve = 0, sum_admm = 0, sum_rho = 0, sum_iters = 0;
//     for(int i = 0; i < num_trials; i++) {
//         sum_solve += stats->solve_times[i];
//         sum_admm += stats->admm_times[i];
//         sum_rho += stats->rho_times[i];
//         sum_iters += stats->iterations[i];
//     }
    
//     stats->avg_solve_time = sum_solve / num_trials;
//     stats->avg_admm_time = sum_admm / num_trials;
//     stats->avg_rho_time = sum_rho / num_trials;
//     stats->avg_iters = sum_iters / num_trials;
    
//     // Compute standard deviations
//     float sum_sq_solve = 0, sum_sq_iters = 0;
//     for(int i = 0; i < num_trials; i++) {
//         sum_sq_solve += pow(stats->solve_times[i] - stats->avg_solve_time, 2);
//         sum_sq_iters += pow(stats->iterations[i] - stats->avg_iters, 2);
//     }
    
//     stats->std_solve_time = sqrt(sum_sq_solve / num_trials);
//     stats->std_iters = sqrt(sum_sq_iters / num_trials);
// }

// void print_stats(const char* method, SolverStats* stats) {
//     Serial.println("\n=== " + String(method) + " Statistics ===");
//     Serial.println("Average solve time: " + String(stats->avg_solve_time) + " µs");
//     Serial.println("Average ADMM time: " + String(stats->avg_admm_time) + " µs");
//     Serial.println("Average rho time: " + String(stats->avg_rho_time) + " µs");
//     Serial.println("Average iterations: " + String(stats->avg_iters));
//     Serial.println("Std Dev solve time: " + String(stats->std_solve_time) + " µs");
//     Serial.println("Std Dev iterations: " + String(stats->std_iters));
// }

// void setup() {
//     Serial.begin(115200);
//     delay(3000);
    
//     Serial.println("Starting MPC Benchmark Test");
//     Serial.println("Method,Trial,SolveTime,ADMMTime,RhoTime,Iterations,FinalRho");
    
//     // Initialize problem & params
//     tiny_problem problem;
//     tiny_params params;
    
//     // Set ADMM tolerances
//     params.abs_pri_tol = 1e-2f;
//     params.abs_dua_tol = 1e-2f;
//     params.max_iter = 500;
    
//     // Initialize RhoAdapter
//     RhoAdapter adapter;
//     adapter.rho_base = 85.0f;
//     adapter.rho_min = 70.0f;
//     adapter.rho_max = 100.0f;
//     adapter.tolerance = 1.1f;
//     adapter.clip = true;
    
//     const int NUM_TRIALS = 1000;
//     SolverStats fixed_stats = {0};
//     SolverStats adaptive_stats = {0};
    
//     // Add storage for references
//     Matrix<float, NSTATES, NHORIZON> stored_Xref;
//     Matrix<float, NINPUTS, NHORIZON-1> stored_Uref;
    
//     // First do hover test
//     Serial.println("\n=== Starting Hover Tests ===");
    
//     // Setup hover conditions
//     float hover_thrust = (MASS * G) / (4.0f * KT);
    
//     problem.x.setZero();
//     problem.x.col(0) << 0.2f, 0.2f, -0.2f,  // position offset
//                         1.0f, 0.0f, 0.0f,    // roll offset
//                         0.0f, 0.0f, 0.0f,    // zero velocity
//                         0.0f, 0.0f, 0.0f;    // zero angular rates
    
//     // Modify reference generation based on USE_WIND
//     if (USE_WIND) {
//         WindState wind = generate_wind(0.0f);  // start at t=0
//         Trajectory traj = generate_wind_trajectory(0.0f, 2.0f);
//         params.Xref = traj.Xref;
//         params.Uref = traj.Uref;
//     } else {
//         params.Xref.setZero();  // target hover state
//         params.Uref.setZero();
//         // Set hover thrust for all timesteps
//         for(int k = 0; k < NHORIZON-1; k++) {
//             params.Uref.col(k) << hover_thrust, hover_thrust, hover_thrust, hover_thrust;
//         }
//     }

//     // Test fixed rho for hover
//     Serial.println("\n=== Hover with Fixed Rho ===");
//     problem.status = 0;
//     problem.iter = 0;
//     solve_admm(&problem, &params);
//     Serial.print("Fixed Hover,");
//     Serial.print("-1,");  // special trial number for hover
//     Serial.print(problem.fixed_timings.total_time);
//     Serial.print(",");
//     Serial.print(problem.fixed_timings.admm_time);
//     Serial.print(",");
//     Serial.print(problem.fixed_timings.rho_time);
//     Serial.print(",");
//     Serial.print(problem.iter);
//     Serial.print(",");
//     Serial.println(params.rho);
    
//     // Test adaptive rho for hover
//     Serial.println("\n=== Hover with Adaptive Rho ===");
//     problem.status = 0;
//     problem.iter = 0;
//     params.rho = adapter.rho_base;
//     params.compute_cache_terms();
//     solve_admm_adaptive(&problem, &params, &adapter);
//     Serial.print("Adaptive Hover,");
//     Serial.print("-1,");
//     Serial.print(problem.adaptive_timings.total_time);
//     Serial.print(",");
//     Serial.print(problem.adaptive_timings.admm_time);
//     Serial.print(",");
//     Serial.print(problem.adaptive_timings.rho_time);
//     Serial.print(",");
//     Serial.print(problem.iter);
//     Serial.print(",");
//     Serial.println(params.rho);
    
//     delay(1000);
    
//     // Then run trials with fixed rho
//     for(int i = 0; i < NUM_TRIALS; i++) {
//         Serial.println("\n=== Starting Fixed Rho Trial " + String(i) + " ===");
//         //delay(100);
        
//         // Reset problem
//         problem.status = 0;
//         problem.iter = 0;
//         problem.fixed_timings.total_time = 0;
//         problem.fixed_timings.admm_time = 0;
//         problem.fixed_timings.rho_time = 0;
        
//         // Reset rho to base value for fixed trials
//         params.rho = adapter.rho_base;
//         params.compute_cache_terms();
        
//         // Set test conditions
//         problem.x.setZero();
//         problem.x.col(0) << 1.0f, 2.0f, 3.0f, 4.0f;
//         problem.u.setRandom();

//         if (USE_WIND) {
//             float t = i * 0.02f;  // time step
//             Trajectory traj = generate_wind_trajectory(t, 2.0f);
//             params.Xref = traj.Xref;
//             params.Uref = traj.Uref;
//         } else {
//             params.Xref.setRandom();
//             params.Uref.setRandom();
//         }
        
//         // Store references for adaptive trials
//         stored_Xref = params.Xref;
//         stored_Uref = params.Uref;
        
//         solve_admm(&problem, &params);
        
//         // Output results
//         Serial.print("Fixed,");
//         Serial.print(i);
//         Serial.print(",");
//         Serial.print(problem.fixed_timings.total_time);
//         Serial.print(",");
//         Serial.print(problem.fixed_timings.admm_time);
//         Serial.print(",");
//         Serial.print(problem.fixed_timings.rho_time);
//         Serial.print(",");
//         Serial.print(problem.iter);
//         Serial.print(",");
//         Serial.println(params.rho);
        
//         // Store stats
//         fixed_stats.solve_times[i] = problem.fixed_timings.total_time;
//         fixed_stats.admm_times[i] = problem.fixed_timings.admm_time;
//         fixed_stats.rho_times[i] = problem.fixed_timings.rho_time;
//         fixed_stats.iterations[i] = problem.iter;
        
//         delay(500);
//     }
    
//     // Then run trials with adaptive rho
//     for(int i = 0; i < NUM_TRIALS; i++) {
//         Serial.println("\n=== Starting Adaptive Rho Trial " + String(i) + " ===");
        
//         // Reset problem
//         problem.status = 0;
//         problem.iter = 0;
//         problem.adaptive_timings.total_time = 0;
//         problem.adaptive_timings.admm_time = 0;
//         problem.adaptive_timings.rho_time = 0;

//         // Reset rho to base value
//         params.rho = adapter.rho_base;
//         params.compute_cache_terms();
        
//         // Set test conditions
//         problem.x.setZero();
//         problem.x.col(0) << 1.0f, 2.0f, 3.0f, 4.0f;
//         problem.u.setRandom();

//         if (USE_WIND) {
//             float t = i * 0.02f;  // time step
//             Trajectory traj = generate_wind_trajectory(t, 2.0f);
//             params.Xref = traj.Xref;
//             params.Uref = traj.Uref;
//         } else {
//             // Use same references as fixed method
//             params.Xref = stored_Xref;
//             params.Uref = stored_Uref;
//         }
        

        
//         solve_admm_adaptive(&problem, &params, &adapter);
        
//         Serial.print("Adaptive,");
//         Serial.print(i);
//         Serial.print(",");
//         Serial.print(problem.adaptive_timings.total_time);
//         Serial.print(",");
//         Serial.print(problem.adaptive_timings.admm_time);
//         Serial.print(",");
//         Serial.print(problem.adaptive_timings.rho_time);
//         Serial.print(",");
//         Serial.print(problem.iter);
//         Serial.print(",");
//         Serial.println(params.rho);
        
//         // Store stats
//         adaptive_stats.solve_times[i] = problem.adaptive_timings.total_time;
//         adaptive_stats.admm_times[i] = problem.adaptive_timings.admm_time;
//         adaptive_stats.rho_times[i] = problem.adaptive_timings.rho_time;
//         adaptive_stats.iterations[i] = problem.iter;
        
//         // Simple tracking of max iteration cases
//         if (problem.iter >= 499) {  // Near max iterations
//             max_iter_count++;
//             Serial.println("\n!!! Max iterations reached in trial " + String(i));
//         }

//         delay(500);
//     }
    
//     // Compute and print statistics
//     compute_stats(&fixed_stats, NUM_TRIALS);
//     compute_stats(&adaptive_stats, NUM_TRIALS);
    
//     print_stats("Fixed Rho", &fixed_stats);
//     print_stats("Adaptive Rho", &adaptive_stats);
    
    
//     // Add summary of problem cases
//     Serial.println("\n=== Problem Cases Summary ===");
//     Serial.print("Total problem cases found: "); 
//     Serial.println(num_problem_cases);

//     Serial.println("Benchmark Complete!");
// }

// void loop() {
//     // Empty
// }  
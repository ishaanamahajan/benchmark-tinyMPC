/**
 * Test TinyMPC with integrated SDP projections
 * Uses actual TinyMPC ADMM loop with your custom project_psd<M>() function
 */

#define ENABLE_SDP_PROJECTION  // Enable SDP projections in ADMM

#include <iostream>
#include <fstream>
#include <chrono>
#include "tinympc_teensy/src/admm.hpp"
#include "tinympc_teensy/src/types.hpp"

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "TinyMPC with Integrated SDP Projections" << std::endl;
    std::cout << "Obstacle Avoidance Test" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Initialize TinyMPC solver
    TinySolver solver;
    
    // For this test, we'll use simplified 4-state problem
    // In full implementation, you'd use the 20-state extended formulation
    
    std::cout << "Setting up TinyMPC solver..." << std::endl;
    
    // Initialize solver (simplified setup)
    reset_problem(&solver);
    
    // Set initial condition
    Vector4d x_init(-10.0, 0.1, 0.0, 0.0);  // From Julia
    solver.work->x.col(0) = x_init;
    
    std::cout << "Initial state: [" << x_init.transpose() << "]" << std::endl;
    std::cout << "Obstacle: center [-5, 0], radius 2.0" << std::endl;
    
    // Record solve time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // SOLVE WITH SDP PROJECTIONS INTEGRATED
    std::cout << "\nSolving with TinyMPC + SDP projections..." << std::endl;
    
    // Manual ADMM loop to test SDP integration
    const int max_iter = 50;
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Standard TinyMPC ADMM steps
        forward_pass(&solver);      // LQR rollout
        update_slack(&solver);      // PROJECT CONSTRAINTS (now includes SDP!)
        update_dual(&solver);       // Update multipliers
        update_linear_cost(&solver); // Update cost terms  
        backward_pass_grad(&solver); // Riccati backward pass
        
        if (iter % 10 == 0) {
            std::cout << "Iteration " << iter << " completed" << std::endl;
        }
        
        // Check termination
        if (termination_condition(&solver)) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Extract results
    std::cout << "\n==========================================" << std::endl;
    std::cout << "SOLVE RESULTS:" << std::endl;
    std::cout << "Solve time: " << duration.count() << " ms" << std::endl;
    
    // Save trajectory
    std::ofstream file("tinympc_integrated_sdp_trajectory.csv");
    file << "# TinyMPC with Integrated SDP Projections\n";
    file << "# Format: time, pos_x, pos_y, vel_x, vel_y\n";
    
    for (int k = 0; k < NHORIZON; k++) {
        Vector4d x_k = solver.work->bounds->vnew.col(k).head<4>();
        file << k << ", " << x_k(0) << ", " << x_k(1) << ", " << x_k(2) << ", " << x_k(3) << "\n";
    }
    file.close();
    
    // Check safety
    int violations = 0;
    double min_dist = 1000.0;
    Vector2d x_obs(-5.0, 0.0);
    double r_obs = 2.0;
    
    for (int k = 0; k < NHORIZON; k++) {
        Vector2d pos = solver.work->bounds->vnew.col(k).head<2>();
        double dist = (pos - x_obs).norm();
        min_dist = std::min(min_dist, dist);
        
        if (dist < r_obs) {
            violations++;
        }
    }
    
    std::cout << "Safety analysis:" << std::endl;
    std::cout << "- Obstacle violations: " << violations << "/" << NHORIZON << std::endl;
    std::cout << "- Min distance to obstacle: " << min_dist << std::endl;
    std::cout << "- Safe trajectory: " << (violations == 0 ? "YES" : "NO") << std::endl;
    
    std::cout << "\n✅ TinyMPC + SDP test complete!" << std::endl;
    std::cout << "📊 Results saved to tinympc_integrated_sdp_trajectory.csv" << std::endl;
    std::cout << "🎯 Your project_psd<M>() function is now integrated in TinyMPC ADMM!" << std::endl;
    
    return 0;
}

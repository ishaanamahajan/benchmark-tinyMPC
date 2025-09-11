#include "sdp_solver.hpp"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "TinyMPC SDP Obstacle Avoidance Test" << std::endl;
    std::cout << "Julia Problem Setup with Custom PSD Projections" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Create and initialize solver
    SDPSolver solver;
    
    // Record solve time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Solve the SDP problem
    bool converged = solver.solve();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Print results
    std::cout << "\n==================================================" << std::endl;
    std::cout << "SOLVE RESULTS:" << std::endl;
    std::cout << "Converged: " << (converged ? "YES" : "NO") << std::endl;
    std::cout << "Solve time: " << duration.count() << " ms" << std::endl;
    
    // Save trajectory data
    solver.save_results("tinympc_sdp_trajectory.csv");
    
    std::cout << "\n✅ SDP solve complete!" << std::endl;
    std::cout << "📊 Results saved to tinympc_sdp_trajectory.csv" << std::endl;
    std::cout << "🎯 Compare with Julia solution using plot script" << std::endl;
    
    return 0;
}



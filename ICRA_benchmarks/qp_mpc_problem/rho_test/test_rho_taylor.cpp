
#include <iostream>
#include <iomanip>
#include <cmath>
#include "rho_benchmark.hpp"

void print_matrix(const char* name, float* matrix, int rows, int cols) {
    std::cout << name << ":\n";
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(6) 
                      << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    // Initialize cache with base values
    initialize_benchmark_cache();
    
    // Print initial values
    std::cout << "Initial state:\n";
    print_matrix("Kinf", (float*)Kinf, BENCH_NU, BENCH_NX);
    print_matrix("C1", (float*)C1, BENCH_NU, BENCH_NU);
    
    // Run Taylor adaptation with some test residuals
    RhoBenchmarkResult result;
    float pri_res = 0.5f;
    float dual_res = 2.0f;
    std::cout << "Running adaptation with pri_res = " << pri_res << ", dual_res = " << dual_res << "\n";
    benchmark_rho_adaptation(pri_res, dual_res, &result);
    
    // Print results
    std::cout << "\nAdaptation results:\n";
    std::cout << "Initial rho: " << result.initial_rho << "\n";
    std::cout << "Final rho: " << result.final_rho << "\n";
    
    // Print final matrices
    std::cout << "\nFinal state:\n";
    print_matrix("Kinf", (float*)Kinf, BENCH_NU, BENCH_NX);
    print_matrix("C1", (float*)C1, BENCH_NU, BENCH_NU);
    
    return 0;
}
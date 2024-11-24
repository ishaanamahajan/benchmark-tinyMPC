// test_rho.cpp
#include <iostream>
#include <iomanip>
#include <cmath>
#include "rho_benchmark_recompute.hpp"

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
    std::cout << "Initial values at rho = 85.0:\n";
    print_matrix("Kinf", (float*)Kinf, BENCH_NU, BENCH_NX);
    print_matrix("C1", (float*)C1, BENCH_NU, BENCH_NU);
    
    // Recompute with new rho
    float new_rho = 90.0f;
    std::cout << "Recomputing for rho = " << new_rho << "\n";
    recompute_cache(new_rho);
    
    // Print new values
    std::cout << "Recomputed values:\n";
    print_matrix("Kinf", (float*)Kinf, BENCH_NU, BENCH_NX);
    print_matrix("C1", (float*)C1, BENCH_NU, BENCH_NU);
    
    return 0;
}
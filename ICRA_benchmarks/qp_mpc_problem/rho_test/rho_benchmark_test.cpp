#include "rho_benchmark.hpp"
#include <Arduino.h>

void setup() {
    Serial.begin(115200);
    while(!Serial) {
        ; // Wait for serial connection
    }

    // Test data
    float x_prev[BENCH_NX] = {0};  // Fill with test values
    float u_prev[BENCH_NU] = {0};  // Fill with test values
    float z_prev[BENCH_NX] = {0};  // Fill with test values
    
    RhoBenchmarkResult result;

    // Test Taylor version
    benchmark_rho_adaptation(x_prev, u_prev, z_prev, 1.0, 0.5, &result);
    Serial.println("Taylor Version Results:");
    Serial.print("Initial rho: "); Serial.println(result.initial_rho);
    Serial.print("Final rho: "); Serial.println(result.final_rho);
    Serial.print("Time (us): "); Serial.println(result.time_us);
    
    // Test Recompute version
    benchmark_rho_recompute(x_prev, u_prev, z_prev, 1.0, 0.5, &result);
    Serial.println("\nRecompute Version Results:");
    Serial.print("Initial rho: "); Serial.println(result.initial_rho);
    Serial.print("Final rho: "); Serial.println(result.final_rho);
    Serial.print("Time (us): "); Serial.println(result.time_us);
}

void loop() {
    // Nothing to do here
}
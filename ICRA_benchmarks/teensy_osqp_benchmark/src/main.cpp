/**
 * Blink
 *
 * Turns on an LED on for one second,
 * then off for one second, repeatedly.
 */
#include "Arduino.h"

#include <stdio.h>
#include <iostream>
#include "osqp.h"
#include "osqp_data_workspace.h"
#include "rand_prob_osqp_xbar.h"
#include "math.h"

#ifndef LED_BUILTIN
#define LED_BUILTIN 13
#endif

// Memory measurement functions for Teensy 4.1 (adapted from CVXPYgen)
// Return an estimate of free RAM by measuring gap between heap and stack
static uint32_t freeRam() {
  uint32_t stack_top;
  stack_top = (uint32_t)&stack_top;    // current stack pointer
  void *heap_ptr = malloc(4);          // current heap end after small alloc
  if (heap_ptr == NULL) return 0;      // Safety check
  uint32_t heap_top = (uint32_t)heap_ptr;
  free(heap_ptr);
  
  // Safety check for pointer arithmetic overflow
  if (stack_top < heap_top) return 0;
  
  uint32_t free_ram = stack_top - heap_top;
  
  // Sanity check: Teensy 4.1 has 512KB total, so free RAM should be reasonable
  if (free_ram > 600 * 1024) {
    // If measurement seems too high, fall back to max allocatable block method
    return findMaxAllocation();
  }
  
  return free_ram;         // remaining free RAM
}

// Test if we can allocate a given amount of memory
static bool testMemoryAllocation(size_t bytes) {
  void* test_ptr = malloc(bytes);
  if (test_ptr != NULL) {
    free(test_ptr);
    return true;
  }
  return false;
}

// Find maximum allocatable block using binary search
static size_t findMaxAllocation() {
  size_t max_size = 0;
  size_t test_size = 512 * 1024; // Start with 512KB (Teensy 4.1 max)
  
  // Binary search for largest allocatable block
  size_t min_size = 0;
  size_t max_test = test_size;
  
  while (min_size < max_test - 1) {
    size_t mid = (min_size + max_test) / 2;
    if (testMemoryAllocation(mid)) {
      min_size = mid;
      max_size = mid;
    } else {
      max_test = mid;
    }
  }
  
  return max_size;
}

void printMemoryInfo(const char* label) {
  uint32_t free_mem = freeRam();
  
  Serial.print("# ");
  Serial.print(label);
  Serial.print(" - Free RAM: ");
  Serial.print(free_mem);
  Serial.print(" bytes (");
  Serial.print(free_mem / 1024.0, 2);
  Serial.println(" KB)");
  Serial.flush(); // Ensure output is sent immediately
}

void add_noise(OSQPFloat x[])
{
  for (int i = 0; i < NSTATES; ++i)
  {
    OSQPFloat noise = (float(rand() / RAND_MAX) - 0.5)*2 * 0.01;
    x[i] += noise;
  }
}

void print_vector(OSQPFloat xn[], int n)
{
  for (int i = 0; i < n; ++i)
  {
    Serial.println(xn[i]);
  }
}

void matrix_vector_mult(int n1,
                        int n2,
                        const OSQPFloat matrix[],
                        const OSQPFloat vector[],
                        OSQPFloat result_vector[])
{
  // n1 is rows of matrix
  // n2 is cols of matrix, or vector
  int i, j; // i = row; j = column;
  for (i = 0; i < n1; i++)
  {
    for (j = 0; j < n2; j++)
    {
      result_vector[i] += matrix[i * n2 + j] * vector[j];
    }
  }
}

void matrix_vector_reset_mult(int n1,
                              int n2,
                              const OSQPFloat matrix[],
                              const OSQPFloat vector[],
                              OSQPFloat result_vector[])
{
  // n1 is rows of matrix
  // n2 is cols of matrix, or vector
  int i, j; // i = row; j = column;
  for (i = 0; i < n1; i++)
  {
    result_vector[i] = 0.0;
    for (j = 0; j < n2; j++)
    {
      result_vector[i] += matrix[i * n2 + j] * vector[j];
    }
  }
}

void system_dynamics(OSQPFloat xn[], const OSQPFloat x[], const OSQPFloat u[], const OSQPFloat A[], const OSQPFloat B[])
{
  matrix_vector_reset_mult(NSTATES, NSTATES, A, x, xn);
  matrix_vector_mult(NSTATES, NINPUTS, B, u, xn);
}

void compute_q(OSQPFloat q[], const OSQPFloat Q_data[], const OSQPFloat Qf_data[], const OSQPFloat Xref[])
// Xref is Nh x Nx, pointer at current step
// q is Nh x (Nx + 1) + Nh x Nu
// Q_data is Nx x Nx
{
  for (int i = 0; i < NHORIZON; ++i)
  {
    matrix_vector_reset_mult(NSTATES, NSTATES, Q_data, Xref + (i * NSTATES), q + (i * NSTATES));
  }
  matrix_vector_reset_mult(NSTATES, NSTATES, Qf_data, Xref + ((NHORIZON)*NSTATES), q + ((NHORIZON)*NSTATES));
}

void compute_bound(OSQPFloat bnew[], OSQPFloat xn[], OSQPFloat xb, OSQPFloat ub, int size)
{
  for (int i = 0; i < NSTATES; ++i)
  {
    bnew[i] = -xn[i]; // only the first is current state
  }
  for (int i = (NHORIZON + 1) * NSTATES; i < (NHORIZON + 1) * NSTATES * 2; ++i)
  {
    bnew[i] = xb; // bounds on x
  }
  for (int i = (NHORIZON + 1) * NSTATES * 2; i < size; ++i)
  {
    bnew[i] = ub; // bounds on u
  }
}

OSQPFloat compute_norm(const OSQPFloat x[], const OSQPFloat x_bar[])
{
  OSQPFloat res = 0.0f;
  for (int i = 0; i < NSTATES; ++i)
  {
    res += (x[i] - x_bar[i]) * (x[i] - x_bar[i]);
  }
  return sqrt(res);
}

OSQPInt exitflag;
OSQPFloat xn[NSTATES] = {0};
OSQPFloat x[NSTATES] = {0};
OSQPFloat q_new[SIZE_Q] = {0};
OSQPFloat l_new[SIZE_LU] = {0};
OSQPFloat u_new[SIZE_LU] = {0};
OSQPFloat xmin = -10000;
OSQPFloat xmax = 10000;
OSQPFloat umin = -3;
OSQPFloat umax = 3;

// Memory tracking variables
static uint32_t baseline_ram = 0;
static uint32_t post_setup_ram = 0;
static uint32_t post_solve_ram = 0;
static bool solver_initialized = false;

void setup()
// int main()
{
  srand(123);
  // initialize LED digital pin as an output.
  pinMode(LED_BUILTIN, OUTPUT);

  // start serial terminal
  Serial.begin(9600);
  delay(1000);
  while (!Serial)
  { // wait to connect
    continue;
  }

  Serial.println("# ========================================");
  Serial.println("# Teensy 4.1 OSQP Memory Analysis");
  Serial.println("# ========================================");
  
  Serial.print("# Horizon: ");
  Serial.println(NHORIZON);
  Serial.print("# States: ");
  Serial.println(NSTATES);
  Serial.print("# Inputs: ");
  Serial.println(NINPUTS);
  Serial.print("# Total iterations planned: ");
  Serial.println(NTOTAL - NHORIZON);
  
  // Get baseline memory info
  baseline_ram = freeRam();
  size_t max_allocation = findMaxAllocation();
  
  Serial.print("# Baseline free RAM: ");
  Serial.print(baseline_ram);
  Serial.print(" bytes (");
  Serial.print(baseline_ram / 1024.0, 2);
  Serial.println(" KB)");
  
  Serial.print("# Maximum allocatable block: ");
  Serial.print(max_allocation);
  Serial.print(" bytes (");
  Serial.print(max_allocation / 1024.0, 2);
  Serial.println(" KB)");
  
  // Estimate OSQP memory requirements (rough calculation)
  size_t estimated_osqp_memory = 0;
  // OSQP typically needs memory for:
  // - Problem matrices (sparse format)
  // - Working vectors
  // - Factorization workspace
  size_t qp_vars = (NHORIZON + 1) * NSTATES + NHORIZON * NINPUTS;
  size_t qp_constraints = SIZE_LU;
  estimated_osqp_memory = qp_vars * qp_vars * sizeof(OSQPFloat) / 10; // Sparse matrices
  estimated_osqp_memory += qp_constraints * sizeof(OSQPFloat) * 5; // Constraint data
  estimated_osqp_memory += qp_vars * sizeof(OSQPFloat) * 20; // Working arrays
  
  Serial.print("# Estimated OSQP memory need: ");
  Serial.print(estimated_osqp_memory);
  Serial.print(" bytes (");
  Serial.print(estimated_osqp_memory / 1024.0, 2);
  Serial.println(" KB)");
  
  if (estimated_osqp_memory > max_allocation) {
    Serial.println("# ⚠ WARNING: Estimated memory exceeds available!");
    Serial.println("# ⚠ OSQP initialization may fail or crash!");
    Serial.println("# ⚠ Consider reducing horizon or problem size.");
    Serial.println("# ⚠ Attempting initialization anyway...");
  } else {
    Serial.println("# ✓ Estimated memory within available limits");
  }
  
  delay(2000); // Give time to read warnings
  
  Serial.println("# ========================================");
  Serial.println("# Attempting OSQP benchmark...");
  
  post_setup_ram = freeRam();
  printMemoryInfo("AFTER setup");
  
  bool first_solve_done = false;
  
  for (int step = 0; step < NTOTAL - NHORIZON; step++)
  // for (int step = 0; step < 5; step++)
  {
    compute_q(q_new, mQ, mQf, &(Xref_data[(step) * NSTATES]));
    compute_bound(l_new, x, xmin, umin, SIZE_LU);
    compute_bound(u_new, x, xmax, umax, SIZE_LU);
    osqp_update_data_vec(&osqp_data_solver, q_new, l_new, u_new);
    
    if (!first_solve_done) {
      Serial.println("# Attempting first solve (CRITICAL MOMENT)...");
      Serial.println("# If this hangs/crashes, problem is too big for available memory");
      uint32_t pre_solve_ram = freeRam();
      Serial.print("# RAM just before first solve: ");
      Serial.print(pre_solve_ram);
      Serial.print(" bytes (");
      Serial.print(pre_solve_ram / 1024.0, 2);
      Serial.println(" KB)");
    }
    
    unsigned long start = micros();
    exitflag = osqp_solve(&osqp_data_solver);
    unsigned long end = micros();
    
    // Measure memory after first OSQP solve
    if (!first_solve_done) {
      post_solve_ram = freeRam();
      Serial.println("# ✓ SUCCESS: First solve completed!");
      printMemoryInfo("AFTER first OSQP solve");
      
      // Calculate OSQP-specific allocation
      if (baseline_ram > post_solve_ram) {
        uint32_t total_alloc = baseline_ram - post_solve_ram;
        Serial.print("# TOTAL OSQP ALLOCATION: ");
        Serial.print(total_alloc);
        Serial.print(" bytes (");
        Serial.print(total_alloc / 1024.0, 2);
        Serial.println(" KB)");
        
        Serial.print("# Memory efficiency: ");
        Serial.print((total_alloc * 100.0) / max_allocation, 1);
        Serial.println("% of available memory used");
        
        if (total_alloc < max_allocation / 2) {
          Serial.println("# ✓ GOOD: Plenty of memory headroom remaining");
        } else if (total_alloc < max_allocation * 0.8) {
          Serial.println("# ⚠ CAUTION: High memory usage but workable");
        } else {
          Serial.println("# ⚠ WARNING: Very high memory usage, near limits");
        }
      }
      
      first_solve_done = true;
      solver_initialized = true;
      Serial.println("# ========================================");
      Serial.println("# Continuing benchmark...");
    }
    
    OSQPFloat norm = compute_norm(x, &(Xref_data[step * NSTATES]));

    system_dynamics(xn, x, (osqp_data_solver.solution->x) + (NHORIZON + 1) * NSTATES, A, B);
    // printf("control\n");
    // print_vector((osqp_data_solver.solution->x) + (NHORIZON + 1) * NSTATES, NINPUTS);
    add_noise(xn);
    memcpy(x, xn, NSTATES * (sizeof(OSQPFloat)));
    printf("%10.4f %10d %10d\n", norm, osqp_data_solver.info->iter, end - start);
  }
  
  Serial.println("# ========================================");
  Serial.println("# FINAL MEMORY ALLOCATION RESULTS");
  Serial.println("# ========================================");
  
  if (solver_initialized && baseline_ram > post_solve_ram) {
    uint32_t total_allocation = baseline_ram - post_solve_ram;
    Serial.print("# SUCCESS: OSQP requires ");
    Serial.print(total_allocation);
    Serial.print(" bytes (");
    Serial.print(total_allocation / 1024.0, 2);
    Serial.println(" KB) of RAM");
    
    Serial.print("# Available RAM on Teensy 4.1: ");
    Serial.print(baseline_ram);
    Serial.print(" bytes (");
    Serial.print(baseline_ram / 1024.0, 2);
    Serial.println(" KB)");
    
    Serial.print("# Remaining after OSQP: ");
    Serial.print(post_solve_ram);
    Serial.print(" bytes (");
    Serial.print(post_solve_ram / 1024.0, 2);
    Serial.println(" KB)");
  }
  
  Serial.println("# Benchmark completed successfully!");
}

void loop()
{
}
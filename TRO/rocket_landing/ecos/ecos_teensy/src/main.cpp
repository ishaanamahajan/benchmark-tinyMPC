#include "Arduino.h"
// #include "cpp_compat.h"
#include <stdio.h>
#include <stdlib.h>
#include "math.h"

extern "C" {
#include "cpg_workspace.h"
#include "cpg_solve.h"
#include "cpg_problem.h"
}

// Forward declaration for freeRam helper
static uint32_t freeRam();

/*
// UNCOMMENT THIS BLOCK FOR CRASH PREDICTION
// Use this when Teensy crashes to estimate memory requirements

static bool testMemoryAllocation(size_t bytes);
static size_t findMaxAllocation();

// Test if we can allocate a given amount of memory
static bool testMemoryAllocation(size_t bytes) {
  void* test_ptr = malloc(bytes);
  if (test_ptr != NULL) {
    free(test_ptr);
    return true;
  }
  return false;
}

// Find maximum allocatable block
static size_t findMaxAllocation() {
  size_t max_size = 0;
  size_t test_size = 1024 * 1024; // Start with 1MB
  
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

// Estimate ECOS solver memory requirements
static size_t estimateECOSMemory() {
  // ECOS memory estimation for QP problems
  // Based on ECOS documentation and typical usage patterns
  
  size_t n_vars = NSTATES * NHORIZON + NINPUTS * NHORIZON; // Decision variables
  size_t n_eq = NSTATES * (NHORIZON - 1);  // Equality constraints (dynamics)
  size_t n_ineq = NINPUTS * NHORIZON * 2;  // Inequality constraints (input bounds)
  size_t n_cone = 0;  // Second-order cone constraints (if any)
  
  printf("# Problem dimensions:\n");
  printf("#   Variables: %d\n", (int)n_vars);
  printf("#   Equality constraints: %d\n", (int)n_eq);
  printf("#   Inequality constraints: %d\n", (int)n_ineq);
  printf("#   States: %d, Inputs: %d, Horizon: %d\n", NSTATES, NINPUTS, NHORIZON);
  
  // ECOS workspace memory estimation (based on ECOS source)
  size_t estimated_memory = 0;
  
  // Main matrices (approximate sizes)
  estimated_memory += n_vars * n_vars * sizeof(float);        // KKT system matrix
  estimated_memory += (n_eq + n_ineq) * n_vars * sizeof(float); // Constraint matrices
  estimated_memory += n_vars * sizeof(float) * 10;            // Working vectors (multiple copies)
  estimated_memory += (n_eq + n_ineq) * sizeof(float) * 5;    // Constraint working vectors
  estimated_memory += n_vars * n_vars * sizeof(float);        // Cholesky factorization
  
  // ECOS internal workspace
  estimated_memory += 50000; // Base ECOS overhead (~50KB)
  
  return estimated_memory;
}
*/

static int i;

void add_noise(float x[], float var)
{
  for (int i = 0; i < NSTATES; ++i)
  {
    float noise = ((rand() / (double)RAND_MAX) - 0.5) * 2; // random -1 to 1
    x[i] += noise * var;
  }
}

void print_vector(float xn[], int n)
{
  for (int i = 0; i < n; ++i)
  {
    // Serial.println(xn[i]);
    printf("%f, ", xn[i]);
  }
  printf("\n");
}

void matrix_vector_mult(int n1,
                        int n2,
                        float matrix[],
                        float vector[],
                        float result_vector[])
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
                              float matrix[],
                              float vector[],
                              float result_vector[])
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

void system_dynamics(float xn[], float x[], float u[], float A[], float B[], float f[])
{
  matrix_vector_reset_mult(NSTATES, NSTATES, A, x, xn);
  matrix_vector_mult(NSTATES, NINPUTS, B, u, xn);
  for (int i = 0; i < NSTATES; ++i)
  {
    xn[i] += f[i];
  }
}

float compute_norm(float x[], float x_bar[])
{
  float res = 0.0f;
  for (int i = 0; i < NSTATES; ++i)
  {
    res += (x[i] - x_bar[i]) * (x[i] - x_bar[i]);
  }
  return sqrt(res);
}

const float xref0[] = {4, 2, 20, -3, 2, -4.5};

float xn[NSTATES] = {0};
float x[NSTATES] = {4.4, 2.2, 22, -3.3, 2.2, -4.95};
float u[NINPUTS] = {0};
float temp = 0;

// Memory tracking variables
static uint32_t free_ram_before = 0;
static uint32_t free_ram_after = 0;

int main(int argc, char *argv[]){
  // delay for 4 seconds
  delay(500);
  Serial.begin(115200);  // Initialize serial communication
  while (!Serial) {      // Wait for serial port to connect
    delay(10);
  }
  // printf("Start ECOS Rocket Landing\n");
  printf("========================\n");
  printf("# Horizon: %d\n", NHORIZON);
  
  /*
  // UNCOMMENT THIS SECTION FOR CRASH PREDICTION ANALYSIS
  // When Teensy crashes, uncomment this to see memory requirements
  
  printf("# ========================================\n");
  printf("# ECOS Memory Requirement Analysis\n");
  printf("# ========================================\n");
  
  uint32_t baseline = freeRam();
  size_t max_alloc = findMaxAllocation();
  size_t estimated = estimateECOSMemory();
  
  printf("# Available dynamic memory: %d bytes (%.2f KB)\n", (int)max_alloc, max_alloc / 1024.0);
  printf("# Estimated ECOS memory need: %d bytes (%.2f KB)\n", (int)estimated, estimated / 1024.0);
  
  if (estimated > max_alloc) {
    printf("# ⚠ WARNING: Estimated memory (%.1f KB) exceeds available (%.1f KB)!\n", 
           estimated / 1024.0, max_alloc / 1024.0);
    printf("# ⚠ ECOS initialization will likely crash!\n");
    printf("# ⚠ Memory deficit: %.1f KB\n", (estimated - max_alloc) / 1024.0);
  } else {
    printf("# ✓ Estimated memory within available limits\n");
    printf("# ✓ Expected to work with %.1f KB headroom\n", (max_alloc - estimated) / 1024.0);
  }
  
  printf("# ========================================\n");
  delay(3000); // Give time to read analysis
  */
  
  // Record baseline RAM before solver initialization
  free_ram_before = freeRam();
  //printf("# Free RAM before ECOS init: %u\n", free_ram_before);

  // Configure ECOS solver - this is where dynamic allocation happens
  cpg_set_solver_abstol(1e-2);
  cpg_set_solver_reltol(1e-3);
  cpg_set_solver_maxit(500);
  
  printf("# Starting ECOS benchmark...\n");
  printf("# Format: [solver_iter] [solve_time_us]\n");

  srand(1);
  // WE WILL ONLY SOLVE ONCE WITH THE GENERATED DATA
  for (int k = 0; k < 1; ++k) {
    //// Update current measurement
    // for (int i = 0; i < NSTATES; ++i)
    // {
    //   cpg_update_param1(i, x[i]);
    // }
    // printf("x = ");
    // print_vector(x, NSTATES);

    //// Update references
    // for (int i = 0; i < NHORIZON; ++i)
    // {
    //   for (int j = 0; j < NSTATES; ++j)
    //   {
    //     if (k+i >= NTOTAL)
    //     {
    //       temp = 0.0;
    //     }
    //     else
    //     {
    //       temp = xref0[j] + (0-xref0[j]) * (float)(k+i) / (NTOTAL);
    //     }
    //     // printf("temp = %f\n", temp);
    //     cpg_update_param3(i*(NSTATES+NINPUTS) + j, -Q_single * temp);
    //   }
    // }
    
    // Solve the problem instance - CRITICAL: Dynamic allocation happens here on first solve
    unsigned long start = micros();
    cpg_solve();
    unsigned long end = micros();
    
    printf("%3d %8d\n",  CPG_Info.iter, (int)(end - start));

    // Get data from the result
    // for (i=0; i<NINPUTS; i++) {
    //   u[i] = CPG_Result.prim->var2[i+NSTATES];
    // }
    // printf("u = ");
    // print_vector(u, NINPUTS);

    // Simulate the system
    // for (int i = 0; i < NSTATES; ++i)
    // {
    //   x[i] = (float)cpg_params_vec[i + NSTATES*NHORIZON + NINPUTS*(NHORIZON-1)];
    // }
    // print_vector(x, NSTATES);
    // system_dynamics(xn, x, u, A, B, f);
    // printf("xn = ");
    // print_vector(xn, NSTATES);

    // Update the state
    // memcpy(x, xn, NSTATES * (sizeof(float)));
    // add_noise(x, 0.01);
  }
  
  // Record final RAM and calculate dynamic allocation
  free_ram_after = freeRam();
  printf("# Dynamic memory allocated by ECOS: %u bytes (%.2f KB)\n", 
         free_ram_before - free_ram_after, 
         (free_ram_before - free_ram_after) / 1024.0);
  
  
  if (free_ram_before > free_ram_after) {
    uint32_t allocation = free_ram_before - free_ram_after;
    printf("# ✓ ECOS dynamic allocation: %u bytes (%.2f KB)\n", allocation, allocation / 1024.0);
  } else {
    printf("# ⚠ Memory measurement anomaly detected\n");
  }
  printf("# Benchmark completed successfully!\n");
  
  return 0;
}

// Return an estimate of free RAM by measuring gap between heap and stack
static uint32_t freeRam() {
  uint32_t stack_top;
  stack_top = (uint32_t)&stack_top;    // current stack pointer
  void *heap_ptr = malloc(4);          // current heap end after small alloc
  uint32_t heap_top = (uint32_t)heap_ptr;
  free(heap_ptr);
  return stack_top - heap_top;         // remaining free RAM
}

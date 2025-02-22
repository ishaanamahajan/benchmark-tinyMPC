#pragma once
#include "main.h"
#include "types_stm32.h"

// Keep same ADMM structs but without Arduino
struct tiny_problem {
    Matrix<float, NSTATES, NHORIZON> x;
    Matrix<float, NINPUTS, NHORIZON-1> u;
    int status;
    int iter;
    struct {
        uint32_t total_time;
        uint32_t admm_time;
        uint32_t rho_time;
    } fixed_timings;
    struct {
        uint32_t total_time;
        uint32_t admm_time;
        uint32_t rho_time;
    } adaptive_timings;
};

struct tiny_params {
    float rho;
    float abs_pri_tol;
    float abs_dua_tol;
    int max_iter;
    Matrix<float, NSTATES, NHORIZON> Xref;
    Matrix<float, NINPUTS, NHORIZON-1> Uref;
};

// Function declarations
void solve_admm_stm32(tiny_problem* prob, tiny_params* params);
void solve_admm_adaptive_stm32(tiny_problem* prob, tiny_params* params, RhoAdapter* adapter);
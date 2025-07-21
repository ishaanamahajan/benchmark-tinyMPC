/*
 * TinyMPC Global Options for Rocket Landing Benchmark
 */

#pragma once

typedef float tinytype;

#define NSTATES 6
#define NINPUTS 3
#define NHORIZON 256
#define NTOTAL 301

// Constraint definitions for rocket landing problem
#define NUM_INPUT_CONES 1  // One thrust cone constraint
#define NUM_STATE_CONES 0  // No state cone constraints

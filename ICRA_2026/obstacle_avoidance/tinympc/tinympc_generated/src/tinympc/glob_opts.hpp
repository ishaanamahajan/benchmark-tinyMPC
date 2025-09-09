#pragma once

typedef float tinytype;  // should be double if you want to generate code

#define NSTATES 6
#define NINPUTS 3

#define NUM_INPUT_CONES 1
#define NUM_STATE_CONES 1

// SDP constraint definitions for obstacle avoidance
#define NUM_SDP_CONS 1      // number of LMI constraints, per time step
#define SDP_DIM 4           // matrix size of each LMI (4x4 for sphere)

#define NHORIZON 16  // vary this for benchmarking
#define NTOTAL 301

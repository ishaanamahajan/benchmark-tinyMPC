#pragma once

typedef float tinytype;  // should be double if you want to generate code

#define NSTATES 6
#define NINPUTS 3

#define NUM_INPUT_CONES 1
#define NUM_STATE_CONES 1

// SDP constants for projection function
#define SDP_DIM 4           // matrix size for sphere LMI (4x4)

#define NHORIZON 16  // vary this for benchmarking
#define NTOTAL 301

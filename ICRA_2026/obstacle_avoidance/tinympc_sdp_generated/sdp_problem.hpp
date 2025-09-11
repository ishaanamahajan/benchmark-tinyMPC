#pragma once
// Auto-generated SDP Obstacle Avoidance Problem Parameters
// Generated from Julia tinysdp_big.jl

// Problem dimensions
#define NX_PHYSICAL 4      // Physical state dimension
#define NU_PHYSICAL 2      // Physical control dimension  
#define NX_EXTENDED 20  // Extended state dimension (includes second moments)
#define NU_EXTENDED 22  // Extended control dimension
#define NHORIZON 31          // Time horizon

// Obstacle parameters
#define OBS_CENTER_X -5.0f
#define OBS_CENTER_Y 0.0f  
#define OBS_RADIUS 2.0f

// Initial condition
#define X0_POS_X -10.0f
#define X0_POS_Y 0.1f
#define X0_VEL_X 0.0f
#define X0_VEL_Y 0.0f

// Cost weights
#define Q_XX 0.1f
#define R_XX 10.0f
#define R_UU 500.0f

// Enable SDP projection
#define ENABLE_SDP_PROJECTION 1
#define SDP_MATRIX_SIZE 4  // Size of PSD constraint matrix [1 x'; x XX]

// Obstacle avoidance constraint function
// Returns constraint value: should be >= 0 for feasibility
// constraint = tr(XX[0:1, 0:1]) - 2*obs_center'*x[0:1] + obs_center'*obs_center - radius^2
inline float obstacle_constraint(const float* x, const float* XX_flat) {
    // Extract position
    float px = x[0];
    float py = x[1];
    
    // Extract relevant second moments XX[0:1, 0:1] 
    // XX is stored as flattened 4x4 matrix in column-major order
    float XX_00 = XX_flat[0];   // XX(0,0)
    float XX_11 = XX_flat[5];   // XX(1,1) 
    
    // Compute constraint: tr(XX[0:1,0:1]) - 2*obs'*x + obs'*obs - r^2
    float trace_XX = XX_00 + XX_11;
    float obs_term = -5.0f * -5.0f + 0.0f * 0.0f;
    float pos_term = 2.0f * (-5.0f * px + 0.0f * py);
    
    return trace_XX - pos_term + obs_term - 2.0f * 2.0f;
}

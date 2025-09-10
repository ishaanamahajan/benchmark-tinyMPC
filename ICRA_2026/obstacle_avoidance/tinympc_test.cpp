#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "tinympc/tinympc_teensy/src/Eigen.h"
#include "tinympc/tinympc_teensy/src/admm.hpp"
#include "tinympc/tinympc_teensy/src/types.hpp"
#include "tinympc/tinympc_teensy/src/glob_opts.hpp"

using namespace std;
using namespace Eigen;

// Problem parameters from tinysdp_big.jl
const int NSTEPS = 31;
const Vector4f x_initial(-10.0f, 0.1f, 0.0f, 0.0f);
const Vector2f x_obs(-5.0f, 0.0f);
const float r_obs = 2.0f;

// Use TinyMPC's actual types
static TinyBounds bounds;
static TinySocs socs;
static TinyWorkspace work;

void setup_tinympc_problem() {
    cout << "Setting up TinyMPC obstacle avoidance problem..." << endl;
    
    // System dynamics (double integrator)
    float h = 0.1f;
    work.Adyn << 1, 0, h, 0,
                 0, 1, 0, h,
                 0, 0, 1, 0,
                 0, 0, 0, 1;
    
    work.Bdyn << 0.5f*h*h, 0,
                 0, 0.5f*h*h,
                 h, 0,
                 0, h;
    
    work.fdyn.setZero();
    
    // Cost matrices
    work.Q << 0.1f, 0.1f, 0.1f, 0.1f;  // Diagonal Q
    work.R << 10.0f, 10.0f;             // Diagonal R
    
    // Bounds
    Vector4f x_min(-15.0f, -10.0f, -5.0f, -5.0f);
    Vector4f x_max(15.0f, 10.0f, 5.0f, 5.0f);
    Vector2f u_min(-3.0f, -3.0f);
    Vector2f u_max(3.0f, 3.0f);
    
    work.bounds->x_min = x_min.replicate(1, NHORIZON);
    work.bounds->x_max = x_max.replicate(1, NHORIZON);
    work.bounds->u_min = u_min.replicate(1, NHORIZON - 1);
    work.bounds->u_max = u_max.replicate(1, NHORIZON - 1);
    
    // Connect structures
    work.bounds = &bounds;
    work.socs = &socs;
    
    cout << "✅ TinyMPC problem setup complete" << endl;
    cout << "Ad =\n" << work.Adyn << endl;
    cout << "Bd =\n" << work.Bdyn << endl;
}

bool check_collision(const Vector4f& x) {
    Vector2f position = x.head<2>();
    float dist = (position - x_obs).norm();
    return dist >= r_obs;
}

void test_sphere_lmi_projection(const Vector4f& x) {
    Vector2f position = x.head<2>();
    Vector2f p_minus_c = position - x_obs;
    
    // Create 3x3 sphere LMI: [[t, (p-c)^T]; [p-c, I2]]
    Matrix3f lmi_matrix;
    lmi_matrix(0, 0) = r_obs;  // Use radius as slack
    lmi_matrix.block<1, 2>(0, 1) = p_minus_c.transpose();
    lmi_matrix.block<2, 1>(1, 0) = p_minus_c;
    lmi_matrix.block<2, 2>(1, 1) = Matrix2f::Identity();
    
    cout << "Position: [" << position.transpose() << "]" << endl;
    cout << "Distance to obstacle: " << (position - x_obs).norm() << endl;
    cout << "LMI matrix:\n" << lmi_matrix << endl;
    
    // Test our TinyMPC projection function
    Matrix3f projected = project_psd<3>(lmi_matrix, 1e-6f);
    cout << "Projected matrix:\n" << projected << endl;
    
    // Check eigenvalues
    SelfAdjointEigenSolver<Matrix3f> solver(projected);
    cout << "Projected eigenvalues: " << solver.eigenvalues().transpose() << endl;
    
    float diff_norm = (lmi_matrix - projected).norm();
    cout << "Projection needed: " << (diff_norm > 1e-3f ? "YES" : "NO") << endl;
    cout << "Difference norm: " << diff_norm << endl;
}

void simulate_obstacle_avoidance() {
    cout << "\n=== Simulating with TinyMPC-style Setup ===" << endl;
    
    vector<Vector4f> states;
    vector<Vector2f> controls;
    vector<bool> safety_status;
    
    Vector4f x_current = x_initial;
    Vector4f x_goal(0.0f, 0.0f, 0.0f, 0.0f);
    
    states.push_back(x_current);
    
    for (int k = 0; k < NSTEPS - 1; k++) {
        cout << "\n--- Step " << k << " ---" << endl;
        
        // Set reference trajectory towards goal with obstacle avoidance
        for (int i = 0; i < NHORIZON; i++) {
            float alpha = float(k + i) / float(NSTEPS);
            Vector4f ref_state = (1.0f - alpha) * x_current + alpha * x_goal;
            
            // Add repulsive reference away from obstacle
            Vector2f ref_pos = ref_state.head<2>();
            Vector2f to_obs = ref_pos - x_obs;
            float dist_to_obs = to_obs.norm();
            
            if (dist_to_obs < 4.0f * r_obs) {
                float push_magnitude = 1.0f / (dist_to_obs + 0.1f);
                ref_pos += push_magnitude * to_obs.normalized();
                ref_state.head<2>() = ref_pos;
            }
            
            work.Xref.col(i) = ref_state;
        }
        
        // Set control reference to zero
        work.Uref.setZero();
        
        // Set initial state
        work.x.col(0) = x_current;
        
        // Test SDP constraint at current state
        test_sphere_lmi_projection(x_current);
        
        // Simple control law (since we're not running full ADMM)
        Vector2f u_goal = -0.1f * x_current.head<2>() - 0.05f * x_current.tail<2>();
        
        // Obstacle avoidance
        Vector2f pos = x_current.head<2>();
        Vector2f to_obs = pos - x_obs;
        float dist = to_obs.norm();
        
        Vector2f u_avoid(0.0f, 0.0f);
        if (dist < 3.0f * r_obs) {
            float force_mag = 2.0f / (dist * dist + 0.1f);
            u_avoid = force_mag * to_obs.normalized();
        }
        
        Vector2f u_total = u_goal + u_avoid;
        // Clamp to bounds
        u_total = u_total.cwiseMax(work.bounds->u_min.col(0)).cwiseMin(work.bounds->u_max.col(0));
        
        controls.push_back(u_total);
        
        // Check safety
        bool safe = check_collision(x_current);
        safety_status.push_back(safe);
        
        cout << "Control: [" << u_total.transpose() << "]" << endl;
        cout << "Safe: " << (safe ? "YES" : "NO") << endl;
        
        // Simulate forward
        x_current = work.Adyn * x_current + work.Bdyn * u_total;
        states.push_back(x_current);
        
        // Check if reached goal
        if (x_current.head<2>().norm() < 0.5f) {
            cout << "✅ Reached goal at step " << k + 1 << endl;
            break;
        }
    }
    
    // Save results
    ofstream file("tinympc_results.csv");
    file << "step,pos_x,pos_y,vel_x,vel_y,ctrl_x,ctrl_y,safe,obs_x,obs_y,obs_radius" << endl;
    
    for (size_t k = 0; k < states.size(); k++) {
        file << k << "," << states[k](0) << "," << states[k](1) 
             << "," << states[k](2) << "," << states[k](3);
        
        if (k < controls.size()) {
            file << "," << controls[k](0) << "," << controls[k](1);
        } else {
            file << ",0,0";
        }
        
        if (k < safety_status.size()) {
            file << "," << (safety_status[k] ? 1 : 0);
        } else {
            file << ",1";
        }
        
        file << "," << x_obs(0) << "," << x_obs(1) << "," << r_obs << endl;
    }
    file.close();
    
    cout << "\n=== Results Summary ===" << endl;
    int safe_count = 0;
    for (bool safe : safety_status) if (safe) safe_count++;
    
    cout << "Total steps: " << safety_status.size() << endl;
    cout << "Safe steps: " << safe_count << "/" << safety_status.size() << endl;
    cout << "Final position: [" << states.back().head<2>().transpose() << "]" << endl;
    cout << "Results saved to tinympc_results.csv" << endl;
    cout << "\n✅ TinyMPC SDP test complete! Use Python to plot the CSV data." << endl;
}

int main() {
    cout << "TinyMPC SDP Obstacle Avoidance Test" << endl;
    cout << "===================================" << endl;
    
    setup_tinympc_problem();
    simulate_obstacle_avoidance();
    
    return 0;
}

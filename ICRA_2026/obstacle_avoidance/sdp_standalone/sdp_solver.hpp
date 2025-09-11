#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>
#include <fstream>

using namespace Eigen;
using tinytype = double;

// Problem dimensions from Julia formulation
const int nx = 4;          // Physical state dimension
const int nu = 2;          // Physical control dimension  
const int nx_ext = 20;     // Extended state dimension (4 + 16)
const int nu_ext = 22;     // Extended control dimension (2 + 8 + 8 + 4)
const int NHORIZON = 31;   // Time horizon from Julia

// ADMM parameters (from safety_filter.py)
const double rho = 1e2;
const double abs_pri_tol = 1.0e-2;
const double abs_dual_tol = 1.0e-2;
const int max_iter = 50;
const int check_termination = 1;

// Problem parameters (from Julia)
const double h = 1.0;      // Time step (Julia uses discrete time)
const Vector2d x_obs(-5.0, 0.0);  // Obstacle center
const double r_obs = 2.0;          // Obstacle radius

// System matrices (from Julia - discrete double integrator)
const Matrix4d Ad = (Matrix4d() << 
    1, 0, 1, 0,
    0, 1, 0, 1,
    0, 0, 1, 0,
    0, 0, 0, 1).finished();

const Matrix<double, 4, 2> Bd = (Matrix<double, 4, 2>() <<
    0.5, 0,
    0, 0.5,
    1, 0,
    0, 1).finished();

// Cost weights (from Julia)
const double q_xx = 0.1;
const double r_xx = 10.0;
const double R_xx = 500.0;
const double reg = 1e-6;

/**
 * PSD projection function - your exact implementation
 * Projects matrix onto positive semidefinite cone using eigendecomposition
 */
template<int M>
EIGEN_STRONG_INLINE Matrix<tinytype, M, M>
project_psd(const Matrix<tinytype, M, M>& S_in, tinytype eps = tinytype(1e-8))
{
    using MatM = Matrix<tinytype, M, M>;
    
    // 1) Make symmetric to avoid numerical asymmetry
    MatM S = tinytype(0.5) * (S_in + S_in.transpose());

    // 2) Eigendecomposition (self-adjoint is fastest & most stable)
    SelfAdjointEigenSolver<MatM> es;
    es.compute(S, ComputeEigenvectors);
    
    // 3) Clamp eigenvalues to be nonnegative (or eps floor)
    Matrix<tinytype, M, 1> d = es.eigenvalues();
    for (int i = 0; i < M; ++i) {
        d(i) = d(i) < eps ? eps : d(i);
    }

    // 4) Reconstruct: V * diag(d) * V^T
    const MatM& V = es.eigenvectors();
    return V * d.asDiagonal() * V.transpose();
}

/**
 * SDP Solver Class
 * Implements the Julia SDP formulation with TinyMPC-style ADMM + your PSD projections
 */
class SDPSolver {
public:
    // Extended system matrices (using Kronecker products from Julia)
    Matrix<double, nx_ext, nx_ext> A_ext;
    Matrix<double, nx_ext, nu_ext> B_ext;
    
    // Cost matrices
    Matrix<double, nx_ext, nx_ext> Q_ext;
    Matrix<double, nu_ext, nu_ext> R_ext;
    
    // Problem data
    Vector<double, nx_ext> x_initial;
    Vector<double, nx_ext> x_goal;
    
    // ADMM variables
    Matrix<double, nx_ext, NHORIZON> x;      // States
    Matrix<double, nu_ext, NHORIZON-1> u;    // Controls
    Matrix<double, nx_ext, NHORIZON> v;      // State slack variables
    Matrix<double, nu_ext, NHORIZON-1> z;    // Control slack variables
    Matrix<double, nx_ext, NHORIZON> g;      // State dual variables
    Matrix<double, nu_ext, NHORIZON-1> y;    // Control dual variables
    
    // Trajectory storage
    Matrix<double, nx, NHORIZON> x_traj;     // Physical trajectory
    Matrix<double, nu, NHORIZON-1> u_traj;   // Physical controls
    
    SDPSolver();
    void setup_extended_system();
    void setup_cost_matrices();
    void setup_initial_condition();
    bool solve();
    void extract_physical_trajectory();
    void save_results(const std::string& filename);
    
private:
    void admm_iteration();
    void update_primal();
    void update_slack_with_sdp_projection();
    void update_dual();
    bool check_convergence();
    double compute_primal_residual();
    double compute_dual_residual();
    
    // SDP projection helpers
    void project_moment_matrix(int k);
    void project_terminal_matrix();
    Matrix<double, 7, 7> build_moment_matrix(int k);
    Matrix<double, 5, 5> build_terminal_matrix();
    void extract_from_moment_matrix(const Matrix<double, 7, 7>& M, int k);
    void extract_from_terminal_matrix(const Matrix<double, 5, 5>& M);
    
    // Obstacle constraint projection
    void project_obstacle_constraint(int k);
    
    // Obstacle constraint
    double obstacle_constraint_value(const Vector4d& x_phys, const Matrix4d& XX);
    
    int iteration;
    double primal_residual;
    double dual_residual;
};

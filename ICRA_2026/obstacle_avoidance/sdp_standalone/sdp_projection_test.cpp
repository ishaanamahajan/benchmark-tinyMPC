// g++ -O2 -std=c++17 sdp_projection_test.cpp -I /usr/include/eigen3 -o sdp_proj_test
// Run: ./sdp_proj_test
//
//
// Verification:
//  1) eigmin(M_proj) >= -tol
//  2) Schur complement nonnegativity (see functions below)
//  3) Idempotence: ||Proj(Proj(M)) - Proj(M)|| small

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using std::cout;
using std::endl;

using tinytype = double;
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector2d;
using Eigen::Vector4d;
using Eigen::Matrix2d;
using Eigen::Matrix4d;

// ------------------------------- Problem setup ------------------------------ //
static constexpr int N  = 31;  // horizon
static constexpr int nx = 4;   // state: [px, py, vx, vy]
static constexpr int nu = 2;   // control: [ax, ay]

// Discrete double integrator (same as your Julia/C++ code)
const Matrix4d Ad = (Matrix4d() <<
    1, 0, 1, 0,
    0, 1, 0, 1,
    0, 0, 1, 0,
    0, 0, 0, 1).finished();

const Eigen::Matrix<double, 4, 2> Bd = (Eigen::Matrix<double, 4, 2>() <<
    0.5, 0.0,
    0.0, 0.5,
    1.0, 0.0,
    0.0, 1.0).finished();

const Vector4d x0(-10.0, 0.1, 0.0, 0.0);

// ----------------------------- Utility helpers ----------------------------- //

template<int M>
inline tinytype eigmin(const Matrix<tinytype, M, M>& A) {
    Eigen::SelfAdjointEigenSolver<Matrix<tinytype, M, M>> es(A);
    return es.eigenvalues().minCoeff();
}

inline tinytype eigmin_dyn(const MatrixXd& A) {
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(A);
    return es.eigenvalues().minCoeff();
}

template<int M>
inline Matrix<tinytype, M, M> symmetrize(const Matrix<tinytype, M, M>& A) {
    return tinytype(0.5) * (A + A.transpose());
}

// Your projection: symmetrize → eigendecomp → clamp → reconstruct
template<int M>
Matrix<tinytype, M, M> project_psd(const Matrix<tinytype, M, M>& S_in, tinytype eps = 1e-8)
{
    using MatM = Matrix<tinytype, M, M>;
    MatM S = symmetrize<M>(S_in);

    Eigen::SelfAdjointEigenSolver<MatM> es;
    es.compute(S, Eigen::ComputeEigenvectors);

    Matrix<tinytype, M, 1> d = es.eigenvalues();
    for (int i = 0; i < M; ++i) d(i) = d(i) < eps ? eps : d(i);

    const MatM& V = es.eigenvectors();
    return V * d.asDiagonal() * V.transpose();
}

// Idempotence check: ||Proj(Proj(M)) - Proj(M)||_F / max(1, ||Proj(M)||_F)
template<int M>
tinytype projection_idempotence_score(const Matrix<tinytype, M, M>& Mproj, tinytype eps = 1e-8) {
    Matrix<tinytype, M, M> Mproj2 = project_psd<M>(Mproj, eps);
    tinytype num = (Mproj2 - Mproj).norm();
    tinytype den = std::max<tinytype>(1.0, Mproj.norm());
    return num / den;
}

// ---------------------- Moment-matrix constructors (LaTeX) ------------------ //
//
// Running step (k = 0..N-2), 7x7:
//
//   M_k = [ 1   x'    u'
//           x   XX    XU
//           u   UX    UU ]
//
// Terminal step (k = N-1), 5x5:
//
//   M_N = [ 1   x'
//           x   XX ]
//

// 7x7 builder
inline Matrix<tinytype, 7, 7>
build_moment_7(const Vector4d& x, const Vector2d& u,
               const Matrix4d* XX_in = nullptr,
               const Eigen::Matrix<double,4,2>* XU_in = nullptr,
               const Eigen::Matrix<double,2,4>* UX_in = nullptr,
               const Matrix2d* UU_in = nullptr)
{
    Matrix<tinytype, 7, 7> M; M.setZero();

    // default "consistent" moments (rank-1)
    Matrix4d XX = XX_in ? *XX_in : (x * x.transpose());
    Eigen::Matrix<double,4,2> XU = XU_in ? *XU_in : (x * u.transpose());
    Eigen::Matrix<double,2,4> UX = UX_in ? *UX_in : (u * x.transpose());
    Matrix2d UU = UU_in ? *UU_in : (u * u.transpose());

    M(0,0) = 1.0;
    M.block<1,4>(0,1) = x.transpose();
    M.block<1,2>(0,5) = u.transpose();

    M.block<4,1>(1,0) = x;
    M.block<4,4>(1,1) = XX;
    M.block<4,2>(1,5) = XU;

    M.block<2,1>(5,0) = u;
    M.block<2,4>(5,1) = UX;
    M.block<2,2>(5,5) = UU;

    return M;
}

// 5x5 terminal builder
inline Matrix<tinytype, 5, 5>
build_terminal_5(const Vector4d& x, const Matrix4d* XX_in = nullptr)
{
    Matrix<tinytype, 5, 5> M; M.setZero();
    Matrix4d XX = XX_in ? *XX_in : (x * x.transpose());

    M(0,0) = 1.0;
    M.block<1,4>(0,1) = x.transpose();

    M.block<4,1>(1,0) = x;
    M.block<4,4>(1,1) = XX;
    return M;
}

// --------------------- Corruption (make matrices indefinite) ---------------- //

inline void corrupt_second_moments_inplace(Matrix<tinytype,7,7>& M, tinytype gamma = 1.5)
{
    // Lower-right 6x6 block: [XX XU; UX UU]
    M.block<6,6>(1,1) -= gamma * Matrix<tinytype,6,6>::Identity();
    M = symmetrize<7>(M);
}

inline void corrupt_second_moments_inplace(Matrix<tinytype,5,5>& M, tinytype gamma = 1.5)
{
    // Lower-right 4x4 block: XX
    M.block<4,4>(1,1) -= gamma * Matrix<tinytype,4,4>::Identity();
    M = symmetrize<5>(M);
}

// ------------------------- Schur-complement checks -------------------------- //
//
// For M = [ 1  b'
//           b  C ], PSD  ⇔  1>0 and  C - (1)^{-1} b b'  ⪰ 0
//
// 7x7: b = [x; u] ∈ R^6, C = [XX XU; UX UU] ∈ R^{6×6}
// 5x5: b = x ∈ R^4,     C = XX ∈ R^{4×4}
//

inline std::pair<tinytype, tinytype>
schur_min_7(const Matrix<tinytype,7,7>& M)
{
    const tinytype alpha = M(0,0);
    const Matrix<tinytype,6,1> b = M.block<6,1>(1,0);
    const Matrix<tinytype,6,6> C = M.block<6,6>(1,1);

    Matrix<tinytype,6,6> S = C - (b * b.transpose()) / alpha;
    tinytype lambda_min = eigmin<6>(S);
    return {alpha, lambda_min};
}

inline std::pair<tinytype, tinytype>
schur_min_5(const Matrix<tinytype,5,5>& M)
{
    const tinytype alpha = M(0,0);
    const Matrix<tinytype,4,1> b = M.block<4,1>(1,0);
    const Matrix<tinytype,4,4> C = M.block<4,4>(1,1);

    Matrix<tinytype,4,4> S = C - (b * b.transpose()) / alpha;
    tinytype lambda_min = eigmin<4>(S);
    return {alpha, lambda_min};
}

// ------------------------------ Trajectory --------------------------------- //

struct Rollout {
    std::vector<Vector4d> x;  // size N
    std::vector<Vector2d> u;  // size N-1
};

Rollout rollout_trajectory()
{
    Rollout R;
    R.x.resize(N);
    R.u.resize(N-1);

    R.x[0] = x0;
    for (int k = 0; k < N-1; ++k) {
        Vector2d pos = R.x[k].head<2>();
        Vector2d vel = R.x[k].tail<2>();

        // Same seed logic used in your C++: goal attraction with saturation
        Vector2d u_total = -0.15 * pos - 0.08 * vel;
        u_total = u_total.cwiseMax(-1.0).cwiseMin(1.0);

        R.u[k] = u_total;
        R.x[k+1] = Ad * R.x[k] + Bd * u_total;
    }
    return R;
}

// ---------------------------------- Main ----------------------------------- //

int main() {
    std::cout.setf(std::ios::fixed); std::cout << std::setprecision(6);

    const tinytype eps = 1e-8;   // clamp floor
    const tinytype tol = 1e-8;   // numerical tolerance for checks
    const tinytype gamma = 1.5;  // severity of corruption

    cout << "============================================================\n";
    cout << "SDP Projection Algorithm Test (C++ Version)\n";
    cout << "Testing your project_psd<M>() on Julia's exact matrices\n";
    cout << "============================================================\n";

    auto R = rollout_trajectory();

    int ok_psd_7 = 0, ok_schur_7 = 0;
    tinytype worst_after_7 = +1e9, worst_schur_7 = +1e9; // track minima
    tinytype worst_idemp_7 = 0.0;

    std::cout << "== 7x7 moment-matrix projection across horizon ==\n";
    for (int k = 0; k < N-1; ++k) {
        const Vector4d& x = R.x[k];
        const Vector2d& u = R.u[k];

        // 1) Build rank-1-consistent moment matrix
        Matrix<tinytype,7,7> M = build_moment_7(x, u);

        // 2) Corrupt second-moment block to force indefiniteness
        corrupt_second_moments_inplace(M, gamma);

        tinytype eig_before = eigmin<7>(M);

        // 3) Project
        Matrix<tinytype,7,7> Mp = project_psd<7>(M, eps);
        tinytype eig_after = eigmin<7>(Mp);

        // 4) Schur complement
        auto [alpha, schur_min] = schur_min_7(Mp);

        // 5) Idempotence
        tinytype idemp = projection_idempotence_score<7>(Mp, eps);

        if (k < 3) {
            std::cout << " step " << k
                      << ": eigmin before = " << std::setw(9) << eig_before
                      << "   after = "       << std::setw(9) << eig_after
                      << "   alpha = "       << std::setw(9) << alpha
                      << "   Schur λmin = "  << std::setw(9) << schur_min
                      << "   idemp = "       << idemp << "\n";
        }

        if (eig_after >= -tol)  { ok_psd_7++; }
        if (schur_min >= -tol)  { ok_schur_7++; }
        worst_after_7 = std::min(worst_after_7, eig_after);
        worst_schur_7 = std::min(worst_schur_7, schur_min);
        worst_idemp_7 = std::max(worst_idemp_7, idemp);
    }

    std::cout << " summary: " << (N-1) << " moment matrices\n"
              << "   PSD after projection     : " << ok_psd_7   << "/" << (N-1) << " (tol=" << tol << ")\n"
              << "   Schur complement ≥ 0     : " << ok_schur_7 << "/" << (N-1) << " (tol=" << tol << ")\n"
              << "   worst eigmin after       : " << worst_after_7 << "\n"
              << "   worst Schur λmin         : " << worst_schur_7 << "\n"
              << "   worst idempotence score  : " << worst_idemp_7 << "\n";

    // Terminal 5x5
    std::cout << "\n== 5x5 terminal matrix projection ==\n";
    Matrix<tinytype,5,5> MT = build_terminal_5(R.x[N-1]);
    corrupt_second_moments_inplace(MT, gamma);
    tinytype eig_before_T = eigmin<5>(MT);

    Matrix<tinytype,5,5> MTp = project_psd<5>(MT, eps);
    tinytype eig_after_T = eigmin<5>(MTp);
    auto [alphaT, schur_min_T] = schur_min_5(MTp);
    tinytype idemp_T = projection_idempotence_score<5>(MTp, eps);

    std::cout << " terminal: eigmin before = " << eig_before_T
              << "   after = " << eig_after_T
              << "   alpha = " << alphaT
              << "   Schur λmin = " << schur_min_T
              << "   idemp = " << idemp_T << "\n";

    bool ok7 = (ok_psd_7 == (N-1)) && (ok_schur_7 == (N-1));
    bool ok5 = (eig_after_T >= -tol) && (schur_min_T >= -tol);

    std::cout << "\n============================================================\n";
    std::cout << "FINAL VALIDATION RESULTS\n";
    std::cout << "============================================================\n";
    std::cout << "7x7 moment matrices: " << (ok7 ? "PASS" : "FAIL") << "\n";
    std::cout << "5x5 terminal matrix: " << (ok5 ? "PASS" : "FAIL") << "\n";
    
    if (ok7 && ok5) {
        std::cout << "\n Project_psd<M>() algorithm works perfectly!\n";
    } else {
        std::cout << "\nFailed\n";
    }

    return (ok7 && ok5) ? 0 : 1;
}


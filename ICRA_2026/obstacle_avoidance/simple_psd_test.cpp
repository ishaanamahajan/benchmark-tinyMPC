#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef float tinytype;

// Simple PSD projection function for testing
template<int M>
Matrix<tinytype, M, M> project_psd(const Matrix<tinytype, M, M>& S_in, tinytype eps = 0.0f)
{
    using MatM = Matrix<tinytype, M, M>;
    
    // 1) Make symmetric
    MatM S = tinytype(0.5) * (S_in + S_in.transpose());

    // 2) Eigendecomposition
    SelfAdjointEigenSolver<MatM> es(S);
    
    // 3) Clamp eigenvalues
    Matrix<tinytype, M, 1> d = es.eigenvalues();
    for (int i = 0; i < M; ++i) {
        d(i) = d(i) < eps ? eps : d(i);
    }

    // 4) Reconstruct
    const MatM& V = es.eigenvectors();
    return V * d.asDiagonal() * V.transpose();
}

int main() {
    cout << "Testing PSD Projection Function" << endl;
    cout << "================================" << endl;
    
    // Test 1: 2x2 matrix with one negative eigenvalue
    cout << "\nTest 1: 2x2 matrix" << endl;
    Matrix2f test1;
    test1 << 1.0f, 2.0f,
             2.0f, 1.0f;  // Eigenvalues should be 3, -1
    
    cout << "Input matrix:" << endl << test1 << endl;
    
    // Check original eigenvalues
    SelfAdjointEigenSolver<Matrix2f> original_solver(test1);
    cout << "Original eigenvalues: " << original_solver.eigenvalues().transpose() << endl;
    
    Matrix2f projected1 = project_psd<2>(test1, 0.0f);
    cout << "Projected matrix:" << endl << projected1 << endl;
    
    // Check projected eigenvalues
    SelfAdjointEigenSolver<Matrix2f> solver1(projected1);
    cout << "Projected eigenvalues: " << solver1.eigenvalues().transpose() << endl;
    
    // Test 2: 4x4 matrix (like the Julia SDP)
    cout << "\nTest 2: 4x4 matrix (Julia SDP style)" << endl;
    Matrix4f test2;
    test2 << 1.0f,  -2.0f,  1.5f,  0.5f,
            -2.0f,   2.0f, -1.0f,  0.3f,
             1.5f,  -1.0f,  3.0f, -0.8f,
             0.5f,   0.3f, -0.8f,  1.5f;
    
    cout << "Input 4x4 matrix:" << endl << test2 << endl;
    
    // Check original eigenvalues
    SelfAdjointEigenSolver<Matrix4f> original_solver2(test2);
    cout << "Original eigenvalues: " << original_solver2.eigenvalues().transpose() << endl;
    
    Matrix4f projected2 = project_psd<4>(test2, 1e-6f);
    cout << "Projected 4x4 matrix:" << endl << projected2 << endl;
    
    // Check projected eigenvalues
    SelfAdjointEigenSolver<Matrix4f> solver2(projected2);
    cout << "Projected eigenvalues: " << solver2.eigenvalues().transpose() << endl;
    
    cout << "\nAll tests completed successfully!" << endl;
    return 0;
}

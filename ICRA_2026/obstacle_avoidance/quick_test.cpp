#include "tinympc/tinympc_teensy/src/admm.cpp"
#include <iostream>
int main() {
    Eigen::Matrix4f test;
    test << 1, -2, 1.5, 0.5, -2, 2, -1, 0.3, 1.5, -1, 3, -0.8, 0.5, 0.3, -0.8, 1.5;
    auto result = project_psd<4>(test, 1e-6f);
    std::cout << "TinyMPC projection test passed!" << std::endl;
    return 0;
}

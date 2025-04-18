#include <iostream>
#include <vector>
#include <cmath>
using namespace std;
int main() {
    // 定义矩阵大小
    const int N = 100; // 可以修改 N 的值来改变矩阵大小
    const int max_iter = 10;
    const float tolerance = 1e-6;

    // 初始化系数矩阵 A 和向量 b
    std::vector<float> A(N * N, 0.0f);
    std::vector<float> b(N, 0.0f);
    std::vector<float> x(N, 0.0f);     // 初始解
    std::vector<float> x_new(N, 0.0f); // 更新后的解

    // 自动初始化 A 和 b
    for (int i = 0; i < N; ++i) {
        A[i * N + i] = 4.0f; // 对角线元素，确保对角占优

        if (i > 0) {
            A[i * N + i - 1] = -1.0f; // 下三角元素
        }
        if (i < N - 1) {
            A[i * N + i + 1] = -1.0f; // 上三角元素
        }

        b[i] = 1.0f; // 初始化向量 b，可根据需要修改
    }

    bool converged = false;
    int iter = 0;

    while (!converged && iter < max_iter) {
        // 计算新的解向量 x_new
        for (int i = 0; i < N; ++i) {
            float sigma = 0.0f;
            // 计算当前 x_new[i]
            for (int j = 0; j < N; ++j) {
                if (j != i) {
                    sigma += A[i * N + j] * x[j];
                }
            }
            x_new[i] = (b[i] - sigma) / A[i * N + i];
        }

        // 计算收敛性
        float max_error = 0.0f;
        for (int i = 0; i < N; ++i) {
            max_error = std::max(max_error, std::fabs(x_new[i] - x[i]));
        }

        if (max_error < tolerance) {
            converged = true;
        }

        // 更新 x
        for (int i = 0; i < N; ++i) {
            x[i] = x_new[i];
        }

        ++iter;
    }

    // 输出结果
    std::cout << "迭代次数: " << iter << std::endl;
    std::cout << "解向量 x:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

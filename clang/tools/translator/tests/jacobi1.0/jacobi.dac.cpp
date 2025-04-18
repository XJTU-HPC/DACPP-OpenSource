#include <iostream>
#include <vector>
#include <cmath>
#include "ReconTensor.h"
// 定义矩阵大小
const int N = 100; // 可以修改 N 的值来改变矩阵大小
const int max_iter = 10000;
const float tolerance = 1e-6;
namespace dacpp {
    typedef std::vector<std::any> list;
}


shell dacpp::list jacobiShell(const dacpp::Matrix<float>& A, 
                                const dacpp::Vector<float>& b, 
                                const dacpp::Vector<float>& x, 
                                dacpp::Vector<float>& x_new,
                                const dacpp::Vector<int>& nums) {
    dacpp::index idx1;
//     dacpp::Index idx2("idx1");
//     dacpp::Index idx3("idx1");
//     dacpp::Index idx4("idx1");
//    // dacpp::Index idx5("idx1");
//     binding(idx1,idx2);
//     binding(idx2,idx3);
//     binding(idx3,idx4);


    dacpp::list dataList{A[{idx1}][{}], b[{idx1}], x[{}], x_new[{idx1}], nums[{idx1}]};
    return dataList;
}

calc void jacobi(float* a, 
                float* b, 
                float* x, 
                float* x_new, 
                int* num) {
    float sigma = 0;
    for(int i = 0;i < N;++i) {
        if(i != num[0]) {
            sigma += a[i] * x[i];
        }
    }
    x_new[0] = (b[0] - sigma) / a[num[0]];
    
}

int main() {


    // 初始化系数矩阵 A 和向量 b
    std::vector<float> mat_A(N * N, 0.0f);
    std::vector<float> vec_b(N, 0.0f);
    std::vector<float> vec_x(N, 0.0f);     // 初始解
    std::vector<float> vec_x_new(N, 0.0f); // 更新后的解

    // 自动初始化 A 和 b
    for (int i = 0; i < N; ++i) {
        mat_A[i * N + i] = 4.0f; // 对角线元素，确保对角占优

        if (i > 0) {
            mat_A[i * N + i - 1] = -1.0f; // 下三角元素
        }
        if (i < N - 1) {
            mat_A[i * N + i + 1] = -1.0f; // 上三角元素
        }

        vec_b[i] = 1.0f; // 初始化向量 b，可根据需要修改
    }

    // std::vector<int> A_shape = {100, 100};
    // std::vector<int> b_shape = {100};
    // std::vector<int> x_shape = {100};
    // std::vector<int> x_new_shape = {100};
    dacpp::Matrix<float> A({100, 100}, mat_A);
    dacpp::Vector<float> b(vec_b);
    dacpp::Vector<float> x(vec_x);
    dacpp::Vector<float> x_new(vec_x_new);
    
    bool converged = false;
    int iter = 0;
    std::vector<int> nums(100);
    // 使用 std::iota 填充 nums，值从 0 开始
    for(int i = 0;i < N;  i++){
        nums[i] = i;
    }
    //std::vector<int> nums_shape = {100};
    dacpp::Vector<int> tensor_nums(nums);
    float* data = new float[1 * 100];
    float* data2 = new float[1 * 100];

    while (!converged && iter < max_iter) {
        jacobiShell(A, b, x, x_new, tensor_nums) <-> jacobi;
        
        x.tensor2Array(data);
        x_new.tensor2Array(data2);

        float max_error = 0.0f;
        for (int i = 0; i < N; ++i) {
            max_error = std::max(max_error, std::fabs(data2[i] - data[i]));
        }

        if (max_error < tolerance) {
            converged = true;
        }

        // 更新 x
        x=x_new;

        ++iter;
    }


    // 输出结果
    std::cout << "迭代次数: " << iter << std::endl;
    std::cout << "解向量 x:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << data2[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

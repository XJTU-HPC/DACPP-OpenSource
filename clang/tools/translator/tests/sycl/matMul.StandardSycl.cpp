#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
    // 定义矩阵A和B
    float A[4][5] = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20}
    };
    
    float B[5][4] = {
        {1, 5, 9, 13},
        {17, 2, 6, 10},
        {14, 18, 3, 7},
        {11, 15, 19, 4},
        {8, 12, 16, 20}
    };
    
    // 结果矩阵C
    float C[4][4] = {0};

    // 创建一个默认队列
    queue q;
    
    {
        // 使用缓冲区将数据传递到设备
        buffer<float, 2> A_buf(&A[0][0], range<2>(4, 5));
        buffer<float, 2> B_buf(&B[0][0], range<2>(5, 4));
        buffer<float, 2> C_buf(&C[0][0], range<2>(4, 4));

        // 提交任务到队列
        q.submit([&](handler& h) {
            // 获取A、B、C的访问器
            auto A_acc = A_buf.get_access<access::mode::read>(h);
            auto B_acc = B_buf.get_access<access::mode::read>(h);
            auto C_acc = C_buf.get_access<access::mode::write>(h);

            // 确保计算过程是按矩阵乘法的规则进行的
            h.parallel_for(range<2>(4, 4), [=](id<2> idx) {
                int i = idx[0]; // C的行
                int j = idx[1]; // C的列
                float sum = 0.0f;
                
                // 计算C[i][j]
                for (int k = 0; k < 5; ++k) {
                    sum += A_acc[i][k] * B_acc[k][j];
                }
                C_acc[i][j] = sum;
            });
        });
    }

    // 输出结果矩阵C
    std::cout << "Matrix C (Result of A * B):\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

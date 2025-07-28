#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <CL/sycl.hpp>

using namespace std;
using Complex = complex<double>;  // 复数类型别名

// 离散傅里叶变换（DFT）
void dft(const vector<Complex>& input, vector<Complex>& output) {
    int N = input.size();
    output.resize(N);

    // DFT 公式：X[k] = Σ (x[n] * e^(-2πi * k * n / N)), k=0 to N-1
    for (int k = 0; k < N; ++k) {
        //std::cout << "k=" << k << std::endl;
        Complex sum(0, 0);
        for (int n = 0; n < N; ++n) {
            double angle = -2.0 * M_PI * k * n / N;
            Complex W_n(cos(angle), sin(angle));  // 旋转因子
            sum += input[n] * W_n;  // 累加每一项
        }
        output[k] = sum;  // 傅里叶变换结果
    }
}

int main() {
    double total_program = 0;
    auto program_start = std::chrono::high_resolution_clock::now();
    // 定义一个输入信号（长度为8的复数序列）
    int N = 64;
    vector<Complex> input(N);
    
    // 初始化输入数据（可以是任何时间域信号）
    for (int i = 0; i < N; ++i) {
        input[i] = Complex(i, 0);  // 以复数形式填充数据，这里只是简单的填充数据
    }

    // 输出原始数据
    //cout << "原始数据（时间域）:" << endl;
    // for (const auto& val : input) {
    //     cout << val << endl;
    // }

    // 计算离散傅里叶变换
    vector<Complex> output;
    dft(input, output);

    // 输出傅里叶变换后的数据（频域）
    //cout << "\n傅里叶变换后的数据（频域）:" << endl;
    std::cout << "{";
    for (size_t i = 0; i < output.size(); ++i) {
        cout << "(" << output[i].real() << "," << output[i].imag() << ")";
        if (i != output.size() - 1) {
            cout << ", ";  // 避免最后一个元素后面多一个逗号
        }
    }
    std::cout << "}\n";

    auto program_end = std::chrono::high_resolution_clock::now();
    total_program += std::chrono::duration<double>(program_end - program_start).count();
    std::cout << "total_program: " << total_program << "s" << std::endl;

    return 0;
}
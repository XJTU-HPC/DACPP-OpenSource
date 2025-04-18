#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

using namespace std;
using Complex = std::complex<double>;  // 复数类型别名
const int N = 8;


shell dacpp::list DFT(const dacpp::Vector<std::complex<double>>& input,
                        dacpp::Vector<std::complex<double>>& output, 
                    const dacpp::Vector<int>& vec) {
    dacpp::index i;
    // dacpp::split s(3,1);
    // binding(i, s);
    dacpp::list dataList{input[{}], output[i], vec[i]};
    return dataList;
}

calc void dft(std::complex<double>* input,
                std::complex<double>* output, 
                int* vec) {
    Complex sum(0, 0);
    for (int n = 0; n < N; ++n) {
        double angle = -2.0 * M_PI * vec[0] * n / N;
        Complex W_n(std::cos(angle), std::sin(angle));  // 旋转因子
        sum += input[n] * W_n;  // 累加每一项
    }
    output[0] = sum;  // 傅里叶变换结果
}

// 离散傅里叶变换（DFT）
void dftfunc(const vector<std::complex<double>>& input, vector<std::complex<double>>& output) {
    int N = input.size();
    output.resize(N);

    std::vector<int> vec(N);

    // 使用 for 循环初始化 vector，元素从 1 到 N
    for (int i = 0; i < N; ++i) {
        vec[i] = i;  // 赋值为 1 到 N
    }
    dacpp::Vector<int> vec_tensor(vec);
    dacpp::Vector<std::complex<double>> input_tensor(input);
    dacpp::Vector<std::complex<double>> output_tensor(output);

    // DFT 公式：X[k] = Σ (x[n] * e^(-2πi * k * n / N)), k=0 to N-1
    DFT(input_tensor, output_tensor, vec_tensor) <-> dft;
    output_tensor.print();
}

int main() {
    // 定义一个输入信号（长度为8的复数序列）

    vector<std::complex<double>> input(N);
    
    // 初始化输入数据（可以是任何时间域信号）
    for (int i = 0; i < N; ++i) {
        input[i] = Complex(i, 0);  // 以复数形式填充数据，这里只是简单的填充数据
    }

    // 输出原始数据
    std::cout << "原始数据（时间域）:" << std::endl;
    for (const auto& val : input) {
        std::cout << val << std::endl;
    }

    // 计算离散傅里叶变换
    vector<Complex> output(N);
    dftfunc(input, output);

    // // 输出傅里叶变换后的数据（频域）
    // cout << "\n傅里叶变换后的数据（频域）:" << endl;
    // for (const auto& val : output) {
    //     cout << val << endl;
    // }

    return 0;
}

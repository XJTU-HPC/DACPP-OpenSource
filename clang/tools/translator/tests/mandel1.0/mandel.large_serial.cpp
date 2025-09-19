#include <iostream>
#include <vector>
#include <complex>

using namespace std;

// 全局变量定义
int row_count = 256, col_count = 256, max_iterations = 10000;
vector<complex<double>> complex_points;  // 一维向量表示复数点
vector<int> mandelbrot_flags;           // 一维数组表示是否属于 Mandelbrot 集
int total_points = 0;                   // 总点数
int mandelbrot_count = 0;               // 属于 Mandelbrot 集的点数量

// 初始化复数点向量
void InitializeComplexPoints() {
    total_points = row_count * col_count;  // 总点数
    complex_points.resize(total_points);

    for (int i = 0; i < row_count; ++i) {
        for (int j = 0; j < col_count; ++j) {
            int index = i * col_count + j;  // 一维向量索引
            double real = -1.5 + (i * (2.0 / row_count));  // 将行索引映射到实部
            double imag = -1.0 + (j * (2.0 / col_count));  // 将列索引映射到虚部
            complex_points[index] = complex<double>(real, imag);
        }
    }
}

// 计算某个点在 Mandelbrot 集中的迭代次数
int ComputePoint(const complex<double>& c) {
    complex<double> z = 0;
    for (int i = 0; i < max_iterations; ++i) {
        if ((std::sqrt(z.real()*z.real()+z.imag()*z.imag())) > 2.0) return i;  // 如果超出边界，停止迭代
        z = z * z + c;
    }
    return max_iterations;  // 如果未发散，返回最大迭代次数
}

// 计算 Mandelbrot 集
void ComputeMandelbrot() {
    mandelbrot_flags.resize(total_points, 0);  // 初始化一维数组为 0

    for (int index = 0; index < total_points; ++index) {
        const complex<double>& c = complex_points[index];  // 获取复数点
        int iterations = ComputePoint(c);
        if (iterations == max_iterations) {
            mandelbrot_flags[index] = 1;  // 设置为 1，表示属于 Mandelbrot 集
        }
    }

    // 统计数组中 1 的个数
    mandelbrot_count = 0;
    for (int flag : mandelbrot_flags) {
        if (flag == 1) mandelbrot_count++;
    }
}

// 打印统计信息
void PrintStats() {
    cout << "Mandelbrot Set Statistics:\n";
    cout << "Total points: " << total_points << "\n";
    cout << "Points in the Mandelbrot set: " << mandelbrot_count << "\n";
}

int main() {
    // 初始化复数点向量
    InitializeComplexPoints();

    // 计算 Mandelbrot 集
    ComputeMandelbrot();

    // 打印统计信息
    PrintStats();

    return 0;
}

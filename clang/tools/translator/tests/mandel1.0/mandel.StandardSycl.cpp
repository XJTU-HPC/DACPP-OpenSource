// mandel.StandardSycl.cpp
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
using namespace sycl;
using namespace std;

const int row_count = 8;
const int col_count = 8;
const int max_iterations = 1000;

// 初始化复数点向量
void InitializeComplexPoints(vector<complex<float>>& complex_points) {
    int total_points = row_count * col_count;
    complex_points.resize(total_points);

    for (int i = 0; i < row_count; ++i) {
        for (int j = 0; j < col_count; ++j) {
            int index = i * col_count + j;

            float real = -1.5f + (i * (2.0f / row_count));
            float imag = -1.0f + (j * (2.0f / col_count));

            complex_points[index] = complex<float>(real, imag);
        }
    }
}

int main() {

    int total_points = row_count * col_count;

    vector<complex<float>> complex_points;
    vector<int> mandelbrot_flags(total_points, 0);

    InitializeComplexPoints(complex_points);

    // SYCL queue
    queue q;

    {
        buffer<complex<float>> buf_points(complex_points.data(), range<1>(total_points));
        buffer<int> buf_flags(mandelbrot_flags.data(), range<1>(total_points));

        // 一个全局 kernel：计算每个点是否属于 Mandelbrot
        q.submit([&](handler& h) {
            auto points = buf_points.get_access<access::mode::read>(h);
            auto flags  = buf_flags.get_access<access::mode::write>(h);

            h.parallel_for(range<1>(total_points), [=](id<1> idx) {
                int tid = idx[0];
                complex<float> c = points[tid];

                complex<float> z = 0;
                int it = 0;

                for (int i = 0; i < max_iterations; ++i) {
                    // <<< 仅此处做了等价替换，避免 sqrt 冲突并提升性能 >>>
                    // float norm = z.real()*z.real() + z.imag()*z.imag();
                    // if (norm > 4.0f) {
                    //     it = i;
                    //     break;
                    // }
                    if (std::sqrt(z.real()*z.real() + z.imag()*z.imag()) > 2.0f) {
                        it = i;
                        break;
                    }

                    z = z * z + c;
                    it = max_iterations;
                }

                if (it == max_iterations) {
                    flags[tid] = 1;  // 属于 Mandelbrot 集
                }
            });
        });
        q.wait();
    }

    // 统计
    int mandelbrot_count = 0;
    for (int f : mandelbrot_flags) {
        if (f == 1) mandelbrot_count++;
    }

    // 输出结果
    std::cout << "Mandelbrot Set Statistics:\n";
    std::cout << "Total points: " << total_points << "\n";
    std::cout << "Points in the Mandelbrot set: " << mandelbrot_count << "\n";

    return 0;
}

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace sycl;
using namespace std;

const double dt = 0.1;
const double T = 5.0;
const size_t numIsotopes = 10000;

int main() {
    queue q{default_selector_v};

    // 初始化 lambdas 与 N0s
    vector<double> lambdas(numIsotopes);
    vector<double> N0s(numIsotopes, 1000.0);
    for (size_t i = 0; i < numIsotopes; ++i)
        lambdas[i] = 0.01 + 0.01 * i;

    // local_A 存储每轮计算结果
    vector<double> local_A(numIsotopes, 0.0);

    // 结果矩阵 A[T/dt][numIsotopes]
    size_t steps = static_cast<size_t>(T / dt);
    vector<double> A((steps + 1) * numIsotopes, 0.0);

    double t = 0.0;

    // 写入初始状态 A[0]
    for (size_t i = 0; i < numIsotopes; ++i)
        A[0 * numIsotopes + i] = N0s[i];

    // SYCL buffers
    buffer<double> bufN0s(N0s.data(), N0s.size());
    buffer<double> bufLambdas(lambdas.data(), lambdas.size());
    buffer<double> bufLocalA(local_A.data(), local_A.size());

    size_t step = 0;

    while (t <= T) {
        double current_t = t;

        // 提交核函数：计算 local_A[i] = N0s[i] * exp(-lambda[i] * t)
        q.submit([&](handler& h) {
            auto N0sAcc = bufN0s.get_access<access::mode::read>(h);
            auto lambdaAcc = bufLambdas.get_access<access::mode::read>(h);
            auto localAAcc = bufLocalA.get_access<access::mode::write>(h);

            h.parallel_for(range<1>(numIsotopes), [=](id<1> idx) {
                size_t i = idx[0];
                localAAcc[i] = N0sAcc[i] * sycl::exp(-lambdaAcc[i] * current_t);
            });
        });

        q.wait();

        // 读回结果
        {
            auto acc = bufLocalA.get_access<access::mode::read>();
            for (size_t i = 0; i < numIsotopes; ++i) {
                A[step * numIsotopes + i] = acc[i];
            }
        }

        t += dt;
        step++;
        if (step > steps) break;
    }

    // 只打印 DAC 对应的 A[1]（第一个时间步）
    cout << "{";
    for (size_t i = 0; i < numIsotopes; ++i) {
        cout << A[1 * numIsotopes + i];
        if (i + 1 < numIsotopes) cout << ", ";
    }
    cout << "}" << std::endl;

    return 0;
}

#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

shell dacpp::list matrixMultiply_shell(const dacpp::Matrix<int>& matA,
                                       const dacpp::Matrix<int>& matB,
                                       dacpp::Matrix<long long>& matC) {
    dacpp::index idx1, idx2;
    dacpp::list dataList{matA[idx1][{}], matB[{}][idx2], matC[idx1][idx2]};
    return dataList;
}

calc void matrixMultiply_calc(dacpp::Vector<int>& vecA,
                              dacpp::Vector<int>& vecB,
                              long long* dotProduct) {
    for (int i = 0; i < 2048; ++i) {
        dotProduct[0] += static_cast<long long>(vecA[i]) * vecB[i];
    }
}

int main() {
    constexpr int N = 2048;
    std::vector<int> dataA(static_cast<std::size_t>(N) * N, 1);
    std::vector<int> dataB(static_cast<std::size_t>(N) * N, 1);
    std::vector<long long> dataC(static_cast<std::size_t>(N) * N, 0);

    dacpp::Matrix<int> matA({N, N}, dataA);
    dacpp::Matrix<int> matB({N, N}, dataB);
    dacpp::Matrix<long long> matC({N, N}, dataC);

    auto start = std::chrono::steady_clock::now();
    matrixMultiply_shell(matA, matB, matC) <-> matrixMultiply_calc;
    auto end = std::chrono::steady_clock::now();

    std::vector<long long> result;
    matC.tensor2Array(result);
    long long checksum = 0;
    for (long long value : result) {
        checksum += value;
    }

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "checksum=" << checksum << "\n";
    std::cout << "kernel_call_ms=" << elapsed_ms << "\n";
    return 0;
}

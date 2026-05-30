#pragma once

#include <CL/sycl.hpp>
#include "Tensor.hpp"

namespace dacpp {

template <typename ImplType>
ImplType* sycl_add(sycl::buffer<ImplType, 1>& bufferA, sycl::buffer<ImplType, 1>& bufferB) {
    sycl::queue q;
    size_t size = bufferA.get_count();

    ImplType* resultData = new ImplType[size];

    {
        sycl::buffer<ImplType, 1> resultBuffer(resultData, sycl::range<1>(size));

        q.submit([&](sycl::handler& cgh) {
            auto accA = bufferA.template get_access<sycl::access::mode::read>(cgh);
            auto accB = bufferB.template get_access<sycl::access::mode::read>(cgh);
            auto accResult = resultBuffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class add_kernel>(sycl::range<1>(size), [=](sycl::id<1> i) {
                accResult[i] = accA[i] + accB[i];
            });
        });

        q.wait();
    }

    return resultData;
}

template <typename ImplType>
std::vector<ImplType> sycl_subtract(sycl::buffer<ImplType, 1>& bufferA, sycl::buffer<ImplType, 1>& bufferB) {
    sycl::queue q;
    size_t size = bufferA.get_count();

    std::vector<ImplType> resultData(size);
    {
        sycl::buffer<ImplType, 1> resultBuffer(resultData.data(), sycl::range<1>(size));

        q.submit([&](sycl::handler& cgh) {
            auto accA = bufferA.template get_access<sycl::access::mode::read>(cgh);
            auto accB = bufferB.template get_access<sycl::access::mode::read>(cgh);
            auto accResult = resultBuffer.template get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class subtract_kernel>(sycl::range<1>(size), [=](sycl::id<1> i) {
                accResult[i] = accA[i] - accB[i];
            });
        });

        q.wait();
    }

    return resultData;
}

template <typename ImplType>
void MatrixMultiplySYCL_1D(ImplType* A, ImplType* B, ImplType* C, int M, int N, int K) {
    sycl::queue queue;

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<1>{static_cast<size_t>(M * K)},
            [=](sycl::id<1> idx) {
                int row = idx[0] / K;
                int col = idx[0] % K;
                ImplType sum = 0;

                for (int n = 0; n < N; ++n) {
                    sum += A[row * N + n] * B[n * K + col];
                }

                C[row * K + col] = sum;
            });
    }).wait();
}

template <typename ImplType>
void MatrixMultiplySYCL(ImplType* A, ImplType* B, ImplType* C, int M, int N, int K) {
    constexpr int BLOCK_SIZE = 16;
    sycl::queue queue;

    queue.submit([&](sycl::handler& cgh) {
        // Use local tiles for the blocked GEMM path.
        sycl::accessor<ImplType, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            Asub(sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);
        sycl::accessor<ImplType, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            Bsub(sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>{static_cast<size_t>((M + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE),
                                             static_cast<size_t>((K + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE)},
                              sycl::range<2>{BLOCK_SIZE, BLOCK_SIZE}),
            [=](sycl::nd_item<2> item) {
                int row = item.get_global_id(0);
                int col = item.get_global_id(1);
                int local_row = item.get_local_id(0);
                int local_col = item.get_local_id(1);

                ImplType sum = 0;

                for (int m = 0; m < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {

                    if (row < M && m * BLOCK_SIZE + local_col < N) {
                        Asub[local_row * BLOCK_SIZE + local_col] = A[row * N + m * BLOCK_SIZE + local_col];
                    } else {
                        Asub[local_row * BLOCK_SIZE + local_col] = 0;
                    }

                    if (col < K && m * BLOCK_SIZE + local_row < N) {
                        Bsub[local_row * BLOCK_SIZE + local_col] = B[(m * BLOCK_SIZE + local_row) * K + col];
                    } else {
                        Bsub[local_row * BLOCK_SIZE + local_col] = 0;
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    for (int e = 0; e < BLOCK_SIZE; ++e) {
                        sum += Asub[local_row * BLOCK_SIZE + e] * Bsub[e * BLOCK_SIZE + local_col];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (row < M && col < K) {
                    C[row * K + col] = sum;
                }
            });
    }).wait();
}

template <typename ImplType>
void Add(sycl::buffer<ImplType, 1>& bufferA, sycl::buffer<ImplType, 1>& bufferB, sycl::buffer<ImplType, 1>& resultBuffer, size_t size) {
    sycl::queue q;

    q.submit([&](sycl::handler& cgh) {
        auto accA = bufferA.template get_access<sycl::access::mode::read>(cgh);
        auto accB = bufferB.template get_access<sycl::access::mode::read>(cgh);
        auto accResult = resultBuffer.template get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class add_kernel>(sycl::range<1>(size), [=](sycl::id<1> i) {
            accResult[i] = accA[i] + accB[i];
        });
    });

    q.wait();
}

template <typename ImplType>
std::vector<ImplType> sycl_divide(sycl::buffer<ImplType, 1>& bufferA, sycl::buffer<ImplType, 1>& bufferB) {
    sycl::queue q;
    size_t size = bufferA.get_count();

    std::vector<ImplType> resultData(size);
    {
        sycl::buffer<ImplType, 1> resultBuffer(resultData.data(), sycl::range<1>(size));
        q.submit([&](sycl::handler& cgh) {
            auto accA = bufferA.template get_access<sycl::access::mode::read>(cgh);
            auto accB = bufferB.template get_access<sycl::access::mode::read>(cgh);
            auto accResult = resultBuffer.template get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for<class divide_kernel>(sycl::range<1>(size), [=](sycl::id<1> i) {
                accResult[i] = accA[i] / accB[i];
            });
        });
        q.wait();
    }

    return resultData;
}

template <typename ImplType>
std::vector<ImplType> sycl_modulo(sycl::buffer<ImplType, 1>& bufferA, sycl::buffer<ImplType, 1>& bufferB) {
    sycl::queue q;
    size_t size = bufferA.get_count();

    std::vector<ImplType> resultData(size);
    {
        sycl::buffer<ImplType, 1> resultBuffer(resultData.data(), sycl::range<1>(size));
        q.submit([&](sycl::handler& cgh) {
            auto accA = bufferA.template get_access<sycl::access::mode::read>(cgh);
            auto accB = bufferB.template get_access<sycl::access::mode::read>(cgh);
            auto accResult = resultBuffer.template get_access<sycl::access::mode::write>(cgh);
            cgh.parallel_for<class modulo_kernel>(sycl::range<1>(size), [=](sycl::id<1> i) {
                accResult[i] = accA[i] % accB[i];
            });
        });
        q.wait();
    }

    return resultData;
}

}

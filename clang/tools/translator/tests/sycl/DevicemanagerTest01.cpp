#include "DeviceManager.h"  // 你提供的代码可以放在这个头文件里
#include <chrono>

int main() {
    constexpr size_t N = 1 << 20; // 1M 个元素

    // 创建 SYCL 队列以分配 USM
    sycl::queue queue{sycl::default_selector{}};

    // 申请 USM 共享内存
    int* A = sycl::malloc_shared<int>(N, queue);
    int* B = sycl::malloc_shared<int>(N, queue);
    int* C = sycl::malloc_shared<int>(N, queue);

    // 初始化数据
    for (size_t i = 0; i < N; ++i) {
        A[i] = 1;
        B[i] = 2;
        C[i] = 0;
    }

    // 构建 DataHandle
    auto handleA = std::make_shared<DataHandle>();
    handleA->host_ptr = A;
    handleA->size = N * sizeof(int);

    auto handleB = std::make_shared<DataHandle>();
    handleB->host_ptr = B;
    handleB->size = N * sizeof(int);

    auto handleC = std::make_shared<DataHandle>();
    handleC->host_ptr = C;
    handleC->size = N * sizeof(int);

    // 构建 DataView
    DataView viewA{handleA, 0, N * sizeof(int), false};
    DataView viewB{handleB, 0, N * sizeof(int), false};
    DataView viewC{handleC, 0, N * sizeof(int), true};

    // 创建 Task
    Task task;
    task.name = "vector_add";
    task.inputs = {viewA, viewB};
    task.outputs = {viewC};

    task.kernel = [=](sycl::handler& cgh, const std::vector<void*>& args) {
        int* a = static_cast<int*>(args[0]);
        int* b = static_cast<int*>(args[1]);
        int* c = static_cast<int*>(args[2]);
        cgh.parallel_for<class VectorAdd>(sycl::range<1>(N), [=](sycl::id<1> idx) {
            c[idx] = a[idx] + b[idx];
        });
    };

    // 提交任务
    DeviceManager manager;

    auto start = std::chrono::high_resolution_clock::now();
    manager.submit_task(task);
    bool ok = manager.wait_all();
    
    auto end = std::chrono::high_resolution_clock::now();

    // 校验结果（直接访问 C）
    bool correct = true;
    for (size_t i = 0; i < N; ++i) {
        if (C[i] != 3) {
            std::cout << "错误: C[" << i << "] = " << C[i] << std::endl;
            correct = false;
            break;
        }
    }
    std::cout << "校验 " << (correct ? "通过 ✅" : "失败 ❌") << std::endl;
    std::cout << "总耗时: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    // 释放 USM
    sycl::free(A, queue);
    sycl::free(B, queue);
    sycl::free(C, queue);

    return 0;
}
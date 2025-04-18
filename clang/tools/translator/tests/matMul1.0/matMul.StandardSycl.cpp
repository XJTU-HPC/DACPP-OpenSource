#include <vector>

#include "ReconTensor.h"

using dacpp::Tensor;

namespace dacpp {
typedef std::vector<std::any> list;
}





#include <sycl/sycl.hpp>
#include "DataReconstructor.h"

using namespace sycl;

void matMul(int* vecA, int* vecB, int* dotProduct) 
{
    for (int i = 0; i < 5; i++) {
        dotProduct[0] += vecA[i] * vecB[i];
    }
}


// 生成函数调用
void matMulSplit(const std::vector<int> & matA, const std::vector<int> & matB, std::vector<int> & matC) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    
    // 设备内存分配
    int *d_matA=malloc_device<int>(20,q);
    // 设备内存分配
    int *d_matB=malloc_device<int>(20,q);
    // 设备内存分配
    int *d_matC=malloc_device<int>(16,q);
    // 数据移动
    
    // 数据移动
    q.memcpy(d_matA,matA.data(),20*sizeof(int)).wait();
    // 数据移动
    q.memcpy(d_matB,matB.data(),20*sizeof(int)).wait();   
    // 内核执行
    
    //工作项划分
    sycl::range<3> local(1, 1, 16);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto i=(item_id/4+(0)+4)%4;
            const auto j=(item_id+(0)+4)%4;
            // 嵌入计算
			
            matMul(d_matA+(i*5),d_matB+(j*5),d_matC+(i*4+j*1));
        });
    }).wait();
    
    
    // 归约
    
    // 返回计算结果
    
    // 归并结果返回
    q.memcpy(matC.data(), d_matC, 16*sizeof(int)).wait();
    // 内存释放
    
    sycl::free(d_matA, q);
    sycl::free(d_matB, q);
    sycl::free(d_matC, q);
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now(); // 开始时间测量

    std::vector<int> dataA{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    std::vector<int> dataB{1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20};

    std::vector<int> dataC{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    matMulSplit(dataA, dataB, dataC);
    
    for(int i=0;i<4;i++) {
        for(int j=0;j<4;j++) {
            std::cout << dataC[i*4+j] << " ";
        }
        std::cout << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now(); // 结束时间测量
    std::chrono::duration<double> duration = end_time - start_time; // 计算持续时间
    std::cout << "总执行时间: " << duration.count() << " 秒" << std::endl; // 输出执行时间
    return 0;
}
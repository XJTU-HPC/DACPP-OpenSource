#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <random>

using namespace sycl;

constexpr size_t NUM_NEURONS = 8;
constexpr size_t INPUT_SIZE  = 8;

int main() {
    // 初始化梯度矩阵
    std::vector<float> host_grads(NUM_NEURONS * INPUT_SIZE);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for(int i=0;i<NUM_NEURONS;i++){
        for(int j=0;j<INPUT_SIZE;j++){
            host_grads[i*INPUT_SIZE+j]=j;
        }
    }
 
    queue q;


    // 外层循环每个神经元
        // buffer 存放内层输入
    std::vector<float> partial_sum(INPUT_SIZE, 0.0f);
    {    // 分配 device buffer
        buffer<float, 1> grad_buf(host_grads.data(), range<1>(NUM_NEURONS * INPUT_SIZE));
        buffer<float, 1> partial_buf(partial_sum.data(), range<1>(INPUT_SIZE));

        q.submit([&](handler &h){
            auto grads_acc = grad_buf.get_access<access::mode::read>(h);
            auto sum_acc   = partial_buf.get_access<access::mode::discard_write>(h);
            h.parallel_for(range<1>(NUM_NEURONS), [=](id<1> i){ 
                for(int n=0;n<INPUT_SIZE;n++)
                    sum_acc[i] += grads_acc[n * NUM_NEURONS + i];
            });
        }).wait();
    } // partial_buf 生命周期结束，会自动同步 host

    // 输出
    std::cout << "First 5 neuron gradient sums:\n";
    for (size_t i = 0; i < std::min(size_t(5), partial_sum.size()); ++i)
        std::cout << partial_sum[i] << " ";
    std::cout << std::endl;

    return 0;
}

#include <CL/sycl.hpp>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

using namespace sycl;

constexpr size_t NUM_NEURONS = 8192;
constexpr size_t INPUT_SIZE  = 8;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 检查进程数是否是 NUM_NEURONS 的约数
    if (NUM_NEURONS % size != 0) {
        if(rank==0)
            std::cerr << "Error: MPI size must be a divisor of NUM_NEURONS=" 
                      << NUM_NEURONS << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t neurons_per_proc = NUM_NEURONS / size;

    // 初始化梯度矩阵
    std::vector<float> host_grads(NUM_NEURONS * INPUT_SIZE);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for(int i=0;i<NUM_NEURONS;i++){
        for(int j=0;j<INPUT_SIZE;j++){
            host_grads[i*INPUT_SIZE+j] = i; // 可以用随机值 dist(gen) 替换
        }
    }

    queue q;

    // 每个进程负责的列索引范围
    size_t col_start = rank * neurons_per_proc;
    size_t col_end   = col_start + neurons_per_proc;

    std::vector<float> local_sum(neurons_per_proc, 0.0f);

    {
        buffer<float,1> grad_buf(host_grads.data(), range<1>(NUM_NEURONS*INPUT_SIZE));
        buffer<float,1> sum_buf(local_sum.data(), range<1>(neurons_per_proc));

        q.submit([&](handler &h){
            auto grads_acc = grad_buf.get_access<access::mode::read>(h);
            auto sum_acc   = sum_buf.get_access<access::mode::discard_write>(h);

            h.parallel_for<class kernel1>(range<1>(neurons_per_proc), [=](id<1> i){
                size_t neuron = col_start + i;
                float tmp = 0.0f;
                for(size_t n=0; n<INPUT_SIZE; n++)
                    tmp += grads_acc[neuron*INPUT_SIZE + n];
                sum_acc[i] = tmp;
            });
        }).wait();
    } // partial_buf 生命周期结束，会同步 host

    // 聚合所有进程结果到 rank 0
    std::vector<float> global_sum(NUM_NEURONS, 0.0f);
    MPI_Gather(local_sum.data(), neurons_per_proc, MPI_FLOAT,
               global_sum.data(), neurons_per_proc, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    if(rank==0){
        int a=5;
        if(NUM_NEURONS<5) a=NUM_NEURONS;
        std::cout << "First 5 neuron gradient sums:\n";
        for(size_t i=0;i<a;i++)
            std::cout << global_sum[i] << " ";
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}

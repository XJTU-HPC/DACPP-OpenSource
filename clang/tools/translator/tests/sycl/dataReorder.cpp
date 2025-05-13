#include <sycl/sycl.hpp>
using namespace sycl;
#include <vector>
#include<iostream>

int main() {
    auto selector = gpu_selector_v;
    queue q(selector);

    std::vector<int> vecA;
    std::vector<int> vecB;
    int N=16;
    for(int i=0;i<N;i++)
    {
        vecA.push_back(i+1);
        vecB.push_back(0);
    }
    // 将数据传到设备
    int *d_vecA=malloc_device<int>(16,q);
    q.memcpy(d_vecA,&vecA[0],N*sizeof(int)).wait();
    int *d_vecB=malloc_device<int>(36,q);
    q.memcpy(d_vecB,&vecB[0],N*sizeof(int)).wait();
    
    int block_size = 4;
    int lane_stride[2]={2,1}; 
    int op_move[2]={1,1};// 算子划分的移位
    int lane_shape[2] = {2,2};
    int block_stride[2]={3,1};
    int block_shape[2]={3,3};
    int vecA_stride[2] = {4,1};
    sycl::range<3> local(1, 1, 36);
    sycl::range<3> global(1, 1, 1);
    q.submit([&](handler &h) {
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            int total_block_id = item_id / block_size;
            int total_lane_id = item_id % block_size;
            int lane_id[2];
            for(int i=0;i<2;i++) {
                lane_id[i] = total_lane_id / lane_stride[i] % lane_shape[i];
            }
            int block_id[2];
            for(int i=0;i<2;i++) {
                block_id[i] = total_block_id / block_stride[i] % block_shape[i];
            }
            int pos[2];
            int total_pos = 0;
            for(int i=0;i<2;i++) {
                pos[i]=block_id[i]*op_move[i]+lane_id[i];
                total_pos += pos[i]*vecA_stride[i];
            }
            
            d_vecB[item_id] = d_vecA[total_pos];
        });
    }).wait();
    
    q.memcpy(&vecB[0],d_vecB,36*sizeof(int)).wait();
    for(int i=0;i<N;i++) std::cout<<vecA[i]<<" ";
    std::cout<<std::endl;
    for(int i=0;i<36;i++) std::cout<<vecB[i]<<" ";
    std::cout<<std::endl;

   int item_id = 16;
    int total_block_id = item_id / block_size;
    int total_lane_id = item_id % block_size;
    int lane_id[2];
    for(int i=0;i<2;i++) {
        lane_id[i] = total_lane_id / lane_stride[i] % lane_shape[i];
    }
    int block_id[2];
    for(int i=0;i<2;i++) {
        block_id[i] = total_block_id / block_stride[i] % block_shape[i];
    }
    int pos[2];
    int total_pos = 0;
    for(int i=0;i<2;i++) {
        pos[i]=block_id[i]*op_move[i]+lane_id[i];
        total_pos += pos[i]*vecA_stride[i];
    }
    std::cout<<total_pos<<std::endl;
    std::cout<<vecA[total_pos]<<std::endl;
    return 0;
}


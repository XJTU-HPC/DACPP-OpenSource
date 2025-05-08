#ifndef UTILS_H
#define UTILS_H

#include "DataReconstructor.new.h"

#include <sycl/sycl.hpp>
using namespace sycl;

struct VirtualMapParams {
    int block_size;
    int dim_num;
    sycl::accessor<int, 1, sycl::access::mode::read_write> start;
    sycl::accessor<int, 1, sycl::access::mode::read_write> data_shape;
    sycl::accessor<int, 1, sycl::access::mode::read_write> data_stride;
    sycl::accessor<int, 1, sycl::access::mode::read_write> view_shape;
    sycl::accessor<int, 1, sycl::access::mode::read_write> view_stride;
    sycl::accessor<int, 1, sycl::access::mode::read_write> block_shape;
    sycl::accessor<int, 1, sycl::access::mode::read_write> block_stride;
    sycl::accessor<int, 1, sycl::access::mode::read_write> block_move;
    sycl::accessor<int, 1, sycl::access::mode::read_write> grid_shape;
    sycl::accessor<int, 1, sycl::access::mode::read_write> grid_stride;
};

template<typename T>
SYCL_EXTERNAL T virtual_to_physical_read(
        T* data,
        int index,
        const VirtualMapParams& p)
{
    int total_block_id = index / p.block_size;
    int total_lane_id = index % p.block_size;
    int lane_id[20];
    int block_id[20];
    int total_pos = 0;
    for (int i = 0; i < p.dim_num; i++) {
        lane_id[i] = total_lane_id / p.block_stride[i] % p.block_shape[i];
        block_id[i] = total_block_id / p.grid_stride[i] % p.grid_shape[i];
        int pos = (block_id[i] * p.block_move[i] + lane_id[i]);
        if (pos < -p.start[i] || pos > -p.start[i] + p.data_shape[i] - 1) {
            return (T)0;
        }
        total_pos += (p.start[i] + pos) * p.data_stride[i];
    }
    return data[total_pos];
}

template<typename T>
SYCL_EXTERNAL T& virtual_to_physical(
        T* data,
        int index,
        const VirtualMapParams& p)
{
    int total_block_id = index / p.block_size;
    int total_lane_id = index % p.block_size;
    int lane_id[20];
    int block_id[20];
    int total_pos = 0;
    for (int i = 0; i < p.dim_num; i++) {
        lane_id[i] = total_lane_id / p.block_stride[i] % p.block_shape[i];
        block_id[i] = total_block_id / p.grid_stride[i] % p.grid_shape[i];
        int pos = (block_id[i] * p.block_move[i] + lane_id[i]);
        if (pos < -p.start[i] || pos > -p.start[i] + p.data_shape[i] - 1) {
            static const T zero = 0;  // 返回只读的静态常量
            return const_cast<T&>(zero);  //  需要强转以匹配返回类型
        }
        total_pos += (p.start[i] + pos) * p.data_stride[i];
    }
    return data[total_pos];
}

// template<typename T>
// void Slice(DataReconstructor<T> &tool,
//            int start,
//            std::vector<int> data_shape)
// {
//     tool.set_start(start);
//     tool.set_data_shape(data_shape);
// }

template<typename ImplType>
void Slice(ImplType* res, ImplType* d_a, std::vector<int> shape, std::vector<Range> region, sycl::queue& q) {    
    // 初始化切片起点
    int dimNum = shape.size();
    std::vector<int> pos;
    for (int i=0;i<dimNum;i++) {
        pos.push_back(region[i].start);
    }

    // 计算切片坐标的线性索引
    auto computeLinearIndex = [&](const std::vector<int>& coord) -> int {
        int index = 0;
        for (int i = 0; i < coord.size(); ++i) {
            index = index * shape[i] + coord[i];
        }
        return index;
    };
    std::vector<int> sliceIndex;
    while(true) {
        sliceIndex.push_back(computeLinearIndex(pos));
        int now = dimNum-1;
        while(now>=0) {
            pos[now]++;
            if (pos[now]<region[now].end) break;
            pos[now] = region[now].start;
            now--;
        }
        if(now<0) break;
    }
    // std::cout<<"sliceIndex: \n";
    // for(int i=0;i<sliceIndex.size();i++) cout<<sliceIndex[i]<<" ";
    // std::cout<<std::endl<<std::endl;
    sycl::buffer<int> sliceIndexbuffer(sliceIndex.data(), sycl::range<1>(sliceIndex.size()));
    // 并行切片赋值
    sycl::range<3> local(1, 1, sliceIndex.size());
    sycl::range<3> global(1, 1, 1);
    q.submit([&](handler &h) {
        auto sliceIndexAccessor = sliceIndexbuffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            res[item_id]=d_a[sliceIndexAccessor[item_id]];
        });
    }).wait();
}

template<typename ImplType>
void SetValue(ImplType* d_a, ImplType value, std::vector<int> shape, std::vector<Range> region, sycl::queue& q) {    
    // 初始化切片起点
    int dimNum = shape.size();
    std::vector<int> pos;
    for (int i=0;i<dimNum;i++) {
        pos.push_back(region[i].start);
    }

    // 计算切片坐标的线性索引
    auto computeLinearIndex = [&](const std::vector<int>& coord) -> int {
        int index = 0;
        for (int i = 0; i < coord.size(); ++i) {
            index = index * shape[i] + coord[i];
        }
        return index;
    };
    std::vector<int> sliceIndex;
    while(true) {
        sliceIndex.push_back(computeLinearIndex(pos));
        int now = dimNum-1;
        while(now>=0) {
            pos[now]++;
            if (pos[now]<region[now].end) break;
            pos[now] = region[now].start;
            now--;
        }
        if(now<0) break;
    }
    sycl::buffer<int> sliceIndexbuffer(sliceIndex.data(), sycl::range<1>(sliceIndex.size()));
    
    // 并行赋值
    sycl::range<3> local(1, 1, sliceIndex.size());
    sycl::range<3> global(1, 1, 1);
    q.submit([&](handler &h) {
        auto sliceIndexAccessor = sliceIndexbuffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            d_a[sliceIndexAccessor[item_id]] = value;
        });
    }).wait();
}

template<typename ImplType>
void SetValue(ImplType* d_a, ImplType* value, std::vector<int> shape, std::vector<Range> region, sycl::queue& q) {    
    // 初始化切片起点
    int dimNum = shape.size();
    std::vector<int> pos;
    for (int i=0;i<dimNum;i++) {
        pos.push_back(region[i].start);
    }

    // 计算切片坐标的线性索引
    auto computeLinearIndex = [&](const std::vector<int>& coord) -> int {
        int index = 0;
        for (int i = 0; i < coord.size(); ++i) {
            index = index * shape[i] + coord[i];
        }
        return index;
    };
    std::vector<int> sliceIndex;
    while(true) {
        sliceIndex.push_back(computeLinearIndex(pos));
        int now = dimNum-1;
        while(now>=0) {
            pos[now]++;
            if (pos[now]<region[now].end) break;
            pos[now] = region[now].start;
            now--;
        }
        if(now<0) break;
    }
    sycl::buffer<int> sliceIndexbuffer(sliceIndex.data(), sycl::range<1>(sliceIndex.size()));
    
    // 并行赋值
    sycl::range<3> local(1, 1, sliceIndex.size());
    sycl::range<3> global(1, 1, 1);
    q.submit([&](handler &h) {
        auto sliceIndexAccessor = sliceIndexbuffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            d_a[sliceIndexAccessor[item_id]] = value[item_id];
        });
    }).wait();
}

template<typename ImplType>
void Swap(ImplType* r_a, ImplType* r_b, ImplType* d_a, ImplType* d_b, DataReconstructor<ImplType>& a_tool, DataReconstructor<ImplType>& b_tool, sycl::queue& q) {
    a_tool.UpdateData(r_a,d_a,q);
    b_tool.UpdateData(r_b,d_b,q);
    swap(d_a, d_b);
    a_tool.Reconstruct(r_a,d_a,q);
    b_tool.Reconstruct(r_b,d_b,q);
}
#endif
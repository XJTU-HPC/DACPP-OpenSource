#ifndef UTILS_H
#define UTILS_H

#include "DataReconstructor.new.h"

#include <sycl/sycl.hpp>
using namespace sycl;

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
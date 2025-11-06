#ifndef DATARECONSTRUCTOR_H
#define DATARECONSTRUCTOR_H

#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>
#include<map>
#include "ReconTensor.h"
#include "dacInfo.h"

#include <sycl/sycl.hpp>
using namespace sycl;


struct Range
{
    int start;
    int end;
};

struct DataInfo
{
    int dim;
    std::vector<int> dimLength;
};

struct PosNumber
{
    std::vector<int> number;   // 数据索引
    std::vector<int> pos;      // 物理位置
    std::vector<Range> region; // 数据单元的区域
};


/*
    数据重组器类，用于数据重组。
*/
template<typename ImplType>
class DataReconstructor{
    private:
        //dacpp::Tensor<ImplType> myTensor;      // 数据
        DataInfo myDataInfo;                   // 数据信息 形状
        Dac_Ops ops;                           // 作用于数据的算子组
        std::vector<PosNumber> posNumberList;  // 数据索引与物理位置的映射   以弃用
        // std::vector<int> myIdx;
        sycl::buffer<int> myIdxBuffer;         // 用于在SYCL中传递 myIdx  
        int* myIdx = nullptr;   // USM 设备指针
        int  myIdxSize = 0;     // 元素数量
    public:
        DataReconstructor() : myIdxBuffer(nullptr, sycl::range<1>(0)) {
        // 初始化为空 buffer，稍后可以在 init 函数中重新分配
        }

        /*
            通过 数据形状，作用于数据的算子，初始化数据重组器
        */
        // 计算张量的 strides（行优先存储情况下的步幅数组）
        static inline std::vector<int> compute_strides(const DataInfo &info) {
            int d = info.dim;                       // 维度数
            std::vector<int> strides(d);            // 存储每一维的 stride
            int s = 1;                              // 累积乘积（从最后一维开始）
            for (int i = d - 1; i >= 0; --i) {      // 从最后一维往前推
                strides[i] = s;                     // 当前维度 stride = 后续所有维度长度的乘积
                s *= info.dimLength[i];             // 更新累积值
            }
            return strides;                         // 返回 stride 数组
        }

    void init(DataInfo dataInfo, Dac_Ops ops, sycl::queue &q) {
        this->myDataInfo = dataInfo;
        this->ops = ops;

        const int dim = this->myDataInfo.dim;
        // 计算总元素数
        int total = 1;
        for (int i = 0; i < dim; ++i) total *= this->myDataInfo.dimLength[i];

        if (total == 0) {
            // 旧行为兼容：清空
            this->myIdxSize = 0;
            // keep myIdxBuffer as empty buffer (or leave as-is)
            this->myIdxBuffer = sycl::buffer<int>(sycl::range<1>(0));
            if (this->myIdx) { sycl::free(this->myIdx, q); this->myIdx = nullptr; } // 保持兼容（若以前分配过 USM）
            return;
        }

        // compute strides on host
        std::vector<int> stride_src = compute_strides(this->myDataInfo);

        // --- 计算 blockCount、每个 block 的 region（host 串行部分） ---
        int K = (int)ops.DacOps.size();
        std::vector<int> blockCount(K);
        for (int oi = 0; oi < K; ++oi) {
            const Dac_Op &op = ops[oi];
            int dimlen = this->myDataInfo.dimLength[op.dimId];
            if (op.size > dimlen) blockCount[oi] = 0;
            else blockCount[oi] = ((dimlen - op.size) / op.stride) + 1;
        }

        int totalBlocks = 1;
        for (int oi = 0; oi < K; ++oi) totalBlocks *= std::max(1, blockCount[oi]);

        std::vector<int> regionStart((size_t)totalBlocks * dim);
        std::vector<int> localSize((size_t)totalBlocks * dim);
        std::vector<int> regionElems((size_t)totalBlocks);

        std::vector<int> blockIdx(K, 0);
        for (int b = 0; b < totalBlocks; ++b) {
            // default region: whole tensor
            for (int d = 0; d < dim; ++d) {
                regionStart[b * dim + d] = 0;
                localSize[b * dim + d] = this->myDataInfo.dimLength[d];
            }

            // adjust per op
            for (int oi = 0; oi < K; ++oi) {
                const Dac_Op &op = ops[oi];
                int dimId = op.dimId;
                int bid = blockIdx[oi];
                if (blockCount[oi] <= 0) {
                    regionStart[b * dim + dimId] = 0;
                    localSize[b * dim + dimId] = 0;
                } else {
                    int now_start = 0;
                    int new_start = now_start + bid * op.stride;
                    int new_end = new_start + op.size;
                    if (new_start < 0) new_start = 0;
                    if (new_end > this->myDataInfo.dimLength[dimId]) new_end = this->myDataInfo.dimLength[dimId];
                    regionStart[b * dim + dimId] = new_start;
                    localSize[b * dim + dimId] = new_end - new_start;
                }
            }

            long long elems = 1;
            for (int d = 0; d < dim; ++d) {
                int ls = localSize[b * dim + d];
                if (ls <= 0) { elems = 0; break; }
                elems *= ls;
            }
            regionElems[b] = (int)elems;

            // increment blockIdx (mixed-radix)
            for (int t = K - 1; t >= 0; --t) {
                blockIdx[t]++;
                if (blockIdx[t] < std::max(1, blockCount[t])) break;
                blockIdx[t] = 0;
            }
        }

        // compute prefix offsets (host)
        std::vector<int> prefix(totalBlocks);
        int sum = 0;
        int maxRegionElems = 0;
        for (int b = 0; b < totalBlocks; ++b) {
            prefix[b] = sum;
            sum += regionElems[b];
            if (regionElems[b] > maxRegionElems) maxRegionElems = regionElems[b];
        }
        int produced = sum;

        // prepare myIdxBuffer (buffer version)
        this->myIdxSize = produced;
        if (this->myIdxSize > 0) {
            this->myIdxBuffer = sycl::buffer<int>(sycl::range<1>((size_t)this->myIdxSize));
        } else {
            this->myIdxBuffer = sycl::buffer<int>(sycl::range<1>(0));
        }

        // create device-side buffers from host arrays (they copy host data into device-accessible storage)
        // using buffer constructor with host pointer copies initial content
        sycl::buffer<int> stride_buf(stride_src.data(), sycl::range<1>((size_t)dim));
        sycl::buffer<int> dimlen_buf(this->myDataInfo.dimLength.data(), sycl::range<1>((size_t)dim));
        sycl::buffer<int> regionStart_buf(regionStart.data(), sycl::range<1>((size_t)totalBlocks * dim));
        sycl::buffer<int> localSize_buf(localSize.data(), sycl::range<1>((size_t)totalBlocks * dim));
        sycl::buffer<int> regionElems_buf(regionElems.data(), sycl::range<1>((size_t)totalBlocks));
        sycl::buffer<int> prefix_buf(prefix.data(), sycl::range<1>((size_t)totalBlocks));

        // kernel: launch totalBlocks * maxRegionElems threads
        size_t global = (size_t)totalBlocks * (size_t)maxRegionElems;
        if (global == 0) {
            // nothing to do; myIdxBuffer already allocated (maybe zero sized)
            return;
        }

    // submit kernel writing into myIdxBuffer via write accessor
    q.submit([&](sycl::handler &h) {
        auto stride_acc = stride_buf.template get_access<sycl::access::mode::read>(h);
        auto dimlen_acc = dimlen_buf.template get_access<sycl::access::mode::read>(h);
        auto regionStart_acc = regionStart_buf.template get_access<sycl::access::mode::read>(h);
        auto localSize_acc = localSize_buf.template get_access<sycl::access::mode::read>(h);
        auto regionElems_acc = regionElems_buf.template get_access<sycl::access::mode::read>(h);
        auto prefix_acc = prefix_buf.template get_access<sycl::access::mode::read>(h);
        auto myIdx_acc = this->myIdxBuffer.template get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::range<1>(global), [=](sycl::id<1> gid) {
            size_t g = gid[0];
            int b = (int)(g / (size_t)maxRegionElems);
            int local = (int)(g % (size_t)maxRegionElems);
            int elemsB = regionElems_acc[b];
            if (local >= elemsB) return;

            int offset = local;
            int src_linear = 0;
            for (int d = dim - 1; d >= 0; --d) {
                int ls = localSize_acc[b * dim + d];
                int c = 0;
                if (ls > 0) {
                    c = offset % ls;
                    offset = offset / ls;
                } else {
                    c = 0;
                }
                int pos = regionStart_acc[b * dim + d] + c;
                src_linear += pos * stride_acc[d];
            }
            int out_idx = prefix_acc[b] + local;
            myIdx_acc[out_idx] = src_linear;
        });
    }).wait(); // wait 确保写入完成（后续 host 访问 myIdxBuffer 时安全）
    // auto host_acc=this->myIdxBuffer.template get_access<sycl::access::mode::read>();
    // std::cout << "---- myIdxBuffer (size = " << this->myIdxSize << ") ----\n";
    //     for (int i = 0; i < std::min(this->myIdxSize, 100); ++i) {
    //         std::cout << host_acc[i] << " ";
    //     }
    //     std::cout << std::endl;
    if (this->myIdxSize != total) {
        std::cerr << "Warning: produced index count (" << this->myIdxSize 
                  << ") != total elements (" << total << ")\n";
    }
}     
//串行的init函数，只有两个参数，重载
// void init(DataInfo dataInfo, Dac_Ops ops) {
//             this->myDataInfo = dataInfo;            // 保存张量信息
//             this->ops = ops;                        // 保存操作集合
        
//             const int dim = this->myDataInfo.dim;   // 维度数
//             // 计算张量总元素个数
//             int total = 1;
//             for (int i = 0; i < dim; ++i) total *= this->myDataInfo.dimLength[i];
        
//             // 如果没有元素，直接返回空
//             if (total == 0) {
//                 this->myIdx.clear();
//                 this->myIdxBuffer = sycl::buffer<int>(nullptr, sycl::range<1>(0));
//                 return;
//             }
        
//             // 预计算 stride，用于多维坐标 → 线性索引的转换
//             std::vector<int> stride_src = compute_strides(this->myDataInfo);
        
//             // 对每个操作计算它能在对应维度产生多少个 block
//             int K = (int)this->ops.size;            // 操作数
//             std::vector<int> blockCount(K);
//             for (int oi = 0; oi < K; ++oi) {
//                 const Dac_Op &op = this->ops[oi];
//                 int dimlen = this->myDataInfo.dimLength[op.dimId];
//                 if (op.size > dimlen) {
//                     blockCount[oi] = 0;             // 如果操作区域大于维度长度，则该维度不能生成 block
//                 } else {
//                     // block 数量 = ((总长度 - 区块大小) / 步幅) + 1
//                     blockCount[oi] = ((dimlen - op.size) / op.stride) + 1;
//                 }
//             }
        
//             // 计算所有操作组合下，总的 block 数量
//             int totalBlocks = 1;
//             for (int oi = 0; oi < K; ++oi) totalBlocks *= std::max(1, blockCount[oi]);
        
//             // 预分配索引数组空间
//             this->myIdx.clear();
//             this->myIdx.reserve((size_t)total);
        
//             // 定义区域和 block 索引
//             std::vector<Range> region(dim);         // 每个维度的范围 [start, end)
//             std::vector<int> blockIdx(K, 0);        // 当前 block 组合的索引
        
//             auto t0 = std::chrono::high_resolution_clock::now();
        
//             // 遍历所有 block 组合
//             for (int b = 0; b < totalBlocks; ++b) {
//                 // 初始时 region 覆盖整个张量
//                 for (int d = 0; d < dim; ++d) {
//                     region[d].start = 0;
//                     region[d].end = this->myDataInfo.dimLength[d];
//                 }
            
//                 // 按操作调整 region，得到当前子区块
//                 for (int oi = 0; oi < K; ++oi) {
//                     const Dac_Op &op = this->ops[oi];
//                     int dimId = op.dimId;           // 操作作用的维度
//                     int bid = blockIdx[oi];         // 当前 block 在该维度的索引
//                     if (blockCount[oi] <= 0) {
//                         // 不存在 block
//                         region[dimId].start = 0;
//                         region[dimId].end = 0;
//                     } else {
//                         // 按 stride 和 size 计算子区域范围
//                         int now_start = region[dimId].start;
//                         int new_start = now_start + bid * op.stride;
//                         int new_end = new_start + op.size;
//                         if (new_start < 0) new_start = 0;
//                         if (new_end > this->myDataInfo.dimLength[dimId]) 
//                             new_end = this->myDataInfo.dimLength[dimId];
//                         region[dimId].start = new_start;
//                         region[dimId].end = new_end;
//                     }
//                 }
            
//                 // 计算子区域的大小
//                 std::vector<int> localPos(dim);     // 当前点的位置
//                 std::vector<int> localSize(dim);    // 子区域在每个维度的大小
//                 int regionElems = 1;
//                 for (int d = 0; d < dim; ++d) {
//                     localPos[d] = region[d].start;
//                     localSize[d] = region[d].end - region[d].start;
//                     regionElems *= std::max(0, localSize[d]);
//                 }
            
//                 // 如果区域为空，跳过
//                 if (regionElems == 0) {
//                     // block 索引进位，寻找下一个组合
//                     for (int t = K - 1; t >= 0; --t) {
//                         blockIdx[t]++;
//                         if (blockIdx[t] < std::max(1, blockCount[t])) break;
//                         blockIdx[t] = 0;
//                     }
//                     continue;
//                 }
            
//                 // 枚举区域内的所有点
//                 for (int e = 0; e < regionElems; ++e) {
//                     // 计算点的线性索引
//                     int src_linear = 0;
//                     for (int d = 0; d < dim; ++d) {
//                         src_linear += localPos[d] * stride_src[d];
//                     }
//                     this->myIdx.push_back(src_linear);
                
//                     // 更新 localPos，相当于多维坐标进位
//                     for (int d = dim - 1; d >= 0; --d) {
//                         localPos[d]++;
//                         if (localPos[d] < region[d].end) break;
//                         localPos[d] = region[d].start;
//                     }
//                 }
            
//                 // block 索引进位，枚举下一个 block 组合
//                 for (int t = K - 1; t >= 0; --t) {
//                     blockIdx[t]++;
//                     if (blockIdx[t] < std::max(1, blockCount[t])) break;
//                     blockIdx[t] = 0;
//                 }
//             }
        
//             auto t1 = std::chrono::high_resolution_clock::now();
//             double gen_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 1e-3;
//             // std::cout << "Block-driven mapping time: " << gen_ms << " ms, produced entries: " << this->myIdx.size() << "\n";
        
//             // 验证索引数量是否与总元素数一致
//             if ((int)this->myIdx.size() != total) {
//                 std::cerr << "Warning: generated myIdx entries != total elements (" 
//                           << this->myIdx.size() << " vs " << total << ")\n";
//             }
        
//             // 将索引写入 SYCL buffer，供设备端使用
//             this->myIdxBuffer = sycl::buffer<int>(this->myIdx.begin(), this->myIdx.end());
        
//             // 调试打印（注释掉了）
//             // {
//             //     sycl::host_accessor acc(this->myIdxBuffer, sycl::read_only);
//             //     std::cout << "myIdxBuffer contents: ";
//             //     for(size_t i = 0; i < acc.size(); i++) {
//             //         std::cout << acc[i] << " ";
//             //     }
//             //     std::cout << std::endl;
//             // }
//             }
            


//  void init(DataInfo dataInfo, Dac_Ops ops, sycl::queue &q) {
//             // std::cout<<"new DataReconstructor init"<<std::endl;
//             this->myDataInfo = dataInfo;
//             this->ops = ops;
        
//             const int dim = this->myDataInfo.dim;
//             // 计算总元素数
//             int total = 1;
//             for (int i = 0; i < dim; ++i) total *= this->myDataInfo.dimLength[i];
        
//             if (total == 0) {
//                 // 旧行为兼容：清空
//                 this->myIdxSize = 0;
//                 if (this->myIdx) { sycl::free(this->myIdx, q); this->myIdx = nullptr; }
//                 return;
//             }
        
//             // 计算 stride (host)
//             std::vector<int> stride_src = compute_strides(this->myDataInfo);
        
//             // --- 计算 blockCount、每个 block 的 region（host 串行部分） ---
//             int K = (int)ops.DacOps.size();
//             std::vector<int> blockCount(K);
//             for (int oi = 0; oi < K; ++oi) {
//                 const Dac_Op &op = ops[oi];
//                 int dimlen = this->myDataInfo.dimLength[op.dimId];
//                 if (op.size > dimlen) blockCount[oi] = 0;
//                 else blockCount[oi] = ((dimlen - op.size) / op.stride) + 1;
//             }
        
//             int totalBlocks = 1;
//             for (int oi = 0; oi < K; ++oi) totalBlocks *= std::max(1, blockCount[oi]);
        
//             // For each block we will record regionStart[d], localSize[d], regionElems[b]
//             std::vector<int> regionStart; // length = totalBlocks * dim
//             std::vector<int> localSize;   // length = totalBlocks * dim
//             std::vector<int> regionElems; // length = totalBlocks
        
//             regionStart.resize((size_t)totalBlocks * dim);
//             localSize.resize((size_t)totalBlocks * dim);
//             regionElems.resize((size_t)totalBlocks);
        
//             // iterate all block combos (same logic as your serial code) and fill arrays
//             std::vector<int> blockIdx(K, 0);
//             for (int b = 0; b < totalBlocks; ++b) {
//                 // default region: whole tensor
//                 for (int d = 0; d < dim; ++d) {
//                     regionStart[b * dim + d] = 0;
//                     localSize[b * dim + d] = this->myDataInfo.dimLength[d];
//                 }
            
//                 // adjust per op
//                 for (int oi = 0; oi < K; ++oi) {
//                     const Dac_Op &op = ops[oi];
//                     int dimId = op.dimId;
//                     int bid = blockIdx[oi];
//                     if (blockCount[oi] <= 0) {
//                         regionStart[b * dim + dimId] = 0;
//                         localSize[b * dim + dimId] = 0;
//                     } else {
//                         int now_start = 0; // original code used region[d].start before; use 0 as baseline as you did
//                         int new_start = now_start + bid * op.stride;
//                         int new_end = new_start + op.size;
//                         if (new_start < 0) new_start = 0;
//                         if (new_end > this->myDataInfo.dimLength[dimId]) new_end = this->myDataInfo.dimLength[dimId];
//                         regionStart[b * dim + dimId] = new_start;
//                         localSize[b * dim + dimId] = new_end - new_start;
//                     }
//                 }
            
//                 // compute regionElems
//                 long long elems = 1;
//                 for (int d = 0; d < dim; ++d) {
//                     int ls = localSize[b * dim + d];
//                     if (ls <= 0) { elems = 0; break; }
//                     elems *= ls;
//                 }
//                 regionElems[b] = (int)elems;
            
//                 // increment blockIdx (mixed-radix)
//                 for (int t = K - 1; t >= 0; --t) {
//                     blockIdx[t]++;
//                     if (blockIdx[t] < std::max(1, blockCount[t])) break;
//                     blockIdx[t] = 0;
//                 }
//             }
        
//             // compute prefix offsets (host)
//             std::vector<int> prefix(totalBlocks);
//             int sum = 0;
//             int maxRegionElems = 0;
//             for (int b = 0; b < totalBlocks; ++b) {
//                 prefix[b] = sum;
//                 sum += regionElems[b];
//                 if (regionElems[b] > maxRegionElems) maxRegionElems = regionElems[b];
//             }
//             // sum 应当等于 total（或小于 total 如果你的 block-driven 会跳过空区域）
//             int produced = sum;
        
//             // allocate device myIdx
//             if (this->myIdx) { sycl::free(this->myIdx, q); this->myIdx = nullptr;}
//             this->myIdxSize = produced;
//             this->myIdx = sycl::malloc_device<int>(this->myIdxSize, q);
//             if (!this->myIdx) throw std::bad_alloc();
        
//             // copy supporting arrays to device (USM)
//             int* stride_d = sycl::malloc_device<int>(dim, q);
//             int* dimlen_d = sycl::malloc_device<int>(dim, q);
//             int* regionStart_d = sycl::malloc_device<int>(totalBlocks * dim, q);
//             int* localSize_d = sycl::malloc_device<int>(totalBlocks * dim, q);
//             int* regionElems_d = sycl::malloc_device<int>(totalBlocks, q);
//             int* prefix_d = sycl::malloc_device<int>(totalBlocks, q);
        
//             // host -> device copies (synchronous submit + wait to simplify)
//             q.memcpy(stride_d, stride_src.data(), sizeof(int) * dim).wait();
//             q.memcpy(dimlen_d, this->myDataInfo.dimLength.data(), sizeof(int) * dim).wait();
//             q.memcpy(regionStart_d, regionStart.data(), sizeof(int) * (totalBlocks * dim)).wait();
//             q.memcpy(localSize_d, localSize.data(), sizeof(int) * (totalBlocks * dim)).wait();
//             q.memcpy(regionElems_d, regionElems.data(), sizeof(int) * totalBlocks).wait();
//             q.memcpy(prefix_d, prefix.data(), sizeof(int) * totalBlocks).wait();
        
//             // kernel: launch totalBlocks * maxRegionElems threads
//             // each global id maps to (b = gid / maxRegionElems, local = gid % maxRegionElems)
//             size_t global = (size_t)totalBlocks * (size_t)maxRegionElems;
//             if (global == 0) {
//                 // nothing to do; free temporaries and return
//                 sycl::free(stride_d, q);
//                 sycl::free(dimlen_d, q);
//                 sycl::free(regionStart_d, q);
//                 sycl::free(localSize_d, q);
//                 sycl::free(regionElems_d, q);
//                 sycl::free(prefix_d, q);
//                 // keep this->myIdx allocated but empty possibly
//                 return;
//             }
        
//             int* myIdx_d_local = this->myIdx; // local copy for kernel capture
//             q.submit([&](sycl::handler &h) {
//                 // capture by value; all pointers are device pointers
//                 h.parallel_for(sycl::range<1>(global), [=](sycl::id<1> gid) {
//                     size_t g = gid[0];
//                     int b = (int)(g / (size_t)maxRegionElems);
//                     int local = (int)(g % (size_t)maxRegionElems);
//                     int elemsB = regionElems_d[b];
//                     if (local >= elemsB) return;
                
//                     // convert local index -> multi-dim offset within this block
//                     int offset = local;
//                     int coord_idx = 0;
//                     int src_linear = 0;
//                     // We'll compute from last dim to first to do mixed-radix conversion
//                     for (int d = dim - 1; d >= 0; --d) {
//                         int ls = localSize_d[b * dim + d];
//                         int c = 0;
//                         if (ls > 0) {
//                             c = offset % ls;
//                             offset = offset / ls;
//                         } else {
//                             c = 0;
//                         }
//                         int pos = regionStart_d[b * dim + d] + c;
//                         src_linear += pos * stride_d[d];
//                     }
//                     int out_idx = prefix_d[b] + local;
//                     myIdx_d_local[out_idx] = src_linear;
//                 });
//             }).wait(); // ensure kernel finished before freeing support arrays or using myIdx on host
        
//             // free temporary device arrays
//             sycl::free(stride_d, q);
//             sycl::free(dimlen_d, q);
//             sycl::free(regionStart_d, q);
//             sycl::free(localSize_d, q);
//             sycl::free(regionElems_d, q);
//             sycl::free(prefix_d, q);
        
//             // Optional: if you want a host copy (for debugging) uncomment:
//             // std::vector<int> hostIdx(this->myIdxSize);
//             // q.memcpy(hostIdx.data(), this->myIdx, sizeof(int) * this->myIdxSize).wait();
//             // // print/check hostIdx...
//             // this->myIdxHostCopy = std::move(hostIdx); // if you want to store a host copy
        
//             // sanity check: produced should equal total (if full coverage)
//             if (this->myIdxSize != total) {
//                 std::cerr << "Warning: produced index count (" << this->myIdxSize 
//                           << ") != total elements (" << total << ")\n";
//             }
//         }
        
        //重写Reconstruct函数支持USM 之前的可能有问题因为多卡过不了
        void Reconstruct(ImplType* res, ImplType* myTensor, sycl::queue& q){
                    // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteRes(cnt, res,this->posNumberList[i].pos,myTensor);
            // }

            int Item_Size = this->myIdxSize;
            sycl::device device = q.get_device();
            int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
            int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
            sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size));
            sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
            q.submit([&](handler &h) {
                auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
                    if(item_id >= Item_Size)
                        return;
                    res[item_id]=myTensor[myIdxAccessor[item_id]];
                    // res[item_id] = 7562;
                });
            }).wait();
            // std::cout<<"Reconstruct finished!"<<std::endl;
        }

        //重载Reconstruct函数支持buffer
void Reconstruct(sycl::buffer<ImplType>& res_buf,sycl::buffer<ImplType>& myTensor_buf,sycl::queue& q) {

    int Item_Size = this->myIdxSize;
    if (Item_Size == 0) return;

    sycl::device device = q.get_device();
    int max_global_size = 256;
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;

    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size));
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);

   q.submit([&](sycl::handler& h) {
        auto myIdx_acc = this->myIdxBuffer.template get_access<sycl::access::mode::read>(h);
        auto myTensor_acc = myTensor_buf.template get_access<sycl::access::mode::read>(h);
        auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
            size_t item_id = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
            if (item_id >= (size_t)Item_Size) return;

            int idx = myIdx_acc[item_id];
            res_acc[item_id] = myTensor_acc[idx];
        });
    });

}

        
        //USM版本的数据逆重组
void UpdateData(ImplType* res, ImplType* myTensor, sycl::queue& q){
            // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteData(cnt,res,this->posNumberList[i].pos,myTensor);
            // }
            int Item_Size = this->myIdxSize;
            sycl::device device = q.get_device();
            int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
            int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
            sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size));
            sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
            q.submit([&](handler &h) {
                int *myIdx_d = this->myIdx;
                // auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
                    if(item_id >= Item_Size)
                        return;
                    myTensor[myIdx_d[item_id]] = res[item_id];
                });
            }).wait();
        }
        
        //Buffer版本的数据逆重组
void UpdateData(sycl::buffer<ImplType, 1> &resBuffer,sycl::buffer<ImplType, 1> &myTensorBuffer,sycl::queue &q){
    int Item_Size = this->myIdxSize;
    sycl::device device = q.get_device();
    int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
    int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;

    sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size));
    sycl::range<3> global(1, 1, (Item_Size <= max_global_size)
                                     ? Item_Size
                                     : work_group_size * max_global_size);

    q.submit([&](sycl::handler &h) {
        // 访问三个 buffer
        auto myIdx_acc = this->myIdxBuffer.template get_access<sycl::access::mode::read>(h);
        auto res_acc = resBuffer.template get_access<sycl::access::mode::read>(h);
        auto myTensor_acc = myTensorBuffer.template get_access<sycl::access::mode::write>(h);

        h.parallel_for(sycl::nd_range<3>(global, local),
                       [=](sycl::nd_item<3> item) {
                           const auto item_id = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);
                           if (item_id >= Item_Size)
                               return;

                           // 数据逆重组
                           myTensor_acc[myIdx_acc[item_id]] = res_acc[item_id];
                       });
    }).wait();

}
void UpdateData(sycl::buffer<ImplType>& r_buf, sycl::buffer<ImplType>& b_buf, sycl::queue& q,int Item_Size1){
            // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteData(cnt,res,this->posNumberList[i].pos,myTensor);
            // }
            int Item_Size = Item_Size1;
            sycl::device device = q.get_device();
            int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
            int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
            sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size));
            sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
            q.submit([&](handler &h) {
                // auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
                auto res = r_buf.template get_access<sycl::access::mode::read_write>(h);
                auto myTensor = b_buf.template get_access<sycl::access::mode::read_write>(h);

                h.parallel_for<class MyKernel4>(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                //h.parallel_for<UpdateDataBufferKernel<ImplType>>(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
                    if(item_id >= Item_Size)
                        return;
                    // myTensor[myIdxAccessor[item_id]] = res[item_id];
                    myTensor[item_id] = res[item_id];
                });
            }).wait();
        }
        /*
            增加一个算子
        */
        void push_back(Dac_Op op) {
            this->ops.push_back(op);
            this->posNumberList.clear();
        }
        void push_back(Dac_Ops ops) {
            for(int i = 0; i < ops.size; i++) {
                this->ops.push_back(ops[i]);
            }
        }
        /*
            减少一个算子
        */
       void pop_back() {
            this->ops.pop_back();
            this->posNumberList.clear();
       }

};


#endif
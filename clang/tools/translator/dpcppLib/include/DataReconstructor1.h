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

template <typename T> struct ReconstructUSMKernel;//​​不完整声明（前向声明） 告诉编译器存在这样一个结构体 但不包含任何的成员 这是为了解决命名内核冲突的问题
template <typename T> struct ReconstructBufferKernel;
template <typename T> struct UpdateDataUSMKernel;
template <typename T> struct UpdateDataBufferKernel;

template<typename ImplType>
class DataReconstructor{
    private:
        //dacpp::Tensor<ImplType> myTensor;      // 数据
        DataInfo myDataInfo;                   // 数据信息 形状
        Dac_Ops ops;                           // 作用于数据的算子组
        std::vector<PosNumber> posNumberList;  // 数据索引与物理位置的映射
        std::vector<int> myIdx;
        sycl::buffer<int> myIdxBuffer;         // 用于在SYCL中传递 myIdx  
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

        void init(DataInfo dataInfo, Dac_Ops ops) {
            this->myDataInfo = dataInfo;            // 保存张量信息
            this->ops = ops;                        // 保存操作集合
        
            const int dim = this->myDataInfo.dim;   // 维度数
            // 计算张量总元素个数
            int total = 1;
            for (int i = 0; i < dim; ++i) total *= this->myDataInfo.dimLength[i];
        
            // 如果没有元素，直接返回空
            if (total == 0) {
                this->myIdx.clear();
                this->myIdxBuffer = sycl::buffer<int>(nullptr, sycl::range<1>(0));
                return;
            }
        
            // 预计算 stride，用于多维坐标 → 线性索引的转换
            std::vector<int> stride_src = compute_strides(this->myDataInfo);
        
            // 对每个操作计算它能在对应维度产生多少个 block
            int K = (int)this->ops.size;            // 操作数
            std::vector<int> blockCount(K);
            for (int oi = 0; oi < K; ++oi) {
                const Dac_Op &op = this->ops[oi];
                int dimlen = this->myDataInfo.dimLength[op.dimId];
                if (op.size > dimlen) {
                    blockCount[oi] = 0;             // 如果操作区域大于维度长度，则该维度不能生成 block
                } else {
                    // block 数量 = ((总长度 - 区块大小) / 步幅) + 1
                    blockCount[oi] = ((dimlen - op.size) / op.stride) + 1;
                }
            }
        
            // 计算所有操作组合下，总的 block 数量
            int totalBlocks = 1;
            for (int oi = 0; oi < K; ++oi) totalBlocks *= std::max(1, blockCount[oi]);
        
            // 预分配索引数组空间
            this->myIdx.clear();
            this->myIdx.reserve((size_t)total);
        
            // 定义区域和 block 索引
            std::vector<Range> region(dim);         // 每个维度的范围 [start, end)
            std::vector<int> blockIdx(K, 0);        // 当前 block 组合的索引
        
            auto t0 = std::chrono::high_resolution_clock::now();
        
            // 遍历所有 block 组合
            for (int b = 0; b < totalBlocks; ++b) {
                // 初始时 region 覆盖整个张量
                for (int d = 0; d < dim; ++d) {
                    region[d].start = 0;
                    region[d].end = this->myDataInfo.dimLength[d];
                }
            
                // 按操作调整 region，得到当前子区块
                for (int oi = 0; oi < K; ++oi) {
                    const Dac_Op &op = this->ops[oi];
                    int dimId = op.dimId;           // 操作作用的维度
                    int bid = blockIdx[oi];         // 当前 block 在该维度的索引
                    if (blockCount[oi] <= 0) {
                        // 不存在 block
                        region[dimId].start = 0;
                        region[dimId].end = 0;
                    } else {
                        // 按 stride 和 size 计算子区域范围
                        int now_start = region[dimId].start;
                        int new_start = now_start + bid * op.stride;
                        int new_end = new_start + op.size;
                        if (new_start < 0) new_start = 0;
                        if (new_end > this->myDataInfo.dimLength[dimId]) 
                            new_end = this->myDataInfo.dimLength[dimId];
                        region[dimId].start = new_start;
                        region[dimId].end = new_end;
                    }
                }
            
                // 计算子区域的大小
                std::vector<int> localPos(dim);     // 当前点的位置
                std::vector<int> localSize(dim);    // 子区域在每个维度的大小
                int regionElems = 1;
                for (int d = 0; d < dim; ++d) {
                    localPos[d] = region[d].start;
                    localSize[d] = region[d].end - region[d].start;
                    regionElems *= std::max(0, localSize[d]);
                }
            
                // 如果区域为空，跳过
                if (regionElems == 0) {
                    // block 索引进位，寻找下一个组合
                    for (int t = K - 1; t >= 0; --t) {
                        blockIdx[t]++;
                        if (blockIdx[t] < std::max(1, blockCount[t])) break;
                        blockIdx[t] = 0;
                    }
                    continue;
                }
            
                // 枚举区域内的所有点
                for (int e = 0; e < regionElems; ++e) {
                    // 计算点的线性索引
                    int src_linear = 0;
                    for (int d = 0; d < dim; ++d) {
                        src_linear += localPos[d] * stride_src[d];
                    }
                    this->myIdx.push_back(src_linear);
                
                    // 更新 localPos，相当于多维坐标进位
                    for (int d = dim - 1; d >= 0; --d) {
                        localPos[d]++;
                        if (localPos[d] < region[d].end) break;
                        localPos[d] = region[d].start;
                    }
                }
            
                // block 索引进位，枚举下一个 block 组合
                for (int t = K - 1; t >= 0; --t) {
                    blockIdx[t]++;
                    if (blockIdx[t] < std::max(1, blockCount[t])) break;
                    blockIdx[t] = 0;
                }
            }
        
            auto t1 = std::chrono::high_resolution_clock::now();
            double gen_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * 1e-3;
            // std::cout << "Block-driven mapping time: " << gen_ms << " ms, produced entries: " << this->myIdx.size() << "\n";
        
            // 验证索引数量是否与总元素数一致
            if ((int)this->myIdx.size() != total) {
                std::cerr << "Warning: generated myIdx entries != total elements (" 
                          << this->myIdx.size() << " vs " << total << ")\n";
            }
        
            // 将索引写入 SYCL buffer，供设备端使用
            this->myIdxBuffer = sycl::buffer<int>(this->myIdx.begin(), this->myIdx.end());
        
            // 调试打印（注释掉了）
            // {
            //     sycl::host_accessor acc(this->myIdxBuffer, sycl::read_only);
            //     std::cout << "myIdxBuffer contents: ";
            //     for(size_t i = 0; i < acc.size(); i++) {
            //         std::cout << acc[i] << " ";
            //     }
            //     std::cout << std::endl;
            // }
            }
            //重写Reconstruct函数支持USM 之前的可能有问题因为多卡过不了
            void Reconstruct(ImplType* res, ImplType* myTensor, sycl::queue& q){
                    // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteRes(cnt, res,this->posNumberList[i].pos,myTensor);
            // }

            int Item_Size = this->myIdx.size();
            sycl::device device = q.get_device();
            int max_global_size = 256;/*device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];*/
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
        void Reconstruct(sycl::buffer<ImplType>& r_buf, sycl::buffer<ImplType>& b_buf, sycl::queue& q){
            // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteRes(cnt, res,this->posNumberList[i].pos,myTensor);
            // }
            int Item_Size = this->myIdx.size();
            sycl::device device = q.get_device();
            int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
            int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
            sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size));
            sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
            q.submit([&](handler &h) {
                auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
                auto res = r_buf.template get_access<sycl::access::mode::read_write>(h);
                auto myTensor = b_buf.template get_access<sycl::access::mode::read_write>(h);
                //h.parallel_for<class MyKernel2>(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                h.parallel_for<ReconstructBufferKernel<ImplType>>(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
                    if(item_id >= Item_Size)
                        return;
                    res[item_id]=myTensor[myIdxAccessor[item_id]];
                });
            }).wait();
        }

        
        //USM版本的数据逆重组
        void UpdateData(ImplType* res, ImplType* myTensor, sycl::queue& q)
        {
            // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteData(cnt,res,this->posNumberList[i].pos,myTensor);
            // }
            int Item_Size = this->myIdx.size();
            sycl::device device = q.get_device();
            int max_global_size = 256;/*device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];*/
            int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
            sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size));
            sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
            q.submit([&](handler &h) {
                auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
                //h.parallel_for<class MyKernel3>(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                h.parallel_for<UpdateDataUSMKernel<ImplType>>(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
                    if(item_id >= Item_Size)
                        return;
                    myTensor[myIdxAccessor[item_id]] = res[item_id];
                });
            }).wait();
        }
        
        //Buffer版本的数据逆重组
        void UpdateData(sycl::buffer<ImplType>& r_buf, sycl::buffer<ImplType>& b_buf, sycl::queue& q){
            // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteData(cnt,res,this->posNumberList[i].pos,myTensor);
            // }
            int Item_Size = this->myIdx.size();
            sycl::device device = q.get_device();
            int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
            int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
            sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size));
            sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
            q.submit([&](handler &h) {
                auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
                auto res = r_buf.template get_access<sycl::access::mode::read_write>(h);
                auto myTensor = b_buf.template get_access<sycl::access::mode::read_write>(h);

                //h.parallel_for<class MyKernel4>(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                h.parallel_for<UpdateDataBufferKernel<ImplType>>(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
                    if(item_id >= Item_Size)
                        return;
                    myTensor[myIdxAccessor[item_id]] = res[item_id];
                });
            }).wait();
        }

        /*
            增加一个算子
        */
    //     void push_back(Dac_Op op) {
    //         this->ops.push_back(op);
    //         this->posNumberList.clear();
    //     }
    //     void push_back(Dac_Ops ops) {
    //         for(int i = 0; i < ops.size; i++) {
    //             this->ops.push_back(ops[i]);
    //         }
    //     }
    //     /*
    //         减少一个算子
    //     */
    //    void pop_back() {
    //         this->ops.pop_back();
    //         this->posNumberList.clear();
    //    }
};

#endif
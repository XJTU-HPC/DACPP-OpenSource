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
    std::vector<int> number;
    std::vector<int> pos;
    std::vector<Range> region;
};

template <typename T> struct ReconstructUSMKernel;
template <typename T> struct ReconstructBufferKernel;
template <typename T> struct UpdateDataUSMKernel;
template <typename T> struct UpdateDataBufferKernel;

template<typename ImplType>
class DataReconstructor{
    private:

        DataInfo myDataInfo;
        Dac_Ops ops;
        std::vector<PosNumber> posNumberList;

        sycl::buffer<int> myIdxBuffer;
        int* myIdx = nullptr;
        int  myIdxSize = 0;
    public:
        DataReconstructor() : myIdxBuffer(nullptr, sycl::range<1>(0)) {

        }

        // Compute row-major strides for flattened index reconstruction.
        static inline std::vector<int> compute_strides(const DataInfo &info) {
            int d = info.dim;
            std::vector<int> strides(d);
            int s = 1;
            for (int i = d - 1; i >= 0; --i) {
                strides[i] = s;
                s *= info.dimLength[i];
            }
            return strides;
        }

        void init(DataInfo dataInfo, Dac_Ops ops, sycl::queue &q)
        {
            this->myDataInfo = dataInfo;
            this->ops = ops;

            const int dim = this->myDataInfo.dim;

            int total = 1;
            for (int i = 0; i < dim; ++i) total *= this->myDataInfo.dimLength[i];

            if (total == 0) {

                this->myIdxSize = 0;

                this->myIdxBuffer = sycl::buffer<int>(sycl::range<1>(0));
                if (this->myIdx) { sycl::free(this->myIdx, q); this->myIdx = nullptr; }
                return;
            }

            std::vector<int> stride_src = compute_strides(this->myDataInfo);

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

            // Build per-block regions on the host before generating the index map.
            std::vector<int> blockIdx(K, 0);
            for (int b = 0; b < totalBlocks; ++b) {

                for (int d = 0; d < dim; ++d) {
                    regionStart[b * dim + d] = 0;
                    localSize[b * dim + d] = this->myDataInfo.dimLength[d];
                }

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

                for (int t = K - 1; t >= 0; --t) {
                    blockIdx[t]++;
                    if (blockIdx[t] < std::max(1, blockCount[t])) break;
                    blockIdx[t] = 0;
                }
            }

            std::vector<int> prefix(totalBlocks);
            int sum = 0;
            int maxRegionElems = 0;
            for (int b = 0; b < totalBlocks; ++b) {
                prefix[b] = sum;
                sum += regionElems[b];
                if (regionElems[b] > maxRegionElems) maxRegionElems = regionElems[b];
            }
            int produced = sum;

            this->myIdxSize = produced;
            if (this->myIdxSize > 0) {
                this->myIdxBuffer = sycl::buffer<int>(sycl::range<1>((size_t)this->myIdxSize));
            } else {
                this->myIdxBuffer = sycl::buffer<int>(sycl::range<1>(0));
            }

            sycl::buffer<int> stride_buf(stride_src.data(), sycl::range<1>((size_t)dim));
            sycl::buffer<int> dimlen_buf(this->myDataInfo.dimLength.data(), sycl::range<1>((size_t)dim));
            sycl::buffer<int> regionStart_buf(regionStart.data(), sycl::range<1>((size_t)totalBlocks * dim));
            sycl::buffer<int> localSize_buf(localSize.data(), sycl::range<1>((size_t)totalBlocks * dim));
            sycl::buffer<int> regionElems_buf(regionElems.data(), sycl::range<1>((size_t)totalBlocks));
            sycl::buffer<int> prefix_buf(prefix.data(), sycl::range<1>((size_t)totalBlocks));

            size_t global = (size_t)totalBlocks * (size_t)maxRegionElems;
            if (global == 0) {

                return;
            }

            // Emit the flattened source index for each reconstructed element.
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
            }).wait();
            if (this->myIdxSize != total) {
                std::cerr << "Warning: produced index count (" << this->myIdxSize
                        << ") != total elements (" << total << ")\n";
            }
        }

        void Reconstruct(ImplType* res, ImplType* myTensor, sycl::queue& q)
        {
            int Item_Size = this->myIdxSize;
            sycl::device device = q.get_device();
            int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
            int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;
            sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size));
            sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
            q.submit([&](handler &h) {
                auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::read>(h);
                h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
                    if(item_id >= Item_Size)
                        return;
                    res[item_id]=myTensor[myIdxAccessor[item_id]];

                });
            }).wait();
        }

        void Reconstruct(sycl::buffer<ImplType>& res_buf,sycl::buffer<ImplType>& myTensor_buf,sycl::queue& q)
        {
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
            }).wait();
        }

        void UpdateData(ImplType* res, ImplType* myTensor, sycl::queue& q,int Item_Size1)
        {
            int Item_Size = Item_Size1;
            sycl::device device = q.get_device();
            auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
            int max_global_size_x = max_sizes[0];
            int max_global_size_y = max_sizes[1];
            int max_global_size_z = max_sizes[2];

            int dim_x = (int)sycl::ceil(sycl::sqrt((float)Item_Size));
            int dim_y = (int)sycl::ceil((float)Item_Size / dim_x);

            int local_x = std::min(16, max_global_size_x);
            int local_y = std::min(16, max_global_size_y);

            int global_x = ((dim_x + local_x - 1) / local_x) * local_x;
            int global_y = ((dim_y + local_y - 1) / local_y) * local_y;

            sycl::range<2> local(local_x, local_y);
            sycl::range<2> global(global_x, global_y);
            q.submit([&](handler &h) {
                h.parallel_for<class MyKernel4>(sycl::nd_range<2>(global, local),[=](sycl::nd_item<2> item) {
                    int gx = item.get_global_id(0);
                    int gy = item.get_global_id(1);
                    int item_id = gx * global[1] + gy;
                    if(item_id >= Item_Size)
                        return;
                    myTensor[item_id] = res[item_id];
                });
            }).wait();
        }

        void UpdateData(sycl::buffer<ImplType>& r_buf, sycl::buffer<ImplType>& b_buf, sycl::queue& q,int Item_Size1)
        {
            int Item_Size = Item_Size1;
            sycl::device device = q.get_device();
            auto max_sizes = device.get_info<sycl::info::device::max_work_item_sizes<3>>();
            int max_global_size_x = max_sizes[0];
            int max_global_size_y = max_sizes[1];
            int max_global_size_z = max_sizes[2];

            int dim_x = (int)sycl::ceil(sycl::sqrt((float)Item_Size));
            int dim_y = (int)sycl::ceil((float)Item_Size / dim_x);

            int local_x = std::min(16, max_global_size_x);
            int local_y = std::min(16, max_global_size_y);

            int global_x = ((dim_x + local_x - 1) / local_x) * local_x;
            int global_y = ((dim_y + local_y - 1) / local_y) * local_y;

            sycl::range<2> local(local_x, local_y);
            sycl::range<2> global(global_x, global_y);
            q.submit([&](handler &h) {
                auto res = r_buf.template get_access<sycl::access::mode::read_write>(h);
                auto myTensor = b_buf.template get_access<sycl::access::mode::read_write>(h);
                h.parallel_for<class MyKernel4>(sycl::nd_range<2>(global, local),[=](sycl::nd_item<2> item) {
                    int gx = item.get_global_id(0);
                    int gy = item.get_global_id(1);
                    int item_id = gx * global[1] + gy;
                    if(item_id >= Item_Size)
                        return;
                    myTensor[item_id] = res[item_id];
                });
            }).wait();
        }

        void push_back(Dac_Op op) {
            this->ops.push_back(op);
            this->posNumberList.clear();
        }
        void push_back(Dac_Ops ops) {
            for(int i = 0; i < ops.size; i++) {
                this->ops.push_back(ops[i]);
            }
        }

        void pop_back() {
            this->ops.pop_back();
            this->posNumberList.clear();
        }

};

#endif

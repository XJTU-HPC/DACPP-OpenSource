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
        DataInfo myDataInfo;                   // 数据信息 形状
        Dac_Ops ops;                           // 作用于数据的算子组
        std::vector<PosNumber> posNumberList;  // 数据索引与物理位置的映射
        std::vector<int> myIdx;
        sycl::buffer<int> myIdxBuffer;         // 用于在SYCL中传递 myIdx

        int dim_num;
        int block_size;
        int block_num;
        std::vector<int> data_shape;
        std::vector<int> data_stride;
        std::vector<int> grid_shape;
        std::vector<int> grid_stride;
        std::vector<int> block_shape;
        std::vector<int> block_stride;
        std::vector<int> block_move;

        sycl::buffer<int> data_shape_buffer;
        sycl::buffer<int> data_stride_buffer;
        sycl::buffer<int> grid_shape_buffer;
        sycl::buffer<int> grid_stride_buffer;
        sycl::buffer<int> block_shape_buffer;
        sycl::buffer<int> block_stride_buffer;
        sycl::buffer<int> block_move_buffer;

        void GetPos(std::vector<int> &pos, Dac_Ops &ops, int now) {
            if (now == this->myDataInfo.dim) {
                GetPosNumber(pos, ops);
                return;
            }
            for (int i = 0; i < this->myDataInfo.dimLength[now]; i++) {
                pos.push_back(i);
                GetPos(pos, ops, now + 1);
                pos.pop_back();
            }
        }
        void GetPosNumber(std::vector<int> &pos, Dac_Ops &ops) {
            int dimNum = this->myDataInfo.dim;
            std::vector<Range> region;
            for (int i = 0; i < dimNum; i++) {
                Range r;
                r.start = 0;
                r.end = this->myDataInfo.dimLength[i];
                region.push_back(r);
            }
            
            std::vector<int> number;   // 存数据索引的中间变量
            RecursiveTraversal(number, pos, region, 0);
        }
        /*
            递归得到物理位置 pos 在当前算子组作用下应该映射的数据索引，以及划分好的数据单元的数据区域。
        */
        void RecursiveTraversal(std::vector<int> &number,std::vector<int> &pos, std::vector<Range> &region, int now) {
            if (now == ops.size) {
                PosNumber posNum;
                posNum.number = number;
                posNum.pos = pos;
                posNum.region = region;
                this->posNumberList.push_back(posNum);
                return;
            }
            Dac_Op op = ops[now];
            int id = pos[op.dimId]-region[op.dimId].start;
            int l;
            if (id-op.size<0) l=0;
            else l = ((id-op.size+1)%op.stride==0) ? (id-op.size+1)/op.stride : ((id-op.size+1+op.stride)&(-op.stride))/op.stride;
            int r = (region[op.dimId].start+(id&(-op.stride))+op.size<region[op.dimId].end) ? (id&(-op.stride))/op.stride : (region[op.dimId].end-op.size-region[op.dimId].start)/op.stride;
            for(int i = l; i <= r; i++) {
                int now_start = region[op.dimId].start;
                int now_end = region[op.dimId].end;
                number.push_back(i);
                region[op.dimId].start = now_start + i * op.stride;
                region[op.dimId].end = now_start + i * op.stride +  op.size;
                RecursiveTraversal(number, pos, region, now + 1);
                region[op.dimId].end = now_end;
                region[op.dimId].start = now_start;
                number.pop_back();
            }
        }
        
    public:
        DataReconstructor()
        : myIdxBuffer(range<1>(1)),
          data_shape_buffer(range<1>(1)),
          data_stride_buffer(range<1>(1)),
          grid_shape_buffer(range<1>(1)),
          grid_stride_buffer(range<1>(1)),
          block_shape_buffer(range<1>(1)),
          block_stride_buffer(range<1>(1)),
          block_move_buffer(range<1>(1))
        {

        }

        /*
            通过 数据形状，作用于数据的算子，初始化数据重组器
        */
        void init(DataInfo dataInfo, Dac_Ops ops){
            this->myDataInfo=dataInfo;
            this->ops=ops;

            this->dim_num = dataInfo.dim;
            this->data_shape = dataInfo.dimLength;
            int total_data_stride = 1;
            int total_block_stride = 1;
            int total_grid_stride = 1;
            for (int i = 0; i < this->dim_num; i++) {
                this->block_shape.push_back(ops[i].size);
                this->block_move.push_back(ops[i].stride);
                this->grid_shape.push_back(ops[i].split_size);
                total_data_stride *= this->data_shape[i];
                total_block_stride *= this->block_shape[i];
                total_grid_stride *= this->grid_shape[i];
            }
            this->block_size = total_block_stride;
            this->block_num = total_grid_stride;
            for (int i = 0; i < this->dim_num; i++) {
                total_data_stride /= this->data_shape[i];
                total_block_stride /= this->block_shape[i];
                total_grid_stride /= this->grid_shape[i];
                this->data_stride.push_back(total_data_stride);
                this->block_stride.push_back(total_block_stride);
                this->grid_stride.push_back(total_grid_stride);
            }
            this->data_shape_buffer = sycl::buffer<int>(this->data_shape.data(), sycl::range<1>(this->dim_num));
            this->data_stride_buffer = sycl::buffer<int>(this->data_stride.data(), sycl::range<1>(this->dim_num));
            this->block_shape_buffer = sycl::buffer<int>(this->block_shape.data(), sycl::range<1>(this->dim_num));
            this->block_stride_buffer = sycl::buffer<int>(this->block_stride.data(), sycl::range<1>(this->dim_num));
            this->block_move_buffer = sycl::buffer<int>(this->block_move.data(), sycl::range<1>(this->dim_num));
            this->grid_shape_buffer = sycl::buffer<int>(this->grid_shape.data(), sycl::range<1>(this->dim_num));
            this->grid_stride_buffer = sycl::buffer<int>(this->grid_stride.data(), sycl::range<1>(this->dim_num));

            // std::vector<int> pos; // 存位置的中间变量
            // GetPos(pos, ops, 0);
            
            // std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.number==b.number)?a.pos<b.pos:a.number<b.number;});
            // for(int i=0;i<this->posNumberList.size();i++) this->myIdx.push_back(0);
            // for(int i=0;i<this->posNumberList.size();i++){
            //     int stride = 1;
            //     for(int j=0;j<this->myDataInfo.dim;j++) stride*=this->myDataInfo.dimLength[j];
            //     int idx = 0;
            //     for(int j=0;j<this->myDataInfo.dim;j++) {
            //         stride/=this->myDataInfo.dimLength[j];
            //         idx+=this->posNumberList[i].pos[j]*stride;
            //     }
            //     this->myIdx[i]=idx;
            // }

            // this->myIdxBuffer = sycl::buffer<int>(this->myIdx.data(), sycl::range<1>(this->myIdx.size()));
        }

        /*
            将重组结果写入res长向量。
        */
        void Reconstruct(ImplType* res, ImplType* myTensor, sycl::queue& q){
            // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteRes(cnt, res,this->posNumberList[i].pos,myTensor);
            // }
            sycl::range<3> local(1, 1, this->posNumberList.size());
            sycl::range<3> global(1, 1, 1);
            q.submit([&](handler &h) {
                auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_local_id(2);
                    res[item_id]=myTensor[myIdxAccessor[item_id]];
                });
            }).wait();
        }

        /*
            将重组结果写入res长向量。
        */
        void Reconstruct_1(ImplType* res, ImplType* myTensor, sycl::queue& q){
            auto block_size = this->block_size;
            auto dim_num = this->dim_num;
            sycl::range<3> local(1, 1, this->block_num * this->block_size);
            sycl::range<3> global(1, 1, 1);
            q.submit([&](handler &h) {
                auto acc_data_shape = data_shape_buffer.get_access<sycl::access::mode::read_write>(h);
                auto acc_data_stride = data_stride_buffer.get_access<sycl::access::mode::read_write>(h);
                auto acc_block_shape = block_shape_buffer.get_access<sycl::access::mode::read_write>(h);
                auto acc_block_stride = block_stride_buffer.get_access<sycl::access::mode::read_write>(h);
                auto acc_block_move = block_move_buffer.get_access<sycl::access::mode::read_write>(h);
                auto acc_grid_shape = grid_shape_buffer.get_access<sycl::access::mode::read_write>(h);
                auto acc_grid_stride = grid_stride_buffer.get_access<sycl::access::mode::read_write>(h);

                h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_local_id(2);
                    int total_block_id = item_id / block_size;
                    int total_lane_id = item_id % block_size;
                    int lane_id[20];
                    int block_id[20];
                    int total_pos = 0;
                    for (int i = 0; i < dim_num; i++) {
                        lane_id[i] = total_lane_id / acc_block_stride[i] % acc_block_shape[i];
                        block_id[i] = total_block_id / acc_grid_stride[i] % acc_grid_shape[i];
                        total_pos += (block_id[i] * acc_block_move[i] + lane_id[i]) * acc_data_stride[i];
                    }

                    res[item_id]=myTensor[total_pos];
                });
            }).wait();
        }

        /*
            用重组结果更新原数据
        */
        void UpdateData(ImplType* res, ImplType* myTensor, sycl::queue& q){
            // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteData(cnt,res,this->posNumberList[i].pos,myTensor);
            // }
            sycl::range<3> local(1, 1, this->posNumberList.size());
            sycl::range<3> global(1, 1, 1);
            q.submit([&](handler &h) {
                auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_local_id(2);
                    myTensor[myIdxAccessor[item_id]] = res[item_id];
                });
            }).wait();
        }

        /*
            用重组结果更新原数据
        */
        void UpdateData_1(ImplType* res, ImplType* myTensor, sycl::queue& q){
            auto block_size = this->block_size;
            auto dim_num = this->dim_num;
            sycl::range<3> local(1, 1, this->block_num * this->block_size);
            sycl::range<3> global(1, 1, 1);
            q.submit([&](handler &h) {
                auto acc_data_shape = data_shape_buffer.get_access<sycl::access::mode::read_write>(h);
                auto acc_data_stride = data_stride_buffer.get_access<sycl::access::mode::read_write>(h);
                auto acc_block_shape = block_shape_buffer.get_access<sycl::access::mode::read_write>(h);
                auto acc_block_stride = block_stride_buffer.get_access<sycl::access::mode::read_write>(h);
                auto acc_block_move = block_move_buffer.get_access<sycl::access::mode::read_write>(h);
                auto acc_grid_shape = grid_shape_buffer.get_access<sycl::access::mode::read_write>(h);
                auto acc_grid_stride = grid_stride_buffer.get_access<sycl::access::mode::read_write>(h);

                h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_local_id(2);
                    int total_block_id = item_id / block_size;
                    int total_lane_id = item_id % block_size;
                    int lane_id[20];
                    int block_id[20];
                    int total_pos = 0;
                    for (int i = 0; i < dim_num; i++) {
                        lane_id[i] = total_lane_id / acc_block_stride[i] % acc_block_shape[i];
                        block_id[i] = total_block_id / acc_grid_stride[i] % acc_grid_shape[i];
                        total_pos += (block_id[i] * acc_block_move[i] + lane_id[i]) * acc_data_stride[i];
                    }
                    sycl::atomic_ref<
                        ImplType,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space
                    > atomic_myTensor(myTensor[total_pos]);
                    atomic_myTensor.store(res[item_id]);
                });
            }).wait();
        }

        /*
            增加一个算子
        */
        void push_back(Dac_Op op) {
            this->ops.push_back(op);
            this->posNumberList.clear();
            std::vector<int> pos; // 存位置的中间变量
            GetPos(pos, this->ops, 0);
            std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.number==b.number)?a.pos<b.pos:a.number<b.number;});
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
            std::vector<int> pos; // 存位置的中间变量
            GetPos(pos, this->ops, 0);
            std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.number==b.number)?a.pos<b.pos:a.number<b.number;});
       }
};

#endif
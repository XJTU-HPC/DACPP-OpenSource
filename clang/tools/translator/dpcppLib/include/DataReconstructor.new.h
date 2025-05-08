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
        // DataInfo myDataInfo;                   // 数据信息 形状
        // Dac_Ops ops;                           // 作用于数据的算子组
        // std::vector<int> data_shape;
        // std::vector<int> data_stride;
        // std::vector<int> grid_shape;
        // std::vector<int> grid_stride;
        // std::vector<int> block_shape;
        // std::vector<int> block_stride;
        // std::vector<int> block_move;
        
    public:
        int dim_num;
        int data_size;
        int block_size;
        int block_num;
        std::vector<int> start;
        std::vector<int> view_shape;
        std::vector<int> view_stride;
        sycl::buffer<int> start_buffer;
        sycl::buffer<int> data_shape_buffer;
        sycl::buffer<int> data_stride_buffer;
        sycl::buffer<int> view_shape_buffer;
        sycl::buffer<int> view_stride_buffer;
        sycl::buffer<int> grid_shape_buffer;
        sycl::buffer<int> grid_stride_buffer;
        sycl::buffer<int> block_shape_buffer;
        sycl::buffer<int> block_stride_buffer;
        sycl::buffer<int> block_move_buffer;
        DataReconstructor()
        : start_buffer(range<1>(1)),
          data_shape_buffer(range<1>(1)),
          data_stride_buffer(range<1>(1)),
          view_shape_buffer(range<1>(1)),
          view_stride_buffer(range<1>(1)),
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
            std::vector<int> start;
            std::vector<int> data_shape;
            std::vector<int> data_stride;
            std::vector<int> grid_shape;
            std::vector<int> grid_stride;
            std::vector<int> block_shape;
            std::vector<int> block_stride;
            std::vector<int> block_move;

            this->dim_num = dataInfo.dim;
            data_shape = dataInfo.dimLength;
            int total_data_stride = 1;
            int total_block_stride = 1;
            int total_grid_stride = 1;
            for (int i = 0; i < this->dim_num; i++) {
                start.push_back(0);
                block_shape.push_back(ops[i].size);
                block_move.push_back(ops[i].stride);
                grid_shape.push_back(ops[i].split_size);
                total_data_stride *= data_shape[i];
                total_block_stride *= block_shape[i];
                total_grid_stride *= grid_shape[i];
            }
            this->data_size = total_data_stride;
            this->block_size = total_block_stride;
            this->block_num = total_grid_stride;
            for (int i = 0; i < this->dim_num; i++) {
                total_data_stride /= data_shape[i];
                total_block_stride /= block_shape[i];
                total_grid_stride /= grid_shape[i];
                data_stride.push_back(total_data_stride);
                block_stride.push_back(total_block_stride);
                grid_stride.push_back(total_grid_stride);
            }
            this->start = start;
            this->view_shape = data_shape;
            this->view_stride = data_stride;
            this->start_buffer = sycl::buffer<int>(start.begin(), start.end());
            this->data_shape_buffer = sycl::buffer<int>(data_shape.begin(), data_shape.end());
            this->data_stride_buffer = sycl::buffer<int>(data_stride.begin(), data_stride.end());
            this->view_shape_buffer = sycl::buffer<int>(data_shape.begin(), data_shape.end());
            this->view_stride_buffer = sycl::buffer<int>(data_stride.begin(), data_stride.end());
            this->block_shape_buffer = sycl::buffer<int>(block_shape.begin(), block_shape.end());
            this->block_stride_buffer = sycl::buffer<int>(block_stride.begin(), block_stride.end());
            this->block_move_buffer = sycl::buffer<int>(block_move.begin(), block_move.end());
            this->grid_shape_buffer = sycl::buffer<int>(grid_shape.begin(), grid_shape.end());
            this->grid_stride_buffer = sycl::buffer<int>(grid_stride.begin(), grid_stride.end());
        }

        void add_start(std::vector<int> start) {
            for (int i = 0; i < this->dim_num; i++) {
                this->start[i] += start[i];
            }
            this->start_buffer = sycl::buffer<int>(this->start.begin(), this->start.end());
            //debug
            // for (int i = 0; i < this->dim_num; i++) {
            //     std::cout<<this->start[i]<<" ";
            // }
            // std::cout<<"\n";
        }

        void sub_start(std::vector<int> start) {
            for (int i = 0; i < this->dim_num; i++) {
                this->start[i] -= start[i];
            }
            this->start_buffer = sycl::buffer<int>(this->start.begin(), this->start.end());
            //debug
            // for (int i = 0; i < this->dim_num; i++) {
            //     std::cout<<this->start[i]<<" ";
            // }
            // std::cout<<"\n";
        }
        void set_data_shape(std::vector<int> data_shape) {
            this->data_shape_buffer = sycl::buffer<int>(data_shape.begin(), data_shape.end());
        }

        /*
            将重组结果写入res长向量。
        */
        void Reconstruct(ImplType* res, ImplType* myTensor, sycl::queue& q){
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
    //     void push_back(Dac_Op op) {
    //         this->ops.push_back(op);
    //         this->posNumberList.clear();
    //         std::vector<int> pos; // 存位置的中间变量
    //         GetPos(pos, this->ops, 0);
    //         std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.number==b.number)?a.pos<b.pos:a.number<b.number;});
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
    //         std::vector<int> pos; // 存位置的中间变量
    //         GetPos(pos, this->ops, 0);
    //         std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.number==b.number)?a.pos<b.pos:a.number<b.number;});
    //    }
};

#endif
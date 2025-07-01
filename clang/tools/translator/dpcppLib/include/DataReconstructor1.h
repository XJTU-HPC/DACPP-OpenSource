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
        std::vector<PosNumber> posNumberList;  // 数据索引与物理位置的映射
        std::vector<int> myIdx;
        sycl::buffer<int> myIdxBuffer;         // 用于在SYCL中传递 myIdx   
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
        
        // /*
        //     将特定位置元素写入res长向量。
        // */
        // void WriteRes(int &cnt, ImplType* res, std::vector<int> pos, const dacpp::TensorBase<ImplType> &myTensor) {
        //     res[cnt++]=myTensor.getElement(pos);
        // }

        // /*
        //     将更新结果写入 myTensor
        // */
        // void WriteData(int &cnt, ImplType* res, std::vector<int> pos, dacpp::TensorBase<ImplType> &myTensor) {
        //     myTensor.reviseValue(res[cnt++],pos);
        // }
        
    public:
        DataReconstructor() : myIdxBuffer(nullptr, sycl::range<1>(0)) {
        // 初始化为空 buffer，稍后可以在 init 函数中重新分配
        }

        /*
            通过 数据形状，作用于数据的算子，初始化数据重组器
        */
        void init(DataInfo dataInfo, Dac_Ops ops){
            this->myDataInfo=dataInfo;
            this->ops=ops;
            std::vector<int> pos; // 存位置的中间变量
            GetPos(pos, ops, 0);
            
            std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.number==b.number)?a.pos<b.pos:a.number<b.number;});
            for(int i=0;i<this->posNumberList.size();i++) this->myIdx.push_back(0);
            for(int i=0;i<this->posNumberList.size();i++){
                int stride = 1;
                for(int j=0;j<this->myDataInfo.dim;j++) stride*=this->myDataInfo.dimLength[j];
                int idx = 0;
                for(int j=0;j<this->myDataInfo.dim;j++) {
                    stride/=this->myDataInfo.dimLength[j];
                    idx+=this->posNumberList[i].pos[j]*stride;
                }
                this->myIdx[i]=idx;
            }

            this->myIdxBuffer = sycl::buffer<int>(this->myIdx.data(), sycl::range<1>(this->myIdx.size()));
        }

        /*
            将重组结果写入res长向量。
        */
        // void Reconstruct(ImplType* res, ImplType* myTensor, sycl::queue& q){
        //     sycl::device device = q.get_device();
        //     int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
        //     if(this->posNumberList.size() < max_global_size){
        //         sycl::range<3> local(1, 1, this->posNumberList.size());
        //         sycl::range<3> global(1, 1, 1);
        //         q.submit([&](handler &h) {

        //             auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
        //             auto range = this->myIdxBuffer.get_range();

        //             sycl::stream out(1024, 256, h);
        //             h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
        //                 const auto item_id = item.get_local_id(2);
                    
        //                     res[item_id]=myTensor[myIdxAccessor[item_id]];
        //             });
        //         }).wait();
        //     }else{
        //         int have_done = 0;
        //         while(have_done < this->posNumberList.size()){

        //             int now_size = std::min(max_global_size, (int)this->posNumberList.size()-have_done);
        //             // printf("now_size: %d\n", now_size);
        //             // printf("have_done: %d\n", have_done);
        //             // printf("this->posNumberList.size(): %d\n", this->posNumberList.size());
        //             sycl::range<3> local(1, 1, now_size);
        //             sycl::range<3> global(1, 1, 1);
        //             q.submit([&](handler &h) {
        //                 auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
        //                 auto range = this->myIdxBuffer.get_range();
        //                 sycl::stream out(1024, 256, h);
        //                 h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
        //                     const auto item_id = item.get_local_id(2);
        //                     res[have_done+item_id]=myTensor[myIdxAccessor[have_done+item_id]];
        //                 });
        //             }).wait();
        //             have_done += now_size;
        //         }
        //     }
        // }

        //重写Reconstruct函数支持USM 之前的可能有问题因为多卡过不了
        void Reconstruct(ImplType* res, ImplType* myTensor, sycl::queue& q)
        {
            // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteRes(cnt, res,this->posNumberList[i].pos,myTensor);
            // }
            int Item_Size = this->posNumberList.size();
            sycl::device device = q.get_device();
            int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
            int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
            sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size));
            sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
            q.submit([&](handler &h) {
                auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
                    if(item_id >= Item_Size)
                        return;
                    res[item_id]=myTensor[myIdxAccessor[item_id]];
                });
            }).wait();
        }

        //重载Reconstruct函数支持buffer
        void Reconstruct(sycl::buffer<ImplType>& r_buf, sycl::buffer<ImplType>& b_buf, sycl::queue& q){
            // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteRes(cnt, res,this->posNumberList[i].pos,myTensor);
            // }
            int Item_Size = this->posNumberList.size();
            sycl::device device = q.get_device();
            int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
            int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
            sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size));
            sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
            q.submit([&](handler &h) {
                auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
                auto res = r_buf.template get_access<sycl::access::mode::read_write>(h);
                auto myTensor = b_buf.template get_access<sycl::access::mode::read_write>(h);
                h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
                    const auto item_id = item.get_group(2)*item.get_local_range(2)+item.get_local_id(2);
                    if(item_id >= Item_Size)
                        return;
                    res[item_id]=myTensor[myIdxAccessor[item_id]];
                });
            }).wait();
        }

        /*
            用重组结果更新原数据
        */
        // void UpdateData(ImplType* res, ImplType* myTensor, sycl::queue& q){
        //     // int cnt=0;
        //     // for (int i=0; i<this->posNumberList.size(); i++) {
        //     //     this->WriteData(cnt,res,this->posNumberList[i].pos,myTensor);
        //     // }
        //     sycl::device device = q.get_device();
        //     int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
        //     if(this->posNumberList.size() < max_global_size){
        //         sycl::range<3> local(1, 1, this->posNumberList.size());
        //         sycl::range<3> global(1, 1, 1);
        //         q.submit([&](handler &h) {
        //             auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
        //             h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
        //                 const auto item_id = item.get_local_id(2);
        //                 myTensor[myIdxAccessor[item_id]] = res[item_id];
        //             });
        //         }).wait();
        //     }else{
        //         int have_done = 0;
        //         while(have_done < this->posNumberList.size()){
        //             int now_size = std::min(max_global_size, (int)this->posNumberList.size()-have_done);
        //             // printf("now_size: %d\n", now_size);
        //             // printf("have_done: %d\n", have_done);
        //             // printf("this->posNumberList.size(): %d\n", this->posNumberList.size());
        //             sycl::range<3> local(1, 1, now_size);
        //             sycl::range<3> global(1, 1, 1);
        //             q.submit([&](handler &h) {
        //                 auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
        //                 h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
        //                     const auto item_id = item.get_local_id(2);
        //                     myTensor[myIdxAccessor[have_done+item_id]] = res[have_done+item_id];
        //                 });
        //             }).wait();
        //             have_done += now_size;
        //         }
        //     }
        // }
        
        //USM版本的数据逆重组
        void UpdateData(ImplType* res, ImplType* myTensor, sycl::queue& q)
        {
            // int cnt=0;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     this->WriteData(cnt,res,this->posNumberList[i].pos,myTensor);
            // }
            int Item_Size = this->posNumberList.size();
            sycl::device device = q.get_device();
            int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
            int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
            sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size));
            sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
            q.submit([&](handler &h) {
                auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
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
            int Item_Size = this->posNumberList.size();
            sycl::device device = q.get_device();
            int max_global_size = device.get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
            int work_group_size = (Item_Size + max_global_size - 1) / max_global_size;  // 计算所需的工作组数量
            sycl::range<3> local(1, 1, std::min(Item_Size, max_global_size));
            sycl::range<3> global(1, 1, (Item_Size <= max_global_size) ? Item_Size : work_group_size * max_global_size);
            q.submit([&](handler &h) {
                auto myIdxAccessor = myIdxBuffer.get_access<sycl::access::mode::write>(h);
                auto res = r_buf.template get_access<sycl::access::mode::read_write>(h);
                auto myTensor = b_buf.template get_access<sycl::access::mode::read_write>(h);

                h.parallel_for(sycl::nd_range<3>(global, local),[=](sycl::nd_item<3> item) {
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
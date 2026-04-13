#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>
#include<map>
#include "Tensor.hpp"
#include "dacInfo.h"
#include "sub_template.h"


struct Range
{
    int start;
    int end;
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
        dacpp::Tensor<ImplType> myTensor;      // 数据
        Dac_Ops ops;                           // 作用于数据的算子组
        std::vector<PosNumber> posNumberList;  // 数据索引与物理位置的映射
        // int cnt;                               // 结果长向量的计数器
        // std::vector<int> number;               // 存编号的中间变量
        // std::vector<int> pos;                  // 存位置的中间变量     
        void GetPos(std::vector<int> &pos, Dac_Ops &ops, int now) {
            if (now == this->myTensor.getDim()) {
                // std::cout<<pos[0]<<" "<<pos[1]<<std::endl;
                GetPosNumber(pos, ops);
                return;
            }
            for (int i = 0; i < this->myTensor.getShape(now); i++) {
                pos.push_back(i);
                GetPos(pos, ops, now + 1);
                pos.pop_back();
            }
        }
        void GetPosNumber(std::vector<int> &pos, Dac_Ops &ops) {
            int dimNum = this->myTensor.getDim();
            std::vector<Range> region;
            for (int i = 0; i < dimNum; i++) {
                Range r;
                r.start = 0;
                r.end = this->myTensor.getShape(i);
                region.push_back(r);
            }
            // if(pos[0]==2&&pos[1]==2) {
            //     for (int i = 0; i < dimNum; i++) {
            //         std::cout<<i<<" "<<region[i].start<<" "<<region[i].end<<std::endl;
            //     }
            // }
            
            // std::cout<<pos[0]<<" "<<pos[1]<<std::endl;
            std::vector<int> number;   // 存数据索引的中间变量
            RecursiveTraversal(number, pos, region, 0);
        }
        /*
            递归得到物理位置 pos 在当前算子组作用下应该映射的数据索引，以及划分好的数据单元的数据区域。
        */
        void RecursiveTraversal(std::vector<int> &number,std::vector<int> &pos, std::vector<Range> &region, int now) {
            if (now == ops.size) {
                // if(pos[0]==2&&pos[1]==2) {
                //     std::cout<<region[0].start<<" "<<region[0].end<<" "<<region[1].start<<" "<<region[1].end<<std::endl;
                // }
                // std::cout<<pos[0]<<" "<<pos[1]<<std::endl;
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
            // while (region[op.dimId].start + l * op.stride + op.size <= id) l++;
            int r = (region[op.dimId].start+(id&(-op.stride))+op.size<region[op.dimId].end) ? (id&(-op.stride))/op.stride : (region[op.dimId].end-op.size-region[op.dimId].start)/op.stride;
            // while (region[op.dimId].start + r * op.stride > id) r--;
            // std::cout<<pos[0]<<" "<<pos[1]<<" "<<op.dimId<<" "<<id<<" "<<l<<" "<<r<<std::endl;
            // if(pos[0]==2&&pos[1]==2) std::cout<<now<<" "<<id<<" "<<l<<" "<<r<<" "<<op.dimId<<" "<<region[op.dimId].end<<std::endl;
            for(int i = l; i <= r; i++) {
                // if(pos[0]==2&&pos[1]==2) std::cout<<"ok"<<i<<"\n";
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
        
        /*
            将特定位置元素写入res长向量。
        */
        void WriteRes(int &cnt, ImplType* res, std::vector<int> pos) {
            res[cnt++]=this->myTensor.getElement(pos);
            // std::cout<<cnt<<" "<<"pos: "<<pos[0]<<" "<<pos[1]<<" "<<res[cnt-1]<<std::endl;
            // std::cout<<this->myTensor.getDim()<<" "<<pos.size()<<std::endl;
            // std::cout<<pos[0]<<" "<<pos[1]<<std::endl;
        }

        /*
            将更新结果写入 myTensor
        */
        void WriteData(int &cnt, ImplType* res, std::vector<int> pos) {
            myTensor.reviseValue(res[cnt++],pos);
        }
        
    public:
        DataReconstructor(){

        }

        /*
            通过 数据，作用于数据的算子，初始化数据重组器
        */
        void init(dacpp::Tensor<ImplType> myTensor, Dac_Ops ops){
            this->myTensor=myTensor;
            this->ops=ops;
            std::vector<int> pos; // 存位置的中间变量
            GetPos(pos, ops, 0);
            
            // std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.pos==b.pos)?a.number<b.number:a.pos<b.pos;});
            // std::cout<<this->posNumberList.size()<<std::endl;
            // for (int i=0; i<this->posNumberList.size(); i++) {
            //     std::cout<<"("<<this->posNumberList[i].pos[0]<<","<<this->posNumberList[i].pos[1]<<"): ("<<this->posNumberList[i].number[0]<<","<<this->posNumberList[i].number[1]<<") "<<this->posNumberList[i].region[0].start<<" "<<this->posNumberList[i].region[0].end<<" "<<this->posNumberList[i].region[1].start<<" "<<this->posNumberList[i].region[1].end<<std::endl;
            // }
            std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.number==b.number)?a.pos<b.pos:a.number<b.number;});
        }

        /*
            将重组结果写入res长向量。
        */
        void Reconstruct(ImplType* res){
            int cnt=0;
            // std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return (a.number==b.number)?a.pos<b.pos:a.number<b.number;});
            // std::cout<<this->posNumberList.size()<<std::endl;
            for (int i=0; i<this->posNumberList.size(); i++) {
                this->WriteRes(cnt, res,this->posNumberList[i].pos);
            }
        }

        /*
            用重组结果更新原数据
        */
        dacpp::Tensor<ImplType> UpdateData(ImplType* res){
            int cnt=0;
            for (int i=0; i<this->posNumberList.size(); i++) {
                this->WriteData(cnt,res,this->posNumberList[i].pos);
            }
            return this->myTensor;
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
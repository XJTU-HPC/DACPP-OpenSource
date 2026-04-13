#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>
#include<map>
#include"DataReconstructor.h"


/*
    通过 数据内容，作用于数据的算子 初始化数据重组器
*/
template<typename ImplType>
void DataReconstructor::init(dacpp::Tensor<ImplType> myTensor, Dac_Ops ops){
    this->myTensor=myTensor;
    this->ops=ops;
    GetPos(ops, 0);     
}

template<typename ImplType>
void DataReconstructor::GetPos(Dac_Ops ops, int now) {
    if (now == this->myTensor.getDim()) {
        GetPosNumber(this->pos, ops);
        return;
    }
    for (int i = 0; i < this->myTensor.getShape(now); i++) {
        this->pos.push_back(i);
        GetPos(ops, now + 1);
        this->pos.pop_back();
    }
}

template<typename ImplType>
void DataReconstructor::GetPosNumber(std::vector<int> pos, Dac_Ops ops) {
    int dimNum = this->myTensor.getDim();
    std::vector<Range> region;
    for (int i = 0; i < dimNum; i++) {
        region[i].start = 0;
        region[i].end = this->myTensor.getShape(i);
    }
    RecursiveTraversal(pos, region, 0);
}

template<typename ImplType>
void DataReconstructor::RecursiveTraversal(std::vector<int> pos, std::vector<Range> region, int now) {
    if (now == ops.size) {
        PosNumber posNum;
        posNum.number = this->number;
        posNum.pos = pos;
        this->posNumberList.push_back(posNum);
        return;
    }
    Dac_Op op = ops[now];
    int id = pos[op.dimId];
    int l = 0;
    while (region[op.dimId].start + l * op.stride + op.size <= id) l++;
    int r = 0;
    while (region[op.dimId].start + r * op.stride < id) r++;
    for(int i = l; i <= r; i++) {
        this->number.push_back(i);
        region[op.dimId].start = region[op.dimId].start + l * op.stride;
        region[op.dimId].end = region[op.dimId].start + (l * op.stride + op.size);
        RecursiveTraversal(pos, region, now + 1);
        region[op.dimId].end = region[op.dimId].start - (l * op.stride + op.size);
        region[op.dimId].start = region[op.dimId].start - l * op.stride;
        this->number.pop_back();
    }
}

/*
    将特定位置元素写入res长向量。
*/
template<typename ImplType>
void DataReconstructor::WriteData(ImplType* res, std::vector<int> pos) {
    res[this->cnt++]=this->myTensor.getElement(pos);
}

/*
    将重组结果写入res长向量。
*/
template<typename ImplType>
void DataReconstructor::Reconstruct(ImplType* res){
    this->cnt=0;
    std::sort(this->posNumberList.begin(),this->posNumberList.end(),[](PosNumber a,PosNumber b){return a.number<b.number;});
    for (int i=0; i<this->posNumberList.size(); i++) {
        this->WriteData(res,this->posNumberList[i].pos);
    }
}
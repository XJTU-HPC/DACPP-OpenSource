#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include"dacInfo.h"
#include<cstring>

Dac_Op::Dac_Op(){}

/*
	通过 算子名称，算子划分数，算子作用的维度 创建算子。
*/
Dac_Op::Dac_Op(std::string Name,int SplitSize,int dimId){
	strcpy(this->name,Name.c_str());
	this->split_size = SplitSize;
	this->dimId = dimId;
}
/*
	设置 算子作用的维度。
*/
void Dac_Op::setDimId(int id){
	this->dimId = id;
}
/*
	设置 算子划分的每份长度。
*/
void Dac_Op::setSplitLength(int len){
	this->split_length = len;
}

/*
	设置 算子划分的划分数。
*/
void Dac_Op::SetSplitSize(int split_size) {
	this->split_size = split_size;
}

Dac_Ops::Dac_Ops(){
	this->size = 0;
}
void Dac_Ops::push_back(Dac_Op op){
	this->DacOps.push_back(op);
	this->size++;
}
void Dac_Ops::push_back(Dac_Ops ops){
	for(int i=0;i<ops.size;i++){
		this->DacOps.push_back(ops[i]);
		this->size++;
	}
}

void Dac_Ops::pop_back() {
	this->DacOps.pop_back();
	this->size--;
}

void Dac_Ops::clear(){
	while(this->size>0) {
		this->pop_back();
	}
}
Dac_Op& Dac_Ops::operator[](int i){
	return this->DacOps[i];
}


DacData::DacData(){

}
/*
	通过 数据名称，数据维数，作用于各个维度的算子 创建数据。
*/
DacData::DacData(std::string Name, int dim, Dac_Ops ops){
	this->name = Name;
	this->dim=dim;
	this->ops = ops;
}
/*
	设置 数据在某维度上的长度。
*/
void DacData::setDimLength(int dimId,int len){
	if(dimId<this->DimLength.size()) this->DimLength[dimId]=len;
}
/*
	得到 数据在某维度上的长度。
*/
int DacData::getDimlength(int dimId){
	if(dimId<this->DimLength.size()) return this->DimLength[dimId];
	else return -1;
}

Args::Args(){
	this->size=0;
}
void Args::push_back(DacData x){
	this->args.push_back(x);
	this->size++;
}
DacData& Args::operator[](int i){
	return this->args[i];
}

RegularSlice::RegularSlice(std::string name, int size, int stride) {
	strcpy(this->name,name.c_str());
	this->stride = stride;
	this->size = size;
}

Index::Index(std::string name) {
	strcpy(this->name,name.c_str());
	this->stride = 1;
	this->size = 1;
}

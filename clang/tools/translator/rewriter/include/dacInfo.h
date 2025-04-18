#ifndef DACINFO_H
#define DACINFO_H

#include<string>
#include<cstring>
#include<iostream>
#include<fstream>
#include<vector>

/*
	Dac 算子类
*/
class Dac_Op{
	public:
		char* name = new char[5]; // 算子名称
		int split_size;           // 划分数    算子作用于不同数据划分数一致
		int split_length;         // 每份长度  算子作用于不同数据划分长度可以不同
		int dimId;                // 算子作用的维度
		int stride;               // 步长
		int size;                 // 作用维度上的每份长度

		Dac_Op();
		/*
			通过 算子名称，算子划分数，算子作用的维度 创建算子。
		*/
		Dac_Op(std::string Name,int SplitSize,int dimId);
		/*
			设置 算子作用的维度。
		*/
		void setDimId(int id);
		/*
			设置 算子划分的每份长度。
		*/
		void setSplitLength(int len);

		/*
			设置 算子的划分份数。
		*/
		void SetSplitSize(int split_size);
};
/*
	Dac 算子组
*/
class Dac_Ops{
	public:
		std::vector<Dac_Op> DacOps;
		int size;

		Dac_Ops();
		void push_back(Dac_Op op);
		void push_back(Dac_Ops ops);
		void pop_back();
		void clear();
		Dac_Op& operator[](int i);
};
/*
	Dac 数据类，表示数据信息。
*/
class DacData{
	public:
		std::string name;                 // 数据名称
		Dac_Ops ops;                      // 被用于该数据的算子组
		int dim;                          // 数据维数
		std::vector<int> DimLength;       // 维上的长度

		DacData();
		/*
			通过 数据名称，数据维数，作用于各个维度的算子 创建数据。
		*/
		DacData(std::string Name,int dim,Dac_Ops ops);
		/*
			设置 数据在某维度上的长度。
		*/
		void setDimLength(int dimId,int len);
		/*
			得到 数据在某维度上的长度。
		*/
		int getDimlength(int dimId);
};
/*
	Dac 参数类，表示嵌入计算时传入的参数。
	其实就是封装的 DAC 数据数组。
*/
class Args{
	public:
		std::vector<DacData> args;
		int size;

		Args();
		void push_back(DacData x);
		DacData& operator[](int i);
};

/*
	RegularSlice 规则分区算子类。
*/
class RegularSlice : public Dac_Op {
	public:

		RegularSlice();
		/*
			通过 算子名称，步长，作用维度上的每份长度 创建规则分区算子。
		*/
		RegularSlice(std::string name, int size, int stride);
};

/*
	Index 降维算子类。
*/
class Index : public Dac_Op {
	public:

		Index();
		/*
			通过 算子名称，步长，作用维度上的每份长度 创建规则分区算子。
		*/
		Index(std::string name);
};

inline Dac_Op::Dac_Op(){}

/*
	通过 算子名称，算子划分数，算子作用的维度 创建算子。
*/
inline Dac_Op::Dac_Op(std::string Name,int SplitSize,int dimId){
	strcpy(this->name,Name.c_str());
	this->split_size = SplitSize;
	this->dimId = dimId;
}
/*
	设置 算子作用的维度。
*/
inline void Dac_Op::setDimId(int id){
	this->dimId = id;
}
/*
	设置 算子划分的每份长度。
*/
inline void Dac_Op::setSplitLength(int len){
	this->split_length = len;
}

/*
	设置 算子划分的划分数。
*/
inline void Dac_Op::SetSplitSize(int split_size) {
	this->split_size = split_size;
}

inline Dac_Ops::Dac_Ops(){
	this->size = 0;
}
inline void Dac_Ops::push_back(Dac_Op op){
	this->DacOps.push_back(op);
	this->size++;
}
inline void Dac_Ops::push_back(Dac_Ops ops){
	for(int i=0;i<ops.size;i++){
		this->DacOps.push_back(ops[i]);
		this->size++;
	}
}

inline void Dac_Ops::pop_back() {
	this->DacOps.pop_back();
	this->size--;
}

inline void Dac_Ops::clear(){
	while(this->size>0) {
		this->pop_back();
	}
}
inline Dac_Op& Dac_Ops::operator[](int i){
	return this->DacOps[i];
}


inline DacData::DacData(){

}
/*
	通过 数据名称，数据维数，作用于各个维度的算子 创建数据。
*/
inline DacData::DacData(std::string Name, int dim, Dac_Ops ops){
	this->name = Name;
	this->dim=dim;
	this->ops = ops;
}
/*
	设置 数据在某维度上的长度。
*/
inline void DacData::setDimLength(int dimId,int len){
	if(dimId<this->DimLength.size()) this->DimLength[dimId]=len;
}
/*
	得到 数据在某维度上的长度。
*/
inline int DacData::getDimlength(int dimId){
	if(dimId<this->DimLength.size()) return this->DimLength[dimId];
	else return -1;
}

inline Args::Args(){
	this->size=0;
}
inline void Args::push_back(DacData x){
	this->args.push_back(x);
	this->size++;
}
inline DacData& Args::operator[](int i){
	return this->args[i];
}

inline RegularSlice::RegularSlice(std::string name, int size, int stride) {
	strcpy(this->name,name.c_str());
	this->stride = stride;
	this->size = size;
}

inline Index::Index(std::string name) {
	strcpy(this->name,name.c_str());
	this->stride = 1;
	this->size = 1;
}
#endif
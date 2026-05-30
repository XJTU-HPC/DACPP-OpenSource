#ifndef DACINFO_H
#define DACINFO_H

#include<string>
#include<cstring>
#include<iostream>
#include<fstream>
#include<vector>

// Basic operator descriptor used by the code-generation templates.
class Dac_Op{
	public:
		char* name = new char[5];
		int split_size;
		int split_length;
		int dimId;
		int stride;
		int size;

		Dac_Op();
Dac_Op(std::string Name,int SplitSize,int dimId);
void setDimId(int id);
void setSplitLength(int len);
void SetSplitSize(int split_size);
};
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
class DacData{
	public:
		std::string name;
		Dac_Ops ops;
		int dim;
		std::vector<int> DimLength;

		DacData();
DacData(std::string Name,int dim,Dac_Ops ops);
void setDimLength(int dimId,int len);
int getDimlength(int dimId);
};

// Ordered argument list passed to generated helper routines.
class Args{
	public:
		std::vector<DacData> args;
		int size;

		Args();
		void push_back(DacData x);
		DacData& operator[](int i);
};
class RegularSlice : public Dac_Op {
	public:

		RegularSlice();
RegularSlice(std::string name, int size, int stride);
};
class Index : public Dac_Op {
	public:

		Index();
Index(std::string name);
};

inline Dac_Op::Dac_Op(){}
inline Dac_Op::Dac_Op(std::string Name,int SplitSize,int dimId){
	strcpy(this->name,Name.c_str());
	this->split_size = SplitSize;
	this->dimId = dimId;
}
inline void Dac_Op::setDimId(int id){
	this->dimId = id;
}
inline void Dac_Op::setSplitLength(int len){
	this->split_length = len;
}
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
inline DacData::DacData(std::string Name, int dim, Dac_Ops ops){
	this->name = Name;
	this->dim=dim;
	this->ops = ops;
}
inline void DacData::setDimLength(int dimId,int len){
	if(dimId<this->DimLength.size()) this->DimLength[dimId]=len;
}
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

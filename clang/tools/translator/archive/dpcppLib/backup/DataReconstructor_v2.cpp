#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>
#include<map>
// #include "dacppLib/sub_template.h"
#include "dacppLib/Tensor.hpp"

/*
    维度类，用于表示数据某一维度的信息。
*/
class Dim{
    private:
        int id;           // 维度号
        bool isIndex;     // 是否被降维
    public:
        int iter;         // 该维度的遍历器

        Dim(){

        }
        /*
            通过 维度号，是否被降维 构造维度类
        */
        Dim(int id,bool isIndex){
            this->id=id;
            this->isIndex=isIndex;
            this->iter=0;
        }
        /*
            设置维度号
        */
        void setId(int id){
            this->id=id;
        }
        /*
            获得维度号
        */
        int getId(){
            return this->id;
        }
        /*
            设置该维度是否被降维
        */
        void setIndex(bool flag){
            this->isIndex=flag;
        }
        /*
            检查该维度是否被降维
        */
        bool checkIndex(){
            return this->isIndex;
        }
        /*
            重载小于号用于数据重组。
            降维的维度先遍历，未降维的后遍历。
        */
        bool operator <(Dim &d){
            if(this->isIndex==d.isIndex) return this->id<d.id;
            else return this->isIndex;
        }
};
/*
    数据重组器类，用于数据重组。
*/
template<typename ImplType>
class DataReconstructor{
    private:
        // DacData myData;                        // 数据的信息
        dacpp::Tensor<ImplType> myTensor;      // 数据
        std::vector<Dim> a;                    // 数据的维度信息
        std::map<int,int> mp;                  // 维度与遍历顺序的映射
        int cnt;                               // 结果长向量的计数器
    public:
        DataReconstructor(){

        }
        /*
            通过 数据内容，数据降维信息 构造数据重组器
        */
        void init(dacpp::Tensor<ImplType> myTensor,std::vector<bool> isIndex){
            // this->myData=myData;
            this->myTensor=myTensor;

            this->a.clear();
            for(int i=0;i<this->myTensor.getDim();i++){
                Dim x;
                x.setId(i);
                x.setIndex(isIndex[i]);
                this->a.push_back(x);
            }
            std::sort(this->a.begin(),this->a.end());
            this->mp.clear();
            for(int i=0;i<this->myTensor.getDim();i++){
                mp[a[i].getId()]=i;
            }
        }
        /* 
            实现数据的遍历与重组。
        */
        void RecursiveTraversal(ImplType* res,int now){
            if(now==this->myTensor.getDim()){
                dacpp::Tensor<ImplType> tensor=this->myTensor;
                for(int i=0;i<this->myTensor.getDim();i++){
                    tensor=tensor.slice(0,this->a[this->mp[i]].iter);
                    // std::cout<<a[this->mp[i]].iter<<" ";
                }
                // std::cout<<" : ";
                tensor.tensor2Array(res+this->cnt);
                // tensor.print();
                this->cnt++;
            }
            else{
                for(a[now].iter=0;a[now].iter<this->myTensor.getShape(a[now].getId());a[now].iter++){
                    RecursiveTraversal(res,now+1);
                }
            }
        }
        /*
            将重组结果写入res长向量。
        */
        void Reconstruct(ImplType* res){
            std::cout<<"数据重组的数据维度遍历顺序: ";
            for(int i=0;i<this->myTensor.getDim();i++){
                std::cout<<a[i].getId()<<" ";
            }
            std::cout<<"\n";

            this->cnt=0;
            RecursiveTraversal(res,0);
        }
};
int main(){
    int* res=(int*)malloc(sizeof(int)*27);
    
    std::vector<int> data{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27};
    std::vector<int> shape{3,3,3};
    dacpp::Tensor<int> myTensor(data,shape);
    myTensor.print();
    
    std::vector<bool> isIndex{true,false,true}; // 表示数据在该维度上是否降维
    DataReconstructor<int> tool;
    tool.init(myTensor,isIndex);
    tool.Reconstruct(res);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
  
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            for(int k=0;k<3;k++){
                std::cout<<res[i*9+j*3+k]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
}
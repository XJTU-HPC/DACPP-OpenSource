#ifndef PARAMETERGENERATION_H
#define PARAMETERGENERATION_H

#include <iostream>
#include <string>
#include <unordered_set>
#include "ReconTensor.h"

class ParameterGeneration
{
    public:
        ParameterGeneration(){
        }
        
        //生成算子的划分数 参数为算子和算子作用的数据信息
        //计算逻辑为(算子作用维度的长度 - 算子的划分大小) / 算子的划分步长 + 1
        //想象一个矩阵的例子帮助理解
        int init_operetor_splitnumber(Dac_Op si,DataInfo data_info)
        {
            int split_num = (data_info.dimLength[si.dimId] - si.size) / si.stride + 1;
            return split_num;
        }

        //生成设备内存的分配大小 支持情况：mat[分区][分区] mat[分区][降维] mat[分区][] mat[降维][]
        //同时也是设备和主机之间内存移动大小 q.memcpy()里面的SIZE
        //参数是一个数据和作用在这个数据上的算子组
        //可以用一个矩阵为例子来理解
        int init_device_memory_size(DataInfo data_info,Dac_Ops ops)
        {
            int result = 1;//初始化结果为1 
            std::unordered_set<int> mySet;//用来存储算子作用的维度 看下这个算子组都作用在数据哪些维度
            for(int i = 0;i < ops.size;i ++)
            {
                int dimId = ops[i].dimId;//拿到算子的维度
                mySet.insert(dimId);//将算子作用的维度放入集合中
                int split_num = (data_info.dimLength[dimId] - ops[i].size) / ops[i].stride + 1;//计算算子的划分数
                int length = split_num * ops[i].size;//划分数乘以划分的大小
                result *= length;
            }
            for(int i = 0;i < data_info.dim;i ++)
            {
                if (mySet.find(i) == mySet.end()) 
                {
                    //i不存在也就是说算子没有作用这个维度 这个维度是保型算子
                    result *= data_info.dimLength[i];
                }
            }
            return result;
        }

        //生成设备内存分配大小 支持情况：mat[][] 对应保型的情况
        //逻辑是每一维个数相乘 可以以矩阵为例帮助理解
        int init_device_memory_size(DataInfo data_info)
        {
            int result = 1;
            for(int i = 0;i < data_info.dim;i ++)
            {
                result *= data_info.dimLength[i]; 
            }
            return result;
        }

        //生成设备内存的分配大小 支持情况：数据重组时中间需要的内存 
        //Dac_Ops ops_in中所有的算子是不同的，由抽象语法树后端进行去重
        int init_device_memory_size(Dac_Ops ops_in,Dac_Ops ops_out,DataInfo data_info)
        {
            int in_op_product = 1;//输入算子划分数的乘积
            for(int i = 0;i < ops_in.size;i ++)
            {
                in_op_product *= ops_in.DacOps[i].split_size; //spilit在前面初始化算子的时候已经完成
            }
            int out_op_product = 1;//输出算子划分数的乘积
            for(int i = 0;i < ops_out.size;i ++)
            {
                out_op_product *= ops_out.DacOps[i].split_size; //spilit在前面初始化算子的时候已经完成
            }
            //return init_device_memory_size(data_info,ops_out) * in_op_product / out_op_product;
            return in_op_product / out_op_product * init_device_memory_size(data_info,ops_out);//先除法再乘法防止爆int
        }

        //生成开辟工作项多少 localsize
        //实际上是输入算子所有划分数的乘积 或者说数据元组的个数（数据单元组成数据元组）
        //由后端对算子组进行去重
        //一份数据元组对应一份工作项 工作项与数据元组相对应
        int init_work_item_size(Dac_Ops in_ops)
        {
            int result = 1;
            for(int i = 0;i < in_ops.size;i ++)
            {
                result *= in_ops.DacOps[i].split_size;
            }
            return result;
        }

        //生成算子的划分长度
        //两个参数分别是算子组和重组之后的数据大小
        void init_op_split_length(Dac_Ops& ops,int size)
        {
            if(ops.size == 0) return;
            ops.DacOps[0].setSplitLength(size / ops.DacOps[0].split_size);//第0维的划分长度是重组后的数据大小除以第0维的划分数
            for(int i = 1;i < ops.size;i ++)
            {
                ops.DacOps[i].setSplitLength(ops.DacOps[i - 1].split_length / ops.DacOps[i].split_size);//第n维的划分长度是第n-1维的划分长度除以第n维的划分数
            }
        }

        //生成SplitLength的矩阵 计算逻辑比较简单
        void init_split_length_martix(int Rows,int Cols,int* matrix,std::vector<Dac_Ops> ops_s)
        {
            for(int i = 0;i < Rows; i++)//row的大小就是ops_s里面包含算子组的个数
            {
                for(int j = 0;j < ops_s[i].size;j ++)//[i][j]访问每个算子组里面的算子 
                {
                    matrix[i * Cols + j] = ops_s[i].DacOps[j].split_length;//将算子的划分数传入到一个矩阵中
                }
            }
        }

        //生成归约中spilits_size的大小 逻辑是输出的算子的划分数除以输出的算子的划分数
        int init_reduction_split_size(Dac_Ops ops_in,Dac_Ops ops_out)
        {
            int in_op_product = 1;//输入算子划分数的乘积
            for(int i = 0;i < ops_in.size;i ++)
            {
                in_op_product *= ops_in.DacOps[i].split_size; //spilit在前面初始化算子的时候已经完成
            }
            int out_op_product = 1;//输出算子划分数的乘积
            for(int i = 0;i < ops_out.size;i ++)
            {
                out_op_product *= ops_out.DacOps[i].split_size; //spilit在前面初始化算子的时候已经完成
            }
            return in_op_product / out_op_product;
        }

        //生成归约中split_length的大小
        //逻辑是某个算子组（输出算子组）最后一个算子的划分数
        int init_reduction_split_length(Dac_Ops ops)
        {
            if(ops.size == 0)
                return 1;
            return ops.DacOps[ops.size - 1].split_length;//返回最后一个算子的划分数
        }

        //计算数据被算子作用后数据单元的形状 本质上就是存了几个数字 比如划分之后是3*3的矩阵 那么这个vector就存了3 3
        //可以用一个矩阵来理解一下
        //对于mat[idx][] mat[][idx]期望是二维的1 N、N 1   mat[idx][idx]期望得到 1 1
        // array[index]期望是1
        // array[]期望是N
        std::vector<int> init_partition_data_shape(DataInfo data_info,Dac_Ops ops) {
            std::vector<int> tmp=data_info.dimLength;//tmp就是数据每个维度的个数
            for(int i=0;i<ops.size;i++) {
                tmp[ops[i].dimId]=ops[i].size;//对tmp中算子作用的维度重新进行赋值
            }
            std::vector<int> res;//存结果的数值
            for(int i=0;i<tmp.size();i++) {
                //if(tmp[i]==1) continue;//原来的数据是几维的 划分后的数据就是几维的 1维的不能忽略
                res.push_back(tmp[i]);
            }
            return res;//返回最终的结果
        }   

        int init_Data_SplitNum(Dac_Ops ops,DataInfo data_info){
            int result = 1;//初始化结果为1 
            for(int i = 0;i < ops.size;i ++)
            {
                int dimId = ops[i].dimId;//拿到算子的维度
                int split_num = (data_info.dimLength[dimId] - ops[i].size) / ops[i].stride + 1;//计算算子的划分数
                // int length = split_num * ops[i].size;//划分数乘以划分的大小
                result *= split_num;
            }
            return result;
        }

        int  init_Data_SplitSize(int data_SplitNum, int Total_Size){
            return Total_Size / data_SplitNum;
        }

        std::vector<int> init_Device_Size(int numDevices,int data_Splitsize, int data_SplitNum){
            bool flag = false;
            if(data_SplitNum % numDevices == 0){
                flag = true;
            }
            std::vector<int> res;
            for(int i = 0; i < numDevices; i++){
                if(i == 0 && !flag){
                    res.push_back(( data_SplitNum / numDevices + 1));
                }else{
                    res.push_back((data_SplitNum / numDevices));
                }
            }
            return res;
        }

        std::vector<int> init_Device_Local(int numDevices,int data_Splitsize, int data_SplitNum){
            bool flag = false;
            if(data_SplitNum % numDevices == 0){
                flag = true;
            }
            std::vector<int> res;
            for(int i = 0; i < numDevices; i++){
                if(i == 0 && !flag){
                    res.push_back(data_SplitNum / numDevices + 1);
                }else{
                    res.push_back(data_SplitNum / numDevices);
                }
            }
            return res;
        }
};

#endif
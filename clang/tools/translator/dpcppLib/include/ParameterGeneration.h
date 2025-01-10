#ifndef PARAMETERGENERATION_H
#define PARAMETERGENERATION_H

#include <iostream>
#include <string>
#include <unordered_set>
// #include "/data/zjx/dacpp/clang/tools/translator/dacppLib/include/Tensor.hpp"
#include "sub_template.h"
#include "DataReconstructor.h"
#include "ReconTensor.h"

template <typename ImplType,int N>
class ParameterGeneration
{
    public:
        ParameterGeneration(){

        }
        // //生成算子的划分数 分区算子
        // int init_operetor_splitnumber(RegularSlice si,dacpp::Tensor<ImplType> tensor)
        // {  
        //     int split_num = (tensor.getShape(si.dimId) - si.size) / si.stride + 1;
        //     return split_num;
        // }

        // //生成算子的划分数 降维算子
        // int init_operetor_splitnumber(Index si,dacpp::Tensor<ImplType> tensor)
        // {  
        //     int split_num = tensor.getShape(si.dimId); //算子作用维度的划分数
        //     return split_num;
        // }

        //生成算子的划分数 不用区分分区算子和降维算子 直接用Dac_Op去计算就可以了
        int init_operetor_splitnumber(Dac_Op si,dacpp::Tensor<ImplType,N> tensor)
        {
            int split_num = (tensor.getShape(si.dimId) - si.size) / si.stride + 1;
            return split_num;
        }

        int init_operetor_splitnumber(Dac_Op si,DataInfo data_info)
        {
            //int split_num = (tensor.getShape(si.dimId) - si.size) / si.stride + 1;
            int split_num = (data_info.dimLength[si.dimId] - si.size) / si.stride + 1;
            return split_num;
        }

        //生成设备内存的分配大小 支持情况：mat[分区][分区] mat[分区][降维] mat[分区][] mat[降维][]
        //同时也是设备和主机之间内存移动大小 q.memcpy()里面的SIZE
        int init_device_memory_size(dacpp::Tensor<ImplType,N> tensor,Dac_Ops ops)
        {
            int result = 1;//初始化结果为1 
            std::unordered_set<int> mySet;//用来存储算子作用的维度
            for(int i = 0;i < ops.size;i ++)
            {
                int dimId = ops[i].dimId;//拿到算子的维度
                mySet.insert(dimId);//将算子作用的维度放入集合中
                int split_num = (tensor.getShape(dimId) - ops[i].size) / ops[i].stride + 1;//计算算子的划分数
                int length = split_num * ops[i].size;//划分数乘以划分的大小
                result *= length;
            }
            for(int i = 0;i < tensor.getDim();i ++)
            {
                if (mySet.find(i) == mySet.end()) 
                {
                    //i不存在也就是说算子没有作用这个维度 这个维度是保型算子
                    result *= tensor.getShape(i);
                }
            }
            return result;
        }

        int init_device_memory_size(DataInfo data_info,Dac_Ops ops)
        {
            int result = 1;//初始化结果为1 
            std::unordered_set<int> mySet;//用来存储算子作用的维度
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

        //生成设备内存分配大小 支持情况：mat[][]
        int init_device_memory_size(dacpp::Tensor<ImplType,N> tensor)
        {
            return tensor.getSize();
        }

        int init_device_memory_size(DataInfo data_info)
        {
            int result = 1;
            for(int i = 0;i <= data_info.dim;i ++)
            {
                result *= data_info.dimLength[i]; 
            }
            return result;
            //return tensor.getSize();
        }

        /*下面函数已废弃  接口太复杂了*/
        //生成设备内存的分配大小 支持情况：数据重组时中间需要的内存 内核函数中localsize的大小 为啥了不知道 反正就这样算的
        //传入四个参数 分别是输入的tensor组 输出的tensor 输入tensor的算子组的数组 输出tensor的算子组
        // int init_device_memory_size(std::vector<dacpp::Tensor<ImplType,N>> tensor_in,dacpp::Tensor<ImplType,N> tensor_out,std::vector<Dac_Ops> ops_in,Dac_Ops ops_out)
        // {
        //     int in_op_product = 1;//输入算子划分数的乘积
        //     std::unordered_set<std::string> mySet;//输入算子是否相同 现在只能拿名字去判断
        //     //下面遍历所有输入算子组里面的算子
        //     for(int i = 0;i < ops_in.size();i ++)
        //     {
        //         for(int j = 0;j < ops_in[i].size;j ++)
        //         {
        //             if(mySet.find(ops_in[i][j].name) == mySet.end()) //如果这个算子不存在于集合中
        //             {
        //                 mySet.insert(ops_in[i][j].name);//插入集合中去
        //                 in_op_product *= init_operetor_splitnumber(ops_in[i].DacOps[j],tensor_in[i]);
        //             }
        //         }
        //     }

        //     int out_op_product = 1;//输出就一个计算比较简单
        //     for(int i = 0;i < ops_out.size;i ++)
        //     {
        //         out_op_product *= init_operetor_splitnumber(ops_out.DacOps[i],tensor_out);
        //     }
        //     return init_device_memory_size(tensor_out,ops_out) * in_op_product / out_op_product;
        // }
        /*上面函数已废弃*/

        //生成设备内存的分配大小 支持情况：数据重组时中间需要的内存 
        //Dac_Ops ops_in中所有的算子是不同的，由抽象语法树后端进行去重
        int init_device_memory_size(Dac_Ops ops_in,Dac_Ops ops_out,dacpp::Tensor<ImplType,N> tensor_out)
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
            return init_device_memory_size(tensor_out,ops_out) * in_op_product / out_op_product;
        }

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
            return init_device_memory_size(data_info,ops_out) * in_op_product / out_op_product;
        }

        //生成开辟工作项多少 localsize
        //实际上是输入算子所有划分数的乘积 或者说数据元组的个数（数据单元组成数据元组）
        //由后端对算子组进行去重
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
                ops.DacOps[i].setSplitLength(ops.DacOps[i - 1].split_length / ops.DacOps[i].split_size);
            }
        }

        //生成SplitLength的矩阵
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
            return ops.DacOps[ops.size - 1].split_length;//返回最后一个算子的划分数
        }

        // bool judge_reduction(Dac_Ops ops_in,Dac_Ops ops_out)
        // {
        //     int in_op_product = 1;//输入算子划分数的乘积
        //     for(int i = 0;i < ops_in.size;i ++)
        //     {
        //         in_op_product *= ops_in.DacOps[i].split_size; //spilit在前面初始化算子的时候已经完成
        //     }
        //     int out_op_product = 1;//输出算子划分数的乘积
        //     for(int i = 0;i < ops_out.size;i ++)
        //     {
        //         out_op_product *= ops_out.DacOps[i].split_size; //spilit在前面初始化算子的时候已经完成
        //     }
        //     return (in_op_product / out_op_product > 1);
        // }
        std::vector<int> init_partition_data_shape(DataInfo data_info,Dac_Ops ops) {
            std::vector<int> tmp=data_info.dimLength;
            for(int i=0;i<ops.size;i++) {
                tmp[ops[i].dimId]=ops[i].size;
            }
            std::vector<int> res;
            for(int i=0;i<tmp.size();i++) {
                if(tmp[i]==1) continue;
                res.push_back(tmp[i]);
            }
            return res;
        }
};

#endif
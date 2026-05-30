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

        int init_operetor_splitnumber(Dac_Op si,DataInfo data_info)
        {
            int split_num = (data_info.dimLength[si.dimId] - si.size) / si.stride + 1;
            return split_num;
        }

        int init_device_memory_size(DataInfo data_info,Dac_Ops ops)
        {
            int result = 1;
            std::unordered_set<int> mySet;
            for(int i = 0;i < ops.size;i ++)
            {
                int dimId = ops[i].dimId;
                mySet.insert(dimId);
                int split_num = (data_info.dimLength[dimId] - ops[i].size) / ops[i].stride + 1;
                int length = split_num * ops[i].size;
                result *= length;
            }
            for(int i = 0;i < data_info.dim;i ++)
            {
                if (mySet.find(i) == mySet.end())
                {

                    result *= data_info.dimLength[i];
                }
            }
            return result;
        }

        int init_device_memory_size(DataInfo data_info)
        {
            int result = 1;
            for(int i = 0;i < data_info.dim;i ++)
            {
                result *= data_info.dimLength[i];
            }
            return result;
        }

        int init_device_memory_size(Dac_Ops ops_in,Dac_Ops ops_out,DataInfo data_info)
        {
            int in_op_product = 1;
            for(int i = 0;i < ops_in.size;i ++)
            {
                in_op_product *= ops_in.DacOps[i].split_size;
            }
            int out_op_product = 1;
            for(int i = 0;i < ops_out.size;i ++)
            {
                out_op_product *= ops_out.DacOps[i].split_size;
            }

            // Divide first to reduce overflow risk in intermediate products.
            return in_op_product / out_op_product * init_device_memory_size(data_info,ops_out);
        }

        int init_work_item_size(Dac_Ops in_ops)
        {
            int result = 1;
            for(int i = 0;i < in_ops.size;i ++)
            {
                result *= in_ops.DacOps[i].split_size;
            }
            return result;
        }

        void init_op_split_length(Dac_Ops& ops,int size)
        {
            if(ops.size == 0) return;
            ops.DacOps[0].setSplitLength(size / ops.DacOps[0].split_size);
            for(int i = 1;i < ops.size;i ++)
            {
                ops.DacOps[i].setSplitLength(ops.DacOps[i - 1].split_length / ops.DacOps[i].split_size);
            }
        }

        void init_split_length_martix(int Rows,int Cols,int* matrix,std::vector<Dac_Ops> ops_s)
        {
            for(int i = 0;i < Rows; i++)
            {
                for(int j = 0;j < ops_s[i].size;j ++)
                {
                    matrix[i * Cols + j] = ops_s[i].DacOps[j].split_length;
                }
            }
        }

        int init_reduction_split_size(Dac_Ops ops_in,Dac_Ops ops_out)
        {
            int in_op_product = 1;
            for(int i = 0;i < ops_in.size;i ++)
            {
                in_op_product *= ops_in.DacOps[i].split_size;
            }
            int out_op_product = 1;
            for(int i = 0;i < ops_out.size;i ++)
            {
                out_op_product *= ops_out.DacOps[i].split_size;
            }
            return in_op_product / out_op_product;
        }

        int init_reduction_split_length(Dac_Ops ops)
        {
            if(ops.size == 0)
                return 1;
            return ops.DacOps[ops.size - 1].split_length;
        }

        std::vector<int> init_partition_data_shape(DataInfo data_info,Dac_Ops ops) {
            std::vector<int> tmp=data_info.dimLength;
            for(int i=0;i<ops.size;i++) {
                tmp[ops[i].dimId]=ops[i].size;
            }
            std::vector<int> res;
            for(int i=0;i<tmp.size();i++) {

                res.push_back(tmp[i]);
            }
            return res;
        }

        int init_Data_SplitNum(Dac_Ops ops,DataInfo data_info){
            int result = 1;
            for(int i = 0;i < ops.size;i ++)
            {
                int dimId = ops[i].dimId;
                int split_num = (data_info.dimLength[dimId] - ops[i].size) / ops[i].stride + 1;

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

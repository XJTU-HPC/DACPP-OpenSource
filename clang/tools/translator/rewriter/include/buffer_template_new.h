#ifndef BUFFER_TEMPLATE_H
#define BUFFER_TEMPLATE_H

#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include <set>              
#include <unordered_map>
#include"dacInfo.h"
#include<unordered_map>
#include<set>

using std::string;

namespace BUFFER_TEMPLATE {
    extern std::string BUFFER_ACCESSOR_LIST;
    extern std::string ACCESSOR_POINTER_LIST;
    extern const char *BUFFER_ACCESSOR_Template;
    extern const char *ACCESSOR_POINTER_Template;
    extern const char *DAC2SYCL_Template_2;
    extern const char *DATA_INFO_INIT_Template;
    extern const char *OP_REGULAR_SLICE_INIT_Template2;
    extern const char *OP_INDEX_INIT_Template2;
    extern const char *PARA_GENE_Template;
    extern const char *OPS_INIT_Template;
    extern const char *ADD_OP2OPS_Template;
    extern const char *DEVICE_MEM_SIZE_Generate_Template1;
    extern const char *DEVICE_MEM_SIZE_Generate_Template2;
    extern const char *DEVICE_MEM_SIZE_Generate_Template3;
    extern const char *INIT_SPLIT_LENGTH_Template;
    extern const char *INIT_SPLIT_LENGTH_MATRIX_Template;
    extern const char *DECLARE_DACOPS_VECTOR_Template;
    extern const char *ADD_DACOPS2VECTOR_Template;
    extern const char *INIT_WORK_ITEM_NUMBER_Template;
    extern const char *INIT_REDUCTION_SPLIT_SIZE_Template;
    extern const char *INIT_REDUCTION_SPLIT_LENGTH_Template;
    extern const char *DEVICE_MEM_ALLOC_Template;
    extern const char *DEVICE_MEM_ALLOC_REDUCTION_Template;
    extern const char *DATA_ASSOC_COMP_Template;
    extern const char *D2B_MOV_BUFFER_Template;
    extern const char *INIT_HOST_MEMORY_Template;
    extern const char *DEVICE_DATA_INIT_Template;
    extern const char *DATA_RECON_BUFFER_Template;
    extern const char *DATA_OPS_INIT_Template;
    extern const char *OP_PUSH_BACK2OPS_Template;
    extern const char *KERNEL_EXECUTE_Template;
    extern const char *ACCESSOR_INIT_Template;
    extern const char *INDEX_INIT_Template;
    extern const char *CALC_EMBED_Template;
    extern const char *REDUCTION_Template_Span;
    extern const char *RESULT_B2H_MOV_Template;

    //函数声明
    void replaceTextInString(std::string& text, const std::string &find, const std::string &replace);
    std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements);
    std::string CodeGen_DAC2SYCL2(std::string dacShellName, std::string dacShellParams, std::string opInit, std::string parameter_generate, std::string deviceMemAlloc, std::string dataAssocComp);
    // std::string CodeGen_DataInfoInit(std::string name);
    std::string CodeGen_DataInfoInit(std::string name, std::string dim);
    std::string CodeGen_RegularSliceInit2(std::string opName, std::string size, std::string stride, std::string dim_id, std::string DATA_INFO_NAME);
    std::string CodeGen_IndexInit2(std::string opName, std::string dim_id, std::string DATA_INFO_NAME);
    // std::string CodeGen_ParameterGenerate(std::string InitOPS, std::string InitDeviceMemorySize, std::string InitSplitLength, std::string InitSpilitLengthMatrix, std::string ItemNumber, std::string InitReductionSplitSize, std::string InitReductionSplitLength);
    // std::string CodeGen_ParameterGenerate(std::string InitOPS, std::string InitDeviceMemorySize, std::string InitSplitLength, std::string InitSpilitLengthMatrix, std::string ItemNumber);
    std::string CodeGen_ParameterGenerate(std::string InitOPS, std::string InitDeviceMemorySize, std::string InitSplitLength, std::string ItemNumber);
    std::string CodeGen_DataOpsInit2(std::string OPS_NAME, std::string ADD_OP2OPS);
    std::string CodeGen_AddOp2Ops(std::string OP_NAME, std::string DIM_ID, std::string OPS_NAME);
    std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string DATA_INFO_NAME, std::string DACOPS_NAME);
    std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string DATA_INFO_NAME);
    std::string CodeGen_DeviceMemSizeGenerate(std::string NAME, std::string IN_DAC_OPS_NAME, std::string OUT_DAC_OPS_NAME, std::string DATA_INFO_NAME);
    std::string CodeGen_Init_Split_Length(std::string OPS_NAME, std::string SIZE);
    std::string CodeGen_Init_Split_Length_Matrix(std::string DECLARE_DACOPS_VECTOR, std::string ROW, std::string COL, std::string OPS_S_NAME);
    std::string CodeGen_Declare_DacOps_Vector(std::string OPSS_NAME, std::string PUSH_BACK_DAC_OPS);
    std::string CodeGen_Add_DacOps2Vector(std::string OPSS_NAME, std::string OPS_NAME);
    std::string CodeGen_Init_Work_Item_Number(std::string NAME, std::string OPS_NAME);
    // std::string CodeGen_Init_Reduction_Split_Size(std::string NAME, std::string OPS_IN, std::string OPS_OUT);
    // std::string CodeGen_Init_Reduction_Split_Length(std::string NAME, std::string OPS_NAME);
    std::string CodeGen_DeviceMemAlloc(std::string type, std::string name, std::string size);
    std::string CodeGen_DeviceMemAllocReduction(std::string type, std::string name, std::string size);
    std::string CodeGen_DataAssocComp(std::string H2DMemMove, std::string dataRecon, std::string kernelExecute, std::string reduction, std::string D2HMemMove);
    std::string CodeGen_D2B_Mov_Buffer(std::string TYPE, std::string NAME, std::string SIZE);
    std::string CodeGen_Init_Host_Memory(std::string TYPE, std::string NAME);
    std::string CodeGen_DeviceDataInit(std::string type, std::string name, std::string size);
    std::string CodeGen_DataReconstruct(std::string type, std::string name, std::string size, std::string dataOpsInit);
    std::string CodeGen_DataReconstruct1(std::string type, std::string name, std::string size, std::string dataOpsInit);
    std::string CodeGen_DataOpsInit(std::string name, std::string opPushBack2Ops);
    std::string CodeGen_OpPushBack2Ops(std::string name, std::string opName, std::string dimId);
    // std::string CodeGen_KernelExecute(std::string SplitSize, std::string AccessorInit, std::string IndexInit, std::string CalcEmbed);
    // std::string CodeGen_KernelExecute(std::string SplitSize, std::string AccessorInit, std::string IndexInit,std::string ACCESSOR_LIST1, std::string ACCESSOR_LIST2, std::string CalcEmbed);
    std::string CodeGen_KernelExecute(std::string SplitSize, std::string AccessorInit, std::string IndexInit, std::string getpos, std::string ACCESSOR_LIST1, std::string ACCESSOR_LIST2, std::string CalcEmbed);
    std::string CodeGen_AccessorInit(std::string name);
    // std::string CodeGen_AccessorInit0(std::string name);
    std::string CodeGen_AccessorInit0_read(std::string name);
    std::string CodeGen_AccessorInit0_write(std::string name);
    std::string CodeGen_AccessorInit1(std::string name);
    std::string CodeGen_IndexInit2(Dac_Ops ops, std::vector<std::string> sets, std::vector<std::string> offsets);
    // std::string CodeGen_CalcEmbed2(std::string Name, Args args, std::vector<std::string> accessor_names);
    std::string CodeGen_CalcEmbed2(std::string Name, std::vector<std::string> splits, std::vector<int> splitNum,std::vector<std::string> accessor_names);
    std::string CodeGen_Reduction_Span(std::string SpanSize, std::string SplitSize, std::string SplitLength, std::string Name, std::string Type, std::string ReductionRule);
    std::string CodeGen_Result_B2H_Mov(std::string NAME, std::string SIZE);
    std::string CodeGen_getpos1(std::string name, std::string opsname, std::string idx);
    std::string CodeGen_getpos0(std::string name, std::string idx);
}

#endif
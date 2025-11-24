#ifndef USM_TEMPLATE_H
#define USM_TEMPLATE_H

#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include <unordered_map>
#include <set>
#include"dacInfo.h"

namespace USM_TEMPLATE {

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
    extern const char *H2D_MEM_MOV_Template3;
    extern const char *H2D_MEM_MOV_Template_Out;
    extern const char *H2D_MEM_MOV_Template;
    extern const char *DATA_RECON_Template2;
    extern const char *DATA_RECON_Template3;
    extern const char *DATA_OPS_INIT_Template;
    extern const char *OP_PUSH_BACK2OPS_Template;
    extern const char *KERNEL_EXECUTE_Template1;
    extern const char *KERNEL_EXECUTE_Template2;
    extern const char *ACCESSOR_INIT_Template;
    extern const char *INDEX_INIT_Template;
    extern const char *CALC_EMBED_Template;
    extern const char *START_OFFSET_INIT_Template;
    extern const char *VIRTUALMAP_INIT_Template;
    extern const char *SWAP_OPERATE_INIT_Template;
    extern const char *START_OFFSET_OPERATE_INIT_Template;
    extern const char *SWAP_START_INIT_Template;
    extern const char *ADD_START_INIT_Template;
    extern const char *SUB_START_INIT_Template;
    extern const char *NEW_DATA_INIT_Template;
    extern const char *REDUCTION_Template_Span;
    extern const char *REDUCTION_Template_Span1;
    extern const char *D2H_MEM_MOV_3_Template;
    extern const char *D2H_MEM_MOV_4_Template;
    extern const char *MEM_FREE_Template;
    extern const char *DEVICE_DATA_INIT_Template;
    
    // 文本处理函数
    void replaceTextInString(std::string& text, const std::string &find, const std::string &replace);
    std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements);

    // SYCL代码生成函数
    std::string CodeGen_DAC2SYCL2(std::string dacShellName, std::string dacShellParams, std::string opInit, std::string parameter_generate, std::string deviceMemAlloc, std::string dataAssocComp, std::string memFree);
    std::string CodeGen_DataInfoInit(std::string name);
    std::string CodeGen_RegularSliceInit2(std::string opName, std::string size, std::string stride, std::string dim_id, std::string DATA_INFO_NAME);
    std::string CodeGen_IndexInit2(std::string opName, std::string dim_id, std::string DATA_INFO_NAME);
    // std::string CodeGen_ParameterGenerate(std::string InitOPS, std::string InitDeviceMemorySize, std::string InitSplitLength, std::string InitSpilitLengthMatrix, std::string ItemNumber, std::string InitReductionSplitSize, std::string InitReductionSplitLength);
    std::string CodeGen_ParameterGenerate(std::string InitOPS, std::string InitDeviceMemorySize, std::string InitSplitLength, std::string InitSpilitLengthMatrix, std::string ItemNumber);
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
    std::string CodeGen_H2DMemMov3(std::string type, std::string name, std::string size);
    std::string CodeGen_H2DMemMov_Out(std::string type, std::string name, std::string size);
    std::string CodeGen_H2DMemMov(std::string type, std::string name, std::string size);
    std::string CodeGen_DataReconstruct(std::string type, std::string name, std::string size, std::string dataOpsInit);
    std::string CodeGen_DataReconstruct_Dacfor(std::string type, std::string name, std::string size, std::string dataOpsInit);
    std::string CodeGen_DataOpsInit(std::string name, std::string opPushBack2Ops);
    std::string CodeGen_OpPushBack2Ops(std::string name, std::string opName, std::string dimId);
    std::string CodeGen_KernelExecute(std::string SplitSize, std::string AccessorInit, std::string IndexInit, std::string CalcEmbed);
    std::string CodeGen_KernelExecute(std::string StartOffsetInit, std::string TimeSteps, std::string AccessorInit, std::string VirtualMapInit, std::string IndexInit, std::string CalcEmbed, std::string SwapOperate, std::string StartOffsetOperate, std::string NewDataInit);
    std::string CodeGen_AccessorInit(std::string name);
    std::string CodeGen_StartOffsetInit(std::string name, std::string num);
    std::string CodeGen_VirtualMapInit(std::string name);
    std::string CodeGen_SwapOperateInit(std::string name1, std::string name2);
    std::string CodeGen_StartOffsetOperateInit(std::string swap_start, std::string add_start, std::string sub_start);
    std::string CodeGen_SwapStartInit(std::string name1, std::string name2);
    std::string CodeGen_AddStartInit(std::string name);
    std::string CodeGen_SubStartInit(std::string name);
    std::string CodeGen_NewDataInit(std::string name, std::string type);
    std::string CodeGen_Reduction_Span(std::string SpanSize, std::string SplitSize, std::string SplitLength, std::string Name, std::string Type, std::string ReductionRule);
    std::string CodeGen_Reduction_Span1(std::string SpanSize, std::string SplitSize, std::string SplitLength, std::string Name, std::string Type, std::string ReductionRule);
    std::string CodeGen_D2HMemMov(std::string Name, std::string Type, std::string Size, bool isReduction);
    std::string CodeGen_MemFree(std::string Name);
    std::string CodeGen_DeviceDataInit(std::string type, std::string name, std::string size);
    std::string CodeGen_IndexInit2(Dac_Ops ops, std::vector<std::string> sets, std::vector<std::string> offsets);
    std::string CodeGen_CalcEmbed2(std::string Name, Args args, std::vector<std::string> accessor_names);
    std::string CodeGen_CalcEmbed3(std::string Name, Args args, std::vector<std::string> accessor_names);

}

#endif
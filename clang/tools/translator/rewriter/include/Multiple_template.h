#ifndef MULTIPLE_TEMPLATE_H
#define MULTIPLE_TEMPLATE_H

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include"dacInfo.h"

namespace MULTIPLE_TEMPLATE {

    void replaceTextInString(std::string& text, const std::string &find, const std::string &replace);
	
    std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements);


    //------------------------------USM-----------------------------------
    
    extern const char *DEVICE_MEM_ALLOC_Template;
    extern const char *DEVICE_MEM_ALLOC_REDUCTION_Template;
    extern const char *H2D_MEM_MOV_Template;
    extern const char *KERNEL_EXECUTE_Template;
    extern const char *Store_Mem_Template; 
    extern const char *REDUCTION_Template_Span;
    extern const char *Load_Mem_Template;
    extern const char *D2H_MEM_MOV_1_Template;
    extern const char *D2H_MEM_MOV_2_Template;
    extern const char* Kernel_Template;
    extern const char *MEM_FREE_Template;
    std::string CodeGen_DataInfoInit(std::string name);
    std::string CodeGen_IndexInit2(std::string opName,std::string dim_id,std::string DATA_INFO_NAME);
    std::string CodeGen_IndexInit2(Dac_Ops ops,std::vector<std::string> sets,std::vector<std::string> offsets);
    std::string CodeGen_RegularSliceInit2(std::string opName,std::string size,std::string stride,std::string dim_id,std::string DATA_INFO_NAME);
    std::string CodeGen_AddOp2Ops(std::string OP_NAME,std::string DIM_ID,std::string OPS_NAME);
    std::string CodeGen_DataOpsInit(std::string name,std::string opPushBack2Ops);
    std::string CodeGen_DataOpsInit2(std::string OPS_NAME,std::string ADD_OP2OPS);
    std::string CodeGen_DeviceMemSizeGenerate(std::string NAME,std::string DATA_INFO_NAME,std::string DACOPS_NAME);
    std::string CodeGen_DeviceMemSizeGenerate(std::string NAME,std::string DATA_INFO_NAME);
    std::string CodeGen_DeviceMemSizeGenerate(std::string NAME,std::string IN_DAC_OPS_NAME,std::string OUT_DAC_OPS_NAME,std::string DATA_INFO_NAME);
    std::string CodeGen_Init_Split_Length(std::string OPS_NAME,std::string SIZE);
    std::string CodeGen_Add_DacOps2Vector(std::string OPSS_NAME,std::string OPS_NAME);
    std::string CodeGen_Declare_DacOps_Vector(std::string OPSS_NAME,std::string PUSH_BACK_DAC_OPS);
    std::string CodeGen_Init_Split_Length_Matrix(std::string DECLARE_DACOPS_VECTOR,std::string ROW,std::string COL,std::string OPS_S_NAME);
    std::string CodeGen_Init_Work_Item_Number(std::string NAME,std::string OPS_NAME);
    std::string CodeGen_Init_Reduction_Split_Size(std::string NAME,std::string OPS_IN,std::string OPS_OUT);
    std::string CodeGen_Init_Reduction_Split_Length(std::string NAME,std::string OPS_NAME);
    std::string CodeGen_ParameterGenerate(std::string InitOPS,std::string InitDeviceMemorySize,std::string InitSplitLength,std::string InitSpilitLengthMatrix,std::string ItemNumber,std::string InitReductionSplitSize,std::string InitReductionSplitLength);
    std::string CodeGen_OpPushBack2Ops(std::string name,std::string opName,std::string dimId);
    std::string CodeGen_DataReconstruct(std::string type,std::string name,std::string size,std::string dataOpsInit, bool isOut);
    std::string CodeGen_CalcEmbed2(std::string Name,Args args, std::vector<std::string> accessor_names,std::vector<bool> isFull);
    std::string CodeGen_DataSplit(std::string Name);
    std::string CodeGen_DataDeviceMalloc(std::string Name,bool isFull);
    std::string CodeGen_DeviceMem(std::string type,std::string name);
    std::string CodeGen_DeviceMemReduction(std::string type,std::string name);
    std::string CodeGen_DeviceDataHaveMalloc(std::string name);
    std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size);
    std::string CodeGen_DeviceMemAllocOutData(std::string type,std::string name,std::string size); 
    std::string CodeGen_DeviceMemAllocReduction(std::string  type,std::string name,std::string size);
    // std::string CodeGen_H2DMemMov(std::string type,std::string name,std::string size);
    std::string CodeGen_H2DMemMov(std::string type,std::string name,std::string size,bool isFull);
    std::string CodeGen_KernelExecute(std::string Calc_Size,std::string SplitSize,std::string AccessorInit,std::string IndexInit,std::string CalcEmbed);
    std::string CodeGen_StoreDeviceMem(std::string name);
    std::string CodeGen_StoreDeviceMemReduction(std::string name);
    std::string CodeGen_LoadDeviceMem(std::string type,std::string name);
    std::string CodeGen_LoadDeviceMemReduction(std::string type,std::string name);
    std::string CodeGen_Reduction_Span(std::string SpanSize,std::string SplitSize,std::string SplitLength,std::string Name,std::string Type,std::string ReductionRule);
    std::string CodeGen_D2HMemMov(std::string Name,std::string Type,std::string Size,bool isReduction);
    std::string CodeGen_MemFree(std::string Name);
    std::string CodeGen_D2HMemMovF(std::string Name,std::string Type,std::string Size);
    std::string CodeGen_JustAMiddle(std::string type,std::string name,std::string size);
    //整合模板
    std::string CodeGen_Kernel(std::string Device_Mem,std::string CantFindBetterWay,std::string DeviceMemAlloc,std::string H2DMemMove,std::string KernelExecute,std::string Store_Device_Mem,std::string Load_Device_Mem,std::string Reduction,std::string D2HMemMove,std::string MemFree);
    std::string CodeGen_DAC2SYCL2(std::string dacShellName, std::string dacShellParams,std::string opInit, std::string parameter_generate, std::string deviceMemAlloc, std::string dataAssocComp);
}
// namespace Multiple_TEMPLATE
#endif
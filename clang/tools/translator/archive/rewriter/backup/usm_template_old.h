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
    extern const char *DEVICE_MEM_ALLOC_Template;
    extern const char *DEVICE_MEM_ALLOC_REDUCTION_Template;
    extern const char *H2D_MEM_MOV_Template;
    extern const char *DEVICE_DATA_INIT_Template;
    extern const char *KERNEL_EXECUTE_Template;
    extern const char *ACCESSOR_INIT_Template;
    extern const char *REDUCTION_Template_Span;
    extern const char *D2H_MEM_MOV_1_Template;
    extern const char *D2H_MEM_MOV_2_Template;
    extern const char *MEM_FREE_Template;

    void replaceTextInString(std::string& text, const std::string &find, const std::string &replace);

    std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements);

    std::string CodeGen_DAC2SYCL2(std::string dacShellName, std::string dacShellParams,std::string opInit, std::string parameter_generate, std::string deviceMemAlloc, std::string dataAssocComp, std::string memFree);

    std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size);

    std::string CodeGen_DeviceMemAllocReduction(std::string  type,std::string name,std::string size);

    std::string CodeGen_H2DMemMov(std::string type,std::string name,std::string size);

    std::string CodeGen_DeviceDataInit(std::string type,std::string name,std::string size);

    std::string CodeGen_KernelExecute(std::string SplitSize,std::string AccessorInit,std::string IndexInit,std::string CalcEmbed);

    std::string CodeGen_AccessorInit(std::string name);

    std::string CodeGen_Reduction_Span(std::string SpanSize,std::string SplitSize,std::string SplitLength,std::string Name,std::string Type,std::string ReductionRule);

    std::string CodeGen_D2HMemMov(std::string Name,std::string Type,std::string Size,bool isReduction);

    std::string CodeGen_MemFree(std::string Name);

    //dac_for用到的内核
    std::string CodeGen_KernelExecute(std::string StartOffsetInit,std::string TimeSteps,std::string AccessorInit,std::string VirtualMapInit,std::string IndexInit,std::string CalcEmbed,std::string SwapOperate,std::string StartOffsetOperate,std::string NewDataInit);

    std::string CodeGen_StartOffsetInit(std::string name,std::string num);

    std::string CodeGen_AccessorInit(std::string name);

    std::string CodeGen_VirtualMapInit(std::string name);

    std::string CodeGen_CalcEmbed2(std::string Name,Args args, std::vector<std::string> accessor_names);

    std::string CodeGen_CalcEmbed3(std::string Name,Args args,std::vector<std::string> accessor_names);

    std::string CodeGen_SwapOperateInit(std::string name1,std::string name2);

    std::string CodeGen_StartOffsetOperateInit(std::string swap_start,std::string add_start,std::string sub_start);

    std::string CodeGen_SwapStartInit(std::string name1,std::string name2);

    std::string CodeGen_AddStartInit(std::string name);

    std::string CodeGen_SubStartInit(std::string name);

    std::string CodeGen_NewDataInit(std::string name,std::string type);



    std::string CodeGen_DataReconstruct_Dacfor(std::string type,std::string name,std::string size,std::string dataOpsInit);

    std::string CodeGen_OpPushBack2Ops(std::string name, std::string opName, std::string dimId);

    std::string CodeGen_OpPushBack2Tool(std::string name, std::string opName, std::string dimId);

    std::string CodeGen_DataOpsInit(std::string name,std::string opPushBack2Ops);

    std::string CodeGen_DataReconstructOpPush(std::string name,std::string opPushBack2Tool);

    std::string CodeGen_OpPopFromTool(std::string name);

    std::string CodeGen_DataReconstructOpPop(std::string name,std::string opPopFromTool);

    std::string CodeGen_D2HMemMov(std::string Name,std::string Type,std::string Size,bool isReduction);

    std::string CodeGen_H2DMemMov3(std::string type,std::string name,std::string size);
    std::string CodeGen_H2DMemMov_Out(std::string type,std::string name,std::string size);

}

#endif
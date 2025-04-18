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

}

#endif
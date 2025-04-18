#ifndef BUFFER_TEMPLATE_H
#define BUFFER_TEMPLATE_H

#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include"dacInfo.h"

using std::string;

namespace BUFFER_TEMPLATE {
    extern const char *DAC2SYCL_Template;
    extern const char *BUFFER_ACCESSOR_Template;
    extern const char *ACCESSOR_POINTER_Template;
    extern const char *DEVICE_MEM_ALLOC_Template;
    extern const char *DEVICE_MEM_ALLOC_REDUCTION_Template;
    extern const char *H2D_MEM_MOV_Template;
    extern const char *DEVICE_DATA_INIT_Template;
    extern const char *ACCESSOR_INIT_Template;
    extern const char *KERNEL_EXECUTE_Template;
    extern const char *REDUCTION_Template;
    extern const char *REDUCTION_Template_Span;
    extern const char *D2H_MEM_MOV_1_Template;
    extern const char *D2H_MEM_MOV_2_Template;

    void replaceTextInString(std::string& text, const std::string &find, const std::string &replace);

    std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements);

    std::string CodeGen_DAC2SYCL(std::string dacShellName,std::string dacShellParams,std::string opInit,std::string dataRecon,
	    std::string deviceMemAlloc,std::string H2DMemMove,std::string kernelExecute,std::string reduction,std::string D2HMemMove);

    std::string CodeGen_DeviceMemAlloc(std::string type,std::string name,std::string size);

    std::string CodeGen_DeviceMemAllocReduction(std::string  type,std::string name,std::string size);

    std::string CodeGen_H2DMemMov(std::string type,std::string name,std::string size);

    std::string CodeGen_DeviceDataInit(std::string type,std::string name,std::string size);

    std::string CodeGen_AccessorInit(std::string name);

    std::string CodeGen_KernelExecute_ArrayList(string SplitSize, std::string AccessorInit, string IndexInit, string CalcEmbed, std::initializer_list<string> values);

    std::string CodeGen_KernelExecute(std::string SplitSize,std::string AccessorInit, std::string IndexInit,std::string CalcEmbed);

    std::string CodeGen_Reduction_Span(std::string SpanSize,std::string SplitSize,std::string SplitLength,std::string Name,std::string Type,std::string ReductionRule);

    std::string CodeGen_D2HMemMov(std::string Name,std::string Type,std::string Size,bool isReduction);
}

#endif